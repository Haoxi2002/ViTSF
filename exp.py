import os
import time

import numpy as np
import torch
from torch import nn, optim

from data_provider.data_factory import data_provider
from models import PatchTST, N2V, DLinear, KAE_Informer, LSTM, Autoformer, MV_DTSF
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual


class Exp(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'PatchTST': PatchTST,
            'DLinear': DLinear,
            'N2V': N2V,
            'KAE_Informer': KAE_Informer,
            'LSTM': LSTM,
            'Autoformer': Autoformer,
            'MV_DTSF': MV_DTSF
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.gpu) if not self.args.use_multi_gpu else str(self.args.devices)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Using GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_num, batch_x_fig, batch_y, batch_x_mark, batch_y_mark, static) in enumerate(vali_loader):
                batch_x_num = batch_x_num.float().to(self.device)
                batch_x_fig = batch_x_fig.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                static = static.float().to(self.device)

                if self.args.use_fig:
                    outputs = self.model(batch_x_fig, static)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.seq_len // 2, :], dec_inp], dim=1).float().to(self.device)
                    outputs = self.model(batch_x_num, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]
                loss = criterion(pred, true)
                total_loss.append(loss.item())
            total_loss = np.average(total_loss)
            self.model.train()
            return total_loss

    def train(self, setting):
        train_data, train_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='vali')
        test_data, test_loader = data_provider(self.args, flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x_num, batch_x_fig, batch_y, batch_x_mark, batch_y_mark, static) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x_num = batch_x_num.float().to(self.device)
                batch_x_fig = batch_x_fig.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                static = static.float().to(self.device)

                if self.args.use_fig:
                    outputs = self.model(batch_x_fig, static)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.seq_len // 2, :], dec_inp], dim=1).float().to(self.device)
                    outputs = self.model(batch_x_num, batch_x_mark, dec_inp, batch_y_mark)
                loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} | Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = data_provider(self.args, flag='test')

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = total_params * 4 / (1024 ** 2)

        inference_times = []
        
        with torch.no_grad():
            for i, (batch_x_num, batch_x_fig, batch_y, batch_x_mark, batch_y_mark, static) in enumerate(test_loader):
                batch_x_num = batch_x_num.float().to(self.device)
                batch_x_fig = batch_x_fig.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                static = static.float().to(self.device)

                if self.args.use_gpu:
                    torch.cuda.synchronize()
                start_time = time.time()
                
                if self.args.use_fig:
                    outputs = self.model(batch_x_fig, static)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    outputs = self.model(batch_x_num, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_gpu:
                    torch.cuda.synchronize()
                end_time = time.time()
                inference_times.append(end_time - start_time)

                outputs = outputs[:, -self.args.pred_len:, :].detach().cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, :].detach().cpu().numpy()

                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
                if self.args.file_name.startswith('ECW'):
                    if i % 30 == 0:
                        for j in range(0, batch_x_num.shape[2] - 5, 5):
                            input = batch_x_num.detach().cpu().numpy()
                            gt = np.concatenate((input[0, :, j + i // 30], true[0, :, j + i // 30]), axis=0)
                            pd = np.concatenate((input[0, :, j + i // 30], pred[0, :, j + i // 30]), axis=0)
                            visual(gt, pd, os.path.join(folder_path, str(i) + '_' + str(j + i // 30) + '.png'))
                else:
                    if i % 20 == 0:
                        input = batch_x_num.detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mean_mse, var_mse, top_mse, bottom_mse, peak_mse, mean_mae, var_mae, top_mae, bottom_mae, peak_mae, rmse, mape, mspe = metric(preds, trues)
        print('mean_mse:{}, var_mse:{}, top_mse:{}, bottom_mse:{}, peak_mse:{}\nmean_mae:{}, var_mae:{}, top_mae:{}, bottom_mae:{}, peak_mae:{}'.format(mean_mse, var_mse, top_mse, bottom_mse, peak_mse, mean_mae, var_mae, top_mae, bottom_mae, peak_mae))

        avg_inference_time = np.mean(inference_times)
        print('Model Size: {:.2f} MB, Inference Time:  {:.4f}s/batch'.format(model_size_mb, avg_inference_time))
        
        f = open('result_forecast.txt', 'a')
        f.write(setting + '\n')
        f.write('mean_mse:{}, var_mse:{}, top_mse:{}, bottom_mse:{}, peak_mse:{}\nmean_mae:{}, var_mae:{}, top_mae:{}, bottom_mae:{}, peak_mae:{}'.format(mean_mse, var_mse, top_mse, bottom_mse, peak_mse, mean_mae, var_mae, top_mae, bottom_mae, peak_mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mean_mse, var_mse, top_mse, bottom_mse, peak_mse, mean_mae, var_mae, top_mae, bottom_mae, peak_mae, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
