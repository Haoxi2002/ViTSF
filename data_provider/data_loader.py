import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class Dataset_Basic(Dataset):
    def __init__(self, args, flag='train'):
        self.args = args

        assert flag in ['train', 'vali', 'test']
        type_map = {'train': 0, 'vali': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.__read_data__()

    def data2Pixel(self, dataXIn, method='plot'):
        if method == 'plot':
            dataX = np.copy(dataXIn.T)
            feature = dataX.shape[0]
            lenX = dataX.shape[1]

            imgX = np.zeros([feature, self.args.h, lenX], dtype=np.float32)
            for i in range(feature):
                if np.min(dataX[i]) < np.max(dataX[i]):
                    data_line = 1 - (dataX[i] - np.min(dataX[i])) / (np.max(dataX[i]) - np.min(dataX[i]))
                    data_line = np.round(data_line * (self.args.h - 1)).astype(int)
                    for j in range(lenX):
                        center = data_line[j]
                        for h in range(self.args.h):
                            distance = abs(h - center)
                            if distance <= 2:
                                imgX[i][h][j] = np.exp(-distance / 2.0)
                else:
                    center = self.args.h // 2
                    imgX[i, center, :] = 1

                canvas = FigureCanvasAgg(plt.figure(figsize=(lenX / 100, self.args.h / 100)))
                plt.plot(dataX[i])
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['bottom'].set_visible(False)
                plt.gca().spines['left'].set_visible(False)
                plt.axis('off')
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                plt.margins(0, 0)
                canvas.draw()
                buf = canvas.buffer_rgba()
                img = np.dot(np.asarray(buf)[:, :, :3] / 255, [0.299, 0.587, 0.114])
                imgX[i, :img.shape[0], :img.shape[1]] = img
                plt.close()
            return imgX
        else:  # elif method == 'GAF':
            dataXIn = np.array(dataXIn)
            min_vals = np.min(dataXIn, axis=0, keepdims=True)
            max_vals = np.max(dataXIn, axis=0, keepdims=True)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1
            dataXIn = 2 * (dataXIn - min_vals) / range_vals - 1
            sqrt_terms = np.sqrt(1 - dataXIn**2)
            outer_a = np.einsum('ik,jk->kij', dataXIn, dataXIn)
            outer_sqrt = np.einsum('ik, jk->kij', sqrt_terms, sqrt_terms)
            return outer_a - outer_sqrt

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(str(os.path.join(self.args.data_dir, self.args.file_name)))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.args.seq_len, len(df_raw) - num_test - self.args.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.args.file_name == 'ECW.csv':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:  # self.args.features == 'S':
            df_data = df_raw[[self.args.target]]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        self.data_num = self.scaler.transform(df_data[border1:border2].values)
        if self.args.use_fig and self.args.method == 'plot':
            self.data_fig = self.data2Pixel(df_data.values, self.args.method)[:, :, border1:border2]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        self.data_stamp = df_stamp.drop(['date'], axis=1).values

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.args.seq_len
        r_begin = s_end
        r_end = r_begin + self.args.pred_len

        seq_x_num = self.data_num[s_begin:s_end]
        seq_x_fig = static = 0
        if self.args.use_fig:
            if self.args.method == 'plot':
                seq_x_fig = self.data_fig[:, :, s_begin:s_end]
            else:  # elif method == 'GAF':
                seq_x_fig = self.data2Pixel(seq_x_num, self.args.method)
            static = np.concatenate([
                np.amax(seq_x_num, axis=0)[:, np.newaxis],
                np.amin(seq_x_num, axis=0)[:, np.newaxis],
                np.median(seq_x_num, axis=0)[:, np.newaxis],
                np.mean(seq_x_num, axis=0)[:, np.newaxis],
                np.percentile(seq_x_num, 25, axis=0)[:, np.newaxis],
                np.percentile(seq_x_num, 75, axis=0)[:, np.newaxis],
                np.std(seq_x_num, axis=0)[:, np.newaxis]
            ], axis=1)
        seq_y = self.data_num[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x_num, seq_x_fig, seq_y, seq_x_mark, seq_y_mark, static

    def __len__(self):
        return len(self.data_num) - self.args.seq_len - self.args.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)