import argparse
import random

import numpy as np
import torch

from exp import Exp

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='ViTSF')

    # basic config
    parser.add_argument('--task_id', type=str, default='test', help='task id')
    parser.add_argument('--data_dir', type=str, default='./data', help='root path to dataset')
    parser.add_argument('--file_name', type=str, default='ECW.csv', help='data file')
    parser.add_argument('--model', type=str, default='ViTSF', help='model name')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # data loader
    parser.add_argument('--target', type=str, default='plan_cpu', help='target feature in univariate task')
    parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=24, help='label sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

    # visual model define
    parser.add_argument('--h', type=int, default=48, help='height of figure')
    parser.add_argument('--hidden_dim', type=int, default=16, help='hidden dimension')
    parser.add_argument('--patch_size', type=int, nargs='+', default=(8, 8), help='patch size')
    parser.add_argument('--token_mlp_dim', type=int, default=512, help='token mlp dimension')
    parser.add_argument('--channel_mlp_dim', type=int, default=128, help='channel mlp dimension')
    parser.add_argument('--n_blocks', type=int, default=4, help='block numbers of backbone')

    # numerical model define
    parser.add_argument('--enc_in', type=int, default=797, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=797, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=797, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', default=False, action='store_true', help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2', help='device ids of multiple gpus')
    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        args.device = torch.device('cpu')
        print('Using CPU')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.use_fig = True if args.model in ['ViTSF', 'MV_DTSF'] else False

    print('Args: {}'.format(args))

    exp = Exp(args)
    setting = '{}_{}_{}_{}_{}_h{}_hd{}_ps{}_tmd{}_cmd{}_nb{}_dropout{}_lr{}'.format(
        args.task_id,
        args.model,
        args.file_name.split('.')[0],
        args.seq_len,
        args.pred_len,
        args.h,
        args.hidden_dim,
        args.patch_size,
        args.token_mlp_dim,
        args.channel_mlp_dim,
        args.n_blocks,
        args.dropout,
        args.learning_rate
    )
    print('>>>>>>>>>>>start training : {}>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>>>>>>>>>>testing : {}>>>>>>>>>>>>>>>>>'.format(setting))
    exp.test(setting)
    torch.cuda.empty_cache()
