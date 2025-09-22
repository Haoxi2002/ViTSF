from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Basic


def data_provider(args, flag):
    shuffle_flag = False if flag == 'test' else True
    batch_size = 1 if flag == 'test' else args.batch_size
    data_set = Dataset_Basic(args, flag)
    print(flag, len(data_set))
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False
    )
    return data_set, data_loader
