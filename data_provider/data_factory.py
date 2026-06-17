from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom


def data_provider(args, flag):
    if args.data != 'custom':
        raise ValueError('This cleaned benchmark keeps only custom CSV datasets.')

    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = False if flag == 'test' else True

    data_set = Dataset_Custom(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
    )

    if args.percent < 1.0 and flag == 'train':
        import torch
        num_samples = int(len(data_set) * args.percent)
        indices = torch.randperm(len(data_set))[:num_samples]
        data_set = torch.utils.data.Subset(data_set, indices)
        print(f"Few-shot sampling: {args.percent * 100}% of data, {len(data_set)} samples")

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return data_set, data_loader
