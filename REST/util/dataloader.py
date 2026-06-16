import numpy as np
from torch.utils.data import DataLoader

from model.images import LabeledImagesDataset, TextDataset
from util.util import BalancedBatchSampler


def get_online_dataloder(ng_index,global_step,data_dir,train_ok_set,train_false_ng_set,old_dataset,args):
    train_ng_set = None
    print(ng_index[:(global_step // args.save_step + 1) * args.ng_num])
    for index in ng_index[:(global_step // args.save_step + 1) * args.ng_num]:
        if train_ng_set is None:
            train_ng_set = LabeledImagesDataset(f'{data_dir}/train/ng/train_ng_{index}*',
                                                feature_folder=args.feature_folder,
                                                label=2, args=args)
        else:
            train_ng_set += LabeledImagesDataset(f'{data_dir}/train/ng/train_ng_{index}*',
                                                 feature_folder=args.feature_folder,
                                                 label=2, args=args)
    trainset = train_ok_set + train_false_ng_set + train_ng_set + old_dataset

    print(len(train_ok_set), len(train_false_ng_set), len(train_ng_set), len(old_dataset))

    idx_list = [np.arange(0, len(train_ok_set)),
                np.arange(len(train_ok_set), len(train_ok_set) + len(train_false_ng_set)),
                np.arange(len(train_ok_set) + len(train_false_ng_set),
                          len(train_ok_set) + len(train_false_ng_set) + len(train_ng_set)),
                np.arange(len(train_ok_set) + len(train_false_ng_set) + len(train_ng_set), len(trainset))]
    # Create sampler, dataset, loader

    train_sampler = BalancedBatchSampler(trainset, idx_list, batch_size_list=args.batch_size_list)

    train_loader = DataLoader(trainset, batch_sampler=train_sampler, num_workers=1, pin_memory=True)
    return train_loader

def get_online_text_dataloder(ng_index,global_step,data_dir,data_info,train_ok_set,train_false_ng_set,old_dataset,args):
    train_ng_set = None
    print(ng_index[:(global_step // args.save_step + 1) * args.ng_num])
    for index in ng_index[:(global_step // args.save_step + 1) * args.ng_num]:
        if train_ng_set is None:
            train_ng_set = TextDataset(f'{data_dir}/train/ng/train_ng_{index}*', data_info,
                                       feature_folder=args.feature_folder,
                                       label=2, args=args)
        else:
            train_ng_set += TextDataset(f'{data_dir}/train/ng/train_ng_{index}*',
                                        data_info, feature_folder=args.feature_folder,
                                        label=2, args=args)
    trainset = train_ok_set + train_false_ng_set + train_ng_set + old_dataset

    print(len(train_ok_set), len(train_false_ng_set), len(train_ng_set), len(old_dataset))

    idx_list = [np.arange(0, len(train_ok_set)),
                np.arange(len(train_ok_set), len(train_ok_set) + len(train_false_ng_set)),
                np.arange(len(train_ok_set) + len(train_false_ng_set),
                          len(train_ok_set) + len(train_false_ng_set) + len(train_ng_set)),
                np.arange(len(train_ok_set) + len(train_false_ng_set) + len(train_ng_set), len(trainset))]
    # Create sampler, dataset, loader

    train_sampler = BalancedBatchSampler(trainset, idx_list, batch_size_list=args.batch_size_list)

    train_loader = DataLoader(trainset, batch_sampler=train_sampler, num_workers=1, pin_memory=True)
    return train_loader