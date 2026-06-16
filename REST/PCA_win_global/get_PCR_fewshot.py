# 参数设定
import argparse
import os
import sys
import time

import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from model.images import LabeledImagesDataset, SampleAugDataset, SampleDataset
from model.detection_model import PatchCore_residual

from util.save_code import save_dependencies_files
from util.util import fix_seed


def build_match_keys(path):
    basename = os.path.basename(path)
    stem, _ = os.path.splitext(basename)
    keys = []

    for token_list in [basename.split('_'), stem.split('_')]:
        for i in range(len(token_list)):
            underscored = '_'.join(token_list[i:])
            joined = ''.join(token_list[i:])
            if underscored and underscored not in keys:
                keys.append(underscored)
            if joined and joined not in keys:
                keys.append(joined)
    return keys


def build_ok_index_map(filenames):
    ok_key_to_idx = {}
    for idx, path in enumerate(filenames):
        for key in build_match_keys(path):
            ok_key_to_idx.setdefault(key, idx)
    return ok_key_to_idx


def find_train_ok_query_id(path, ok_key_to_idx):
    for key in build_match_keys(path):
        if key in ok_key_to_idx:
            return ok_key_to_idx[key]
    return -1


def filter_dataset_by_ok_subset(dataset, ok_key_to_idx):
    keep_indices = [
        idx for idx, path in enumerate(dataset.filenames)
        if find_train_ok_query_id(path, ok_key_to_idx) != -1
    ]
    if len(keep_indices) == len(dataset.filenames):
        return dataset

    dataset.filenames = [dataset.filenames[idx] for idx in keep_indices]
    dataset.mask_filenames = [dataset.mask_filenames[idx] for idx in keep_indices]
    if dataset.feature_filenames is not None:
        dataset.feature_filenames = [
            [paths[idx] for idx in keep_indices] for paths in dataset.feature_filenames
        ]
    return dataset


parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', type=str, default='../data/defect_512/mvtec')
parser.add_argument('--dataset', type=str, default='bottle')
parser.add_argument('--context', type=bool, default=False, help='if context position context hist')
parser.add_argument('--patchcore-add-pos-embed', type=bool, default=False, help='if add sin cos position embedding')
parser.add_argument('--feature-folder', type=str, default=None)
parser.add_argument('--save-folder', type=str, default='position_weight0.5_residual')
parser.add_argument('--target-embed-dimension', type=int, default=1024)
parser.add_argument('--pos-weight', type=float, default=1)
parser.add_argument('--k-ratio', type=float, default=0.1)
parser.add_argument('--pca-com', type=float)
parser.add_argument('--global-pca', type=int)
parser.add_argument('--global-win', type=int)
parser.add_argument('--global-resize_stride', type=int, default=8)
parser.add_argument('--train-nn-num', type=int, default=1)
parser.add_argument('--aug-num', type=int, default=1)
parser.add_argument('--min-global-nn', type=int, default=64)
parser.add_argument('--p', nargs='+', type=float, help='sample possibility for n near')
parser.add_argument('--local-layers', nargs='+')
parser.add_argument('--mode', type=str, default='gpu')
parser.add_argument('--global-nn-strategy', type=str, default='max')
parser.add_argument('--vi', default=False, action='store_true')
parser.add_argument('--patchsize', type=int, default=3)
parser.add_argument('--bank-name')
parser.add_argument('--few-shot', type=int, default=1)
parser.add_argument('--exp', type=int, default=0)
args = parser.parse_args()
print("No Square!")
print(args)
TRAIN_BATCH_SIZE = 1

backbone_name = 'wideresnet50'

save_dependencies_files(os.path.join(args.root_dir, args.dataset, os.path.join(sys.argv[0])), args)
bank_path = os.path.join(args.root_dir, args.dataset, f'{args.bank_name}.pkl') if args.bank_name is not None else None
SEED = args.exp

fix_seed(SEED)
data_dir = f'{args.root_dir}/{args.dataset}'
few_shot_train_ok_set = SampleDataset(
    f'{args.root_dir}/{args.dataset}/train/ok',
    label=0,
    seed=SEED,
    split='train',
    num_sample=args.few_shot,
)
few_shot_train_ok_bank_set = SampleAugDataset(
    f'{args.root_dir}/{args.dataset}/train/ok',
    label=0,
    seed=SEED,
    num_sample=args.few_shot,
)
train_ok_loader = DataLoader(few_shot_train_ok_bank_set, batch_size=8)
ok_key_to_idx = build_ok_index_map(few_shot_train_ok_bank_set.filenames)

train_ng_set = LabeledImagesDataset(f'{data_dir}/train/ng', feature_folder=args.feature_folder, label=1)
train_false_ng_set = LabeledImagesDataset(f'{data_dir}/train/false_ng', feature_folder=args.feature_folder, label=1)
train_false_ng_set = filter_dataset_by_ok_subset(train_false_ng_set, ok_key_to_idx)

testset = LabeledImagesDataset(f'{data_dir}/test/ng', feature_folder=args.feature_folder, label=1) + \
          LabeledImagesDataset(f'{data_dir}/test/ok', feature_folder=args.feature_folder, label=0)
print(
    f'few-shot train ok: {len(few_shot_train_ok_set)}, '
    f'augmented bank ok: {len(few_shot_train_ok_bank_set)}, '
    f'train false ng: {len(train_false_ng_set)}, '
    f'train ng: {len(train_ng_set)}, test: {len(testset)}'
)

new_image_loader = DataLoader(testset + train_ng_set, batch_size=TRAIN_BATCH_SIZE)
need_index_loader = DataLoader(train_false_ng_set + few_shot_train_ok_set, batch_size=TRAIN_BATCH_SIZE)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open(f'{data_dir}/input_size.txt', 'r') as f:
    h, w = [int(c) for c in f.readline().split(',')]
start_time = time.time()
model = PatchCore_residual(
    (3, h, w),
    backbone_name,
    args.local_layers,
    device,
    train_ok_loader=train_ok_loader,
    target_embed_dimension=args.target_embed_dimension,
    pos_weight=args.pos_weight,
    mode=args.mode,
    patchcore_add_pos=args.patchcore_add_pos_embed,
    context=args.context,
    k_ratio=args.k_ratio,
    pca_com=args.pca_com,
    visible=args.vi,
    min_global_nn=args.min_global_nn,
    global_win=args.global_win,
    global_pca=args.global_pca,
    global_resize_stride=args.global_resize_stride,
    global_nn_strategy=args.global_nn_strategy,
    bank_path=bank_path,
    patchsize=args.patchsize,
)
model.to(device)


model.eval()
with torch.no_grad():
    for batch in tqdm(new_image_loader):
        images = batch['image'].to(device)
        residual = model(images, nn_num=args.train_nn_num, p=args.p)
        filenames = batch['filename']
        for i, path in enumerate(filenames):
            bsn = os.path.basename(path)
            save_dir = path.replace(bsn, args.save_folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, bsn.split('.')[0])
            np.save(save_path, residual[i].cpu().numpy())
    for batch in tqdm(need_index_loader):
        images = batch['image'].to(device)
        filenames = batch['filename']
        train_ok_query_id = find_train_ok_query_id(filenames[0], ok_key_to_idx)
        residual = model(images, train_ok_query_id=train_ok_query_id, nn_num=args.train_nn_num, p=args.p)
        for i, path in enumerate(filenames):
            bsn = os.path.basename(path)
            save_dir = path.replace(bsn, args.save_folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, bsn.split('.')[0])
            np.save(save_path, residual[i].cpu().numpy())
