# 参数设定
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torchvision.transforms import transforms
from tqdm import tqdm

from model.images import LabeledImagesDataset, SampleDataset

from model.detection_model import PatchCore_residual

from util.save_code import save_dependencies_files
from util.util import fix_seed


#
parser = argparse.ArgumentParser()
parser.add_argument('--root-dir',type=str, default='../data/defect_512/mvtec')
parser.add_argument('--dataset',type=str, default='bottle')
parser.add_argument('--context',type=bool, default=False,help='if context position context hist')
parser.add_argument('--patchcore-add-pos-embed',type=bool, default=False,help='if add sin cos position embedding')
parser.add_argument('--feature-folder',type=str, default=None)
parser.add_argument('--save-folder',type=str, default='position_weight0.5_residual')
# parser.add_argument('--method',type=str,default='sub')
parser.add_argument('--target-embed-dimension',type=int, default=1024)
parser.add_argument('--pos-weight',type=float, default=1)
parser.add_argument('--k-ratio',type=float, default=0.1)
parser.add_argument('--pca-com',type=float)
parser.add_argument('--global-pca',type=int)
parser.add_argument('--global-win',type=int)
parser.add_argument('--global-resize_stride',type=int,default=8)
parser.add_argument('--train-nn-num',type=int, default=1)
parser.add_argument('--aug-num',type=int, default=1)
parser.add_argument('--min-global-nn',type=int,default=64)
parser.add_argument('--p', nargs='+',type=float,help='sample possibility for n near')
parser.add_argument('--local-layers',nargs='+')
parser.add_argument('--mode' ,type=str,default='gpu')
parser.add_argument('--global-nn-strategy',type=str,default='max')
parser.add_argument('--vi', default=False,action='store_true')
parser.add_argument('--patchsize',type=int,default=3)
parser.add_argument('--bank-name')
args = parser.parse_args()
print("No Square!")
print(args)
TRAIN_BATCH_SIZE = 1

backbone_name = 'wideresnet50'

# 图像简单处理

for single_dataset in [args.dataset]:
    save_dependencies_files(os.path.join(args.root_dir, single_dataset, os.path.join(sys.argv[0])), args)
    bank_path = os.path.join(args.root_dir, single_dataset, f'{args.bank_name}.pkl') if args.bank_name is not None else None
    # for single_dataset in sorted(os.listdir('data/mvtec')):#[args.dataset]:
    # 设定种子
    SEED = 0

    fix_seed(SEED)
    data_dir  = f'{args.root_dir}/{single_dataset}'
    bank_set = LabeledImagesDataset(f'{args.root_dir}/{single_dataset}/train/ok',label=0)
    if len(bank_set) > 8000:
        bank_set = SampleDataset(f'{args.root_dir}/{single_dataset}/train/ok',
                                     label=0, seed=SEED, split='train', num_sample=8000)
    train_ok_loader = DataLoader(bank_set,batch_size=8)
    train_ok_set = LabeledImagesDataset(f'{args.root_dir}/{single_dataset}/train/ok',
                                    label=0)
    ok_filenames = []
    for sample in bank_set:
        ok_filenames.append("".join(os.path.basename(sample['filename']).split('_')[3:]))
    train_ng_set = LabeledImagesDataset(f'{data_dir}/train/ng', feature_folder=args.feature_folder, label=1)
    train_false_ng_set = LabeledImagesDataset(f'{data_dir}/train/false_ng',feature_folder=args.feature_folder,label=1)

    testset = LabeledImagesDataset(f'{data_dir}/test/ng',feature_folder=args.feature_folder,  label=1) +\
              LabeledImagesDataset(f'{data_dir}/test/ok',feature_folder=args.feature_folder,  label=0)
    print(f'train ok: {len(train_ok_set)}, train false ng: {len(train_false_ng_set)}, train ng: {len(train_ng_set)}, test: {len(testset)}')

    new_image_loader = DataLoader(testset+train_ng_set,
                              batch_size=TRAIN_BATCH_SIZE)

    need_index_loader = DataLoader(train_false_ng_set+train_ok_set,
                              batch_size=TRAIN_BATCH_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(f'{data_dir}/input_size.txt', 'r') as f:
        h, w = [int(c) for c in f.readline().split(',')]
    start_time = time.time()
    model = PatchCore_residual((3,h,w),backbone_name,args.local_layers,device,train_ok_loader=train_ok_loader,
                               target_embed_dimension=args.target_embed_dimension,pos_weight = args.pos_weight,mode=args.mode,
                               patchcore_add_pos=args.patchcore_add_pos_embed,context=args.context,k_ratio=args.k_ratio,
                               pca_com=args.pca_com,visible=args.vi,
                               min_global_nn=args.min_global_nn,global_win=args.global_win,global_pca=args.global_pca,
                               global_resize_stride=args.global_resize_stride,
                               global_nn_strategy=args.global_nn_strategy,bank_path=bank_path,patchsize=args.patchsize)
    model.to(device)


    model.eval()
    with torch.no_grad():
        for batch in tqdm(new_image_loader):
            images = batch['image'].to(device)
            residual = model(images,nn_num=args.train_nn_num,p=args.p) # b,2048,32,32
            # print(residual.shape)
            residual = residual
            filenames = batch['filename']
            for i,path in enumerate(filenames):
                # print(residual[i].shape,path.split('.')[0])
                bsn = os.path.basename(path)
                save_dir = path.replace(bsn,args.save_folder)
                os.makedirs(save_dir,exist_ok=True)
                save_path = os.path.join(save_dir,bsn.split('.')[0])
                np.save(save_path,residual[i].cpu().numpy())
        for batch in tqdm(need_index_loader):
            images = batch['image'].to(device)
            filenames = batch['filename']
            if 'false_ng' in filenames[0]:
                q = "".join(os.path.basename(filenames[0]).split('_')[4:])

            else:
                q = "".join(os.path.basename(filenames[0]).split('_')[3:])
            if q in ok_filenames:
                train_ok_query_id = ok_filenames.index(q)
                # print(train_ok_query_id)
            else:
                train_ok_query_id = -1
            residual = model(images, train_ok_query_id=train_ok_query_id,nn_num=args.train_nn_num,p=args.p)  # b,2048,32,32
            # print(f'residual time: {time.time() - start_time}')

            residual = residual
            for i, path in enumerate(filenames):
                # print(residual[i].shape,path.split('.')[0])
                bsn = os.path.basename(path)
                save_dir = path.replace(bsn, args.save_folder)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, bsn.split('.')[0])
                np.save(save_path, residual[i].cpu().numpy())
