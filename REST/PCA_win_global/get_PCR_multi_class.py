# 参数设定
import argparse
import glob
import os
import random
import sys
import time

import numpy as np
import torch

from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torchvision.transforms import transforms, InterpolationMode
from tqdm import tqdm

from model.images import LabeledImagesDataset, SampleDataset

from model.detection_model import PatchCore_residual

from util.save_code import save_dependencies_files
from util.util import fix_seed


#
parser = argparse.ArgumentParser()
parser.add_argument('--root-dir',type=str, default='../data/defect_512/mvtec')
# parser.add_argument('--dataset',type=str, default='bottle')
parser.add_argument('--context',type=bool, default=False,help='if context position context hist')
parser.add_argument('--patchcore-add-pos-embed',type=bool, default=False,help='if add sin cos position embedding')
parser.add_argument('--feature-folder',type=str, default=None)
parser.add_argument('--save-folder',type=str, default='position_weight0.5_residual')
# parser.add_argument('--method',type=str,default='sub')
parser.add_argument('--target-embed-dimension',type=int, default=1024)
parser.add_argument('--pos-weight',type=float, default=1)
parser.add_argument('--k-ratio',type=float, default=0.1)
parser.add_argument('--pca-com',type=float)
parser.add_argument('--topk',type=int, default=1)
parser.add_argument('--global-pca',type=int,default=16)
parser.add_argument('--global-win',type=int,default=4)
parser.add_argument('--global-resize_stride',type=int,default=4)
parser.add_argument('--aug-num',type=int, default=1)
parser.add_argument('--min-global-nn',type=int,default=64)
parser.add_argument('--local-layers',nargs='+')
parser.add_argument('--mode' ,type=str,default='gpu')
parser.add_argument('--global-nn-strategy',type=str,default=None)
parser.add_argument('--vi', default=False,action='store_true')
parser.add_argument('--patchsize',type=int,default=3)
parser.add_argument('--bank-name')
parser.add_argument('--backbone-name', type=str, default='wideresnet50')
parser.add_argument('--image-size', nargs='+', type=int, default=[512,512])
parser.add_argument('--crop-size', nargs='+', type=int, default=[512,512])
parser.add_argument('--num-groups',type=int, default=2)
args = parser.parse_args()
print(args)
TRAIN_BATCH_SIZE = 1

# backbone_name = 'wideresnet50'

SEED = 0

fix_seed(SEED)
save_dependencies_files(os.path.join('../work_dirs', os.path.join(sys.argv[0])), args)
bank_path = os.path.join('../work_dirs', f'{args.bank_name}.pkl') if args.bank_name is not None else None

train_transform = transforms.Compose([
    transforms.Resize(args.image_size, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(args.crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_mask = transforms.Compose([
    transforms.Resize(args.image_size, interpolation=InterpolationMode.BILINEAR),
transforms.CenterCrop(args.crop_size),
    transforms.ToTensor()])

all_train_ok_set = LabeledImagesDataset(f'{args.root_dir}/*/train/ok',feature_folder=args.feature_folder,label=0,
                                        train_transforms=train_transform,mask_transforms=transform_mask)

if len(all_train_ok_set) > 8000:
    all_train_ok_set = SampleDataset(f'{args.root_dir}/*/train/ok',
                                 label=0, seed=SEED, split='train', num_sample=8000)

print(len(all_train_ok_set))
all_train_ok_loader = DataLoader(all_train_ok_set,batch_size=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# with open([*glob.glob(f'{args.root_dir}/*/input_size.txt')][0],'r') as f:
#     h,w = [ int(c) for c in f.readline().split(',')]
model = PatchCore_residual((3,*args.crop_size),args.backbone_name,args.local_layers,device,train_ok_loader=all_train_ok_loader,
                           target_embed_dimension=args.target_embed_dimension,
                           pretrain_embed_dimension=1024,
                           pos_weight = args.pos_weight,mode=args.mode,
                           patchcore_add_pos=args.patchcore_add_pos_embed,context=args.context,k_ratio=args.k_ratio,
                           pca_com=args.pca_com,visible=args.vi,
                           min_global_nn=args.min_global_nn,global_win=args.global_win,global_pca=args.global_pca,
                           global_resize_stride=args.global_resize_stride,
                           global_nn_strategy=args.global_nn_strategy,bank_path=bank_path,patchsize=args.patchsize)
model.to(device)


model.eval()

for single_dataset in  sorted(os.listdir(args.root_dir)):

    data_dir  = f'{args.root_dir}/{single_dataset}'

    train_ok_set = LabeledImagesDataset(f'{args.root_dir}/{single_dataset}/train/ok', feature_folder=args.feature_folder,
                                    label=0,train_transforms=train_transform,mask_transforms=transform_mask)
    # ok_filenames = []
    # for sample in train_ok_set:
    #     ok_filenames.append(sample['filename'].split('_')[-1])
    train_ng_set = LabeledImagesDataset(f'{data_dir}/train/ng', feature_folder=args.feature_folder, label=1,
                                        train_transforms=train_transform,mask_transforms=transform_mask)
    train_false_ng_set = LabeledImagesDataset(f'{data_dir}/train/false_ng',feature_folder=args.feature_folder,label=1,
                                              train_transforms=train_transform,mask_transforms=transform_mask)

    testset = LabeledImagesDataset(f'{data_dir}/test/ng',feature_folder=args.feature_folder,  label=1,
                                   train_transforms=train_transform,mask_transforms=transform_mask) +\
              LabeledImagesDataset(f'{data_dir}/test/ok',feature_folder=args.feature_folder,  label=0,
                                   train_transforms=train_transform,mask_transforms=transform_mask)
    print(f'{single_dataset} train ok: {len(train_ok_set)}, train false ng: {len(train_false_ng_set)}, train ng: {len(train_ng_set)}, test: {len(testset)}')

    image_loader = DataLoader(testset+train_ng_set + train_ok_set + train_false_ng_set,
                              batch_size=TRAIN_BATCH_SIZE)

    # need_index_loader = DataLoader(train_false_ng_set+train_ok_set,
    #                           batch_size=TRAIN_BATCH_SIZE)

    with torch.no_grad():
        for batch in tqdm(image_loader):
            images = batch['image'].to(device)
            residual = model(images) # b,2048,32,32
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
