# 参数设定
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn import metrics

from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from get_other_model import get_lavt, get_avg_text_embedding, get_pca_text_embedding, get_anomaly_avg_text_embedding
from model.feature_detector import Swin_detector
from model.images import LabeledImagesDataset
import logging

from model.patchcore.metrics import compute_ap_torch, compute_pixel_auc_torch, compute_image_auc_torch, compute_pro_torch
from model.text_detector import Swin_text_detector
from model.text_tokenizer import get_text_token

from util.util import fix_seed, predict, predict_with_fg, predict_text

# os.system("set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32")
parser = argparse.ArgumentParser()
parser.add_argument('--data-root',type=str, default='data/defect/mvtec')
parser.add_argument('--dataset',type=str, default='bottle')
# parser.add_argument('--patchcore-patchsize', type=int,default=3)
parser.add_argument('--exp', type=int,default=1)
parser.add_argument('--batch-size', type=int,default=1)
parser.add_argument('--exp-name', type=str,default='tmp')
parser.add_argument('--detection-model', type=str,default='swin',choices=['vit','swin'])
parser.add_argument('--feature-channel', type=int,default=1024)
parser.add_argument('--slide_window', type=int,default=None)
parser.add_argument('--slide_stride', type=int,default=None)
parser.add_argument('--num-heads', type=int,default=32)
parser.add_argument('--depths', type=int,default=4)
parser.add_argument('--num-classes', type=int,default=2)
parser.add_argument('--window-size', type=int,default=4)
parser.add_argument('--num-steps', type=int,default=2000)
parser.add_argument('--eval-step', type=int,default=100)
parser.add_argument('--n-anomaly', type=int,default=10)
parser.add_argument('--only-ap', type=bool,default=False)
parser.add_argument('--feature-folder', nargs='+',type=str)
parser.add_argument('--gt-size', nargs='+',type=int,default=[256,256])
parser.add_argument('--gt-resize-mode',type=str)
# parser.add_argument('--aug-feature-folder',type=str)
parser.add_argument('--mode', type=str,default='supervised')
parser.add_argument('--gaussian', type=int,default=1)

parser.add_argument('--with-fg', type=bool,default=False)
parser.add_argument('--fg-dir', type=str,
                    default='work_dirs/mvtec_retreival_foreground_12_640_features.denseblock2_features.denseblock2_DensenetPM')
parser.add_argument('--fg-knn', type=int,default=10)
parser.add_argument('--residual-method',type=str,default='square')

parser.add_argument('--lavt-weights',type=str,default='work_dirs/refcoco.pth')
parser.add_argument('--text',type=str,default='anomaly_avg' )
parser.add_argument('--text-dir',type=str,default='data/mvtec_text' )
args = parser.parse_args()

# 设定种子
SEED = 0
fix_seed(SEED)
# backbone_name =  'wideresnet50' # 'work_dirs/vit_L_kd_patchify_wideres50/iter-10000.pth'
# layers_to_extract_from = ['layer2', 'layer3'] # ['head']
# 图像简单处理
gt_size = args.gt_size
print(args)
if args.gt_resize_mode == 'nearest':
    transform_mask = transforms.Compose([
    transforms.Resize(gt_size,interpolation=InterpolationMode.NEAREST) ,
    transforms.ToTensor()])

if args.gt_resize_mode == 'bilinear':
    transform_mask = transforms.Compose([
    transforms.Resize(gt_size,interpolation=InterpolationMode.BILINEAR) ,
    transforms.ToTensor()])
os.makedirs('work_dirs/results',exist_ok=True)

exp_name = f'{args.exp_name}_exp{args.exp}'

for single_dataset in [args.dataset]:

    data_dir  = f'{args.data_root}/{single_dataset}'
    work_dir = f'work_dirs/{exp_name}_{single_dataset}'

    # work_dir = f'work_dirs/{exp_name}'

    os.makedirs(work_dir,exist_ok=True)
    logging.basicConfig(filename=f'{work_dir}/log.log',level=logging.INFO)

    trainset = LabeledImagesDataset(f'{data_dir}/train/ok',feature_folder=args.feature_folder,mask_transforms=transform_mask,label=0)

    # train_ok_loader = DataLoader(trainset,batch_size=4)
    train_ng_set = LabeledImagesDataset(f'{data_dir}/train/ng', feature_folder=args.feature_folder,mask_transforms=transform_mask,
                                        label=1) if args.n_anomaly <= 10 else \
        LabeledImagesDataset(f'data/defect_512/{os.path.basename(args.data_root)}_{args.n_anomaly}/{args.dataset}/train/ng',
                             feature_folder=args.feature_folder,mask_transforms=transform_mask,
                             label=1)
    if args.mode == 'supervised':
        print('Supervised!')
        trainset += train_ng_set

    trainset += LabeledImagesDataset(f'{data_dir}/train/false_ng', feature_folder=args.feature_folder,mask_transforms=transform_mask,
                                                  label=1)
    testset = LabeledImagesDataset(f'{data_dir}/test/ng',feature_folder=args.feature_folder,mask_transforms=transform_mask,label=1)
    testset += LabeledImagesDataset(f'{data_dir}/test/ok', feature_folder=args.feature_folder,
                                    mask_transforms=transform_mask, label=0) \
        if args.n_anomaly <= 10 else \
        LabeledImagesDataset(f'data/defect_512/{os.path.basename(args.data_root)}_{args.n_anomaly}/{args.dataset}/test/ng',
                             feature_folder=args.feature_folder,mask_transforms=transform_mask,
                             label=1)
    if  args.mode == 'unsupervised':
        print('Unsupervised!')
        testset += train_ng_set

    print(len(testset))
    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)

    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.batch_size)

    test_loader = DataLoader(testset,
                            sampler=test_sampler,
                            batch_size=args.batch_size) if testset is not None else None



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(f'{data_dir}/input_size.txt', 'r') as f:
        h, w = [int(c) for c in f.readline().split(',')]
        image_size = (3,h,w)
    # print('Load LAVT model...')
    tokenizer, bert_model, mm_swin = get_lavt(args, device)
    model = Swin_text_detector(image_size, stride=8, patch_size=(1, 1),residual_method=args.residual_method,
                          slide_window=args.slide_window, slide_stride=args.slide_stride,
                          in_chans=args.feature_channel,
                          num_classes=args.num_classes, embed_dim=1024, window_size=args.window_size, depths=[args.depths],
                          num_heads=[args.num_heads])
    model.vit.mm_swin = mm_swin

    if not args.only_ap:
        f = open(f'{work_dir}/{os.path.basename(work_dir)}_{args.eval_step}_{args.gt_resize_mode}_{gt_size[0]}_{gt_size[1]}_metric.csv', 'w', encoding='utf-8')
    else:
        f = open(f'{work_dir}/{os.path.basename(work_dir)}_{args.gt_resize_mode}_metric.csv', 'w', encoding='utf-8')

    f.write('iteration,ap,pixel_auroc,pro,image_auroc,test_underkill,test_overkill,train_underkill,train_overkill,image_auroc1\n')
    # outputsize = (h // model.out_stride, w // model.out_stride)
    if args.text == 'avg':    embeddings, attention_masks = get_avg_text_embedding(single_dataset,tokenizer,bert_model)
    if args.text == 'anomaly_avg':    embeddings, attention_masks = get_anomaly_avg_text_embedding(single_dataset,tokenizer,bert_model,root_dir=args.text_dir)
    if args.text == 'pca':    embeddings, attention_masks = get_pca_text_embedding(single_dataset,tokenizer,bert_model)
    for iter in range(args.eval_step,args.num_steps+args.eval_step,args.eval_step):
        checkpoint = torch.load(f'{work_dir}/iter-{iter}.pth',map_location='cpu')
        msg = model.vit.load_state_dict(checkpoint,strict=False)
        # print(msg)
        model.to(device)

        model.eval()
    ###################

        if args.with_fg:
            origin_path = f'{data_dir}/origin_path.json'
            with open(origin_path, 'r', encoding='utf-8') as f1:
                path_info = json.load(f1)
            knn_info_path = f'{args.fg_dir}/{args.dataset}/r_result.json'
            with open(knn_info_path,'r', encoding='utf-8') as f2:
                knn_info = json.load(f2)
            test_segs, test_gts, test_scores, test_anomaly_label = predict_text(test_loader, model, model.vit.mm_swin, embeddings,
                                                                                attention_masks, args,
                                                                                path_info=path_info, knn_info=knn_info)
        else:
            test_segs, test_gts, test_scores, test_anomaly_label = predict_text(test_loader, model,model.vit.mm_swin,embeddings,attention_masks,args)
        test_gts = test_gts.int().cuda()
        test_segs = test_segs.cuda()
        ap = compute_ap_torch(test_gts, test_segs)

        if args.only_ap:
            f.write(f'{iter},{ap}\n')
            print(f'iter:{iter}, ap:{ap:.4f}')
            continue
        # print(test_gts.shape,test_segs.shape)
        pixel_auc = compute_pixel_auc_torch(test_gts, test_segs)
        image_auc = compute_image_auc_torch(test_anomaly_label.cuda(), test_scores)
        start_time = time.time()
        pro = compute_pro_torch(test_gts, test_segs)
        print('pro time=', time.time() - start_time)
        # val_segs,val_gts,val_scores, val_anomaly_label = predict(train_loader,model,args)
        #
        # ls, gs, train_underkill, train_overkill, image_auc1, image_ap1 = cal_kill(val_scores, test_scores, val_segs,
        #                                                                           test_segs,
        #                                                                           val_anomaly_label, test_anomaly_label,
        #                                                                           strategy='max_rectangle')
        # print(f'iter:{iter}')
        # print(f'{single_dataset} Train underkill:{train_underkill:.4f}, overkill:{train_overkill:.4f}')
        # print(f'{single_dataset} Test underkill:{ls:.4f},overkill:{gs:.4f}, ap:{ap:.4f},pro:{pro:.4f}')

        print(f'iter: {iter}, ap={ap},pixel_auc={pixel_auc},pro={pro},image_auc={image_auc}\n')
        f.write(f'{iter},{ap},{pixel_auc},{pro},{image_auc}\n')