import argparse
import copy
import json
import math
import os
import pickle

import numpy as np
import torch

from torch.utils.data import  DataLoader
from tqdm import tqdm

from get_other_model import get_lavt, get_batch_random_text_embeddings
from model.images import TextDataset
from model.text_detector import Swin_text_detector
from model.text_tokenizer import get_text_token
from util.data_util import get_info
from util.dataloader import get_online_text_dataloder

from util.schedule import AverageMeter, WarmupCosineSchedule
import logging

from util.util import fix_seed, BalancedBatchSampler,WeightEMA

parser = argparse.ArgumentParser()
parser.add_argument('--data-root',type=str, default='data/defect_512/mvtec')
parser.add_argument('--dataset',type=str, default='screw')

parser.add_argument('--exp', type=int,default=1)
parser.add_argument('--exp-name', type=str,required=True)
parser.add_argument('--focal-loss-alpha', type=float,default=0.5)
parser.add_argument('--focal-loss-beta', type=bool,default=False)
parser.add_argument('--focal-loss-gamma', type=float,default=2)
parser.add_argument('--feature-channel', type=int,default=1024)
parser.add_argument('--num-heads', type=int,default=32,help='number of Swin head')
parser.add_argument('--depths', type=int,default=4)
parser.add_argument('--window-size', type=int,default=8)
parser.add_argument('--slide_window', type=int,default=None) # slide detection
parser.add_argument('--slide_stride', type=int,default=None)
parser.add_argument('--k', type=int,default=1)
parser.add_argument('--normal-var', type=float)
parser.add_argument('--p', nargs='+',type=float,help='sample possibility for n near')
parser.add_argument('--sample-patch', type=int,default=None)
parser.add_argument('--num-steps', type=int,default=2000)
parser.add_argument('--save-step', type=int,default=100)
parser.add_argument('--batch-size-list', type=list,default=[3,1,1,1])# ok:flaseng:ng:base data
parser.add_argument('--lr', type=float,default=None)
parser.add_argument('--gt-thres1', type=float,default=0.25)
parser.add_argument('--gt-thres2', type=float,default=0.08)
# parser.add_argument('--semi-pos-thres', type=float,default=1.)
# parser.add_argument('--semi-neg-thres', type=float,default=0.9)
# parser.add_argument('--n-anomaly', type=int,default=10)
parser.add_argument('--feature-folder', nargs='+',type=str)

parser.add_argument('--mask-ratio', type=float,default=None)
parser.add_argument('--ema-decay', type=float,default=None)
parser.add_argument('--weight-decay', type=float,default=0.05)

parser.add_argument('--aug', type=bool,default=False)

parser.add_argument('--pretrained-path', type=str,default=None)
parser.add_argument('--start-step', type=int,default=0)
parser.add_argument('--ng-num',type=int,help="indexs of true ng")
parser.add_argument('--text-avg',type=str,default='anomaly_avg')
parser.add_argument('--residual-method',type=str,default='square')

parser.add_argument('--lavt-weights',type=str,default='work_dirs/refcoco.pth')
parser.add_argument('--with-fg', type=bool,default=False)
parser.add_argument('--fg-dir', type=str,
                    default='work_dirs/mvtec_retreival_foreground_12_640_features.denseblock2_features.denseblock2_DensenetPM')
parser.add_argument('--fg-knn', type=int,default=10)
args = parser.parse_args()
args.batch_size_list = [int(i) for i in args.batch_size_list]
TRAIN_BATCH_SIZE = sum(args.batch_size_list)
gradient_accumulation_steps = 1

LEARNING_RATE =  1e-3*TRAIN_BATCH_SIZE*gradient_accumulation_steps/256 # 3e-3
if args.lr is not None:
    LEARNING_RATE = args.lr
LAYER_DECAY = 0.75 # vit

SEED = args.exp
fix_seed(SEED)
exp = args.exp

work_dir = f'work_dirs/{args.exp_name}_exp{args.exp}_{args.dataset}'
os.makedirs(work_dir,exist_ok=True)
logging.basicConfig(filename=f'{work_dir}/log.log',level=logging.INFO)
print(args)
logging.info(args)
old_dataset=[]


data_dir  = f'{args.data_root}/{args.dataset}'
data_info = get_info(data_dir,args.dataset,args)
train_ok_set = TextDataset(f'{data_dir}/train/ok',data_info,feature_folder=args.feature_folder,label=0,args=args)

train_false_ng_set = TextDataset(f'{data_dir}/train/false_ng', data_info,feature_folder=args.feature_folder,
                                              label=1,args=args)
train_ng_set = TextDataset(f'{data_dir}/train/ng/train_ng_*', data_info,
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

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
with open(f'{data_dir}/input_size.txt','r') as f:
    h,w = [ int(c) for c in f.readline().split(',')]
image_size = (3,h,w)
print('Load LAVT model...')
tokenizer,bert_model,mm_swin = get_lavt(args,device)

model = Swin_text_detector(image_size, stride = 8,patch_size=(1,1),residual_method=args.residual_method,
                           focal_loss_gamma=args.focal_loss_gamma,focal_loss_alpha=args.focal_loss_alpha,
                      slide_window=args.slide_window, slide_stride=args.slide_stride,in_chans=args.feature_channel,
                      num_classes=2,embed_dim=1024,window_size=args.window_size,depths=[args.depths],num_heads=[args.num_heads])
print(args.feature_channel)
base_params = [p for p in model.vit.parameters()]
model.vit.mm_swin = mm_swin
# print(len(model.vit.mm_swin.parameters()))
# print(len(model.vit.mm_swin.num_features.fusion.parameters()))
for p in model.vit.mm_swin.parameters():
    p.requires_grad = False
pwam_params = []
for layer in model.vit.mm_swin.layers:
    for p  in layer.fusion.parameters():
        p.requires_grad = True
        pwam_params.append(p)
parameters = [{'params':base_params},
              {'params':pwam_params,'lr':1e-6}]
optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE,weight_decay=args.weight_decay)
# scheduler = WarmupCosineSchedule(optimizer, warmup_steps=math.ceil(args.num_steps*0.05), t_total=args.num_steps)

if args.pretrained_path is not None:
    msg = model.vit.load_state_dict(torch.load(args.pretrained_path,map_location='cpu'),strict= False)
    print(msg)
if args.ema_decay is not None:
    ema_model = copy.deepcopy(model.vit)
    for param in ema_model.parameters():
        param.detach_()
    ema_model = ema_model.cuda()
    ema_optimizer= WeightEMA(model.vit.cuda(), ema_model,lr=LEARNING_RATE, alpha=args.ema_decay)
model.vit = torch.nn.DataParallel(model.vit).cuda()
model.zero_grad()
losses = AverageMeter()
global_step = args.start_step
ng_index = np.arange(10)
np.random.shuffle(ng_index)
while True:
    # if global_step % args.save_step == 0:
    #     train_loader = get_online_text_dataloder(ng_index, global_step, data_dir,data_info, train_ok_set, train_false_ng_set,
    #                                              old_dataset, args)
        # scheduler = WarmupCosineSchedule(optimizer, warmup_steps=math.ceil(args.save_step * 0.05),
        #                                  t_total=args.save_step)

    model.train()
    epoch_iterator = tqdm(train_loader,
                          desc="Training (X / X Steps) (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        features = batch['feature']
        masks = batch['mask']
        label=batch['label']


        img = batch['image']
        sens = batch['sens']
        fg = batch['fg']
        # with torch.no_grad():

        p = np.random.uniform()
        if p < 0.5:
            embedding, attention_mask_list = get_batch_random_text_embeddings(sens,tokenizer,bert_model)
        else:
            embedding = batch['avg_emb'].cuda()
            attention_mask_list = batch['avg_amask'].cuda()

        fusion_features = model.vit.module.mm_swin(img.cuda(), embedding, l_mask=attention_mask_list)
            # f1 = torch.nn.functional.interpolate(f1,model.feature_size,mode='bilinear',align_corners=False)
            # fusion_features = torch.cat([f1,f2],dim=1)
        # for p in model.vit.module.mm_swin.parameters():
        #     print(p.requires_grad)

        loss = model(features,fusion_features, masks,fg, if_train=True, args=args, iteration=global_step)
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            losses.update(loss.item() * gradient_accumulation_steps)
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f) (lr=%.8f)" % (
                    global_step, args.num_steps, losses.val, optimizer.state_dict()['param_groups'][0]['lr'])
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if args.ema_decay is not None: ema_optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            logging.info(f"Training ({global_step} / {args.num_steps} Steps) (loss={losses.val}) ")
            if global_step % args.save_step == 0:
                if args.ema_decay is not None:
                    torch.save(ema_model.state_dict(), f"{work_dir}/iter-{global_step}.pth")
                else:
                    torch.save(model.vit.module.state_dict(), f"{work_dir}/iter-{global_step}.pth")


        model.train()

        if global_step == args.num_steps:
            break


    losses.reset()
    if global_step == args.num_steps:
        break


