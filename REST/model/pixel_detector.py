import math
from collections import OrderedDict
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from model.loss import focal_loss, semi_loss, consistency_loss, sigmoid_adaptive_focal_loss, NormalizedFocalLossSigmoid
from model.models_swin import Swin


from timm.models.layers import trunc_normal_


class Swin_pixel_detector(nn.Module):
    def __init__(self,image_size, stride = 8,patch_size=(1,1),nfl_alpha=0.5,nfl_gamma=2,
                 slide_window=None,slide_stride=None,in_chans=1024,num_classes=2,embed_dim=1024,window_size=4,depths=(4,),num_heads=(32,)):
        super(Swin_pixel_detector, self).__init__()

        feature_size = (image_size[1] // stride,image_size[2]//stride)
        self.slide_window = slide_window
        if slide_window is not None:
            self.slide_window = (slide_window,slide_window)
            # feature_size = (slide_window * math.floor(feature_size[0] / slide_window), slide_window * math.floor(feature_size[1] / slide_window))
            self.vit_img_size=self.slide_window
        else:
            # feature_size = (window_size*math.floor(feature_size[0]/window_size),window_size*math.floor(feature_size[1]/window_size))
            self.vit_img_size=feature_size

        self.patch_size = patch_size
        self.feature_size = feature_size
        self.slide_stride = slide_stride

        self.out_stride = stride // len(depths)
        self.vit = Swin(img_size=self.vit_img_size, stride=self.out_stride, in_chans=in_chans, num_classes=num_classes,
                        window_size=window_size,
                        patch_size=patch_size, embed_dim=embed_dim, depths=depths,
                        num_heads=num_heads)
        self.loss_f = NormalizedFocalLossSigmoid(alpha=nfl_alpha, gamma=nfl_gamma)

        # manually initialize fc layer
        trunc_normal_(self.vit.head.weight, std=2e-5)


    def shake(self,sample_num,feature_channel,normal_var):
        patch_index = np.random.choice(np.arange(2), sample_num, (0.4, 0.6))
        dim_p = np.random.choice(np.arange(0.2, 0.6, 0.1))
        dim_index = np.random.choice(np.arange(2), (sample_num,feature_channel), (1 - dim_p, dim_p))
        dim_index[patch_index == 0] = 0
        dim_index = torch.Tensor(dim_index)
        # print(dim_index.shape,dim_index)
        s =  torch.pow(
            torch.exp(torch.normal(0., math.sqrt(normal_var), (sample_num,feature_channel)).clamp(-0.223, 0.223)),
            dim_index)
        # print(torch.unique(s))
        return s
    def get_label_kind(self,masks):
        # 有标签 1，无标签0
        label_kinds = torch.zeros_like(masks)
        label_kinds[masks == 1] = 1
        label_kinds[masks == 0] = 1
        return label_kinds
    def aug(self,feature_residual,args):
        # augment
        feature_residual = feature_residual.permute(0, 2, 3, 1)  # b,h,w,3*c

        aug_residuals = []
        b, h, w = feature_residual.shape[:3]
        n = b * h * w
        feature_residual = feature_residual.reshape(n, -1, args.feature_channel)
        assert feature_residual.shape[1] >= len(args.p), feature_residual.shape  # 近邻数量
        # 独立k次数据增强
        for i in range(0, args.k):
            # 按概率随机选择近邻
            index = np.random.choice(np.arange(feature_residual.shape[1]), n, p=args.p)
            aug_residual0 = feature_residual[np.arange(n), index]
            if args.normal_var is not None:
                aug_residual0 = aug_residual0 * self.shake(aug_residual0.shape[0], aug_residual0.shape[1],
                                                       args.normal_var).to('cuda', non_blocking=True)
            # np.save(os.path.join(save_dir, f'{args.dataset}_noise'), aug_residual0.numpy())
            aug_residuals.append(aug_residual0)

        feature_residual = torch.cat(aug_residuals, dim=-1).view(b, h, w, -1).permute(0, 3, 1, 2)
        del aug_residual0, aug_residuals
        return feature_residual


    def forward(self, feature_residual,masks=None, if_train=False,args=None,iteration=None,semi=False,semi_label=False):
        feature_residual = feature_residual.to('cuda',non_blocking=True)
        # if img_gts is not None:
        #
        #     masks = self.get_gt(img_gts, semi_label, args)
        if if_train and args.aug:
            feature_residual = self.aug(feature_residual,args)

        if self.slide_window is not None:

            if self.slide_stride == 1:
                sample_features, sample_masks = [],[]
                for s in range(args.sample_patch):
                    r1 = random.randint(0,feature_residual.shape[0]-1)
                    r2 = random.randint(0,feature_residual.shape[2]-args.slide_window)
                    r3 = random.randint(0,feature_residual.shape[3]-args.slide_window)

                    sample_features.append(feature_residual[r1,:,r2:r2+args.slide_window,r3:r3+args.slide_window].unsqueeze(0))
                    sample_masks.append(masks[r1,:,r2:r2+args.slide_window,r3:r3+args.slide_window].unsqueeze(0))

                feature_residual = torch.cat(sample_features,dim=0)
                masks = torch.cat(sample_masks,dim=0)
                # print(sample_features[0].shape,feature_residual.shape,masks.shape)
            else:
                c = feature_residual.shape[1]


                feature_residual = torch.nn.functional.unfold(feature_residual, self.slide_window,
                                                              stride=self.slide_stride).transpose(1, 2).flatten(0, 1)
                feature_residual = feature_residual.view(-1, c, *self.slide_window)

                if if_train :

                    masks = torch.nn.functional.unfold(masks,
                                                    (self.slide_window[0] * self.out_stride, self.slide_window[1] * self.out_stride),
                                                      stride=self.slide_stride * self.out_stride).transpose(1, 2).flatten(0,1)
                    masks = masks.view(-1, 1, self.slide_window[0] * self.out_stride, self.slide_window[1] * self.out_stride)
                    # sample some slide window
                    if args.sample_patch is not None:

                        sample = np.random.choice(feature_residual.shape[0],feature_residual.shape[0]//args.sample_patch,False)
                        feature_residual = feature_residual[sample]
                        masks = masks[sample]
                #
        # feature_residual = feature_residual.to('cuda',non_blocking=True)

        if if_train:
            if args.aug and not semi:
                feature_residual = feature_residual.view(-1, args.feature_channel, *feature_residual.shape[2:])
                masks = torch.cat([masks]*args.k,dim=1)
                masks = masks.view(feature_residual.shape[0],-1,*masks.shape[2:])# torch.Size([75, 2, 32, 32])
            if semi:

                feature_residual,masks,label_kinds  = self.guess_mixup(feature_residual,masks,args)
            masks = masks.to('cuda',non_blocking=True)

        logits = self.vit(feature_residual)

        if if_train:
            B, N, C = logits.shape
            grid_size = self.feature_size

            logits = logits.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])
            logits = torch.nn.functional.interpolate(logits,masks.shape[-2:],mode='bilinear',align_corners=False)

            return self.loss_f(logits, masks.cuda()).mean()


        return logits



