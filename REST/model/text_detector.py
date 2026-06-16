import math
import time
from collections import OrderedDict
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from model.loss import focal_loss, semi_loss, consistency_loss, sigmoid_adaptive_focal_loss, NormalizedFocalLossSigmoid
from model.models_swin import Swin, TextSwin

from timm.models.layers import trunc_normal_

class Swin_text_detector(nn.Module):
    def __init__(self,image_size, stride = 8,patch_size=(1,1),residual_method='square',focal_loss_alpha=0.5,focal_loss_gamma=2,
                 slide_window=None,slide_stride=None,in_chans=1024,num_classes=2,embed_dim=1024,window_size=4,depths=(4,),num_heads=(32,)):
        super(Swin_text_detector, self).__init__()

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
        self.vit = TextSwin(img_size=self.vit_img_size, stride=self.out_stride, in_chans=in_chans, num_classes=num_classes,
                        window_size=window_size,residual_method=residual_method,
                        patch_size=patch_size, embed_dim=embed_dim, depths=depths,
                        num_heads=num_heads)

        # manually initialize fc layer
        # trunc_normal_(self.vit.head.weight, std=2e-5)
        # print(f'Focal loss: alpha={focal_loss_alpha},gamma={focal_loss_gamma}')
        print(f'NormalizedFocalLossSigmoid: alpha={focal_loss_alpha},gamma={focal_loss_gamma}')
        self.loss_f = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)

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

    def get_gt(self,img_gts,semi_label,args):
        # if masks.shape[1] != self.feature_size[0]*self.out_stride or masks.shape[2] != self.feature_size[1]*self.out_stride:
        #     masks = nn.functional.interpolate(masks, size=(self.feature_size[0]*self.out_stride,self.feature_size[1]*self.out_stride),mode='nearest')
        if img_gts.shape[2] == self.feature_size[0] and img_gts.shape[3] == self.feature_size[1]:
            # masks = -torch.ones_like(img_gts)
            # masks[img_gts == 1] = 1
            # masks[img_gts == -1] = 0
            return img_gts
        # 512*512分成8*8小块
        #
        if semi_label:
            # print(img_gts.shape)
            # masks = -torch.ones([*img_gts.shape[:2]])
            # masks[torch.mean((img_gts > 0.49).float(), dim=-1) >= args.semi_pos_thres] = 1  # 正样本
            # masks[torch.mean((img_gts < 0.49).float(), dim=-1) >= args.semi_neg_thres] = 0  # 负样本
            masks = torch.nn.functional.interpolate(
                img_gts,
                size=self.feature_size,
                mode="bilinear",
                align_corners=False,
            )

        else:
            img_gts = torch.nn.functional.unfold(img_gts, self.patch_size[0] * self.out_stride,
                                                 stride=self.patch_size[0] * self.out_stride).transpose(1, 2)
            masks = -torch.ones([*img_gts.shape[:2]])
            img_gts = torch.mean(img_gts, dim=-1)  # img_gts的值是0或1， 8*8一个单元计算缺陷面积占比
            masks[img_gts >= args.gt_thres1] = 1
            masks[img_gts <= args.gt_thres2] = 0
            masks = masks.view(masks.shape[0], 1, self.feature_size[0], self.feature_size[1])

        # plt.imshow(masks[0].view(64, 64))
        # plt.show()
        return masks

    def forward(self, feature_residual,fusion_feature,img_gts=None,fg=None, if_train=False,args=None,iteration=None,semi=False,semi_label=False):
        # feature_residual = feature_residual.cuda()

        if img_gts is not None:

            masks = self.get_gt(img_gts, semi_label, args)
        if if_train and args.aug:
            feature_residual = self.aug(feature_residual,args)

        if self.slide_window is not None:

            c = feature_residual.shape[1]
            feature_residual = torch.nn.functional.unfold(feature_residual, self.slide_window,
                                                          stride=self.slide_stride).transpose(1, 2).flatten(0, 1)
            feature_residual = feature_residual.view(-1, c, *self.slide_window)

            c = fusion_feature.shape[1]
            fusion_feature = torch.nn.functional.unfold(fusion_feature, self.slide_window,
                                                          stride=self.slide_stride).transpose(1, 2).flatten(0, 1)
            fusion_feature = fusion_feature.view(-1, c, *self.slide_window)
            if if_train :

                masks = torch.nn.functional.unfold(masks,
                                                (self.slide_window[0] , self.slide_window[1] ),
                                                  stride=self.slide_stride).transpose(1, 2).flatten(0,1)
                masks = masks.view(-1, 1, self.slide_window[0] , self.slide_window[1])


                if args.sample_patch is not None:
                    if args.with_fg:
                        fg = torch.nn.functional.unfold(fg,
                                                        (self.slide_window[0], self.slide_window[1]),
                                                        stride=self.slide_stride).transpose(1, 2)
                        fg = torch.mean(fg, dim=-1)  # 6,25
                        p =np.array(fg).flatten()
                        p = p/p.sum()
                        batch_sample = np.random.choice(feature_residual.shape[0],feature_residual.shape[0]//args.sample_patch,p=p,replace=False)


                    else:
                        batch_sample = np.random.choice(feature_residual.shape[0],feature_residual.shape[0]//args.sample_patch,False)
                    feature_residual = feature_residual[batch_sample]
                    fusion_feature = fusion_feature[batch_sample]
                    masks = masks[batch_sample]

        if if_train:
            if args.aug and not semi:
                feature_residual = feature_residual.view(-1, args.feature_channel, *feature_residual.shape[2:])
                masks = torch.cat([masks]*args.k,dim=1)
                masks = masks.view(feature_residual.shape[0],-1,*masks.shape[2:])# torch.Size([75, 2, 32, 32])
            if semi:

                feature_residual,masks,label_kinds  = self.guess_mixup(feature_residual,masks,args)
            masks = masks.to('cuda',non_blocking=True)
        # print(fusion_feature.shape)
        if not if_train or args.mask_ratio is None:


            logits = self.vit(feature_residual,fusion_feature)

            # print(time.time()-s)

            # torch.cuda.synchronize()
            # print(time.time() - sttatee)

        else:
            mask_ratio = random.uniform(0, args.mask_ratio)
            logits,mask_pos = self.vit(feature_residual,fusion_feature,mask_ratio)
            mask_pos = mask_pos.view(*logits.shape[:2])
            logits = logits[mask_pos==1]
            if masks.ndim == 4:
                mask_pos = mask_pos.view(*masks.shape)
            if masks.ndim == 2:
                mask_pos = mask_pos.view(masks.shape[0])

            masks = masks[mask_pos==1]
        logits = torch.nn.functional.interpolate(
            logits,
            size=(self.feature_size[0]*self.out_stride,self.feature_size[1]*self.out_stride),
            mode="bilinear",
            align_corners=False,
        )
        if if_train:

            # print(logits.shape)
            # logits = logits.view(-1,2)
            #
            # masks = masks.view(-1)
            # logits = torch.cat([logits[masks == 1], logits[masks == 0]])
            # masks = torch.cat([masks[masks == 1], masks[masks == 0]])
            # loss = focal_loss(logits, masks, alpha=args.focal_loss_alpha,
            #                   gamma=args.focal_loss_gamma,beta=args.focal_loss_beta)
            # return loss
            # print(ann_logits.shape,slide_gt.shape)
            # print(logits.shape)
            return self.loss_f(logits, img_gts.cuda()).mean()
            # loss = focal_loss(logits.permute(0,2,3,1).flatten(0,2),img_gts.cuda().flatten(),args.focal_loss_alpha,args.focal_loss_gamma)
            # return loss
        return logits



