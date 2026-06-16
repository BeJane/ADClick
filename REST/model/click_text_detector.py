import math
from collections import OrderedDict
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from model.isegm import DistMaps,  get_ori_pred
from model.loss import focal_loss, semi_loss, consistency_loss, sigmoid_adaptive_focal_loss, NormalizedFocalLossSigmoid
from model.models_swin import Swin, SwinTwoHead, ClickTextSwin, ClickTextSwin1

from timm.models.layers import trunc_normal_

from model.patchcore.common import RescaleSegmentor
from model.point import get_points, get_first_points
from model.util import get_iou


class ClickTextDetector(nn.Module):
    def __init__(self,image_size, stride = 8,patch_size=(1,1),norm_radius=5,ad_training=False,no_head=False,nfl_alpha=0.5,nfl_gamma=2,fusion_feature_channels=[128,256,512,1024],
                 slide_window=None,slide_stride=None,in_chans=1024,num_classes=2,embed_dim=1024,window_size=4,depths=(4,),num_heads=(32,)):
        super(ClickTextDetector, self).__init__()
        print(f'AD training: {ad_training}')
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
        self.out_stride = stride//len(depths)
        if no_head:
            print('No Conv. Neck and MLP Head.')
            self.vit = ClickTextSwin(img_size=self.vit_img_size, stride=self.out_stride, ad_training=ad_training,
                                      in_chans=in_chans, num_classes=num_classes, window_size=window_size,
                                      patch_size=patch_size, embed_dim=embed_dim, depths=depths,
                                      num_heads=num_heads)
        else:
            print('With Conv. Neck and MLP Head.')
            self.vit = ClickTextSwin1(img_size=self.vit_img_size, stride=self.out_stride,ad_training=ad_training,
                                      fusion_feature_channels=fusion_feature_channels,in_chans=in_chans,num_classes=num_classes,window_size=window_size,
                             patch_size=patch_size, embed_dim=embed_dim, depths=depths,
                       num_heads=num_heads)


        self.loss_f = NormalizedFocalLossSigmoid(alpha=nfl_alpha, gamma=nfl_gamma)

        self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0,
                                  cpu_mode=False, use_disks=True)
        self.ad_training = ad_training

        self.anomaly_segmentor = RescaleSegmentor(
        device='cuda', target_size=image_size[1:],gaussian=True
    )



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
                align_corners=True,
            )
            # plt.imshow(masks.view(64,64))
            # plt.show()
            # print((masks==1).sum())
        else:
            img_gts = torch.nn.functional.unfold(img_gts, self.patch_size[0] * self.out_stride,
                                                 stride=self.patch_size[0] * self.out_stride).transpose(1, 2)
            masks = -torch.ones([*img_gts.shape[:2]])
            img_gts = torch.mean(img_gts, dim=-1)  # img_gts的值是0或1， 8*8一个单元计算缺陷面积占比
            masks[img_gts >= args.gt_thres1] = 1
            masks[img_gts <= args.gt_thres2] = 0
            masks = masks.view(masks.shape[0], 1, self.feature_size[0], self.feature_size[1])
        return masks

    def slide(self,feature_residual,fusion_feature,args,masks=None):
        if self.slide_stride == 1:
            sample_features, sample_masks = [], []
            for s in range(args.sample_patch):
                r1 = random.randint(0, feature_residual.shape[0] - 1)
                r2 = random.randint(0, feature_residual.shape[2] - args.slide_window)
                r3 = random.randint(0, feature_residual.shape[3] - args.slide_window)

                sample_features.append(
                    feature_residual[r1, :, r2:r2 + args.slide_window, r3:r3 + args.slide_window].unsqueeze(0))
                sample_masks.append(masks[r1, :, r2:r2 + args.slide_window, r3:r3 + args.slide_window].unsqueeze(0))

            feature_residual = torch.cat(sample_features, dim=0)
            masks = torch.cat(sample_masks, dim=0)
            # print(sample_features[0].shape,feature_residual.shape,masks.shape)
        else:
            c = feature_residual.shape[1]

            feature_residual = torch.nn.functional.unfold(feature_residual, self.slide_window,
                                                          stride=self.slide_stride).transpose(1, 2).flatten(0, 1)
            feature_residual = feature_residual.view(-1, c, *self.slide_window)
            if isinstance(fusion_feature,torch.Tensor) and fusion_feature.shape == feature_residual.shape:
                c = fusion_feature.shape[1]

                fusion_feature = torch.nn.functional.unfold(fusion_feature, self.slide_window,
                                                              stride=self.slide_stride).transpose(1, 2).flatten(0, 1)
                fusion_feature = fusion_feature.view(-1, c, *self.slide_window)

            if masks is not None:

                masks = torch.nn.functional.unfold(masks, self.slide_window,
                                                   stride=self.slide_stride).transpose(1, 2).flatten(0, 1)
                masks = masks.view(-1, 1, *self.slide_window)
                # sample some slide window


            #
                return feature_residual,fusion_feature,masks
            return feature_residual,fusion_feature

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        if self.slide_window is not None:
            c = coord_features.shape[1]

            coord_features = torch.nn.functional.unfold(coord_features,
                                                        (self.slide_window[0] * self.out_stride, self.slide_window[1] * self.out_stride),
                                                          stride=self.slide_stride * self.out_stride).transpose(1, 2).flatten(0,
                                                                                                            1)
            coord_features= coord_features.view(-1, c, self.slide_window[0] * self.out_stride, self.slide_window[1] * self.out_stride)

        return coord_features

    def forward(self, feature_residual,fusion_feature, img_gts=None, if_train=False, args=None, iteration=None, semi=False,
                semi_label=False):
        feature_residual = feature_residual.to('cuda', non_blocking=True)
        orig_gt_mask = img_gts.clone()
        # plt.imshow(orig_gt_mask[-1,0])
        # plt.show()
        if img_gts is not None:
            masks = self.get_gt(img_gts, semi_label, args)

        if if_train and args.aug:
            feature_residual = self.aug(feature_residual, args)

        if self.slide_window is not None:
            if img_gts is not None:
                feature_residual,fusion_feature, masks = self.slide(feature_residual,fusion_feature, args, masks)
                slide_gt = torch.nn.functional.unfold(img_gts,
                                                            (self.slide_window[0] * self.out_stride,
                                                             self.slide_window[1] * self.out_stride),
                                                            stride=self.slide_stride * self.out_stride).transpose(1,
                                                                                                                  2).flatten(0,1)
                slide_gt = slide_gt.view(-1, 1, self.slide_window[0] * self.out_stride,
                                                     self.slide_window[1] * self.out_stride)
                img_gts = slide_gt

            else:
                feature_residual = self.slide(feature_residual, args)

        if if_train:
            if args.aug and not semi:
                feature_residual = feature_residual.view(-1, args.feature_channel, *feature_residual.shape[2:])
                masks = torch.cat([masks] * args.k, dim=1)
                masks = masks.view(feature_residual.shape[0], -1, *masks.shape[2:])  # torch.Size([75, 2, 32, 32])
            masks = masks.to('cuda', non_blocking=True)  # 37,1,32,32
            pred = torch.zeros_like(orig_gt_mask).cuda()

            # print(pred.shape)
            with torch.no_grad():
                if if_train:
                    self.vit.eval()

                num_iters = random.randint(0, args.max_num_next_clicks)

                points = -torch.ones((orig_gt_mask.shape[0], 48, 3)).to('cuda', non_blocking=True)
                if not  self.ad_training:
                    points = get_first_points(pred,orig_gt_mask,points)

                for click_indx in range(1,num_iters):
                    coord_features = self.get_coord_features(orig_gt_mask, pred, points)

                    pred = self.vit((feature_residual,fusion_feature,coord_features),only_ann=True)
                    pred = torch.nn.functional.interpolate(
                        pred,
                        size=(img_gts.shape[2], img_gts.shape[3]),
                        mode="bilinear",
                        align_corners=True,
                    )
                    pred = torch.sigmoid(pred)
                    if args.slide_window is not None:
                        pred = pred.reshape(orig_gt_mask.shape[0], -1,pred.shape[2], pred.shape[3])
                        out = torch.zeros(orig_gt_mask.shape[0],orig_gt_mask.shape[2],orig_gt_mask.shape[3]).cuda()
                        t = torch.zeros(orig_gt_mask.shape[2],orig_gt_mask.shape[3]).cuda()
                        index = 0
                        # print(out.shape,pred.shape)
                        for i in range(0, out.shape[1] - args.slide_window*self.out_stride + 1, args.slide_stride*self.out_stride):
                            for j in range(0, out.shape[2] - args.slide_window*self.out_stride + 1, args.slide_stride*self.out_stride):
                                # print(out[:, i:i + args.slide_window*self.out_stride, j:j + args.slide_window*self.out_stride].shape,pred[:,index].shape)
                                out[:, i:i + args.slide_window*self.out_stride, j:j + args.slide_window*self.out_stride] += pred[:, index]
                                t[i:i + args.slide_window*self.out_stride, j:j + args.slide_window*self.out_stride] += 1
                                index += 1
                        pred = out / t
                        pred = pred.unsqueeze(1)


                    points = get_points(pred, orig_gt_mask, points, click_indx + 1,args)

                if if_train:
                    self.vit.train()

        if args.mask_ratio is None:
            coord_features = self.get_coord_features(orig_gt_mask, pred, points)
            # net_input = torch.cat((feature_residual, prev_output), dim=1)
            if self.slide_window is not  None:

                if args.sample_patch is not None:
                    sample = np.random.choice(feature_residual.shape[0], feature_residual.shape[0] // args.sample_patch,
                                              False)
                    feature_residual = feature_residual[sample]
                    masks = masks[sample]
                    coord_features = coord_features[sample]
                    img_gts = img_gts[sample]

            ann_logits = self.vit((feature_residual,fusion_feature,coord_features), only_ann=True,train=True)
            # if args.block_loss:
                # return self.loss_f(ann_logits,masks).mean()
            ann_logits =  torch.nn.functional.interpolate(
                        ann_logits,
                        size=(img_gts.shape[2], img_gts.shape[3]),
                        mode="bilinear",
                        align_corners=True,
                    )

            # print(ann_logits.shape,slide_gt.shape)
            return self.loss_f(ann_logits, img_gts.cuda()).mean()

    def annotation(self, feature_residual,fusion_feature,img_gts=None,args=None,seg_model=None):
        # feature_residual = feature_residual.to('cuda', non_blocking=True)
        self._object_roi = None
        orig_gt_mask = img_gts.clone()
        if self.slide_window is not None:
            feature_residual,fusion_feature = self.slide(feature_residual,fusion_feature, args)
            slide_gt = torch.nn.functional.unfold(img_gts,
                                                  (self.slide_window[0] * self.out_stride,
                                                   self.slide_window[1] * self.out_stride),
                                                  stride=self.slide_stride * self.out_stride).transpose(1,
                                                                                                        2).flatten(0, 1)
            slide_gt = slide_gt.view(-1, 1, self.slide_window[0] * self.out_stride,
                                     self.slide_window[1] * self.out_stride)
            img_gts = slide_gt
        # prev_output = torch.zeros_like(orig_gt_mask, dtype=torch.float32)
        points = -torch.ones((orig_gt_mask.shape[0], 48, 3)).to('cuda', non_blocking=True)
        pred = torch.zeros_like(orig_gt_mask).cuda()
        coord_features = self.get_coord_features(orig_gt_mask, pred, points)
        with torch.no_grad():
            if seg_model is not  None:
                pred = seg_model.vit(feature_residual)
                pred = torch.softmax(pred, dim=2)[:, :, 1]  # 100,256,1
            else:
                pred = self.vit((feature_residual,fusion_feature,coord_features))
            pred = torch.nn.functional.interpolate(
                pred,
                size=(img_gts.shape[2], img_gts.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
            pred = torch.sigmoid(pred)
            if args.slide_window is not None:
                pred = pred.reshape(orig_gt_mask.shape[0], -1, pred.shape[2], pred.shape[3])
                out = torch.zeros(orig_gt_mask.shape[0], orig_gt_mask.shape[2], orig_gt_mask.shape[3]).cuda()
                t = torch.zeros(orig_gt_mask.shape[2], orig_gt_mask.shape[3]).cuda()
                index = 0
                # print(out.shape,pred.shape)
                for i in range(0, out.shape[1] - args.slide_window * self.out_stride + 1,
                               args.slide_stride * self.out_stride):
                    for j in range(0, out.shape[2] - args.slide_window * self.out_stride + 1,
                                   args.slide_stride * self.out_stride):
                        # print(out[:, i:i + args.slide_window*self.out_stride, j:j + args.slide_window*self.out_stride].shape,pred[:,index].shape)
                        out[:, i:i + args.slide_window * self.out_stride,
                        j:j + args.slide_window * self.out_stride] += pred[:, index]
                        t[i:i + args.slide_window * self.out_stride, j:j + args.slide_window * self.out_stride] += 1
                        index += 1
                pred = out / t
                pred = pred.unsqueeze(1)

            ious_list = []
            best_iou = get_iou(orig_gt_mask, pred.cpu() > args.pred_thres)
            # best_pred = pred
            if args.max_num_next_clicks == 0:
                return pred, self.dist_maps(img_gts, points), best_iou


            for click_indx in range(1, 1+ args.max_num_next_clicks):
                points = get_points(pred, orig_gt_mask, points, click_indx, args,pred_thresh=args.pred_thres)

                coord_features = self.get_coord_features(orig_gt_mask, pred, points)

                pred = self.vit((feature_residual,fusion_feature, coord_features),only_ann=True)
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=(img_gts.shape[2], img_gts.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                pred = torch.sigmoid(pred)
                if args.slide_window is not None:
                    pred = pred.reshape(orig_gt_mask.shape[0], -1, pred.shape[2], pred.shape[3])
                    out = torch.zeros(orig_gt_mask.shape[0], orig_gt_mask.shape[2], orig_gt_mask.shape[3]).cuda()
                    t = torch.zeros(orig_gt_mask.shape[2], orig_gt_mask.shape[3]).cuda()
                    index = 0
                    # print(out.shape,pred.shape)
                    for i in range(0, out.shape[1] - args.slide_window * self.out_stride + 1,
                                   args.slide_stride * self.out_stride):
                        for j in range(0, out.shape[2] - args.slide_window * self.out_stride + 1,
                                       args.slide_stride * self.out_stride):
                            # print(out[:, i:i + args.slide_window*self.out_stride, j:j + args.slide_window*self.out_stride].shape,pred[:,index].shape)
                            out[:, i:i + args.slide_window * self.out_stride,
                            j:j + args.slide_window * self.out_stride] += pred[:, index]
                            t[i:i + args.slide_window * self.out_stride, j:j + args.slide_window * self.out_stride] += 1
                            index += 1
                    pred = out / t
                    pred = pred.unsqueeze(1)


                iou = get_iou(orig_gt_mask,pred.cpu()>args.pred_thres)
                if iou > best_iou:
                    best_pred = pred
                    best_iou = iou
                ious_list.append(iou)
            # pred,self._object_roi = get_ori_pred(pred, points, args.pred_thres, self._object_roi, recompute_thresh_iou=0.5)
            # iou = get_iou(orig_gt_mask, pred.cpu() > args.pred_thres)
            return pred,self.dist_maps(orig_gt_mask, points),iou,ious_list

