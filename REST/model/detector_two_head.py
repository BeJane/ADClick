import math
from collections import OrderedDict
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from model.isegm import DistMaps, get_ori_pred
from model.loss import focal_loss, semi_loss, consistency_loss, sigmoid_adaptive_focal_loss, NormalizedFocalLossSigmoid

from model.models_swin import Swin, SwinTwoHead, ClickSwin

from timm.models.layers import trunc_normal_

from model.models_vit import VitTwoHead
from model.point import get_points, get_first_points
from model.util import get_iou, random_crop_feature


class DetectorTwoHead(nn.Module):
    def __init__(self,image_size, stride = 8,patch_size=(1,1),norm_radius=5,click_chans=3,ad_training=False,no_head=False,
                 slide_window=None,slide_stride=None,in_chans=1024,num_classes=2,embed_dim=1024,window_size=4,depths=(4,),num_heads=(32,),
                 backbone='swin'):
        super(DetectorTwoHead, self).__init__()

        feature_size = (image_size[1] // stride,image_size[2]//stride)
        self.slide_window = slide_window
        self.ad_training = ad_training
        if self.ad_training:        print(f'AD Training! Reduce the number of clicks!')
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
        if backbone == 'swin':
            if no_head:
                print("No Neck and MLP head!")
                self.vit = ClickSwin(img_size=self.vit_img_size, stride=self.out_stride,in_chans=in_chans,num_classes=num_classes,window_size=window_size,
                                     patch_size=patch_size, embed_dim=embed_dim, depths=depths,click_chans=click_chans,
                               num_heads=num_heads)
            else:
                self.vit = SwinTwoHead(img_size=self.vit_img_size, stride=self.out_stride, in_chans=in_chans,
                                     num_classes=num_classes, window_size=window_size,
                                     patch_size=patch_size, embed_dim=embed_dim, depths=depths, click_chans=click_chans,
                                     num_heads=num_heads)

        # if backbone == 'mamba':
        #     if no_head:
        #         print("No Neck and MLP head!")
        #         print("No implement")
        #     else:
        #         self.vit = MambaTwoHead(img_size=self.vit_img_size, stride=self.out_stride, in_chans=in_chans,
        #                                num_classes=num_classes,
        #                                patch_size=patch_size, embed_dim=embed_dim, depth=depths[0],
        #                                click_chans=click_chans,
        #                                num_heads=num_heads[0])
        self.loss_f = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)

        self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0,
                                  cpu_mode=False, use_disks=True)



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

    def slide(self,feature_residual,args,masks=None):
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

            if masks is not None:

                masks = torch.nn.functional.unfold(masks, self.slide_window,
                                                   stride=self.slide_stride).transpose(1, 2).flatten(0, 1)
                masks = masks.view(-1, 1, *self.slide_window)
                # sample some slide window


            #
                return feature_residual,masks
            return feature_residual
    def get_coord_features(self, prev_mask, points):
        coord_features = self.dist_maps(prev_mask, points)
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

    def forward(self, feature_residual, img_gts=None, image=None,if_train=False, args=None, iteration=None, semi=False,
                semi_label=False):
        feature_residual = feature_residual.to('cuda', non_blocking=True)
        if img_gts is not None:
            masks = self.get_gt(img_gts, semi_label, args)

        if if_train and args.aug:
            feature_residual = self.aug(feature_residual, args)

        if self.slide_window is not None:
            if img_gts is not None:
                feature_residual, masks = self.slide(feature_residual, args, masks)
            else:
                feature_residual = self.slide(feature_residual, args)

        if if_train:
            if args.aug and not semi:
                feature_residual = feature_residual.view(-1, args.feature_channel, *feature_residual.shape[2:])
                masks = torch.cat([masks] * args.k, dim=1)
                masks = masks.view(feature_residual.shape[0], -1, *masks.shape[2:])  # torch.Size([75, 2, 32, 32])
            #
            # feature_residual, img_gts = random_crop_feature(feature_residual,
            #                                                 (feature_residual.shape[2] // 2, feature_residual.shape[2]),
            #                                                 feature_residual.shape[2:], [img_gts])
            # img_gts = img_gts[0]
            # img_gts[img_gts > 0.5] = 1
            # img_gts[img_gts < 1] = 0

            orig_gt_mask = img_gts.clone()
            pred = torch.zeros_like(orig_gt_mask).cuda()

            # print(pred.shape)
            with torch.no_grad():
                if if_train:
                    self.vit.eval()

                num_iters = random.randint(1, args.max_num_next_clicks)

                points = -torch.ones((img_gts.shape[0], 48, 3)).to('cuda', non_blocking=True)
                if not self.ad_training:
                    points = get_first_points(pred,orig_gt_mask,points)
                # else:
                #     if np.random.uniform() < 0.5:num_iters=0
                # print(torch.unique(points))
                # points = get_points(pred, orig_gt_mask, points, 1,args)

                for click_indx in range(1,num_iters+1):
                    coord_features = self.get_coord_features(pred, points)
                    pred = self.vit((feature_residual,coord_features,image),only_ann=True)

                    pred = torch.sigmoid(pred)

                    if args.slide_window is not None:
                        pred = pred.reshape(orig_gt_mask.shape[0], -1, args.slide_window, args.slide_window)
                        out = torch.zeros((orig_gt_mask.shape[0], *self.feature_size), device='cuda')
                        t = torch.zeros(self.feature_size, device='cuda')
                        index = 0
                        for i in range(0, self.feature_size[0] - args.slide_window + 1, args.slide_stride):
                            for j in range(0, self.feature_size[1] - args.slide_window + 1, args.slide_stride):
                                out[:, i:i + args.slide_window, j:j + args.slide_window] += pred[:, index]
                                t[i:i + args.slide_window, j:j + args.slide_window] += 1
                                index += 1
                        pred = out / t
                        pred = pred.unsqueeze(1)
                    pred = torch.nn.functional.interpolate(
                        pred,
                        size=(img_gts.shape[2], img_gts.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    points = get_points(pred, orig_gt_mask, points, click_indx + 1,args)

                if if_train:
                    self.vit.train()

        if args.mask_ratio is None:
            coord_features = self.get_coord_features( pred, points)
            # net_input = torch.cat((feature_residual, prev_output), dim=1)
            if self.slide_window is not  None:
                slide_gt = torch.nn.functional.unfold(orig_gt_mask,
                                                            (self.slide_window[0] * self.out_stride,
                                                             self.slide_window[1] * self.out_stride),
                                                            stride=self.slide_stride * self.out_stride).transpose(1,
                                                                                                                  2).flatten(0,1)
                slide_gt = slide_gt.view(-1, 1, self.slide_window[0] * self.out_stride,
                                                     self.slide_window[1] * self.out_stride)
                img_gts = slide_gt
                if args.sample_patch is not None:
                    sample = np.random.choice(feature_residual.shape[0], feature_residual.shape[0] // args.sample_patch,
                                              False)
                    feature_residual = feature_residual[sample]
                    coord_features = coord_features[sample]
                    img_gts = img_gts[sample]


            ann_logits = self.vit((feature_residual,coord_features,image), only_ann=True,train=True)

            ann_logits =  torch.nn.functional.interpolate(
                        ann_logits,
                        size=(img_gts.shape[2], img_gts.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
            # print(ann_logits.shape,img_gts.shape)
            return self.loss_f(ann_logits, img_gts.cuda()).mean()

    def annotation(self, feature_residual,img_gts=None,image=None,args=None,seg_model=None):

        self._object_roi = None
        if self.slide_window is not None:
            feature_residual= self.slide(feature_residual, args)
        orig_gt_mask = img_gts.clone()
        # prev_output = torch.zeros_like(orig_gt_mask, dtype=torch.float32)
        points = -torch.ones((img_gts.shape[0], 48, 3)).to('cuda', non_blocking=True)
        pred = torch.zeros_like(orig_gt_mask).cuda()
        coord_features = self.get_coord_features( pred, points)
        with torch.no_grad():
            if seg_model is not None:
                pred = seg_model.vit(feature_residual)
                pred = torch.softmax(pred, dim=2)[:, :, 1]  # 100,256,1
            else:
                pred = self.vit((feature_residual, coord_features,image))

            # pred = torch.sigmoid(pred)
            if args.slide_window is not None:
                pred = pred.reshape(orig_gt_mask.shape[0], -1, args.slide_window, args.slide_window)
                out = torch.zeros((orig_gt_mask.shape[0], *self.feature_size), device='cuda')
                t = torch.zeros(self.feature_size, device='cuda')
                index = 0
                for i in range(0, self.feature_size[0] - args.slide_window + 1, args.slide_stride):
                    for j in range(0, self.feature_size[1] - args.slide_window + 1, args.slide_stride):
                        out[:, i:i + args.slide_window, j:j + args.slide_window] += pred[:, index]
                        t[i:i + args.slide_window, j:j + args.slide_window] += 1
                        index += 1
                pred = out / t
                pred = pred.unsqueeze(1)
            pred = torch.nn.functional.interpolate(
                pred,
                size=(img_gts.shape[2], img_gts.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
            pred = torch.sigmoid(pred)
            ious_list = []
            best_iou = get_iou(orig_gt_mask, pred.cpu() > args.pred_thres)
            # best_pred = pred
            if args.max_num_next_clicks == 0:
                return pred, self.dist_maps(img_gts, points), best_iou

            for click_indx in range(1, 1 + args.max_num_next_clicks):
                points = get_points(pred, orig_gt_mask, points, click_indx, args, pred_thresh=args.pred_thres)
                # _, self._object_roi = get_ori_pred(pred, points, args.pred_thres, self._object_roi,
                #                                       recompute_thresh_iou=0.5)
                #
                # rmin, rmax, cmin, cmax = self._object_roi

                #
                # new_pred = torch.nn.functional.interpolate(
                #     pred[:,:,rmin:rmax+1,cmin:cmax+1],
                #     size=(pred.shape[2], pred.shape[3]),
                #     mode="bilinear",
                #     align_corners=True,
                # )
                # new_point = points.clone()
                # for i in range(points.shape[1]):
                #     if points[0,i,0] != -1:
                #         new_point[0,i,0] = new_pred.shape[2] * (points[0,i,0]  - rmin) / (rmax - rmin + 1)
                #         new_point[0,i,1]  = new_pred.shape[3] * (points[0,i,1]  - cmin) / (cmax - cmin + 1)
                # print(points[0,i],new_point[0,i],rmin,rmax)
                coord_features = self.get_coord_features( pred, points)
                # coord_features = self.get_coord_features( new_pred, new_point)
                # coord_features= torch.nn.functional.interpolate(
                #     coord_features[:,:,rmin:rmax+1,cmin:cmax+1],
                #     size=(pred.shape[2], pred.shape[3]),
                #     mode="bilinear",
                #     align_corners=True,
                # )

                # print(rmin,rmin//self.out_stride)
                # crop_feature_residual = torch.nn.functional.interpolate(
                #     feature_residual[:,:,round(rmin/self.out_stride):round((rmax+1)/self.out_stride),
                #     round(cmin / self.out_stride):round((cmax+1)/self.out_stride)],
                #     size=(feature_residual.shape[2], feature_residual.shape[3]),
                #     mode="bilinear",
                #     align_corners=True,
                # )
                # if isinstance(fusion_feature,(list,tuple)):
                #     crop_fusion_feature = []
                #     scale = [4,8,16,32]
                #     for i,f in enumerate(fusion_feature):
                #         crop_fusion_feature.append(
                #             torch.nn.functional.interpolate(
                #                 f[:, :, round(rmin / scale[i]):round((rmax +1) / scale[i]),
                #                 round(cmin / scale[i]):round((cmax+1) / scale[i])],
                #                 size=(f.shape[2], f.shape[3]),
                #                 mode="bilinear",
                #                 align_corners=True,
                #             )
                #         )
                #
                # else:
                #     crop_fusion_feature = torch.nn.functional.interpolate(
                #         fusion_feature[:,:,round(rmin/self.out_stride):round((rmax+1)/self.out_stride),
                #     round(cmin / self.out_stride):round((cmax+1)/self.out_stride)],
                #         size=(feature_residual.shape[2], feature_residual.shape[3]),
                #         mode="bilinear",
                #         align_corners=True,
                #     )
                pred = self.vit((feature_residual,  coord_features,image), only_ann=True)
                # plt.imshow(pred[0,0].cpu())
                # plt.show()
                # pred = torch.softmax(pred, dim=2)[:, :, 1]  # 100,256,1
                if args.slide_window is not None:
                    pred = pred.reshape(orig_gt_mask.shape[0], -1, args.slide_window, args.slide_window)
                    out = torch.zeros((orig_gt_mask.shape[0], *self.feature_size), device='cuda')
                    t = torch.zeros(self.feature_size, device='cuda')
                    index = 0
                    for i in range(0, self.feature_size[0] - args.slide_window + 1, args.slide_stride):
                        for j in range(0, self.feature_size[1] - args.slide_window + 1, args.slide_stride):
                            out[:, i:i + args.slide_window, j:j + args.slide_window] += pred[:, index]
                            t[i:i + args.slide_window, j:j + args.slide_window] += 1
                            index += 1
                    pred = out / t
                    pred = pred.unsqueeze(1)
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=(img_gts.shape[2], img_gts.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                pred = torch.sigmoid(pred)
                # new_pred = torch.zeros_like(orig_gt_mask)
                # new_pred[:,:,rmin:rmax+1,cmin:cmax+1] = pred
                # pred = new_pred.cuda()
                # pred = self.anomaly_segmentor.convert_to_segmentation(pred.squeeze(1))
                # pred = torch.tensor(np.array(pred)).unsqueeze(1).cuda()
                iou = get_iou(orig_gt_mask, pred.cpu() > args.pred_thres)
                if iou > best_iou:
                    best_pred = pred
                    best_iou = iou
                ious_list.append(iou)
            # pred,self._object_roi = get_ori_pred(pred, points, args.pred_thres, self._object_roi, recompute_thresh_iou=0.5)
            # iou = get_iou(orig_gt_mask, pred.cpu() > args.pred_thres)
            return pred, self.dist_maps(img_gts, points), iou,ious_list


