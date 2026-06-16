import math
from collections import OrderedDict
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from model.loss import focal_loss, semi_loss, consistency_loss
from model.model_unet import DiscriminativeSubNetwork
from model.models_swin import Swin


from timm.models.layers import trunc_normal_

from model.models_vit import ViT
from semilearn.algorithms.freematch.freematch import entropy_loss
from semilearn.core.hooks import Hook, get_priority
from semilearn.freematch import FreeMatchThresholingHook
from semilearn.hooks import PseudoLabelingHook


class Base_detector(nn.Module):
    def __init__(self,image_size, stride = 8,patch_size=(1,1),
                 slide_window=None,slide_stride=None):
        super(Base_detector, self).__init__()

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

        self.vit = None

    def freematch_init(self):

        self.clip_thresh = False
        # set common hooks during training
        self._hooks = []  # record underlying hooks

        self.hooks_dict = OrderedDict() # actual object to be used to call hooks
        self.set_hooks()

        self.distributed = False
        self.use_quantile = True
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
    def guess_label(self,feature_residual,args):
        """
        follow MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)
        Args:
            feature_residual:
            args:

        Returns:

        """
        with torch.no_grad():
            out = self.vit(feature_residual)
            out = torch.softmax(out, dim=-1)
            out = out.view(-1, args.k, *out.shape[1:])
            out = torch.mean(out, dim=1)
            # sharpen
            out = torch.pow(out, 1. / args.T)
            # print(torch.sum(out,dim=2,keepdim=True).shape)
            out /= torch.sum(out, dim=2, keepdim=True)
            out = out.detach().cpu()  # 25,1024,2
            out = out.permute(0, 2, 1)
            out = out.view(*out.shape[:2], *self.vit_img_size)  # 25,2,32,32

            return out
    def guess_mixup(self,feature_residual,masks,args):
        feature_residual = feature_residual.view(-1, args.feature_channel, *feature_residual.shape[2:])
        out = self.guess_label(feature_residual, args)

        all_targets = torch.cat([torch.ones_like(masks) - masks, masks], dim=1)

        label_kinds = self.get_label_kind(all_targets)  # 75 2 32 32
        all_targets[label_kinds == 0] = out[label_kinds == 0]
        n, c, h, w = feature_residual.shape  # torch.Size([75, 1024, 32, 32])
        all_targets = torch.cat([all_targets] * args.k, dim=1)
        all_targets = all_targets.reshape(n, 2, h, w)

        label_kinds = torch.cat([label_kinds] * args.k, dim=1)
        label_kinds = label_kinds.view(n, 2, h, w)
        all_inputs = feature_residual.permute(0, 2, 3, 1).flatten(0, 2)
        all_targets = all_targets.permute(0, 2, 3, 1).flatten(0, 2)

        label_kinds = label_kinds.permute(0, 2, 3, 1).flatten(0, 2)
        # print(all_targets.shape,all_inputs.shape)# 76800,1024
        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1 - l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        feature_residual = l * input_a + (1 - l) * input_b
        masks = l * target_a + (1 - l) * target_b

        sample_idx = idx[label_kinds[:, 0] == 1]
        sample_idx = sample_idx[math.floor(sample_idx.shape[0] * args.mixup_ratio):]
        feature_residual[sample_idx] = input_a[sample_idx]
        masks[sample_idx] = target_a[sample_idx]
        feature_residual = feature_residual.view(n, h, w, c).permute(0, 3, 1, 2)
        return feature_residual,masks,label_kinds
    def get_gt(self,img_gts,semi_label,args):
        # if masks.shape[1] != self.feature_size[0]*self.out_stride or masks.shape[2] != self.feature_size[1]*self.out_stride:
        #     masks = nn.functional.interpolate(masks, size=(self.feature_size[0]*self.out_stride,self.feature_size[1]*self.out_stride),mode='nearest')

        # 512*512分成8*8小块
        #
        out_h,out_w = img_gts.shape[2]//args.block_size,img_gts.shape[3]//args.block_size
        img_gts = torch.nn.functional.unfold(img_gts, args.block_size,
                                             stride=args.block_size).transpose(1, 2)
        if semi_label:
            masks = -torch.ones([*img_gts.shape[:2]])
            masks[torch.mean((img_gts == 1).float(), dim=-1) >= args.semi_pos_thres] = 1  # 正样本
            masks[torch.mean((img_gts == -1).float(), dim=-1) >= args.semi_neg_thres] = 0  # 负样本
            # print((masks==1).sum())
        else:
            masks = torch.mean(img_gts, dim=-1)  # img_gts的值是0或1， 8*8一个单元计算缺陷面积占比
            masks[masks >= args.gt_thres1] = 1
            masks[masks <= args.gt_thres2] = 0

        # print(masks.shape)
        masks = masks.view(masks.shape[0], 1, out_h,out_w)
        if out_h != self.feature_size[0] or out_w != self.feature_size[1]:
            masks = torch.nn.functional.interpolate(masks,self.feature_size,mode='nearest')
        # plt.imshow(masks[-1,0])
        # plt.show()
        return masks

    def register_hook(self, hook, name=None, priority='NORMAL'):
        """
        Ref: https://github.com/open-mmlab/mmcv/blob/a08517790d26f8761910cac47ce8098faac7b627/mmcv/runner/base_runner.py#L263
        Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            hook_name (:str, default to None): Name of the hook to be registered. Default is the hook class name.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        hook.name = name if name is not None else type(hook).__name__

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break

        if not inserted:
            self._hooks.insert(0, hook)

        # call set hooks
        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook

    # follow freematch
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FreeMatchThresholingHook(num_classes=2, momentum=0.999), "MaskingHook")
        # super().set_hooks()

    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            hook_name (str): The specific hook name to be called, such as
                "param_update" or "dist_align", uesed to call single hook in train_step.
        """

        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)

        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

    def freematch(self, feature_residual,img_gts=None,args=None,iteration=None,semi=False,semi_label=False):
        feature_residual = feature_residual.to('cuda',non_blocking=True)
        if img_gts is not None:
            masks = self.get_gt(img_gts, semi_label, args)

        feature_residual = self.aug(feature_residual,args)

        if self.slide_window is not None:
            c = feature_residual.shape[1]
            feature_residual = torch.nn.functional.unfold(feature_residual, self.slide_window,
                                                          stride=self.slide_stride).transpose(1, 2).flatten(0, 1)
            feature_residual = feature_residual.view(-1, c, *self.slide_window)

            masks = torch.nn.functional.unfold(masks, self.slide_window,
                                               stride=self.slide_stride).transpose(1, 2).flatten(0, 1)
            masks = masks.view(-1, 1, *self.slide_window)
            # sample some slide window
            if args.sample_patch is not None:
                sample = np.random.choice(feature_residual.shape[0],feature_residual.shape[0]//args.sample_patch,False)
                feature_residual = feature_residual[sample]
                masks = masks[sample]

        feature_residual = feature_residual.view(-1, args.feature_channel, *feature_residual.shape[2:])
        masks = torch.cat([masks]*args.k,dim=1)
        masks = masks.view(-1)

        label_kinds = self.get_label_kind(masks)

        logits = self.vit(feature_residual)
        # print(logits.shape,masks.shape,feature_residual.shape)# torch.Size([2, 1024, 2]) torch.Size([2048]) torch.Size([2, 1024, 32, 32])

        logits = logits.reshape(-1, 2)

        masks = masks.to('cuda',non_blocking=True)
        # print(label_kinds.shape,logits.shape,masks.shape)
        sup_loss = focal_loss(logits[label_kinds == 1], masks[label_kinds == 1], alpha=args.focal_loss_alpha,
                            gamma=args.focal_loss_gamma, beta=args.focal_loss_beta)if (label_kinds == 1).sum() > 0 else 0

        if (label_kinds == 0).sum() > 0:
            logits_x_ulb_w = logits[label_kinds==0]
            # calculate mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=logits_x_ulb_w,
                                          use_hard_label=True,
                                          T=args.T)
            logits,mask_pos = self.vit(feature_residual,mask_ratio=args.mask_ratio)
            # print(logits.shape,mask_pos.shape) # torch.Size([5, 1024, 2]) torch.Size([5, 1024, 1])
            logits = logits.reshape(-1, 2)
            mask_pos = mask_pos.view(-1)
            logits_x_ulb_s = logits[label_kinds == 0]
            mask_pos = mask_pos[label_kinds == 0]
            if (mask_pos == 1).sum() > 0:
                # 只计算没有mask的loss
                logits_x_ulb_s = logits_x_ulb_s[mask_pos == 1]
                pseudo_label = pseudo_label[mask_pos == 1]
                mask = mask[mask_pos == 1]
                # print(logits_x_ulb_s.shape,pseudo_label.shape,mask.shape)
                # calculate unlabeled loss
                if args.sat == 1:
                    w = args.lambda_u
                    unsup_loss = w*consistency_loss(logits_x_ulb_s,
                                                       pseudo_label,
                                                       'focal_loss',
                                                       mask=mask,args=args)
                if args.sat == -1:
                    w = args.lambda_u * iteration/400
                    unsup_loss = w*consistency_loss(logits_x_ulb_s,
                                                  pseudo_label,
                                                  'focal_loss', args=args)
            else:
                unsup_loss = 0.0
            # print(sup_loss,unsup_loss)
            # calculate entropy loss
            if mask.sum() > 0:
                ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
            else:
                ent_loss = 0.0
            # ent_loss = 0.0
            total_loss = sup_loss + unsup_loss + args.lambda_e * ent_loss

            return total_loss
        return sup_loss
    def forward(self, feature_residual,img_gts=None, if_train=False,args=None,iteration=None,semi=False,semi_label=False):
        feature_residual = feature_residual.to('cuda',non_blocking=True)
        if img_gts is not None:

            masks = self.get_gt(img_gts, semi_label, args)

        if if_train and args.aug:
            feature_residual = self.aug(feature_residual,args)

        if self.slide_window is not None:
            # print(feature_residual.shape)
            c = feature_residual.shape[1]


            feature_residual = torch.nn.functional.unfold(feature_residual, self.slide_window,
                                                          stride=self.slide_stride).transpose(1, 2).flatten(0, 1)
            feature_residual = feature_residual.view(-1, c, *self.slide_window)

            if if_train :
                masks = torch.nn.functional.unfold(masks, self.slide_window,
                                                   stride=self.slide_stride).transpose(1, 2).flatten(0, 1)
                masks = masks.view(-1, 1, *self.slide_window)
                # sample some slide window
                if args.sample_patch is not None:
                    if args.slide_stride == 1:
                        sample = np.random.choice(feature_residual.shape[0], args.sample_patch, False)
                    else:
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
        # print(feature_residual.shape)
        if not if_train or args.mask_ratio is None:
            logits = self.vit(feature_residual)
        else:
            mask_ratio = random.uniform(0, args.mask_ratio)
            logits,mask_pos = self.vit(feature_residual,mask_ratio)
            mask_pos = mask_pos.view(*logits.shape[:2])
            logits = logits[mask_pos==1]
            if masks.ndim == 4:
                mask_pos = mask_pos.view(*masks.shape)
            if masks.ndim == 2:
                mask_pos = mask_pos.view(masks.shape[0])

            masks = masks[mask_pos==1]
            if semi:
                label_kinds = label_kinds[mask_pos.cpu()==1]
        if if_train:

            logits = logits.reshape(-1, 2)



            if semi:


                return semi_loss(logits,masks,label_kinds,iteration,args)

            else:
                # print((masks==1).sum())
                # 样本丢弃
                # plt.imshow(masks[-1,0].cpu())
                # plt.show()
                masks = masks.view(-1)
                logits = torch.cat([logits[masks == 1], logits[masks == 0]])
                masks = torch.cat([masks[masks == 1], masks[masks == 0]])
                loss = focal_loss(logits, masks, alpha=args.focal_loss_alpha,
                                  gamma=args.focal_loss_gamma,beta=args.focal_loss_beta)

                return loss

        return logits


class Swin_detector(Base_detector):
    def __init__(self,image_size, stride = 8,patch_size=(1,1),
                 slide_window=None,slide_stride=None,in_chans=1024,num_classes=2,embed_dim=1024,window_size=4,depths=(4,),num_heads=(32,)):
        super(Swin_detector, self).__init__(image_size, stride,patch_size,
                                            slide_window,slide_stride)
        self.out_stride = stride//len(depths)
        self.vit = Swin(img_size=self.vit_img_size, stride=self.out_stride,in_chans=in_chans,num_classes=num_classes,window_size=window_size,
                             patch_size=patch_size, embed_dim=embed_dim, depths=depths,
                       num_heads=num_heads)


        # manually initialize fc layer
        trunc_normal_(self.vit.head.weight, std=2e-5)
class Vit_detector(Base_detector):
    def __init__(self,image_size, stride = 8,patch_size=(1,1),
                 slide_window=None,slide_stride=None,in_chans=1024,num_classes=2,embed_dim=1024,window_size=4,depths=(4,),num_heads=(32,)):
        super(Vit_detector, self).__init__(image_size, stride,patch_size,
                                            slide_window,slide_stride)
        self.out_stride = stride#//len(depths)
        self.vit = ViT(img_size=self.vit_img_size, stride=self.out_stride,in_chans=in_chans,num_classes=num_classes,
                             patch_size=patch_size, embed_dim=embed_dim, depth=depths,
                       num_heads=num_heads)


        # manually initialize fc layer
        trunc_normal_(self.vit.head.weight, std=2e-5)

class Unet_detector(Base_detector):
    def __init__(self,image_size, stride = 8,patch_size=(1,1),
                 slide_window=None,slide_stride=None,in_chans=1024,num_classes=2,embed_dim=1024,window_size=4,depths=(4,),num_heads=(32,)):
        super(Unet_detector, self).__init__(image_size, stride,patch_size,
                                            slide_window,slide_stride)
        self.out_stride = stride#//len(depths)
        self.vit = DiscriminativeSubNetwork(in_channels=in_chans,out_channels=num_classes,base_channels=embed_dim)
