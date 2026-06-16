 # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from torch.nn import CrossEntropyLoss

from model.is_plainvit_model import SimpleFPN_4_32, SwinTransfomerSegHead
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import adapt_input_conv, resize_pos_embed

from model.util import prepocess_residual, random_masking


class ViT(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, stride=8,global_pool=False,residual_method='square', **kwargs):
        super(ViT, self).__init__(**kwargs)
        self.stride = stride
        self.patch_size = kwargs['patch_size']
        self.img_size = kwargs['img_size']
        self.global_pool = global_pool
        self.residual_method = residual_method
        print(f'model residual:{self.residual_method}')
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x, mask_ratio=None):
        mask_pos = None
        if mask_ratio is not None:
            mask_pos = random_masking(x, mask_ratio)
            mask_pos = mask_pos.view(-1, 1, *x.shape[2:])
            # print(x[mask_pos.squeeze()==1])
            x = x * mask_pos
            # mask_pos = mask_pos.flatten(1,3)
            # print(mask_pos.shape)
            # print(x.shape,mask_pos.shape)
            # print(x[mask_pos.squeeze()==1])

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x

        return outcome, mask_pos

    def forward(self, x, mask_ratio=None):
        x = prepocess_residual(x,self.residual_method)
        x, mask_pos = self.forward_features(x, mask_ratio)
        logits = self.forward_head(x)[:, 1:]
        if self.num_classes == 1:
            logits = torch.sigmoid(logits)
        if mask_ratio is not None:
            return logits, mask_pos
        return logits

class VitTwoHead(timm.models.vision_transformer.VisionTransformer):

    def __init__(self, stride=8, click_chans=3, **kwargs):
        super(VitTwoHead, self).__init__( no_embed_class=True,**kwargs)
        self.patch_embed_coords = PatchEmbed(
            img_size=(kwargs['img_size'][0]*stride,kwargs['img_size'][1]*stride), patch_size=8, in_chans=click_chans, embed_dim=kwargs['embed_dim'])



        neck_params = dict(
            in_dim=1024,#1280,
            out_dims=[240, 480, 960, 1920],
        )

        head_params = dict(
            in_channels=[240, 480, 960, 1920],
            in_index=[0, 1, 2, 3],
            dropout_ratio=0.1,
            num_classes=1,
            loss_decode=CrossEntropyLoss(),
            align_corners=False,
            upsample='x1',
            channels={'x1': 256, 'x2': 128, 'x4': 64}['x1'],
        )
        # self.neck = SimpleFPN(**neck_params)
        self.neck = SimpleFPN_4_32(**neck_params)
        # self.neck = Fusion_coord(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)
        self.maps_transform = nn.Identity()

    def ann_forward_features(self, x,coord_features, mask_ratio=None,train=False):
        mask_pos = None
        if mask_ratio is not None:
            mask_pos = self.random_masking(x, mask_ratio)
            mask_pos = mask_pos.view(-1, 1, *x.shape[2:])
            # print(x[mask_pos.squeeze()==1])
            x = x * mask_pos
            # mask_pos = mask_pos.flatten(1,3)
            # print(mask_pos.shape)
            # print(x.shape,mask_pos.shape)
            # print(x[mask_pos.squeeze()==1])
        # print(x.shape,coord_features.shape)
        # print(self.patch_embed(x).shape, self.patch_embed_coords(coord_features).shape)
        x = self.patch_embed(x)
        # if train:
        #     idx = torch.randperm(x.size(0))
        #     # print(idx[:idx.shape[0]//2])
        #     x[idx[:idx.shape[0]//2]] = x[idx[:idx.shape[0]//2]] + self.patch_embed_coords(coord_features[:idx.shape[0]//2])
        # else:
        x = x + self.patch_embed_coords(coord_features)

        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)  # B L C

        return x, mask_pos

    def forward(self, input, mask_ratio=None, only_ann=False, train=False, backbone_features=None):
        residual, coord_features,image = input
        residual = residual**2
        if image is not None:
            coord_features = torch.cat([coord_features,image],dim=1)
        # if coord_features is None: return self.seg_head(residual)

        coord_features = self.maps_transform(coord_features)
        backbone_features,mask_ratio = self.ann_forward_features(residual,coord_features,mask_ratio,train)
        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1, -2).view(B, C, grid_size[0],grid_size[1])

        multi_scale_features = self.neck(backbone_features)

        return self.head(multi_scale_features)


    def seg_head(self, image, mask_ratio=None):

        x, mask_pos = self.forward_features(image, mask_ratio)
        logits = self.forward_head(x)


        if mask_ratio is not None:
            return logits, mask_pos
        return logits
    def forward_ann_head(self,image, x,coord_features, pre_logits: bool = False):
        # x = image.permute(0,2,3,1).flatten(1,2) + x + self.patch_embed_coords(coord_features)
        x = x + self.patch_embed_coords(coord_features)
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        return x if pre_logits else self.ann_head(x)


    def prepare_input(self, image):
        # prev_mask = None
        prev_mask = image[:, -1:, :, :]
        image = image[:, :-1, :, :]

        return image, prev_mask

