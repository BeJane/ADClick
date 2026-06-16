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

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.layers.patch_embed import PatchEmbed
from timm.models.vision_transformer import adapt_input_conv, resize_pos_embed

from model.fusion import FusionModule, FusionConv1x1
from model.is_plainvit_model import SimpleFPN, SwinTransfomerSegHead, SimpleFPN_4_32, Fusion_coord
from model.isegm import DistMaps
from model.new_swin import SwinTransformer
from model.transformer_helper.cross_entropy_loss import CrossEntropyLoss
from model.util import prepocess_residual, random_masking


class Multi_win_Swin(SwinTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, stride=8, global_pool=False, **kwargs):
        super(Multi_win_Swin, self).__init__(**kwargs)
        self.stride = stride
        self.patch_size = kwargs['patch_size']
        self.img_size = kwargs['img_size']

        self.global_pool = global_pool
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
        x = self.patch_embed(x)

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C

        return x, mask_pos

    def forward(self, x, mask_ratio=None):

        x = x ** 2
        x, mask_pos = self.forward_features(x, mask_ratio)
        logits = self.forward_head(x)
        if self.num_classes == 1:
            logits = torch.sigmoid(logits)
        if mask_ratio is not None:
            return logits, mask_pos
        return logits

class Swin(timm.models.swin_transformer.SwinTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, stride=8, residual_method='square',image_cls=False,**kwargs):
        super(Swin, self).__init__(global_pool='',**kwargs)
        self.stride = stride
        self.patch_size = kwargs['patch_size']
        self.img_size = kwargs['img_size']
        self.residual_method = residual_method
        self.image_cls = image_cls
        if self.image_cls:
            self.image_head = nn.Linear(self.num_features, kwargs['num_classes'])
        print(f'model residual:{self.residual_method}')



    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L = x.shape[0],x.shape[2]*x.shape[3]
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        # ids_keep = ids_shuffle[:, :len_keep]
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 1 is keep, 0 is remove
        mask = torch.zeros([N, L], device=x.device)
        mask[:, :len_keep] = 1
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask

    def forward_features(self, x,mask_ratio=None):
        mask_pos = None
        if mask_ratio is not None:
            mask_pos= self.random_masking(x, mask_ratio)
            mask_pos = mask_pos.view(-1,1,*x.shape[2:])
            # print(x[mask_pos.squeeze()==1])
            x = x * mask_pos
            # mask_pos = mask_pos.flatten(1,3)
            # print(mask_pos.shape)
            # print(x.shape,mask_pos.shape)
            # print(x[mask_pos.squeeze()==1])
        x = self.patch_embed(x)

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C
        
        return x,mask_pos

    def forward(self, x, mask_ratio=None):

        x = prepocess_residual(x, self.residual_method)
        x,mask_pos = self.forward_features(x,mask_ratio)
        logits = self.forward_head(x)
        if self.image_cls:
            logits = (logits,self.image_head(x.mean(dim=1)))
        # if self.num_classes == 1:
        #     logits = torch.sigmoid(logits)
        if mask_ratio is not None:
            return logits,mask_pos
        return logits


class SwinTwoHead(Swin):

    def __init__(self, stride=8, click_chans=3, **kwargs):
        super(SwinTwoHead, self).__init__(stride, **kwargs)
        self.patch_embed_coords = PatchEmbed(
            img_size=(kwargs['img_size'][0]*stride,kwargs['img_size'][1]*stride), patch_size=8, in_chans=3, embed_dim=kwargs['embed_dim'])

        # self.ann_head = nn.Linear(self.num_features, 1)

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
        # self.neck = SimpleFPN_4_32(**neck_params)
        self.neck = Fusion_coord(**neck_params)
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

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
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

        multi_scale_features = self.neck(backbone_features,coord_features)

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





class ClickSwin(Swin):

    def __init__(self, stride=8, click_chans=3, **kwargs):
        super(ClickSwin, self).__init__(stride, **kwargs)
        self.patch_embed_coords = PatchEmbed(
            img_size=(kwargs['img_size'][0]*stride,kwargs['img_size'][1]*stride), patch_size=8, in_chans=click_chans, embed_dim=kwargs['embed_dim'])

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

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C

        return x, mask_pos

    def forward(self, input, mask_ratio=None, only_ann=False, train=False, backbone_features=None):
        residual, coord_features,image = input
        residual = residual**2
        if image is not None:
            coord_features = torch.cat([coord_features,image],dim=1)


        coord_features = self.maps_transform(coord_features)
        backbone_features,mask_ratio = self.ann_forward_features(residual,coord_features,mask_ratio,train)
        logits = self.forward_head(backbone_features)
        B, N, C = logits.shape
        grid_size = self.patch_embed.grid_size

        logits = logits.transpose(-1, -2).view(B, C, grid_size[0],grid_size[1])

        return logits


    # def seg_head(self, image, mask_ratio=None):
    #
    #     x, mask_pos = self.forward_features(image, mask_ratio)
    #     logits = self.forward_head(x)
    #
    #
    #     if mask_ratio is not None:
    #         return logits, mask_pos
    #     return logits
    # def forward_ann_head(self,image, x,coord_features, pre_logits: bool = False):
    #     # x = image.permute(0,2,3,1).flatten(1,2) + x + self.patch_embed_coords(coord_features)
    #     x = x + self.patch_embed_coords(coord_features)
    #     if self.global_pool == 'avg':
    #         x = x.mean(dim=1)
    #     return x if pre_logits else self.ann_head(x)


    # def prepare_input(self, image):
    #     # prev_mask = None
    #     prev_mask = image[:, -1:, :, :]
    #     image = image[:, :-1, :, :]
    #
    #     return image, prev_mask





class ClickTextSwin1(Swin):

    def __init__(self, stride=8,ad_training=False, fusion_feature_channels=[128,256,512,1024], **kwargs):
        super(ClickTextSwin1, self).__init__(stride, **kwargs)
        self.ad_training = ad_training
        self.patch_embed_coords = PatchEmbed(
            img_size=(kwargs['img_size'][0] * stride, kwargs['img_size'][1] * stride), patch_size=8, embed_dim=kwargs['embed_dim'])

        # self.patch_embed_text_img = PatchEmbed(
        #     img_size=kwargs['img_size'], patch_size=1, in_chans=384, embed_dim=kwargs['embed_dim'])

        self.maps_transform = nn.Identity()
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
        self.neck = SimpleFPN_4_32(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)
        self.fusion_module = FusionConv1x1(in_channels=fusion_feature_channels)



    def ann_forward_features(self, x,coord_features, mask_ratio=None):
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
        x = x + self.patch_embed_coords(coord_features) #+ self.patch_embed_text_img(fusion_feature)

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C

        return x, mask_pos

    def forward(self, input, mask_ratio=None,only_ann=True,train=False):
        residual, fusion_feature, coord_features = input
        residual = residual ** 2
        coord_features = self.maps_transform(coord_features)
        backbone_features, mask_ratio = self.ann_forward_features(residual, coord_features, mask_ratio)
        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])

        multi_scale_features = self.neck(backbone_features)
        multi_scale_features = self.fusion_module(multi_scale_features, fusion_feature)
        # print('ad')
        return self.head(multi_scale_features)

  

class ClickTextSwin(Swin):

    def __init__(self, stride=8,ad_training=False,  **kwargs):
        super(ClickTextSwin, self).__init__(stride, **kwargs)
        self.ad_training = ad_training
        self.patch_embed_coords = PatchEmbed(
            img_size=(kwargs['img_size'][0] * stride, kwargs['img_size'][1] * stride), patch_size=8, embed_dim=kwargs['embed_dim'])

        self.patch_embed_text_img = PatchEmbed(
            img_size=kwargs['img_size'], patch_size=1, in_chans=384, embed_dim=kwargs['embed_dim'])

        self.maps_transform = nn.Identity()

    def ann_forward_features(self, x,fusion_feature,coord_features, mask_ratio=None):
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
        x = x + self.patch_embed_coords(coord_features) + self.patch_embed_text_img(fusion_feature)

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C

        return x, mask_pos

    def forward(self, input, mask_ratio=None,only_ann=True,train=False):
        # if self.ad_training:
        residual, fusion_feature, coord_features = input
        residual = residual ** 2
        coord_features = self.maps_transform(coord_features)
        backbone_features, mask_ratio = self.ann_forward_features(residual,fusion_feature, coord_features, mask_ratio)
        logits = self.forward_head(backbone_features)
        B, N, C = logits.shape
        grid_size = self.patch_embed.grid_size

        logits = logits.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])

        return logits


class ClickSwinLAVT12Head(Swin):

    def __init__(self, stride=8,ad_training=False,  **kwargs):
        super(ClickSwinLAVT12Head, self).__init__(stride, **kwargs)
        self.ad_training = ad_training
        self.patch_embed_coords = PatchEmbed(
            img_size=(kwargs['img_size'][0] * stride, kwargs['img_size'][1] * stride), patch_size=8, embed_dim=kwargs['embed_dim'])

        self.patch_embed_text_img = PatchEmbed(
            img_size=kwargs['img_size'], patch_size=1, in_chans=384, embed_dim=kwargs['embed_dim'])

        self.maps_transform = nn.Identity()
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
        self.neck = SimpleFPN_4_32(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)

    def ann_forward_features(self, x,fusion_feature,coord_features, mask_ratio=None):
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
        x = x + self.patch_embed_coords(coord_features) + self.patch_embed_text_img(fusion_feature)

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C

        return x, mask_pos

    def forward(self, input, mask_ratio=None,only_ann=True,train=False):
        # if self.ad_training:
        residual, fusion_feature, coord_features = input
        residual = residual ** 2
        coord_features = self.maps_transform(coord_features)
        backbone_features, mask_ratio = self.ann_forward_features(residual,fusion_feature, coord_features, mask_ratio)
        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])

        multi_scale_features = self.neck(backbone_features)
        # multi_scale_features = self.fusion_module(multi_scale_features, fusion_feature)
        # print('ad')
        return self.head(multi_scale_features)


class TextSwin(Swin):

    def __init__(self, stride=8,  **kwargs):
        super(TextSwin, self).__init__(stride, **kwargs)

        # self.patch_embed_text_img = PatchEmbed(
        #     img_size=kwargs['img_size'], patch_size=1, in_chans=384, embed_dim=kwargs['embed_dim'])
        # neck_params = dict(
        #     in_dim=1024,#1280,
        #     out_dims=[240, 480, 960, 1920],
        # )
        #
        # head_params = dict(
        #     in_channels=[240, 480, 960, 1920],
        #     in_index=[0, 1, 2, 3],
        #     dropout_ratio=0.1,
        #     num_classes=1,
        #     loss_decode=CrossEntropyLoss(),
        #     align_corners=False,
        #     upsample='x1',
        #     channels={'x1': 256, 'x2': 128, 'x4': 64}['x1'],
        # )
        # self.neck = SimpleFPN_4_32(**neck_params)
        # self.head = SwinTransfomerSegHead(**head_params)
        # self.fusion_module = FusionConv1x1()

    def forward_features(self, x,fusion_feature, mask_ratio=None):
        mask_pos = None
        if mask_ratio is not None:
            mask_pos = self.random_masking(x, mask_ratio)
            mask_pos = mask_pos.view(-1, 1, *x.shape[2:])
            # print(x[mask_pos.squeeze()==1])
            x = x * mask_pos
            # fusion_feature = fusion_feature * mask_pos
        x = self.patch_embed(x)

        # x = x + self.patch_embed_text_img(fusion_feature)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C

        return x, mask_pos

    def forward(self, residual,fusion_feature, mask_ratio=None,only_ann=False,train=False):
        if self.residual_method == 'square':
            residual= residual**2

        backbone_features,mask_ratio = self.forward_features(residual,fusion_feature,mask_ratio)
        logits = self.forward_head(backbone_features)
        B, N, C = logits.shape
        grid_size = self.patch_embed.grid_size

        logits = logits.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])
        return logits
        # B, N, C = backbone_features.shape
        # grid_size = self.patch_embed.grid_size

        # backbone_features = backbone_features.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])

        # multi_scale_features = self.neck(backbone_features)
        #
        # multi_scale_features = self.fusion_module(multi_scale_features, fusion_feature)
        #
        #
        # return self.head(multi_scale_features)

class TextWeightSwin(Swin):

    def __init__(self, stride=8,  **kwargs):
        super(TextWeightSwin, self).__init__(stride, **kwargs)

        self.patch_embed_text_img = PatchEmbed(
            img_size=kwargs['img_size'], patch_size=1, in_chans=384, embed_dim=kwargs['embed_dim'])

        self.maps_transform = nn.Identity()

    def forward_features(self, x,text_feature, mask_ratio=None):
        mask_pos = None
        if mask_ratio is not None:
            mask_pos = self.random_masking(x, mask_ratio)
            mask_pos = mask_pos.view(-1, 1, *x.shape[2:])
            # print(x[mask_pos.squeeze()==1])
            x = x * mask_pos

        # x = self.patch_embed(x)
        print(x.shape)
        x = x * text_feature
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C

        return x, mask_pos

    def forward(self, residual,text_feature, mask_ratio=None,only_ann=False,train=False):

        # residual= residual**2

        x,mask_ratio = self.forward_features(residual,text_feature,mask_ratio)
        return self.forward_head(x)
