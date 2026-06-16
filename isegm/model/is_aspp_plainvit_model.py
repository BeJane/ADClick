import math

import torch
import torch.nn as nn
from isegm.utils.serialization import serialize
from .asppl import ASPP
from .is_model import ISModel
from .modeling.models_vit import VisionTransformer, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead
from .models_swin import Swin, make_zero_conv
from .swin_cls_prompt import Swin_ClsPrompt, Swin_ASPP_ClsPrompt


class PlainVitASPPModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params={},
            residual_backbone_params={},
        neck_params={}, 
        # head_params={},
        random_split=False,
            prompt_mode='prompt',
            task='click',
            use_zero_conv=False,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.random_split = random_split
        self.task = task
        if task == 'click':

            self.patch_embed_coords = PatchEmbed(
                img_size= backbone_params['img_size'],
                patch_size=backbone_params['patch_size'],
                in_chans=3 if self.with_prev_mask else 2,
                embed_dim=backbone_params['embed_dim'],
            )

        self.backbone = VisionTransformer(**backbone_params)
        if prompt_mode == 'cls_prompt':
            self.residual_backbone = Swin_ASPP_ClsPrompt(**residual_backbone_params)
        # else:
        #     self.residual_backbone = Swin(**residual_backbone_params)

        self.neck = ASPP(**neck_params)
        self.use_zero_conv = use_zero_conv
        if self.use_zero_conv:
            self.zero_conv = make_zero_conv(256)
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def backbone_forward(self, image,residual,prompt, coord_features=None):

        residual_features = self.residual_backbone.forward_features(residual,coord_features,prompt)
        if self.task == 'click':
            coord_features = self.patch_embed_coords(coord_features)
        else:
            coord_features = None
        backbone_features = self.backbone.forward_backbone(image, coord_features, self.random_split)
        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1,-2).view(B, C, grid_size[0], grid_size[1])
        backbone_features = backbone_features

        multi_scale_features = self.neck(backbone_features)
        if self.use_zero_conv:
            multi_scale_features = self.zero_conv(multi_scale_features)
        multi_scale_features =torch.nn.functional.interpolate(multi_scale_features,residual_features.shape[-2:],
                                                              mode='bilinear',align_corners=True)

        multi_scale_features = multi_scale_features + residual_features
        return {'instances': self.head(multi_scale_features), 'instances_aux': None}
