import math
from functools import partial

import torch
from torch import nn

from timm.models.layers import trunc_normal_
from model.models_swin import Swin
from model.models_vit import ViT
from model.patchcore import backbones, common
from model.patchcore.patchcore import PatchCore
from model.patchcore.sampler import ApproximateGreedyCoresetSampler
from model.patchcore_vit_slide import PatchCore_Base


class PatchCore_Swin(PatchCore_Base):
    def __init__(self,backbone_name,layers_to_extract_from,device,  IMAGE_SIZE,num_classes=2, in_chans=1024,
                 train_ok_loader=None,k_ratio=0.1,patchcore_patchsize=3, stride = 8,patchcore_add_pos=True,
                 num_heads=(6,12),depths=(2,2),embed_dim=1024,window_size=8,swin_patch_size=(1,1),
                 slide_window=None,slide_stride=None):
        super(PatchCore_Swin, self).__init__(backbone_name,layers_to_extract_from,device,  IMAGE_SIZE,
                                                  train_ok_loader=train_ok_loader,k_ratio=k_ratio,patchcore_patchsize=patchcore_patchsize,
                                                  stride=stride,patchcore_add_pos=patchcore_add_pos,slide_window=slide_window,slide_stride=slide_stride)

        self.out_stride = stride//len(depths)

        self.vit = Swin(img_size=self.vit_img_size, stride=self.out_stride,in_chans=in_chans,num_classes=num_classes,window_size=window_size,
                             patch_size=swin_patch_size, embed_dim=embed_dim, depths=depths,
                       num_heads=num_heads)

        self.vit.load_state_dict(torch.load('./content/swin_large_patch4_window7_224_22kto1k.pth',map_location='cpu'),
                                 strict=False)

        # manually initialize fc layer
        trunc_normal_(self.vit.head.weight, std=2e-5)

