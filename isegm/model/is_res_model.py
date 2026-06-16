import math

import torch
import torch.nn as nn
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.models_vit import VisionTransformer, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead
from .models_swin import Swin, make_zero_conv
from .swin_cls_prompt import Swin_ClsPrompt




class ResModel(ISModel):
    @serialize
    def __init__(
        self,
            residual_backbone_params={},
        head_params={},
        random_split=False,
            prompt_mode='prompt',
            task='click',
        **kwargs
        ):

        super().__init__(**kwargs)
        self.random_split = random_split
        self.task = task
        if prompt_mode == 'cls_prompt':
            self.residual_backbone = Swin_ClsPrompt(**residual_backbone_params)
        else:
            self.residual_backbone = Swin(**residual_backbone_params)

        self.head = SwinTransfomerSegHead(**head_params)

    def backbone_forward(self, image,residual,prompt, coord_features=None):

        residual_features = self.residual_backbone.forward_features(residual,coord_features,prompt)

        return {'instances': self.head(residual_features), 'instances_aux': None}
