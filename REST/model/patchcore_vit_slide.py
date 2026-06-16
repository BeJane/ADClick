import math
from functools import partial

from timm.models.layers import trunc_normal_
import torch
from torch import nn

from model import common
from model.loss import focal_loss
from model.models_vit import ViT
from model.patchcore import backbones
from model.patchcore.patchcore import PatchCore
from model.patchcore.sampler import ApproximateGreedyCoresetSampler

class PatchCore_Base(nn.Module):
    def __init__(self,backbone_name,layers_to_extract_from,device,  IMAGE_SIZE,num_classes=2, in_chans=1024,
                 train_ok_loader=None,k_ratio=0.1,patchcore_patchsize=3, stride = 8,patchcore_add_pos=True,
                 num_heads=(6,12),depths=(2,2),embed_dim=1024,window_size=8,swin_patch_size=(1,1),
                 slide_window=None,slide_stride=None):
        super(PatchCore_Base, self).__init__()

        img_size = (IMAGE_SIZE[1] // stride,IMAGE_SIZE[2]//stride)
        self.slide_window = slide_window
        if slide_window is not None:
            self.slide_window = (slide_window,slide_window)
            img_size = (slide_window * math.floor(img_size[0] / slide_window), slide_window * math.floor(img_size[1] / slide_window))
            self.vit_img_size=self.slide_window
        else:
            img_size = (window_size*math.floor(img_size[0]/window_size),window_size*math.floor(img_size[1]/window_size))
            self.vit_img_size=img_size

        if backbone_name in backbones._BACKBONES.keys():
            backbone = backbones.load(backbone_name)

        else:
            backbone = ViT(img_size=img_size, in_chans=3,num_classes=1024,drop_path_rate=0.1,
                             patch_size=16, embed_dim=1024, depth=24,num_heads=16,
                             mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            backbone.load_state_dict(torch.load(backbone_name))
            backbone.eval()
        # backbone.name, backbone.seed = backbone_name, backbone_seed
        nn_method = common.FaissNN(on_gpu=True, num_workers=4)
        sampler = ApproximateGreedyCoresetSampler(k_ratio, device)

        patchcore_instance = PatchCore(device,save_residuals=True,ref_num_patches=img_size,add_pos=patchcore_add_pos)
        patchcore_instance.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=IMAGE_SIZE,
            pretrain_embed_dimension=1024,
            target_embed_dimension=1024,
            patchsize=patchcore_patchsize,
            featuresampler=sampler,
            anomaly_scorer_num_nn=1,
            nn_method=nn_method,
        )

        patchcore_instance.fit(train_ok_loader)
        torch.cuda.empty_cache()
        self.patch_size = swin_patch_size
        self.img_size = img_size
        self.patchcore_instance = patchcore_instance

        self.layers_to_extract_from = layers_to_extract_from
        self.device = device
        self.slide_stride = slide_stride

        self.vit = None
        self.in_chans = in_chans


    def forward(self, x,masks=None, if_train=False,args=None,out_patchcore_map=False):
        if masks is not None:

            if masks.shape[1] != self.img_size[0]*self.out_stride or masks.shape[2] != self.img_size[1]*self.out_stride:
                masks = nn.functional.interpolate(masks, size=(self.img_size[0]*self.out_stride,self.img_size[1]*self.out_stride),mode='nearest')
        feature_residual,maps = self.patchcore_instance._predict(
            x
        ) # batchsize,2048,32,32
        if self.in_chans == 1024:
            feature_residual =feature_residual[:,1024:]
        feature_residual = torch.tensor(feature_residual)

        if self.slide_window is not None:
            b, c = feature_residual.shape[:2]
            feature_residual = torch.nn.functional.unfold(feature_residual, self.slide_window,
                                                          stride=self.slide_stride).transpose(1, 2).flatten(0, 1)
            sub_num = 2 if b % 2 == 0 else 1
            feature_residual = feature_residual.view(-1,sub_num, c, *self.slide_window)
            if masks is not None:
                mask_slide_window = (self.slide_window[0] * self.out_stride, self.slide_window[1] * self.out_stride)
                masks = torch.nn.functional.unfold(masks, mask_slide_window,
                                                   stride=self.slide_stride * self.out_stride).transpose(1, 2).flatten(
                    0, 1)
                masks = masks.view(-1,1,*self.slide_window)
                masks = torch.nn.functional.unfold(masks, self.patch_size[0] * self.out_stride,
                                                   stride=self.patch_size[0] * self.out_stride).transpose(1, 2)
                masks = torch.mean(masks, dim=-1)

        feature_residual = feature_residual
        logits = []
        for i in range(feature_residual.shape[0]):
            logit= self.vit(feature_residual[i].cuda())
            logits.append(logit.cpu())
            torch.cuda.empty_cache()
        logits = torch.cat(logits).cuda()
        if if_train:

            masks[masks >= args.gt_thres1] = 1
            masks[masks < args.gt_thres2] = 0
            logits = logits.reshape(-1, 2)
            masks = masks.view(-1)

            logits = torch.cat([logits[masks == 1], logits[masks == 0]])
            masks = torch.cat([masks[masks == 1], masks[masks == 0]])
            loss = focal_loss(logits, masks.cuda(), alpha=args.focal_loss_alpha,
                              gamma=args.focal_loss_gamma)
            if out_patchcore_map:
                return loss,maps
            return loss
        if out_patchcore_map:
            return logits, maps
        return logits
class PatchCore_Vit_slide(PatchCore_Base):
    def __init__(self,backbone_name,layers_to_extract_from,device,  IMAGE_SIZE,num_classes=2, in_chans=2048,
                 train_ok_loader=None,k_ratio=0.1,patchcore_patchsize=3, stride = 8,patchcore_add_pos=True,
                 patch_size=(1,1), embed_dim=1024, depth=4, num_heads=16,
                 slide_window=None,slide_stride=None):
        super(PatchCore_Vit_slide, self).__init__(backbone_name,layers_to_extract_from,device,  IMAGE_SIZE,
                                                  train_ok_loader=train_ok_loader,k_ratio=k_ratio,patchcore_patchsize=patchcore_patchsize,
                                                  stride=stride,patchcore_add_pos=patchcore_add_pos,slide_window=slide_window,slide_stride=slide_stride)


        self.out_stride = stride

        self.vit = ViT(img_size=self.vit_img_size, stride=self.out_stride,in_chans=in_chans,num_classes=num_classes,drop_path_rate=0.1,
                             patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                       num_heads=num_heads,
                             mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))


        self.vit.load_pretrained('./content/ViT-L_16.npz')


        # manually initialize fc layer
        trunc_normal_(self.vit.head.weight, std=2e-5)

    # def forward(self, x, masks=None,out_patchcore_map=False):
    #     feature_residual,maps = self.patchcore_instance._predict(
    #         x
    #     ) # batchsize,2048,32,32
    #     if self.in_chans == 1024:
    #         feature_residual =feature_residual[:,1024:]
    #
    #     feature_residual = torch.tensor(feature_residual)
    #     b,c =  feature_residual.shape[:2]
    #     feature_residual = torch.nn.functional.unfold(feature_residual, self.slide_window, stride=self.slide_stride).transpose(1, 2).flatten(0,1)
    #
    #     feature_residual =feature_residual.view(-1,c,*self.slide_window).to(self.device)
    #     if masks is not None:
    #         mask_slide_window = (self.slide_window[0]*self.out_stride,self.slide_window[1]*self.out_stride)
    #         masks = torch.nn.functional.unfold(masks, mask_slide_window,
    #                                            stride=self.slide_stride*self.out_stride).transpose(1, 2).flatten(0,1)
    #         masks =masks.view(-1,1,*mask_slide_window)
    #
    #     if out_patchcore_map:
    #         return self.vit(feature_residual, masks),maps
    #     return self.vit(feature_residual, masks)
