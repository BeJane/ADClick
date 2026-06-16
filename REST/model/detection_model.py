import math
import os
import time
from functools import partial


import torch
from matplotlib import pyplot as plt

from timm.models.vision_transformer import PatchEmbed, Block

from torch import nn


from timm.models.layers import trunc_normal_
from model.fusion import FusionBlock

from model.models_vit import ViT
from model.patchcore import backbones
from model.patchcore.common import NetworkFeatureAggregator
from model.patchcore.patchcore import PatchCore
from model.patchcore.retrieval import RetrievalPredictor

from model.pos_embed import get_2d_sincos_pos_embed
from model.vison_transformer import VisionTransformer


class PatchCore_residual(nn.Module):
    def __init__(self,  IMAGE_SIZE,backbone_name,layers_to_extract_from,global_backbone_name,gobal_layers_to_extract_from,
                 device, patchsize=3,global_nn=10,distance_limit=0.15,slice_step=None,
                 train_ok_loader=None,k_ratio=0.1,stride=8,patchcore_add_pos=True,context=True,pretrain_embed_dimension=1024,
                 target_embed_dimension=1024,pos_weight=1):
        super(PatchCore_residual, self).__init__()

        self.train_ok_loader = train_ok_loader
        self.global_nn = global_nn
        self.distance_limit = distance_limit
        self.slice_step =slice_step # slice referenced feature to reduce memory
        if backbone_name in backbones._BACKBONES.keys():
            backbone = backbones.load(backbone_name)

        else:
            backbone = ViT(img_size=IMAGE_SIZE, in_chans=3,num_classes=1024,drop_path_rate=0.1,
                             patch_size=16, embed_dim=1024, depth=24,num_heads=16,
                             mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            backbone.load_state_dict(torch.load(backbone_name))
            backbone.eval()
        img_size = (IMAGE_SIZE[1] // stride,IMAGE_SIZE[2]//stride)

        self.stride =stride
        self.retrieval_predictor = RetrievalPredictor(global_backbone_name,gobal_layers_to_extract_from)
        self.retrieval_predictor.fit(train_ok_loader)


        patchcore_instance = PatchCore(device,save_residuals=True,ref_num_patches=img_size,slice_step=slice_step,
                                       add_pos_embed=patchcore_add_pos,context=context,pos_weight=pos_weight)
        patchcore_instance.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=IMAGE_SIZE,
            pretrain_embed_dimension=pretrain_embed_dimension,
            target_embed_dimension=target_embed_dimension,
            patchsize=patchsize,
            featuresampler=None,
            anomaly_scorer_num_nn=1,
            # nn_method=nn_method,

        )

        patchcore_instance.fit(train_ok_loader)

        self.patchcore_instance = patchcore_instance


    def forward(self, x,pos_context=None,nn_num=1,p=None,method='square',sub=False,train_ok_query_id=-1):
        assert x.shape[0] == 1
        # torch.cuda.empty_cache()

        # torch.cuda.synchronize()
        # st = time.time()
        sorted_indexs = self.retrieval_predictor.transform(x)[2].cpu()
        # torch.cuda.synchronize()
        # print(time.time()-st)
        idxs = []
        i = 0
        while len(idxs) < self.global_nn:
            if sorted_indexs[i] == train_ok_query_id:
                i += 1
                continue
            idxs.append(sorted_indexs[i].item())
            i += 1
        # idxs = torch.tensor(idxs)
        # print(idxs,train_ok_query_id)

        # plt.subplot(1,self.global_nn+1,1)
        # plt.imshow(x[0].cpu().numpy().transpose((1, 2, 0)) * IMAGENET_STD + IMAGENET_MEAN)
        # for i,id in enumerate(idxs[:global_nn]):
        #     plt.subplot(1,global_nn+1,i+2)
        #     # print(self.train_ok_loader.dataset[id]['image'][0].shape)
        #     plt.imshow(self.train_ok_loader.dataset[id]['image'].numpy().transpose((1, 2, 0)) * IMAGENET_STD + IMAGENET_MEAN)
        # # print(idx.shape)
        # plt.show()
        # images = []
        # for id in idxs:
        #     images.append(self.train_ok_loader.dataset[id]['image'][None,:,:,:])
        # images = torch.cat(images)
        # images = torch.nn.functional.unfold(torch.tensor(images), self.stride,
        #                                     stride=self.stride).transpose(1, 2)
        #
        # images = images.reshape(-1,64,64, 3,self.stride,self.stride)[:,::self.slice_step,::self.slice_step,:,:,:].flatten(0,2)
        # print(images.shape)
        output = self.patchcore_instance._predict(
            x,idxs,pos_context,nn_num,p,method,sub,distance_limit=self.distance_limit,vi_images=None,#images
        )

        return output
class PatchCore_cascade_Vit(nn.Module):
    def __init__(self, config, backbone_name,layers_to_extract_from,
                 device,patchcore_result, num_classes=2, in_chans=2048,train_ok_loader=None,k_ratio=0.1):
        super(PatchCore_cascade_Vit, self).__init__()

        self.patchcore_instance = PatchCore_residual(backbone_name,layers_to_extract_from,device,train_ok_loader=train_ok_loader)
        self.max_score = patchcore_result["max_score"]
        self.min_score = patchcore_result["min_score"]
        self.threshold = patchcore_result["threshold"]
        self.layers_to_extract_from = layers_to_extract_from
        self.device = device

        self.patch_size  =9
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(32, 1, in_chans, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 32*32+1, config.hidden_size), requires_grad=False)  # fixed sin-cos embedding

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
            Block(config.hidden_size, config.transformer.num_heads, 4, qkv_bias=True, norm_layer=norm_layer)
            for i in range(config.transformer.num_layers)])
        self.norm = norm_layer(config.hidden_size)

        # Classifier Head
        # self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, masks=None):
        feature_residual,map = self.patchcore_instance(
            x
        ) # batchsize,2048,32,32
        if self.in_chans == 1024:
            feature_residual =feature_residual[:,1024:]

        feature_residual = torch.tensor(feature_residual).to(self.device)

        x = self.patch_embed(feature_residual)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:,1:] # b,1024,c
        x= x.cpu()
        x = x.reshape(*map.shape,-1).permute(0,3,1,2)
        b,c = x.shape[:2]
        x = torch.nn.functional.unfold(x, self.patch_size, stride=1,padding=self.patch_size//2).transpose(1, 2)
        x =x.view(b,-1,c,self.patch_size,self.patch_size).permute(0,1,3,4,2) # b, 1024, 9, 9, 512
        # print(x[0,0,0,0,:])
        masks = nn.functional.interpolate(masks, size=map.shape[-2:]).squeeze(1).flatten(1)
        masks[masks >= 0.5] = 1
        masks[masks < 0.5] = 0
        # print(masks.shape,map.shape)
        map = (map - self.min_score)/(self.max_score - self.min_score)
        map = map.reshape(x.shape[0],x.shape[1])
        binary_map = (map >= self.threshold)
        # index = torch.arange(x.shape[1]).expand(x.shape[0],x.shape[1]).reshape(masks.shape)
        # print(index)
        # print(index.shape,index[map >= self.threshold].shape)
        x = x[[binary_map]]
        if x.shape[0] == 0:
            return x,masks,binary_map,map
        x = x.reshape(x.shape[0],-1,x.shape[-1])

        masks = masks[[binary_map]]
        # print(x.shape,map.shape,masks.shape) # torch.Size([1305, 81, 512]) (32, 1024) torch.Size([1305])
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        torch.cuda.empty_cache()
        x = x.to(self.device)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # print(x.shape)
        x = self.head(x[:,:1])
        return x,masks,binary_map,map


class PatchCore_Vit(nn.Module):
    def __init__(self,config, backbone_name,layers_to_extract_from,device,  IMAGE_SIZE,num_classes=2, in_chans=2048,
                 train_ok_loader=None,k_ratio=0.1,patchcore_patchsize=3, stride = 8,patchcore_add_pos=True):
        super(PatchCore_Vit, self).__init__()
        img_size = (IMAGE_SIZE[1] // stride,IMAGE_SIZE[2]//stride)

        if backbone_name in backbones._BACKBONES.keys():
            backbone = backbones.load(backbone_name)

        else:
            backbone = ViT(img_size=img_size, in_chans=3,num_classes=1024,drop_path_rate=0.1,
                             patch_size=16, embed_dim=1024, depth=24,num_heads=16,
                             mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            backbone.load_state_dict(torch.load(backbone_name))
            backbone.eval()
        # backbone.name, backbone.seed = backbone_name, backbone_seed
        # nn_method = common.FaissNN(on_gpu=True, num_workers=4)
        sampler = None #ApproximateGreedyCoresetSampler(k_ratio, device)

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
            # nn_method=nn_method,
        )

        patchcore_instance.fit(train_ok_loader)
        torch.cuda.empty_cache()
        self.patchcore_instance = patchcore_instance

        self.layers_to_extract_from = layers_to_extract_from
        self.device = device

        self.out_stride = 8
        self.vit = ViT(img_size=img_size, stride=self.out_stride,in_chans=in_chans,num_classes=num_classes,drop_path_rate=0.1,
                             patch_size=config.patches["size"], embed_dim=config.hidden_size, depth=config.transformer.num_layers,
                       num_heads=config.transformer.num_heads,
                             mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # self.vit.load_state_dict(torch.load('./work_dirs/mae/iter-100.pth'))
        if in_chans == 2048:
            model_path = './work_dirs/mae/iter-5000.pth'
        elif in_chans == 1024:
            model_path = './work_dirs/mae1024/iter-5000.pth'
        self.in_chans = in_chans
        # pretrained_dict = torch.load(model_path)
        # model_dict = self.vit.state_dict()
        # # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # self.vit.load_state_dict(model_dict)
        self.vit.load_pretrained('./content/ViT-L_16.npz')


        # manually initialize fc layer
        trunc_normal_(self.vit.head.weight, std=2e-5)

        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.vit.pos_embed.shape[-1], int(self.vit.patch_embed.num_patches**.5), cls_token=True)
        # self.vit.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # self.vit.pos_embed.requires_grad = False
        # print(self.vit.pos_embed.requires_grad)

    def forward(self, x, masks=None,out_patchcore_map=False):
        feature_residual,maps = self.patchcore_instance._predict(
            x
        ) # batchsize,2048,32,32
        if self.in_chans == 1024:
            feature_residual =feature_residual[:,1024:]
        feature_residual = torch.tensor(feature_residual).to(self.device)
        # if masks is not None:
        #     masks = nn.functional.interpolate(masks, size=feature_residual.shape[-2:])
        #     masks[masks >= 0.5] = 1
        #     masks[masks < 0.5] = 0

        if out_patchcore_map:
            return self.vit(feature_residual, masks),maps
        return self.vit(feature_residual, masks)


class PRN_Vit(nn.Module):
    def __init__(self, config, backbone_name,layers_to_extract_from,layers_to_transformer,work_dir,
                 device, num_classes=2, zero_head=False, vis=False,train_ok_loader=None,k_ratio=0.1):
        super(PRN_Vit, self).__init__()

        backbone = backbones.load(backbone_name)
        # backbone.name, backbone.seed = backbone_name, backbone_seed

        self.feature_aggregator = NetworkFeatureAggregator(
            backbone, layers_to_extract_from, device
        )
        multi_layer_prototypes = {}
        if train_ok_loader is not None:
            multi_layer_features = [None] * len(layers_to_extract_from)
            _ = self.feature_aggregator.eval()
            with torch.no_grad():
                for images in train_ok_loader:
                    features = self.feature_aggregator(images['image'].to(device))
                    for i, layer in enumerate(layers_to_extract_from):
                        if multi_layer_features[i] is not None:
                            multi_layer_features[i] = torch.cat([multi_layer_features[i], features[layer]], dim=0)
                        else:
                            multi_layer_features[i] = features[layer]

                # kmeans
                num_clusters = math.ceil(k_ratio * len(train_ok_loader.dataset))

                for i, layer_features in enumerate(multi_layer_features):
                    cluster_ids_x, cluster_centers = kmeans(
                        X=layer_features, num_clusters=num_clusters, distance='l2',  iter_limit=300, tol=1e-10
                    )
                    multi_layer_prototypes[layers_to_extract_from[i]] = cluster_centers
                    torch.save(cluster_centers,
                               os.path.join(work_dir, f'{backbone_name}_{layers_to_extract_from[i]}_{k_ratio}.pt'))
                del multi_layer_features
        else:
            for layer in layers_to_extract_from:
                multi_layer_prototypes[layer] = torch.load(os.path.join(work_dir, f'{backbone_name}_{layer}_{k_ratio}.pt'))
        self.layers_to_extract_from = layers_to_extract_from
        self.layers_to_transformer = layers_to_transformer
        self.multi_layer_prototypes = multi_layer_prototypes
        self.device = device

        self.fusionBlock = FusionBlock()
        img_size = multi_layer_prototypes[layers_to_transformer].shape[-1]
        in_channels = multi_layer_prototypes[layers_to_transformer].shape[1]
        self.vit = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes,in_channels=in_channels*2)
        # self.vit.load_from(np.load("./content/ViT-L_16.npz"))

    def forward(self, x, masks=None):
        with torch.no_grad():
            features_dict = self.feature_aggregator(x)
            residuals = []

            features = []
            for layer in self.layers_to_extract_from:
                feature  = features_dict[layer].detach()
                features.append(feature)
                prototyes = self.multi_layer_prototypes[layer]
                dis = pairwise_l2(feature, prototyes)
                min_index = torch.argmin(dis,dim=1)
                residuals.append(torch.abs((feature - prototyes[min_index].to(feature.device))))

        features = self.fusionBlock(tuple(features))
        residuals = self.fusionBlock(tuple(residuals))
        layer_index = self.layers_to_extract_from.index(self.layers_to_transformer)
        feature_residual = residuals[layer_index]
        # feature_residual = torch.cat([features[layer_index],residuals[layer_index]],dim=1)
        # if masks is not None:
            # masks = nn.functional.interpolate(masks, size=feature_residual.shape[-2:])
            # masks[masks >= 0.5] = 1
            # masks[masks < 0.5] = 0
        return self.vit(feature_residual, masks)

class CNN_Vit(nn.Module):
    def __init__(self, config, backbone_name,layers_to_extract_from,layers_to_transformer,
                 device, num_classes=2, zero_head=False, vis=False,img_size=60,in_channels=128):
        super(CNN_Vit, self).__init__()

        backbone = backbones.load(backbone_name)
        # backbone.name, backbone.seed = backbone_name, backbone_seed

        self.feature_aggregator = patchcore.commonNetworkFeatureAggregator(
            backbone, layers_to_extract_from, device
        )

        self.layers_to_extract_from = layers_to_extract_from
        self.layers_to_transformer = layers_to_transformer
        # self.multi_layer_prototypes = multi_layer_prototypes
        self.device = device

        self.fusionBlock = FusionBlock()
        # img_size = multi_layer_prototypes[layers_to_transformer].shape[-1]
        # in_channels = multi_layer_prototypes[layers_to_transformer].shape[1]
        self.vit = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes,in_channels=in_channels)
        # self.vit.load_from(np.load("./content/ViT-B_16.npz"))

    def forward(self, x, masks=None):
        with torch.no_grad():
            features_dict = self.feature_aggregator(x)
            residuals = []

            features = []
            for layer in self.layers_to_extract_from:
                feature  = features_dict[layer].detach()
                features.append(feature)
                # prototyes = self.multi_layer_prototypes[layer]
                # dis = pairwise_l2(feature, prototyes)
                # min_index = torch.argmin(dis,dim=1)
                # residuals.append(torch.abs((feature - prototyes[min_index].to(self.device))))

        layer_index = self.layers_to_extract_from.index(self.layers_to_transformer)
        feature = self.fusionBlock(tuple(features))[layer_index]
        # residuals = self.fusionBlock(tuple(residuals))
        if masks is not None:
            masks = nn.functional.interpolate(masks, size=feature.shape[-2:],mode='nearest')

        return self.vit(feature, masks)
