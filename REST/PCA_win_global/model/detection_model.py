import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from model.images import IMAGENET_STD, IMAGENET_MEAN
from model.patchcore import backbones, common

from model.patchcore.patchcore import PatchCore
from model.patchcore.retrieval import RetrievalPredictor
from model.patchcore.sampler import ApproximateGreedyCoresetSampler, RandomSampler


class PatchCore_residual(nn.Module):
    def __init__(self,  IMAGE_SIZE,backbone_name,layers_to_extract_from,
                 device, train_ok_loader=None,k_ratio=0.1,stride=8,patchcore_add_pos=True,context=True,pretrain_embed_dimension=1024,
                 target_embed_dimension=1024,pos_weight=1,mode='cpu',pca_com=0.95,visible=False,global_win=4,
                 bank_path=None,global_pca=8,global_resize_stride=8,
                 global_nn_strategy='max',patchsize=3,
                 global_backbone_name='wideresnet50',gobal_layers_to_extract_from=['layer1'],min_global_nn=64):
        super(PatchCore_residual, self).__init__()

        backbone = backbones.load(backbone_name)

        img_size = (IMAGE_SIZE[1] // stride,IMAGE_SIZE[2]//stride)

        sampler =ApproximateGreedyCoresetSampler(k_ratio,device)


        patchcore_instance = PatchCore(device,save_residuals=True,ref_num_patches=img_size,mode=mode,
                                       add_pos_embed=patchcore_add_pos,context=context,pos_weight=pos_weight,pca_com=pca_com,visible=visible)
        patchcore_instance.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=IMAGE_SIZE,
            pretrain_embed_dimension=pretrain_embed_dimension,
            target_embed_dimension=target_embed_dimension,
            patchsize=patchsize,
            featuresampler=sampler,
            anomaly_scorer_num_nn=1

        )

        patchcore_instance.fit(train_ok_loader,bank_path)

        self.retrieval_predictor = RetrievalPredictor(global_backbone_name,gobal_layers_to_extract_from,out_stride=global_resize_stride,
                                                      pca_com=global_pca,window_size=global_win,add_pos=False)
        self.retrieval_predictor.fit(train_ok_loader)

        self.visible = visible
        if visible:
            self.train_ok_loader = train_ok_loader
        self.patchcore_instance = patchcore_instance

        if global_nn_strategy == 'max':
            self.global_nn = min(max(min_global_nn,len(train_ok_loader.dataset)//2),len(train_ok_loader.dataset))
        if global_nn_strategy == 'min':
            self.global_nn = min(len(train_ok_loader.dataset),min_global_nn)
        print(f"Global:{self.global_nn}/{len(train_ok_loader.dataset)}")

        self.latency_list = []


    def forward(self, x,pos_context=None,nn_num=1,p=None,train_ok_query_id=-1):

        assert x.shape[0] == 1

        torch.cuda.synchronize()
        st = time.time()
        sorted_indexs = self.retrieval_predictor.transform(x)

        if train_ok_query_id == -1:
            idxs = sorted_indexs[:self.global_nn]
        else:
            idxs = []
            i = 0
            while len(idxs) < self.global_nn and i < self.global_nn:
                if sorted_indexs[i] == train_ok_query_id:
                    i += 1
                    continue
                idxs.append(sorted_indexs[i].item())
                i += 1
        # torch.cuda.synchronize()
        # print(time.time()-st)
        # vi
        if self.visible:
            global_nn = 10
            plt.subplot(1,global_nn+1,1)
            plt.imshow(x[0].cpu().numpy().transpose((1, 2, 0)) * IMAGENET_STD + IMAGENET_MEAN)
            for i,id in enumerate(idxs[:global_nn]):
                plt.subplot(1,global_nn+1,i+2)
                plt.title(f'pca global {i}')
                # print(self.train_ok_loader.dataset[id]['image'][0].shape)
                plt.imshow(self.train_ok_loader.dataset[id]['image'].numpy().transpose((1, 2, 0)) * IMAGENET_STD + IMAGENET_MEAN)
            # print(idx.shape)
            plt.show()
        output = self.patchcore_instance._predict(
            x,idxs,pos_context,nn_num,p
        )

        torch.cuda.synchronize()
        t = time.time() - st
        self.latency_list.append(t)
        if len(self.latency_list) == 30:
            print(f'Latency: {np.mean(self.latency_list)}')
        return output