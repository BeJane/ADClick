import sys

import tqdm

from .pca_torch import PCA_torch
from model.patchcore import common
from model.patchcore import backbones
import torch

from ..pos_embed import get_2d_sincos_pos_embed


class RetrievalPredictor:
    def __init__(self,backbone_name,layers_to_extract_from , device='cuda:0',add_pos=False,pos_weight=0,pca_com=8,out_stride=4,window_size=4) -> None:
        self.device = device

        backbone = backbones.load(backbone_name)
        self.backbone = backbone.to(device)
        self.backbone = backbone
        self.layers_to_extract_from = layers_to_extract_from # ['layer1']#['features.denseblock1']

        self.device = device

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        self.forward_modules["feature_aggregator"] = feature_aggregator

        self.pca_com = pca_com
        self.stride = out_stride
        self.window_size = window_size

        self.pos_weight = pos_weight
        self.add_pos = add_pos
        self.pos_embed=None

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []


        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for batch in data_iterator:
                if isinstance(batch, dict):
                    image = batch["image"]

                # if self.visible:                images.append(image)
                features.append(_image_to_features(image))
        features = torch.concat(features, dim=0)
        return features

    def _embed(self, images,device='cpu'):
        """Returns feature embeddings for images."""

        def _detach(features):
            if device == 'gpu': return [feature.detach() for feature in features]
            return [feature.detach().cpu() for feature in features]


        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        features = [features[layer] for layer in self.layers_to_extract_from]

        features = torch.concat(_detach(features))
        if self.add_pos:
            if self.pos_embed is None:
                self.pos_embed = get_2d_sincos_pos_embed(features.shape[1],features.shape[2:4], cls_token=False)
                self.pos_embed = torch.tensor(self.pos_embed).permute(1,0).reshape(1,*features.shape[1:])

            features = features + self.pos_embed.type_as(
                features) * self.pos_weight
        return features

    def fit(self, input_data):
        retrieval_features = self._fill_memory_bank(input_data) # ([320, 256, 128, 128])
        N, C, H, W = retrieval_features.shape

        retrieval_features = torch.nn.functional.interpolate(retrieval_features,
                                                             size=(H//self.stride,W//self.stride),mode='bilinear') # 8
        retrieval_features = retrieval_features.permute(0,2,3,1).flatten(0,2).numpy()
        self.pca = PCA_torch(n_components=self.pca_com) # 8
        retrieval_features = self.pca.fit_transform(retrieval_features)
        retrieval_features = retrieval_features.reshape(N,H//self.stride,W//self.stride,self.pca_com)
        retrieval_features = torch.from_numpy(retrieval_features).to(self.device).permute(0,3,1,2)

        self.retrieval_features = torch.nn.functional.unfold(retrieval_features,kernel_size=self.window_size,stride=self.window_size).transpose(1, 2)

        print(self.retrieval_features.shape)
    def transform(self, image):
        assert image.shape[0] == 1

        with torch.no_grad():
            # start_time = time.time()
            features= self._embed(image,device='gpu')
            N, C, H, W = features.shape
            features = torch.nn.functional.interpolate(features,size=(H//self.stride,W//self.stride),mode='bilinear') # 8
            features = features.permute(0,2,3,1).flatten(0,2)

            features = self.pca.transform(features)

            features = features.reshape(N, H // self.stride, W // self.stride, self.pca_com).permute(0, 3, 1, 2)

            features = torch.nn.functional.unfold(features, kernel_size=self.window_size,
                                                                 stride=self.window_size).transpose(1, 2)
            # print(((features-self.retrieval_features)**2).sum(-1).shape,torch.sort(((features-self.retrieval_features)**2).sum(-1),dim=-1)[0][:, :-2].shape)
            _,idx = torch.sort(torch.sort(((features-self.retrieval_features)**2).sum(-1),dim=-1)[0][:, :round(features.shape[1]*0.8)].mean(-1))

        return idx

