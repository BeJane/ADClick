import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import tqdm
from wandb.wandb_torch import torch

from model import common
from model.patchcore import backbones
import torch

class KMeans_Pytorch:
    """使用pytorch推断
    """

    def __init__(self, kmeans: KMeans, device='cuda') -> None:
        self.cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(device)  # n x c

    @torch.no_grad()
    def transform(self, x):
        # return torch.cdist(x, self.cluster_centers_)
        return torch.square((x[:, None].cuda() - self.cluster_centers[None])).sum(-1)

def entropy_pytorch(pk, qk, dim=0):
    return (pk * torch.log(pk / qk)).sum(dim)

class RetrievalPredictor:
    def __init__(self,backbone_name,layers_to_extract_from , device='cuda:0', seed=66, n_clusters=12,input_shape=(3,512,512)) -> None:
        self.device = device
        self.kmeans_f_num = 50000
        self.row = 8 # 5
        self.col = 8 # 5
        self.d_method = 'kl'
        self.l_ratio = 4 / 5
        self.n_clusters = n_clusters
        self.random_state = np.random.RandomState(seed)
        self.lda = LinearDiscriminantAnalysis()
        self.kmeans = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_predictor = None
        self.train_hs = None
        self.idx = None
        self.s_l = None
        self.e_l = None
        self.eye_m = torch.eye(self.n_clusters, device=self.device)

        backbone = backbones.load(backbone_name)
        self.backbone = backbone.to(device)
        self.backbone = backbone
        self.layers_to_extract_from = layers_to_extract_from # ['layer1']#['features.denseblock1']

        self.device = device

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator
        #
        # preprocessing = common.Preprocessing(
        #     feature_dimensions, pretrain_embed_dimension
        # )
        # self.forward_modules["preprocessing"] = preprocessing
        #
        # preadapt_aggregator = common.Aggregator(
        #     target_dim=target_embed_dimension
        # )


    def _transform(self, features):
        H, W = features.shape[-2:]
        pred = self.kmeans_predictor.transform(features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])).reshape(H,
                                                                                                                    W,
                                                                                                                    -1)
        pred_l = pred.argmin(-1)
        # h = torch.stack([F.one_hot(pred_l[s[1]:e[1], s[0]:e[0]].reshape(-1), self.kmeans.n_clusters).float().mean(0) for s, e in zip(self.s_l, self.e_l)])  # row*col x k, 0.4ms
        h = self.eye_m[pred_l[self.idx[..., 0], self.idx[..., 1]]].mean(1)
        return pred_l, pred, h
    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image,context=None):
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

    def _embed(self, images):
        """Returns feature embeddings for images."""


        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        features = [features[layer] for layer in self.layers_to_extract_from]
        # print(len(features))
        # print(torch.concat(features).shape)
        return torch.concat(features).cpu()

    def fit(self, input_data):
        retrieval_features = self._fill_memory_bank(input_data)
        # print(retrieval_features.shape)
        B, C, H, W = retrieval_features.shape
        self.s_l = np.mgrid[:W:W // self.col, :H:H // self.row].transpose(1, 2, 0).reshape(-1, 2)  # xy
        self.e_l = np.mgrid[W // self.col:W + 1:W // self.col, H // self.row:H + 1:H // self.row].transpose(1, 2,
                                                                                                            0).reshape(
            -1, 2)  # xy
        p_idx = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), -1)

        self.idx = torch.stack(
            [p_idx[s[1]:e[1], s[0]:e[0]].reshape(-1, p_idx.shape[-1]) for s, e in zip(self.s_l, self.e_l)]).to(
            self.device)  # (row*col) x n x 2
        image_retrieval_features = retrieval_features.permute(0, 2, 3, 1).cpu().numpy()  # b x h x w x 512
        self.kmeans.fit(
            image_retrieval_features.reshape(-1, C)[self.random_state.permutation(B * H * W)[:self.kmeans_f_num]])
        self.kmeans_predictor = KMeans_Pytorch(self.kmeans, self.device)
        train_hs = []
        for features in retrieval_features:
            train_hs.append(
                self._transform(features[None])[-1]
            )
        self.train_hs = torch.stack(train_hs)


    def transform(self, image):
        assert image.shape[0] == 1
        _ = self.forward_modules.eval()

        with torch.no_grad():
            # start_time = time.time()
            features= self._embed(image)
        # print(features.shape)
        # features: 1 x c x h x w
        pred_l, pred, h = self._transform(features)
        if self.d_method == 'kl':
            idx = torch.argsort(torch.sort(entropy_pytorch(
                h[None] + 1e-8, self.train_hs + 1e-8, -1
            ), -1)[0][:, :int(self.row * self.col * self.l_ratio)].sum(-1))
        else:
            idx = torch.argsort(torch.sort(torch.norm(
                h[None] - self.train_hs, dim=-1
            ), -1)[0][:, :int(self.row * self.col * self.l_ratio)].sum(-1))
        return pred_l, pred, idx

    def transform_fast(self, features, knn: int = 10):
        # features: 1 x c x h x w
        pred_l, pred, h = self._transform(features)
        if self.d_method == 'kl':
            # idx = torch.argsort(torch.topk(entropy_pytorch(h[None] + 1e-8, self.train_hs + 1e-8, -1),
            #            k=int(self.row*self.col*self.l_ratio), dim=-1, largest=False)[0].sum(-1))
            idx = torch.topk(torch.sort(entropy_pytorch(
                h[None] + 1e-8, self.train_hs + 1e-8, -1
            ), -1)[0][:, :int(self.row * self.col * self.l_ratio)].sum(-1), k=knn, largest=False)[1]
        else:
            idx = torch.topk(torch.sort(torch.norm(
                h[None] - self.train_hs, dim=-1
            ), -1)[0][:, :int(self.row * self.col * self.l_ratio)].sum(-1), k=knn, largest=False)[1]
        return pred_l, pred, idx

