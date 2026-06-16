"""PatchCore and PatchCore detection methods."""
import logging
import math
import os
import pickle
import random
import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm


from . import backbones
from . import  common
from . import  sampler
from .sampler import ApproximateGreedyCoresetSampler
from ..images import IMAGENET_STD, IMAGENET_MEAN
from ..pos_embed import get_2d_sincos_pos_embed

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device, save_patch_scores=False,save_residuals=False,ref_num_patches=None,add_pos_embed=True,
                 context=True,context_weight=1,visible=False,pos_weight=1,slice_step=1):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device
        self.save_patch_scores = save_patch_scores
        self.save_residuals = save_residuals
        self.ref_num_patches = ref_num_patches

        self.pos_embed = None
        self.add_pos_embed = add_pos_embed
        self.context = context # 几何特征，直方图
        self.context_weight = context_weight
        self.pos_weight = pos_weight
        self.visible = visible
        self.slice_step = slice_step
        print(f'Slice referenced features, slice step = {self.slice_step}')
        print(f'Add position embedding: {add_pos_embed}! Weight={self.pos_weight} !')
        print(f'Add position context: {context}! Weight={self.context_weight}')
    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=sampler.IdentitySampler(),
        nn_method=common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = common.NearestNeighbourScorer_gpu(
            n_nearest_neighbours=anomaly_score_num_nn
        )
        # self.sub_anomaly_scorer =  common.NearestNeighbourS//corer_gpu(
        #     n_nearest_neighbours=anomaly_score_num_nn/
        # )

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler
        # self.sub_featuresampler = ApproximateGreedyCoresetSampler(0.25, self.device)
        self.ignore_patches = None

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images,context=None, detach=True,CPU=False, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                features = features.detach()
            if CPU:
                features = features.cpu()
                # return [x.detach().cpu().numpy() for x in features]
            return features
        b = images.shape[0]
        start_time = time.time()
        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        # print(features[self.layers_to_extract_from[0]].shape) # b,c,h,w
        # print(features[self.layers_to_extract_from[0]].shape,self.layers_to_extract_from[0])
        # print(features[self.layers_to_extract_from[1]].shape,self.layers_to_extract_from[1])
        features = [features[layer] for layer in self.layers_to_extract_from]
        if features[0].dim() == 3:#vit
            features = [f.transpose(1,2)[:,:,1:] for f in features]
            features = [f.reshape(*f.shape[:2],int(math.sqrt(f.shape[2])),int(math.sqrt(f.shape[2]))) for f in features]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]

        patch_shapes = [x[1] for x in features]
        # print(patch_shapes)# [[32, 32], [16, 16]]
        features = [x[0] for x in features]
        if self.ref_num_patches is None:
            self.ref_num_patches = patch_shapes[0]

        for i in range(0, len(features)):

            _features = features[i]
            patch_dims = patch_shapes[i]
            if (patch_dims[0],patch_dims[1]) == self.ref_num_patches :continue
            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape

            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(self.ref_num_patches[0], self.ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], self.ref_num_patches[0], self.ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features # b,1024,1024,3,3
            # print(features[i].shape)


        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # print(features[0].shape) # 8192,512,3,3
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)# 8192,1024


        if self.add_pos_embed:
            if self.pos_embed is None:
                self.pos_embed = get_2d_sincos_pos_embed(features.shape[-1],self.ref_num_patches,cls_token=False)
                self.pos_embed = torch.tensor(self.pos_embed)
            features = features.reshape(-1,self.pos_embed.shape[0],features.shape[-1]) + self.pos_embed.type_as(features)*self.pos_weight
            features =features.reshape(-1,features.shape[-1])


        if self.context and (not self.add_pos_embed):
            context = context.permute(0,3,1,2)
            context = F.interpolate(
                context,
                size=(self.ref_num_patches[0], self.ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            context = context.permute(0,2,3,1).reshape(features.shape[0],-1)
            # print(context.shape,features.shape)
            features = torch.cat([features, context.type_as(features)*self.context_weight],dim=1)

        # _detach(features)
        # print((time.time()-start_time)/b)
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image,context=None):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                f = self._embed(input_image,context)
                f = f.view(input_image.shape[0],*self.ref_num_patches,-1)[:,0::self.slice_step,
                    0::self.slice_step,::].flatten(0,2).cpu()
                # print(f.shape)
                return f

        features = []
        if self.visible:        images = []
        if self.context: contexts = [] 
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for batch in data_iterator:
                if isinstance(batch, dict):
                    image = batch["image"]
                    if self.context:
                        context = batch["feature"]
                        contexts.append(context)
                    else:
                        context = None
                if self.visible:                images.append(image)
                features.append(_image_to_features(image,context).view(image.shape[0],-1,self.target_embed_dimension))
        self.features = torch.concat(features, dim=0) # 320,4096,1024
        # print(features.shape)
        # if self.featuresampler is not None:
        #     features, sample_indices = self.featuresampler.run(features)

    def _predict(self, images,ids,context=None,nn_num=1,p=None,method='square',sub=False,distance_limit=0.15,vi_images=None ):
        """Infer score and mask for a batch of images."""
        if self.ignore_patches is None:
            grid_h = np.arange(self.ref_num_patches[0], dtype=np.float32)
            grid_w = np.arange(self.ref_num_patches[1], dtype=np.float32)


            grid = np.meshgrid(grid_w, grid_h)  # here w goes first
            grid = np.stack(grid, axis=0).transpose((1,2,0))
            grid = np.expand_dims(grid, 0)
            # print(grid.shape)
            ref_grid = grid[:,::self.slice_step,::self.slice_step,:].reshape(-1,2)
            # print(ref_grid.shape)
            query_grid = grid.reshape(-1,1,2)

            self.ignore_patches = np.sqrt(((ref_grid - query_grid)**2).sum(-1)) > (distance_limit*max(self.ref_num_patches)) # 4096,40960
            self.ignore_patches = np.expand_dims(self.ignore_patches,1)
            self.ignore_patches = self.ignore_patches.repeat(len(ids),axis=1).reshape(self.ignore_patches.shape[0],-1)
            # print(self.ignore_patches.shape)
        features = self.features[ids]# 10,4096,1024

        features = features.to('cuda',non_blocking=True)

        # features = features.view(len(ids),self.ref_num_patches[0],self.ref_num_patches[1],features.shape[-1])[:,::slice_step,::slice_step,:].flatten(1,2)

        # print(features.shape)

        self.anomaly_scorer.fit(detection_features=[features.view(-1,self.target_embed_dimension)])

        _ = self.forward_modules.eval()
        # print(ignore_patches)
        batchsize,c,h,w = images.shape
        with torch.no_grad():
            start_time = time.time()
            features, patch_shapes = self._embed(images,context, provide_patch_shapes=True)
            time2 = time.time() - start_time
            # features = np.asarray(features)
            scales = self.ref_num_patches
            # print(features.shape) # 8192,1024
            # torch.cuda.synchronize()
            # st = time.time()
            patch_scores,residuals,query_nns = self.anomaly_scorer.predict([features],self.ignore_patches,nn_num,p,method)
            # torch.cuda.synchronize()
            # print(time.time() - st)
            residuals = self.patch_maker.unpatch_scores(residuals,batchsize=batchsize)

            residuals = residuals.reshape(batchsize,scales[0],scales[1],-1).permute(0,3,1,2)


            time3 = time.time() - start_time
            # print(time2/batchsize,(time3-time2)/batchsize,time3/batchsize)
            if self.context and self.add_pos_embed:
                context = context.permute(0, 3, 1, 2)
                context = F.interpolate(
                    context,
                    size=(self.ref_num_patches[0], self.ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                context = context.permute(0, 2, 3, 1).reshape(-1, context.shape[1]).numpy()+(1e-10)
                kl = context * np.log(context/(self.contexts[query_nns]+(1e-10)))
                residuals = np.concatenate([residuals,kl],axis=1)

            if vi_images is not None:
                print(vi_images.shape,query_nns.shape)
                stride = h//self.ref_num_patches[0]

                images_nns = vi_images[query_nns.cpu()].view(batchsize,-1,3*stride*stride).transpose(1,2)
                images_nns = torch.nn.functional.fold(images_nns,(h,w),stride,stride=stride)
                # print(images_nns.shape)# torch.Size([8, 3,512,512])
                for i in range(batchsize):
                    plt.subplot(1,2,1)
                    plt.imshow(images[i].cpu().permute(1,2,0)*torch.tensor(IMAGENET_STD)+torch.tensor(IMAGENET_MEAN))
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplot(1, 2, 2)
                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(images_nns[i].cpu().permute(1, 2, 0) * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN))
                    if self.add_pos_embed:
                        title = 'position_code'
                    elif self.context:
                        title = f'context_weight{self.context_weight}_concat'
                    else:
                        title = 'patchcore_residual'
                    plt.title(title)
                    # plt.show()
                    l = [*glob('work_dirs/img/*.png')]
                    plt.show()
                    # plt.savefig(f'work_dirs/img/{title}_{self.images.shape[0]}_{len(l)}.png')
                    plt.close()
            if self.save_residuals:
                return residuals

            image_scores = patch_scores
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)
            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        if self.save_patch_scores:

            return [score for score in image_scores], [mask for mask in masks], [patch_score for patch_score in
                                                                                 patch_scores]


        return [score for score in image_scores], [mask for mask in masks]

    def _fast_predict(self, images, ids, context=None, nn_num=1, p=None, method='square', sub=False, distance_limit=0.15,
                 vi_images=None):
        """Infer score and mask for a batch of images."""
        global_nn_features = self.features[ids].cuda()  # 10,4096,1024
        # # print(features.shape)
        # self.anomaly_scorer.fit(detection_features=[features.view(-1, self.target_embed_dimension)])
        # if self.ignore_patches is None:
        #     grid_h = np.arange(self.ref_num_patches[0], dtype=np.float32)
        #     grid_w = np.arange(self.ref_num_patches[1], dtype=np.float32)
        #     grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        #     grid = np.stack(grid, axis=0).transpose((1, 2, 0))
        #     grid = np.expand_dims(grid, 0)
        #     ref_grid = grid.repeat(len(ids), axis=0).reshape(-1, 2)
        #     query_grid = grid.repeat(images.shape[0], axis=0).reshape(-1, 1, 2)
        #
        #     # print(ref_grid.shape,query_grid.shape)# (10, 64, 64, 2) (1, 64, 64, 2)
        #     self.ignore_patches = np.sqrt(((ref_grid - query_grid) ** 2).sum(-1)) > (
        #                 distance_limit * max(self.ref_num_patches))  # 4096,40960
        _ = self.forward_modules.eval()
        # print(ignore_patches)
        batchsize, c, h, w = images.shape
        with torch.no_grad():
            start_time = time.time()
            features, patch_shapes = self._embed(images, context, provide_patch_shapes=True)
            time2 = time.time() - start_time
            # features = np.asarray(features)
            scales = self.ref_num_patches
            global_nn_features = global_nn_features.permute(0,2,1).view(global_nn_features.shape[0],-1,self.ref_num_patches[0],self.ref_num_patches[1])
            kernal_size = math.floor(distance_limit * max(self.ref_num_patches))
            global_nn_features = torch.nn.functional.pad(global_nn_features,(1,1,1,1))
            # print(global_nn_features.shape)
            global_nn_features = torch.nn.functional.unfold(global_nn_features,
                                                            kernal_size,
                                                            stride=1)
            global_nn_features = global_nn_features.view(global_nn_features.shape[0], -1,
                                                                       kernal_size *kernal_size, self.ref_num_patches[0]*self.ref_num_patches[1] )
            global_nn_features = global_nn_features.permute(3,0,2,1).flatten(1,2)
            # print(global_nn_features.shape,features.shape) # torch.Size([4096, 450, 1024]) torch.Size([4096, 1024])

            feature_distance = torch.cdist(features.unsqueeze(1),global_nn_features, p=2) ** 2
            # feature_distance[ignore_patches] = self.MAX_DIS
            # print(feature_distance.shape)
            _, query_nns = torch.topk(feature_distance.squeeze(1), nn_num, 1, largest=False)
            # global_nn_features = global_nn_features.permute(0,2,3,1).flatten(0,2)
            # print(global_nn_features.shape)
            if nn_num > 1:
                # p = np.zeros_like(query_distances)
                # p[query_distances[:,2] <= query_distances[:,0] * thres] = [0.5,0.3,0.2]
                # p[(query_distances[:,1] <= query_distances[:,0] * thres) * (query_distances[:,2] > query_distances[:,0] * thres)] = [0.7,0.3,0]
                # p[query_distances[:,1] > query_distances[:,0] * thres] = [1,0,0]
                indexs = np.random.choice(np.arange(len(p)), size=features.shape[0], p=p)
                # print(indexs)
                query_single_nns = query_nns[np.arange(features.shape[0]), indexs]

            else:
                query_single_nns = query_nns[:, 0]
            # print(query_nns.shape,self.detection_features[query_single_nns].shape)

            if method == 'square':
                residuals = (global_nn_features[np.arange(global_nn_features.shape[0]),query_single_nns,:] - features) ** 2
            if method == 'abs':
                residuals = torch.abs(global_nn_features[np.arange(global_nn_features.shape[0]),query_single_nns,:] - features)

            if method == 'sub':
                # print(global_nn_features[np.arange(global_nn_features.shape[0]),query_single_nns,:].shape)
                residuals = global_nn_features[np.arange(global_nn_features.shape[0]),query_single_nns,:] - features
            anomaly_scores = residuals.sum(-1)


            residuals = self.patch_maker.unpatch_scores(residuals, batchsize=batchsize)

            residuals = residuals.reshape(batchsize, scales[0], scales[1], -1).permute(0, 3, 1, 2)
            time3 = time.time() - start_time
            torch.cuda.empty_cache()
            # print(time2/batchsize,(time3-time2)/batchsize,time3/batchsize)
            if self.context and self.add_pos_embed:
                context = context.permute(0, 3, 1, 2)
                context = F.interpolate(
                    context,
                    size=(self.ref_num_patches[0], self.ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                context = context.permute(0, 2, 3, 1).reshape(-1, context.shape[1]).numpy() + (1e-10)
                kl = context * np.log(context / (self.contexts[query_nns] + (1e-10)))
                residuals = np.concatenate([residuals, kl], axis=1)

            if vi_images is not None:
                print(vi_images.shape)
                images_nns = vi_images[query_single_nns.cpu()].view(batchsize, -1, 3 * 8 * 8).transpose(1, 2)
                images_nns = torch.nn.functional.fold(images_nns, (h+8, w+8), 8, stride=8)
                # print(images_nns.shape)# torch.Size([8, 3,512,512])
                for i in range(batchsize):
                    plt.subplot(1, 2, 1)
                    plt.imshow(
                        images[i].cpu().permute(1, 2, 0) * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN))
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplot(1, 2, 2)
                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(
                        images_nns[i].cpu().permute(1, 2, 0) * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN))
                    if self.add_pos_embed:
                        title = 'position_code'
                    elif self.context:
                        title = f'context_weight{self.context_weight}_concat'
                    else:
                        title = 'patchcore_residual'
                    plt.title(title)
                    # plt.show()
                    l = [*glob('work_dirs/img/*.png')]
                    plt.show()
                    # plt.savefig(f'work_dirs/img/{title}_{self.images.shape[0]}_{len(l)}.png')
                    plt.close()
            if self.save_residuals:
                return residuals

            image_scores = patch_scores
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)
            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        if self.save_patch_scores:
            return [score for score in image_scores], [mask for mask in masks], [patch_score for patch_score in
                                                                                 patch_scores]

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
