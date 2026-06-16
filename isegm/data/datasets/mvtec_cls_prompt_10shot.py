import os.path
import pickle as pkl
from glob import glob
from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from isegm.data.base import ISDataset
from isegm.data.sample import DSample

categories = ['carpet','grid','leather','tile','wood', 'bottle', 'cable', 'capsule','hazelnut', 'metal_nut','pill', 'screw',
        'toothbrush','transistor', 'zipper' ]
class Mvtec_ClsPrompt_10shot_Dataset(ISDataset):
    def __init__(self, dataset_path,category, split='train', **kwargs):
        super().__init__(**kwargs)
        # assert split in {'train','test'}

        self.dataset_path = Path(dataset_path)
        self.dataset_path_path = self.dataset_path
        self.dataset_split = split
        # self.residual_dir = 'global50_residual_l123'
        self.split_dir = os.path.split(self.dataset_path)
        self.prompt_info = {}
        for c in categories:
            lang_path = os.path.join(self.split_dir[0],self.split_dir[1]+'_text',f'{c}_classifier.pkl')
            with open(lang_path, 'rb') as f:
                self.prompt_info[c] = pkl.load(f)

        if split == 'train':
            self.dataset_samples = []
            for c in categories:
                if c == category:continue

                self.dataset_samples.extend( [*glob(os.path.join(self.dataset_path, c, '*', 'ng', '*.png'))])
                self.dataset_samples.extend([*glob(os.path.join(self.dataset_path, c, '*', 'false_ng', '*.png'))])
            # self.dataset_samples = [*glob(os.path.join(self.dataset_path, category, 'train', 'ok', '*.png'))] + \
            #                        [*glob(os.path.join(self.dataset_path, category, 'train', 'false_ng', '*.png'))]
        if split == 'finetune':

            self.dataset_samples = [*glob(os.path.join(self.dataset_path, category, 'train', 'false_ng', '*.png'))]

        if split == 'train_sup_ad':

            self.dataset_samples = [*glob(os.path.join(self.dataset_path, category, 'train', 'ok', '*.png'))] + \
                                   [*glob(os.path.join(self.dataset_path, category, 'train', 'false_ng', '*.png'))] + \
            [*glob(os.path.join(self.dataset_path, category, 'train', 'ng','train_ng_*', '*.png'))]

        if split == 'test':
            self.dataset_samples = [*glob(os.path.join(self.dataset_path, category, '*', 'ng', '*.png'))]

        if split == 'test_sup_ad':
            self.dataset_samples = [*glob(os.path.join(self.dataset_path, category, 'test', '*', '*.png'))]

        if split == 'test_un_ad':
            self.dataset_samples = [*glob(os.path.join(self.dataset_path, category, 'test', '*', '*.png'))] + \
                                   [*glob(os.path.join(self.dataset_path, category, 'train', 'ng', '*.png'))]

    def get_sample(self, index) -> DSample:
        sample_id = self.dataset_samples[index]


        bsn = os.path.basename(sample_id)

        mask_path = os.path.join(sample_id.replace(bsn,''),'binary',bsn.replace('.png','_pha.png'))
        # print(sample_id)
        image = cv2.imread(sample_id)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)
        image = cv2.resize(image, (512,512))
        instances_mask = cv2.resize(instances_mask, (512,512))
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)


        c = sample_id.replace(os.path.join(*self.split_dir), '').split('/')[1]
        if c in ['carpet','grid','leather','tile','wood']:
            residual_path = os.path.join(sample_id.replace(bsn,''),'10shot_pca0.95_glo64s4pca16w4_pos0.03_l123_residual',bsn.replace('png','npy'))
        else:
            residual_path = os.path.join(sample_id.replace(bsn, ''), '10shot_pca0.95_glo64s4pca16w4_pos0.15_l123_residual',
                                         bsn.replace('png', 'npy'))
        residual = np.load(residual_path)
        residual = torch.from_numpy(residual)**2
        residual = residual.unsqueeze(0)
        # if self.dataset_split == 'test':
            # instance_id = self.instance_ids[index]
        mask = np.zeros_like(instances_mask)
        mask[instances_mask == 0] = 0  # ignored area
        mask[instances_mask == 255] = 1

        instances_mask = mask
        if self.dataset_split in ['test_sup_ad', 'test_un_ad']:
            prompt = []
            # anomaly type is unknow
            for k, v in self.prompt_info[c].items():
                if k == 'all_embedding':continue
                random_index = random.randint(0, len(v) - 1)
                prompt.append(v[random_index])
            prompt = torch.vstack(prompt)


        else:
            # know anomaly type


            anomaly = '_'.join(os.path.basename(sample_id).split('_')[3:-1])
            if 'false_ng' in sample_id: anomaly = 'false_ng'
            if anomaly == 'good':anomaly='all'
            assert f'{anomaly}_embedding' in self.prompt_info[c].keys(), f"{c},{anomaly},{self.prompt_info[c].keys()}"
            prompt_list  = self.prompt_info[c][f'{anomaly}_embedding']
            random_index = random.randint(0, len(prompt_list)-1)

            prompt = prompt_list[random_index]


        return DSample(image, instances_mask,residual, objects_ids=[1], sample_id=index),{'prompt':prompt,'path':sample_id}

    def __getitem__(self, index):
        if self.samples_precomputed_scores is not None:
            index = np.random.choice(self.samples_precomputed_scores['indices'],
                                     p=self.samples_precomputed_scores['probs'])
        else:
            if self.epoch_len > 0:
                index = random.randrange(0, len(self.dataset_samples))

        sample,sampleinfo = self.get_sample(index)
        sample = self.augment_sample(sample)
        sample.remove_small_objects(self.min_object_area)

        self.points_sampler.sample_object(sample)
        points = np.array(self.points_sampler.sample_points())
        mask = self.points_sampler.selected_mask

        output = {
            'images': self.to_tensor(sample.image),
            'points': points.astype(np.float32),
            'instances': mask,
            'residual': sample.residual.squeeze(),
            'prompt':sampleinfo['prompt']
        }

        if self.with_image_info:
            output['image_info'] = sample.sample_id

        return output
