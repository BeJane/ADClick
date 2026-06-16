import os.path
import pickle as pkl
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample

categories = ['01','02','03' ]
class BTADDataset(ISDataset):
    def __init__(self, dataset_path,category, split='train', **kwargs):
        super().__init__(**kwargs)
        assert split in {'train','test'}

        self.dataset_path = Path(dataset_path)
        self.dataset_path_path = self.dataset_path
        self.dataset_split = split
        if split == 'test':
            self.dataset_samples = [*glob(os.path.join(self.dataset_path, category, '*', 'ng', '*.png'))]
        if split == 'train':
            # self.dataset_samples = []
            # for c in categories:
                # if c == category:continue
                # print(c)
                # self.dataset_samples.extend( [*glob(os.path.join(self.dataset_path, c, '*', '*', '*.png'))])
            self.dataset_samples = [*glob(os.path.join(self.dataset_path, category, 'train', 'ok', '*.png'))] + \
                                   [*glob(os.path.join(self.dataset_path, category, 'train', 'false_ng', '*.png'))]

    def get_sample(self, index) -> DSample:
        sample_id = self.dataset_samples[index]
        bsn = os.path.basename(sample_id)
        mask_path = os.path.join(sample_id.replace(bsn,''),'binary',bsn.replace('.png','_pha.png'))
        # print(sample_id)
        image = cv2.imread(sample_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)
        # if self.dataset_split == 'test':
            # instance_id = self.instance_ids[index]
        mask = np.zeros_like(instances_mask)
        mask[instances_mask == 0] = 0  # ignored area
        mask[instances_mask == 255] = 1

        instances_mask = mask
            # print(np.unique(instances_mask))
        # plt.imshow(instances_mask)
        # plt.show()
        if self.dataset_split == 'test':
            return DSample(image, instances_mask, objects_ids=[1], sample_id=index),sample_id
        return DSample(image, instances_mask, objects_ids=[1],sample_id=index)
