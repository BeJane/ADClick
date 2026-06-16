import json
import os
from enum import Enum
from glob import glob

import PIL
import torch
import torchvision
from torchvision import transforms

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize,interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)[0].unsqueeze(0)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "ok"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        with open(self.source,'r',encoding='utf-8') as f:
            data_info = json.load(f)
        for classname in self.classnames_to_use:
            for anomaly in ['ng','ok']:
                if anomaly == 'ng' and self.split == DatasetSplit.TRAIN:continue

                image_paths = [data_info["root_dir"] + p for p in data_info[classname][self.split.value][anomaly]]
                mask_paths = [data_info["root_dir"] + p for p in data_info[classname][self.split.value][f'{anomaly}_binary']]
                if anomaly == 'ng' and self.split == DatasetSplit.VAL:
                    image_paths += [data_info["root_dir"] + p for p in data_info[classname]["train"][anomaly]]
                    mask_paths += [data_info["root_dir"] + p for p in
                                  data_info[classname]["train"][f'{anomaly}_binary']]
                for i, image_path in enumerate(image_paths):
                    data_tuple = [classname, anomaly, image_path]
                    if anomaly != "ok":
                        data_tuple.append(mask_paths[i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return  data_to_iterate
