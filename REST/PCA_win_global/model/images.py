import os
import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
import numpy as np
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def augmentation(image):
    augmented_images = [image]
    for angle in [-45, -33.75, -22.5, -11.25, 11.25, 22.5, 33.75, 45]:
        augmented_images.append(
            TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
        )
    height, width = image.shape[-2:]
    for shift_x, shift_y in [
        (0.2, 0.2),
        (-0.2, 0.2),
        (-0.2, -0.2),
        (0.2, -0.2),
        (0.1, 0.1),
        (-0.1, 0.1),
        (-0.1, -0.1),
        (0.1, -0.1),
    ]:
        augmented_images.append(
            TF.affine(
                image,
                angle=0.0,
                translate=[int(round(width * shift_x)), int(round(height * shift_y))],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
        )
    augmented_images.append(TF.hflip(image))
    augmented_images.append(TF.rgb_to_grayscale(image, num_output_channels=image.shape[0]))
    for k in [1, 2, 3]:
        augmented_images.append(torch.rot90(image, k, dims=(-2, -1)))
    shuffle_indices = torch.randperm(len(augmented_images)).tolist()
    return [augmented_images[idx] for idx in shuffle_indices]


class ImagesDataset(Dataset):
    def __init__(self, root, mode='RGB', transforms=None):
        self.transforms = transforms
        self.mode = mode
        self.filenames = sorted([*glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True),
                                 *glob.glob(os.path.join(root, '**', '*.png'), recursive=True)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            # mode = L : convert img into gray
            img = img.convert(self.mode)
        
        if self.transforms:
            img = self.transforms(img)
        
        return img
class LabeledImagesDataset(Dataset):
    def __init__(self,root,mode='RGB',train_transforms=None,mask_transforms=None,label=None,
                 feature_folder=None,aug_feature_folder=None,semi_label=False,label_num=None,args=None):
        if train_transforms is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = train_transforms
        self.mask_transform = mask_transforms
        self.mode = mode
        self.label = label
        self.semi_label = semi_label
        self.root = root
        self.aug_feature_folder = aug_feature_folder

        self.filenames = sorted([*glob.glob(os.path.join(root, '*.jpg')),
                                 *glob.glob(os.path.join(root, '*.png'))])
        if semi_label: # 模拟人工bbox标注的gt
            self.mask_filenames = sorted([*glob.glob(os.path.join(root, args.semi_label_folder, '*.npy'))])
        elif mask_transforms is not None and os.path.exists(os.path.join(root,'origin_gt')):
            self.mask_filenames = sorted([*glob.glob(os.path.join(root, 'origin_gt', '*.jpg')),
                                          *glob.glob(os.path.join(root, 'origin_gt', '*.png'))])
        else:
            self.mask_filenames = sorted([*glob.glob(os.path.join(root,'binary', '*.jpg')),
                                 *glob.glob(os.path.join(root, 'binary','*.png'))])
        if isinstance(feature_folder,str):
            self.feature_filenames = sorted([*glob.glob(os.path.join(root, feature_folder, '*.npy'))])
            assert len(self.filenames) == len(self.feature_filenames)
            self.feature_filenames = [self.feature_filenames]
        # print(feature_folder)

        elif isinstance(feature_folder,list):
            self.feature_filenames = [sorted([*glob.glob(os.path.join(root, folder, '*.npy'))]) for folder in feature_folder]
            assert len(self.filenames) == len(self.feature_filenames[0]),f"{ len(self.filenames)},{ len(self.feature_filenames[0])}"
        else: self.feature_filenames = None
        self.label_num = label_num
        # if args is not None:
        #     if not args.semi:
        #         self.filenames = self.filenames[:label_num]
        #         self.feature_filenames = self.feature_filenames[:label_num]
        #         self.mask_filenames = self.mask_filenames[:label_num]
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)

        if self.transform:
            img = self.transform(img)
        if self.semi_label:
            mask = np.load(self.mask_filenames[idx])
        else:
            with Image.open(self.mask_filenames[idx]) as mask:
                mask = mask.convert('L')
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)
        if self.label_num is not None:
            if idx < self.label_num:
                mask[mask == 0] = -1 # 负样本
            else:
                mask = torch.zeros_like(mask) # 无标签样本
            # print(self.root, idx,self.label_num,torch.unique(mask))


        item = {"image": img,
                    "mask": mask,

                    "label": self.label,
                    "filename":self.filenames[idx]}
        if self.feature_filenames is not  None:
            features = []
            h,w = 0,0
            for p in self.feature_filenames:
                feature_path = p[idx]
                # print(feature_path)
                if self.aug_feature_folder is not None:
                    aug_feature_path = [*glob.glob(os.path.join(self.root,self.aug_feature_folder,
                                                                os.path.basename(feature_path).replace('.npy','_*.npy')))]
                    aug_feature_path.append(feature_path)
                    assert len(aug_feature_path) ==8,feature_path
                    np.random.shuffle(aug_feature_path)
                    feature_path = aug_feature_path[0]
                # print(feature_path)
                feature = torch.tensor(np.load(feature_path))
                h = max(feature.shape[1],h)
                w = max(feature.shape[2],w)
                features.append(feature)

            # for i in range(len(features)):
            #
            #     if features[i].shape[1] != h and features[i].shape[2] != w:
            #         features[i] = torch.nn.functional.interpolate(features[i].unsqueeze(1),
            #     size=(h,w),
            #     mode="bilinear",
            #     align_corners=False).squeeze(1)
                # print(features[i].shape,h,w)
                # if len(features) > 1:
                    # features[i] = torch.nn.functional.normalize(features[i].permute(1,2,0), p=2.0, dim=2, eps=1e-12, out=None).permute(2,0,1)
                    # print(torch.sqrt(torch.sum(features[i].permute(1,2,0)[0,0]**2)),features[i].permute(1,2,0)[0,0].shape)
            # print(len(features))
            item['feature'] = torch.cat(features).float()
            # print(len(features),item['feature'].shape)
        return  item
class SampleDataset(LabeledImagesDataset):
    def __init__(self, root, mode='RGB', train_transforms=None, mask_transforms=None, label=None,
                 feature_folder=None, aug_feature_folder=None, semi_label=False, label_num=None, args=None,
              seed=None, split='train', num_sample=10):
        super().__init__(root, mode, train_transforms, mask_transforms, label,  feature_folder,
                         aug_feature_folder, semi_label,
                         label_num, args,)

        if seed is not None:
            np.random.RandomState(seed).shuffle(self.filenames)
            np.random.RandomState(seed).shuffle(self.mask_filenames)
            shuffle_indices = np.arange(len(self.filenames))
            np.random.RandomState(seed).shuffle(shuffle_indices)
            if split == 'train':
                self.filenames = self.filenames[:num_sample]
                self.mask_filenames = self.mask_filenames[:num_sample]
                shuffle_indices = shuffle_indices[0:num_sample]
                # print(self.filenames)
            if split == 'test':
                self.filenames = self.filenames[num_sample:]
                self.mask_filenames = self.mask_filenames[num_sample:]
                shuffle_indices = shuffle_indices[num_sample:]
            if feature_folder is not None:
                self.feature_filenames = [[i[j] for j in shuffle_indices] for i in self.feature_filenames]


class SampleAugDataset(LabeledImagesDataset):
    def __init__(self, root, mode='RGB', train_transforms=None, mask_transforms=None, label=None,
                 feature_folder=None, aug_feature_folder=None, semi_label=False, label_num=None, args=None,
                 seed=None, num_sample=10):
        super().__init__(root, mode, train_transforms, mask_transforms, label, feature_folder,
                         aug_feature_folder, semi_label, label_num, args)

        if seed is not None:
            rng = np.random.RandomState(seed)
            shuffle_indices = np.arange(len(self.filenames))
            rng.shuffle(shuffle_indices)
            self.filenames = [self.filenames[idx] for idx in shuffle_indices]
            self.mask_filenames = [self.mask_filenames[idx] for idx in shuffle_indices]
            if self.feature_filenames is not None:
                self.feature_filenames = [[paths[idx] for idx in shuffle_indices] for paths in self.feature_filenames]

        self.filenames = self.filenames[:num_sample]
        self.mask_filenames = self.mask_filenames[:num_sample]
        if self.feature_filenames is not None:
            self.feature_filenames = [paths[:num_sample] for paths in self.feature_filenames]

        self.augmented_data = None
        self.augmented_filenames = None
        if self.label == 0:
            self.augmented_data = []
            self.augmented_filenames = []
            for path in self.filenames:
                with Image.open(path) as img:
                    img = img.convert(self.mode)
                if self.transform:
                    img = self.transform(img)
                augmented_images = augmentation(img)
                self.augmented_data.extend(augmented_images)
                self.augmented_filenames.extend([path] * len(augmented_images))
            self.filenames = self.augmented_filenames

    def __len__(self):
        if self.label == 0 and self.augmented_data is not None:
            return len(self.augmented_data)
        return super().__len__()

    def __getitem__(self, idx):
        if self.label == 0 and self.augmented_data is not None:
            image = self.augmented_data[idx]
            mask = torch.zeros((1, image.shape[1], image.shape[2]))
            return {
                "image": image,
                "mask": mask,
                "label": self.label,
                "filename": self.filenames[idx],
            }
        return super().__getitem__(idx)

