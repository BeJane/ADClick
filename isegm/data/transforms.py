from typing import Optional

import cv2
import random
import numpy as np
import torch

from albumentations.core.serialization import SERIALIZABLE_REGISTRY
from albumentations import ImageOnlyTransform, DualTransform, normalize_bbox, get_random_crop_coords
from albumentations.core.transforms_interface import to_tuple
from albumentations.augmentations import functional as F
from isegm.utils.misc import get_bbox_from_mask, expand_bbox, clamp_bbox, get_labels_with_sizes
class TripleTransform(DualTransform):
    """Transform for segmentation task."""

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,

            "residual":self.apply_to_residual
        }

class UniformRandomResize(TripleTransform):
    def __init__(self, scale_range=(0.9, 1.1), interpolation=cv2.INTER_LINEAR, always_apply=False, p=1,residual_stride=4):
        super().__init__(always_apply, p)
        self.scale_range = scale_range
        self.interpolation = interpolation
        self.residual_stride = residual_stride
    def get_params_dependent_on_targets(self, params):
        scale = random.uniform(*self.scale_range)
        height = int(round(params['image'].shape[0] * scale))
        width = int(round(params['image'].shape[1] * scale))
        return {'new_height': height, 'new_width': width}

    def apply(self, img, new_height=0, new_width=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.resize(img, height=new_height, width=new_width, interpolation=interpolation)
    def apply_to_residual(self, residual, **params):
        return torch.nn.functional.interpolate(residual,
                                               size=(params['new_height']//self.residual_stride, params['new_width']//self.residual_stride),
                                               mode='bilinear', align_corners=False)
    def apply_to_keypoint(self, keypoint, new_height=0, new_width=0, **params):
        scale_x = new_width / params["cols"]
        scale_y = new_height / params["rows"]
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return "scale_range", "interpolation"

    @property
    def targets_as_params(self):
        return ["image"]
class TriPadIfNeeded(TripleTransform):
    """Pad side of the image / max if side is less than desired number.

    Args:
        min_height (int): minimal result image height.
        min_width (int): minimal result image width.
        pad_height_divisor (int): if not None, ensures image height is dividable by value of this argument.
        pad_width_divisor (int): if not None, ensures image width is dividable by value of this argument.
        border_mode (OpenCV flag): OpenCV border mode.
        value (int, float, list of int, lisft of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    lisft of float): padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bbox, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        min_height: Optional[int] = 1024,
        min_width: Optional[int] = 1024,
        pad_height_divisor: Optional[int] = None,
        pad_width_divisor: Optional[int] = None,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=1.0,
            residual_stride=4
    ):
        if (min_height is None) == (pad_height_divisor is None):
            raise ValueError("Only one of 'min_height' and 'pad_height_divisor' parameters must be set")

        if (min_width is None) == (pad_width_divisor is None):
            raise ValueError("Only one of 'min_width' and 'pad_width_divisor' parameters must be set")

        super(TriPadIfNeeded, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.pad_width_divisor = pad_width_divisor
        self.pad_height_divisor = pad_height_divisor
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.residual_stride = residual_stride

    def update_params(self, params, **kwargs):
        params = super(TriPadIfNeeded, self).update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]

        if self.min_height is not None:
            if rows < self.min_height:
                h_pad_top = int((self.min_height - rows) / 2.0)
                h_pad_bottom = self.min_height - rows - h_pad_top
            else:
                h_pad_top = 0
                h_pad_bottom = 0
        else:
            pad_remained = rows % self.pad_height_divisor
            pad_rows = self.pad_height_divisor - pad_remained if pad_remained > 0 else 0

            h_pad_top = pad_rows // 2
            h_pad_bottom = pad_rows - h_pad_top

        if self.min_width is not None:
            if cols < self.min_width:
                w_pad_left = int((self.min_width - cols) / 2.0)
                w_pad_right = self.min_width - cols - w_pad_left
            else:
                w_pad_left = 0
                w_pad_right = 0
        else:
            pad_remainder = cols % self.pad_width_divisor
            pad_cols = self.pad_width_divisor - pad_remainder if pad_remainder > 0 else 0

            w_pad_left = pad_cols // 2
            w_pad_right = pad_cols - w_pad_left

        h_pad_top =  h_pad_top + (self.residual_stride - h_pad_top%self.residual_stride)
        h_pad_bottom = h_pad_bottom + (self.residual_stride - h_pad_bottom%self.residual_stride)
        w_pad_right = w_pad_right + (self.residual_stride - w_pad_right%self.residual_stride)
        w_pad_left = w_pad_left + (self.residual_stride - w_pad_left%self.residual_stride)
        params.update(
            {"pad_top": h_pad_top, "pad_bottom": h_pad_bottom, "pad_left": w_pad_left, "pad_right": w_pad_right}
        )
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right, border_mode=self.border_mode, value=self.value
        )
    def apply_to_residual(self, residual, **params):
        return torch.nn.functional.pad(residual,(params["pad_left"]//self.residual_stride,params["pad_right"]//self.residual_stride,
                                                 params["pad_top"]//self.residual_stride,params["pad_bottom"]//self.residual_stride),
                                       value=0)
    def apply_to_mask(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right, border_mode=self.border_mode, value=self.mask_value
        )

    def apply_to_bbox(self, bbox, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, rows=0, cols=0, **params):
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
        bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
        return normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    # skipcq: PYL-W0613
    def apply_to_keypoint(self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        x, y, angle, scale = keypoint
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return (
            "min_height",
            "min_width",
            "pad_height_divisor",
            "pad_width_divisor",
            "border_mode",
            "value",
            "mask_value",
        )

class Flip(TripleTransform):
    """Flip the input either horizontally, vertically or both horizontally and vertically.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img, d=0, **params):
        """Args:
        d (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                -1 for both vertical and horizontal flipping (which is also could be seen as rotating the input by
                180 degrees).
        """
        return F.random_flip(img, d)
    def apply_to_residual(self, residual, d=0,**params):
        if d == -1:
            return torch.flip(residual, [2,3])
        return torch.flip(residual,[2+d])
    def get_params(self):
        # Random int in the range [-1, 1]
        return {"d": random.randint(-1, 1)}

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_flip(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_flip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()

class TriRandomCrop(TripleTransform):
    """Crop a random part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, always_apply=False, p=1.0,residual_stride=4):
        super(TriRandomCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.residual_stride = residual_stride

    def apply(self, img, h_start=0, w_start=0, **params):
        height, width = img.shape[:2]
        if height <self.height or width < self.width:
            raise ValueError(
                "Requested crop size ({crop_height}, {crop_width}) is "
                "larger than the image size ({height}, {width})".format(
                    crop_height=self.height, crop_width=self.width, height=height, width=width
                )
            )
        x1, y1, x2, y2 = get_random_crop_coords(height, width, self.height, self.width, h_start, w_start)
        img = img[y1:y2, x1:x2]
        return img

    def apply_to_residual(self, residual,h_start=0,w_start=0, **params):
        height, width = residual.shape[2:4]
        height,width = height*self.residual_stride, width*self.residual_stride
        if height <self.height or width < self.width:
            raise ValueError(
                "Requested crop size ({crop_height}, {crop_width}) is "
                "larger than the image size ({height}, {width})".format(
                    crop_height=self.height, crop_width=self.width, height=height, width=width
                )
            )
        x1, y1, x2, y2 = get_random_crop_coords(height, width, self.height, self.width, h_start, w_start)
        residual = residual[:,:,y1//self.residual_stride:y2//self.residual_stride, x1//self.residual_stride:x2//self.residual_stride]

        return residual

    def get_params(self):
        return {"h_start": random.random(), "w_start": random.random()}

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_random_crop(bbox, self.height, self.width, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_random_crop(keypoint, self.height, self.width, **params)

    def get_transform_init_args_names(self):
        return ("height", "width")


class ZoomIn(DualTransform):
    def __init__(
            self,
            height,
            width,
            bbox_jitter=0.1,
            expansion_ratio=1.4,
            min_crop_size=200,
            min_area=100,
            always_resize=False,
            always_apply=False,
            p=0.5,
    ):
        super(ZoomIn, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.bbox_jitter = to_tuple(bbox_jitter)
        self.expansion_ratio = expansion_ratio
        self.min_crop_size = min_crop_size
        self.min_area = min_area
        self.always_resize = always_resize

    def apply(self, img, selected_object, bbox, **params):
        if selected_object is None:
            if self.always_resize:
                img = F.resize(img, height=self.height, width=self.width)
            return img

        rmin, rmax, cmin, cmax = bbox
        img = img[rmin:rmax + 1, cmin:cmax + 1]
        img = F.resize(img, height=self.height, width=self.width)

        return img

    def apply_to_mask(self, mask, selected_object, bbox, **params):
        if selected_object is None:
            if self.always_resize:
                mask = F.resize(mask, height=self.height, width=self.width,
                                interpolation=cv2.INTER_NEAREST)
            return mask

        rmin, rmax, cmin, cmax = bbox
        mask = mask[rmin:rmax + 1, cmin:cmax + 1]
        if isinstance(selected_object, tuple):
            layer_indx, mask_id = selected_object
            obj_mask = mask[:, :, layer_indx] == mask_id
            new_mask = np.zeros_like(mask)
            new_mask[:, :, layer_indx][obj_mask] = mask_id
        else:
            obj_mask = mask == selected_object
            new_mask = mask.copy()
            new_mask[np.logical_not(obj_mask)] = 0

        new_mask = F.resize(new_mask, height=self.height, width=self.width,
                            interpolation=cv2.INTER_NEAREST)
        return new_mask

    def get_params_dependent_on_targets(self, params):
        instances = params['mask']

        is_mask_layer = len(instances.shape) > 2
        candidates = []
        if is_mask_layer:
            for layer_indx in range(instances.shape[2]):
                labels, areas = get_labels_with_sizes(instances[:, :, layer_indx])
                candidates.extend([(layer_indx, obj_id)
                                   for obj_id, area in zip(labels, areas)
                                   if area > self.min_area])
        else:
            labels, areas = get_labels_with_sizes(instances)
            candidates = [obj_id for obj_id, area in zip(labels, areas)
                          if area > self.min_area]

        selected_object = None
        bbox = None
        if candidates:
            selected_object = random.choice(candidates)
            if is_mask_layer:
                layer_indx, mask_id = selected_object
                obj_mask = instances[:, :, layer_indx] == mask_id
            else:
                obj_mask = instances == selected_object

            bbox = get_bbox_from_mask(obj_mask)

            if isinstance(self.expansion_ratio, tuple):
                expansion_ratio = random.uniform(*self.expansion_ratio)
            else:
                expansion_ratio = self.expansion_ratio

            bbox = expand_bbox(bbox, expansion_ratio, self.min_crop_size)
            bbox = self._jitter_bbox(bbox)
            bbox = clamp_bbox(bbox, 0, obj_mask.shape[0] - 1, 0, obj_mask.shape[1] - 1)

        return {
            'selected_object': selected_object,
            'bbox': bbox
        }

    def _jitter_bbox(self, bbox):
        rmin, rmax, cmin, cmax = bbox
        height = rmax - rmin + 1
        width = cmax - cmin + 1
        rmin = int(rmin + random.uniform(*self.bbox_jitter) * height)
        rmax = int(rmax + random.uniform(*self.bbox_jitter) * height)
        cmin = int(cmin + random.uniform(*self.bbox_jitter) * width)
        cmax = int(cmax + random.uniform(*self.bbox_jitter) * width)

        return rmin, rmax, cmin, cmax

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_transform_init_args_names(self):
        return ("height", "width", "bbox_jitter",
                "expansion_ratio", "min_crop_size", "min_area", "always_resize")


def remove_image_only_transforms(sdict):
    if not 'transforms' in sdict:
        return sdict

    keep_transforms = []
    for tdict in sdict['transforms']:
        cls = SERIALIZABLE_REGISTRY[tdict['__class_fullname__']]
        if 'transforms' in tdict:
            keep_transforms.append(remove_image_only_transforms(tdict))
        elif not issubclass(cls, ImageOnlyTransform):
            keep_transforms.append(tdict)
    sdict['transforms'] = keep_transforms

    return sdict
