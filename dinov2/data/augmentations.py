# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

import numpy as np
import random
import torch

logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip (hor, vert)
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


def random_resized_crop_4ch(image, size, scale=(0.08, 1.0)):
    """
    image: np.ndarray of shape (H, W, 4)
    size: output size (int) for square crop (size x size)
    scale: tuple specifying min and max area ratio for the crop
    """
    H, W, C = image.shape
    area = H * W

    for _ in range(10):
        target_area = random.uniform(*scale) * area
        # pick a random aspect ratio
        aspect_ratio = random.uniform(0.75, 1.3333)
        
        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if w <= W and h <= H:
            top = random.randint(0, H - h)
            left = random.randint(0, W - w)
            # Crop
            crop = image[top: top + h, left: left + w, :]
            # Resize to (size, size)
            crop = resize_4ch(crop, (size, size))
            return crop

    # fallback: just center-crop then resize
    min_side = min(H, W)
    center_crop = image[(H - min_side)//2 : (H + min_side)//2,
                        (W - min_side)//2 : (W + min_side)//2, :]
    return resize_4ch(center_crop, (size, size))

def resize_4ch(image, out_size):
    """
    Resize a (H,W,4) image to out_size using e.g. OpenCV or skimage.
    """
    import cv2
    # shape: (H, W, 4) -> do channel-wise (or if OpenCV, we can do it all at once).
    # For example:
    resized = []
    for c in range(image.shape[-1]):
        resized_c = cv2.resize(image[..., c], out_size, interpolation=cv2.INTER_CUBIC)
        resized.append(resized_c[..., None])
    return np.concatenate(resized, axis=-1)

def random_horizontal_flip_4ch(image, p=0.5):
    if random.random() < p:
        return np.ascontiguousarray(image[:, ::-1, :])
    return image

def random_vertical_flip_4ch(image, p=0.5):
    if random.random() < p:
        return np.ascontiguousarray(image[::-1, :, :])
    return image

def to_tensor_4ch(image):
    """
    Convert HWC ndarray to CHW PyTorch tensor
    """
    # shape (H, W, C) -> (C, H, W)
    return torch.from_numpy(image.transpose(2, 0, 1))

def normalize_4ch(tensor, mean, std):
    """
    Normalize each of the 4 channels with the given mean & std (lists of length 4)
    """
    # tensor: shape (4, H, W)
    for c in range(tensor.shape[0]):
        tensor[c] = (tensor[c] - mean[c]) / std[c]
    return tensor

class DataAugmentationDINO_4CH:
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        #calculated for my min-max scaled 255 UINT8 Tiffs in the dataset
        mean = (39.7075, 15.8885, 18.5576,100.3117),
        std = (30.8822,16.7861,15.7978,44.8835)
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.mean = mean
        self.std = std

    def geometric_augmentation_global(self, image):
        # Random resized crop
        image = random_resized_crop_4ch(image, self.global_crops_size, self.global_crops_scale)
        # Random flips
        image = random_horizontal_flip_4ch(image, p=0.5)
        image = random_vertical_flip_4ch(image, p=0.5)
        return image

    def geometric_augmentation_local(self, image):
        image = random_resized_crop_4ch(image, self.local_crops_size, self.local_crops_scale)
        image = random_horizontal_flip_4ch(image, p=0.5)
        image = random_vertical_flip_4ch(image, p=0.5)
        return image

    def global_transfo1(self, image):
        return self.normalize_4ch(image)

    def global_transfo2(self, image):
        return image

    def local_transfo(self, image):
        return image

    def normalize_4ch(self, image):
        tensor = to_tensor_4ch(image)  # (4, H, W)
        tensor = normalize_4ch(tensor, self.mean, self.std)
        return tensor

    def __call__(self, image):
        """
        image: np.ndarray (H,W,4)
        """
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)
        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = []
        for _ in range(self.local_crops_number):
            im_local_base = self.geometric_augmentation_local(image)
            local_crops.append(self.local_transfo(im_local_base))
        output["local_crops"] = local_crops

        # offsets (if you need them, else leave empty)
        output["offsets"] = ()
        return output