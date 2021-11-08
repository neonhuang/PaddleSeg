# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
from PIL import Image
import random
import math
import collections

from paddleseg.cvlibs import manager


@manager.TRANSFORMS.add_component
class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].

    Args:
        transforms (list): A list contains data pre-processing or augmentation. Empty list means only reading images, no transformation.
        to_rgb (bool, optional): If converting image to RGB color space. Default: True.

    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """

    def __init__(self, transforms, cut_height=0, to_rgb=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.to_rgb = to_rgb
        self.cut_height = cut_height

    def __call__(self, im, label=None):
        """
        Args:
            im (str|np.ndarray): It is either image path or image object.
            label (str|np.ndarray): It is either label path or label ndarray.

        Returns:
            (tuple). A tuple including image, image info, and label after transformation.
        """
        if isinstance(im, str):
            im = cv2.imread(im).astype('float32')
            im = im[self.cut_height:, :, :]
        if isinstance(label, str):
            label = np.asarray(Image.open(label))
            if len(label.shape) > 2:
                label = label[:, :, 0]
                label = label[self.cut_height:, :]
        if im is None:
            raise ValueError('Can\'t read The image file {}!'.format(im))
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for op in self.transforms:
            outputs = op(im, label)
            im = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]
        im = np.transpose(im, (2, 0, 1))
        return (im, label)


@manager.TRANSFORMS.add_component
class GroupRandomRotation:
    def __init__(self, degree=(-10, 10), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=None):
        self.degree = degree
        self.interpolation = interpolation
        self.padding = padding
        if self.padding is None:
            self.padding = [0, 0]

    def __call__(self, im, label=None):
        img_group = im, label
        assert (len(self.interpolation) == len(img_group))
        v = random.random()
        if v < 0.5:
            degree = random.uniform(self.degree[0], self.degree[1])
            h, w = img_group[0].shape[0:2]
            center = (w / 2, h / 2)
            map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
            out_images = list()
            for img, interpolation, padding in zip(img_group, self.interpolation, self.padding):
                out_images.append(cv2.warpAffine(
                    img, map_matrix, (w, h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=padding))
                if len(img.shape) > len(out_images[-1].shape):
                    out_images[-1] = out_images[-1][...,
                                                    np.newaxis]  # single channel image
            return out_images
        else:
            return img_group


@manager.TRANSFORMS.add_component
class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, im, label=None, is_flow=False):
        img_group = im, label
        v = random.random()
        if v < 0.5:
            out_images = [np.fliplr(img) for img in img_group]
            if self.is_flow:
                for i in range(0, len(out_images), 2):
                    # invert flow pixel values when flipping
                    out_images[i] = -out_images[i]
            return out_images
        else:
            return img_group


@manager.TRANSFORMS.add_component
class LaneNormalize:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im, label=None):
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = im.astype(np.float32, copy=False)
        im -= mean
        im /= std

        if label is None:
            return (im,)
        else:
            return (im, label)
