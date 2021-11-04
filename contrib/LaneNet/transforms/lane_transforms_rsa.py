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
class LaneComposeRsa:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].

    Args:
        transforms (list): A list contains data pre-processing or augmentation. Empty list means only reading images, no transformation.
        to_rgb (bool, optional): If converting image to RGB color space. Default: False.

    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """

    def __init__(self, transforms, to_rgb=False):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.to_rgb = to_rgb

    def __call__(self, im, label=None, instancelabel=None):
        """
        Args:
            im (str|np.ndarray): It is either image path or image object.
            label (str|np.ndarray): It is either label path or label ndarray.
            instancelabel (str|np.ndarray): It is either label path or instancelabel ndarray.

        Returns:
            (tuple). A tuple including image, image info, and label after transformation.
        """
        grt_instance = None
        if isinstance(im, str):
            im = cv2.imread(im).astype('float32')
            im = im[160:, :, :]
        if isinstance(label, str):
            label = cv2.imread(label, cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
                label = label[160:, :]
            # label = np.asarray(Image.open(label))
            # label = np.array(label / 255.0, dtype=np.uint8)
        # if isinstance(instancelabel, str):
        #     grt_instance = cv2.imread(instancelabel, cv2.IMREAD_GRAYSCALE)
        # if im is None:
        #     raise ValueError('Can\'t read The image file {}!'.format(im))

        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # if grt_instance is not None:
        #     labels = cv2.merge([label, grt_instance])
        # else:
        #     labels = label
        labels = label
        for op in self.transforms:
            outputs = op(im, labels)
            im = outputs[0]
            if len(outputs) == 2:
                labels = outputs[1]

        im = np.transpose(im, (2, 0, 1)).astype('float32')
        label = labels.astype('int64')
        # if grt_instance is not None:
        #     label, grt_instance = cv2.split(labels)
        # else:
        #     label = labels

        # if grt_instance is not None:
        #     # h, w = im.shape[1:]
        #     # grt_instance = cv2.resize(
        #     #     grt_instance, [w, h], interpolation=cv2.INTER_NEAREST)
        #     grt_instance = grt_instance[np.newaxis, ...]

        return im, label


@manager.TRANSFORMS.add_component
class GroupRandomRotation:
    def __init__(self,
                 degree=(-10, 10),
                 interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST),
                 padding=None):
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
            for img, interpolation, padding in zip(
                    img_group, self.interpolation, self.padding):
                out_images.append(
                    cv2.warpAffine(
                        img,
                        map_matrix, (w, h),
                        flags=interpolation,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=padding))
                if len(img.shape) > len(out_images[-1].shape):
                    out_images[-1] = out_images[-1][
                        ..., np.newaxis]  # single channel image
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
class SampleResize(object):
    def __init__(self, size=(640, 368)):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, im, label=None):
        sample = im, label
        out = list()
        out.append(
            cv2.resize(sample[0], self.size, interpolation=cv2.INTER_CUBIC))
        if len(sample) > 1:
            out.append(
                cv2.resize(
                    sample[1], self.size, interpolation=cv2.INTER_NEAREST))
        return out


@manager.TRANSFORMS.add_component
class GroupNormalize(object):
    def __init__(self, mean=(103.939, 116.779, 123.68), std=(1., 1., 1.)):
        # tf.GroupNormalize(mean=(self.cfg.img_norm['mean'], (0,)), std=(
        #     self.cfg.img_norm['std'], (1,))),
        self.mean = ([103.939, 116.779, 123.68], (0, ))
        self.std = ([1., 1., 1.], (1, ))
        # self.std = std

    def __call__(self, im, label=None):
        img_group = im, label
        out_images = list()
        for img, m, s in zip(img_group, self.mean, self.std):
            if len(m) == 1:
                img = img - np.array(m)  # single channel image
                img = img / np.array(s)
            else:
                img = img - np.array(m)[np.newaxis, np.newaxis, ...]
                img = img / np.array(s)[np.newaxis, np.newaxis, ...]
            out_images.append(img)

        return out_images
