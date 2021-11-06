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
class LaneCompose:
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
            im = im[160:, :, :]
        if isinstance(label, str):
            label = cv2.imread(label, cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
                label = label[160:, :]
            # label = np.asarray(Image.open(label))
            # label = np.array(label / 255.0, dtype=np.uint8)
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for op in self.transforms:
            outputs = op(im, label)
            im = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]

        im = np.transpose(im, (2, 0, 1)).astype('float32')
        if label is not None:
            label = label.astype('int64')
        return im, label


@manager.TRANSFORMS.add_component
class LaneRandomRotation:
    def __init__(self,
                 max_rotation=10,
                 im_padding_value=(0, 0, 0),
                 label_padding_value=0):
        self.max_rotation = max_rotation
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, label=None):
        v = random.random()
        if v < 0.5:
            (h, w) = im.shape[:2]
            do_rotation = np.random.uniform(-self.max_rotation, self.max_rotation)
            pc = (w // 2, h // 2)
            r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
            dsize = (w, h)
            im = cv2.warpAffine(
                im,
                r,
                dsize=dsize,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.im_padding_value)
            if label is not None:
                label = cv2.warpAffine(
                    label,
                    r,
                    dsize=dsize,
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=self.label_padding_value)

        if label is None:
            return (im, )
        else:
            return (im, label)


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
            return (im, )
        else:
            return (im, label)
