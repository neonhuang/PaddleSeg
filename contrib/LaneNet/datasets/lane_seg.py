# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np

import paddle
from paddleseg.cvlibs import manager
from transforms import LaneCompose


@manager.DATASETS.add_component
class LaneSeg(paddle.io.Dataset):
    NUM_CLASSES = 2

    def __init__(self, dataset_root=None, transforms=None, mode='train'):
        self.dataset_root = dataset_root
        self.transforms = LaneCompose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "`mode` should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if self.dataset_root is None:
            raise ValueError("`dataset_root` is necessary, but it is None.")

        if mode == 'train':
            file_path = os.path.join(self.dataset_root,
                                     'training/train_part.txt')
        elif mode == 'val':
            file_path = os.path.join(self.dataset_root, 'training/val_part.txt')
        else:
            file_path = os.path.join(self.dataset_root,
                                     'training/test_part.txt')

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 3:
                    if mode == 'train' or mode == 'val':
                        raise Exception(
                            "File list format incorrect! It should be"
                            " image_name label_name\\n")
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = None
                    instance_path = None
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = os.path.join(self.dataset_root, items[1])
                    instance_path = os.path.join(self.dataset_root, items[2])
                self.file_list.append([image_path, label_path, instance_path])

    def __getitem__(self, idx):
        image_path, label_path, instance_path = self.file_list[idx]

        if self.mode == 'test':
            im, _, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            return im, image_path

        elif self.mode == 'val':
            im, label, _ = self.transforms(im=image_path, label=label_path)
            return im, label
        else:
            im, label, instance = self.transforms(
                im=image_path, label=label_path, instancelabel=instance_path)
            return im, label, instance

    def __len__(self):
        return len(self.file_list)
