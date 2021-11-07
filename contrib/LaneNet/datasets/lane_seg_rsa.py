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
import cv2

import paddle
from paddleseg.cvlibs import manager
from transforms import LaneComposeRsa


@manager.DATASETS.add_component
class LaneSegRsa(paddle.io.Dataset):
    NUM_CLASSES = 2

    def __init__(self, dataset_root=None, transforms=None, mode='train'):
        self.dataset_root = dataset_root
        self.transforms = LaneComposeRsa(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.exist_list = []
        self.img_name_list = []
        self.full_img_path_list = []

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
                                     'seg_label/list/train_val_gt.txt')
        elif mode == 'val':
            file_path = os.path.join(self.dataset_root,
                                     'seg_label/list/test_gt.txt')
        else:
            file_path = os.path.join(self.dataset_root,
                                     'training/test_part.txt')

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split()
                self.img_name_list.append(items[0])
                self.full_img_path_list.append(self.dataset_root + items[0])
                if len(items) != 8:
                    if mode == 'train' or mode == 'val':
                        raise Exception(
                            "File list format incorrect! It should be"
                            " image_name label_name\\n")
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = None
                    instance_path = None
                else:
                    image_path = self.dataset_root + items[0]
                    label_path = self.dataset_root + items[1]
                    self.exist_list.append(
                        np.array([int(items[2]), int(items[3]),
                                  int(items[4]), int(items[5]),
                                  int(items[6]), int(items[7])
                                  ]))
                self.file_list.append([image_path, label_path])

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]

        if self.mode == 'test':
            im, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            return im, image_path

        elif self.mode == 'val':
            im, label = self.transforms(im=image_path, label=label_path)
            meta = {'full_img_path': self.full_img_path_list[idx],
                    'img_name': self.img_name_list[idx]}

            data = {'img': im, 'meta': meta}
            return im, label, data
        else:
            exist = self.exist_list[idx]
            im, label, = self.transforms(im=image_path, label=label_path, instancelabel=None)
            return im, label, exist

    def __len__(self):
        return len(self.file_list)

    def fix_gap(self, coordinate):
        if any(x > 0 for x in coordinate):
            start = [i for i, x in enumerate(coordinate) if x > 0][0]
            end = [i for i, x in reversed(list(enumerate(coordinate))) if x > 0][0]
            lane = coordinate[start:end + 1]
            if any(x < 0 for x in lane):
                gap_start = [i for i, x in enumerate(
                    lane[:-1]) if x > 0 and lane[i + 1] < 0]
                gap_end = [i + 1 for i,
                                     x in enumerate(lane[:-1]) if x < 0 and lane[i + 1] > 0]
                gap_id = [i for i, x in enumerate(lane) if x < 0]
                if len(gap_start) == 0 or len(gap_end) == 0:
                    return coordinate
                for id in gap_id:
                    for i in range(len(gap_start)):
                        if i >= len(gap_end):
                            return coordinate
                        if id > gap_start[i] and id < gap_end[i]:
                            gap_width = float(gap_end[i] - gap_start[i])
                            lane[id] = int((id - gap_start[i]) / gap_width * lane[gap_end[i]] + (
                                    gap_end[i] - id) / gap_width * lane[gap_start[i]])
                if not all(x > 0 for x in lane):
                    print("Gaps still exist!")
                coordinate[start:end + 1] = lane
        return coordinate

    def is_short(self, lane):
        start = [i for i, x in enumerate(lane) if x > 0]
        if not start:
            return 1
        else:
            return 0

    def get_lane(self, prob_map, y_px_gap, pts, thresh, resize_shape=None):
        """
        Arguments:
        ----------
        prob_map: prob map for single lane, np array size (h, w)
        resize_shape:  reshape size target, (H, W)

        Return:
        ----------
        coords: x coords bottom up every y_px_gap px, 0 for non-exist, in resized shape
        """
        if resize_shape is None:
            resize_shape = prob_map.shape
        h, w = prob_map.shape
        H, W = resize_shape
        H -= 160 # self.cfg.cut_height

        coords = np.zeros(pts)
        coords[:] = -1.0
        for i in range(pts):
            y = int((H - 10 - i * y_px_gap) * h / H)
            if y < 0:
                break
            line = prob_map[y, :]
            id = np.argmax(line)
            if line[id] > thresh:
                coords[i] = int(id / w * W)
        if (coords > 0).sum() < 2:
            coords = np.zeros(pts)
        self.fix_gap(coords)
        # print(coords.shape)

        return coords

    def probmap2lane(self, seg_pred, exist, resize_shape=(720, 1280), smooth=True, y_px_gap=10, pts=56, thresh=0.6):
        """
        Arguments:
        ----------
        seg_pred:      np.array size (5, h, w)
        resize_shape:  reshape size target, (H, W)
        exist:       list of existence, e.g. [0, 1, 1, 0]
        smooth:      whether to smooth the probability or not
        y_px_gap:    y pixel gap for sampling
        pts:     how many points for one lane
        thresh:  probability threshold

        Return:
        ----------
        coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
        """
        if resize_shape is None:
            resize_shape = seg_pred.shape[1:]  # seg_pred (5, h, w)
        _, h, w = seg_pred.shape
        H, W = resize_shape
        coordinates = []

        for i in range(7 - 1):
            prob_map = seg_pred[i + 1]
            if smooth:
                prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            coords = self.get_lane(prob_map, y_px_gap, pts, thresh, resize_shape)
            if self.is_short(coords):
                continue
            coordinates.append(
                [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
                 range(pts)])

        if len(coordinates) == 0:
            coords = np.zeros(pts)
            coordinates.append(
                [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
                 range(pts)])
        # print(coordinates)

        return coordinates
