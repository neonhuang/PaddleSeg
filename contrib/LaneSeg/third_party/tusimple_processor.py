#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# this code heavily base on
# https://github.com/ZJULearning/resa/blob/main/runner/evaluator/tusimple/tusimple.py
# https://github.com/ZJULearning/resa/blob/main/datasets/tusimple.py

import paddle.nn as nn
import json
import os
import cv2
import numpy as np

from .lane import LaneEval


def split_path(path):
    """split path tree into list"""
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            folders.insert(0, folder)
        else:
            if path != "":
                folders.insert(0, path)
            break
    return folders


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


class TusimpleProcessor:
    def __init__(self,
                 num_classes=2,
                 cut_height=0,
                 thresh=0.6,
                 is_view=False,
                 test_gt_json=None,
                 save_dir='output/'):
        super(TusimpleProcessor, self).__init__()
        self.num_classes = num_classes
        self.cut_height = cut_height
        self.dump_to_json = []
        self.thresh = thresh

        self.save_dir = save_dir
        self.is_view = is_view
        self.test_gt_json = test_gt_json
        self.smooth = True
        self.y_pixel_gap = 10
        self.points_nums = 56
        self.src_height = 720
        self.src_width = 1280
        self.color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                          (255, 0, 255), (0, 255, 125), (50, 100, 50),
                          (100, 50, 100)]

    def get_lane_coords(self, seg_planes):
        lane_coords_list = []
        for batch in range(len(seg_planes)):
            seg = seg_planes[batch]
            lane_coords = self.heatmap2lane(seg)
            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(
                    lane_coords[i], key=lambda pair: pair[1])
            lane_coords_list.append(lane_coords)
        return lane_coords_list

    def heatmap2lane(self, seg_planes):
        coordinates = []
        for i in range(self.num_classes - 1):
            prob_map = seg_planes[i + 1]
            if self.smooth:
                prob_map = cv2.blur(
                    prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            coords = self.get_lane(prob_map)
            sum = np.sum(coords)
            if sum < 1e-8:
                continue
            self.add_coords(coordinates, coords)

        if len(coordinates) == 0:
            coords = np.zeros(self.points_nums)
            self.add_coords(coordinates, coords)
        return coordinates

    def get_lane(self, prob_map):
        dst_height = self.src_height - self.cut_height
        pointCount = 0
        coords = np.zeros(self.points_nums)
        for i in range(self.points_nums):
            y = int((dst_height - 10 - i * self.y_pixel_gap) * prob_map.shape[0]
                    / dst_height)
            if y < 0:
                break
            line = prob_map[y, :]
            id = np.argmax(line)
            val = line[id]
            if val > self.thresh:
                coords[i] = int(id / prob_map.shape[1] * self.src_width)
                pointCount = pointCount + 1
        if pointCount < 2:
            coords = np.zeros(self.points_nums)
        self.process_gap(coords)
        return coords

    def process_gap(self, coordinate):
        if any(x > 0 for x in coordinate):
            start = [i for i, x in enumerate(coordinate) if x > 0][0]
            end = [
                i for i, x in reversed(list(enumerate(coordinate))) if x > 0
            ][0]
            lane = coordinate[start:end + 1]
            # The line segment is not continuous
            if any(x < 0 for x in lane):
                gap_start = [
                    i for i, x in enumerate(lane[:-1])
                    if x > 0 and lane[i + 1] < 0
                ]
                gap_end = [
                    i + 1 for i, x in enumerate(lane[:-1])
                    if x < 0 and lane[i + 1] > 0
                ]
                gap_id = [i for i, x in enumerate(lane) if x < 0]
                if len(gap_start) == 0 or len(gap_end) == 0:
                    return coordinate
                for id in gap_id:
                    for i in range(len(gap_start)):
                        if i >= len(gap_end):
                            return coordinate
                        if id > gap_start[i] and id < gap_end[i]:
                            gap_width = float(gap_end[i] - gap_start[i])
                            # line interpolation
                            lane[id] = int((id - gap_start[i]) / gap_width *
                                           lane[gap_end[i]] +
                                           (gap_end[i] - id) / gap_width *
                                           lane[gap_start[i]])
                if not all(x > 0 for x in lane):
                    print("Gaps still exist!")
                coordinate[start:end + 1] = lane
        return coordinate

    def fix_outliers(self, coords):
        data = [x for i, x in enumerate(coords) if x > 0]
        index = [i for i, x in enumerate(coords) if x > 0]
        if len(data) == 0:
            return coords
        diff = []
        is_outlier = False
        n = 1
        x_gap = abs((data[-1] - data[0]) / (1.0 * (len(data) - 1)))
        for idx, dt in enumerate(data):
            if is_outlier == False:
                t = idx - 1
                n = 1
            if idx == 0:
                diff.append(0)
            else:
                diff.append(abs(data[idx] - data[t]))
                if abs(data[idx] - data[t]) > n * (x_gap * 1.5):
                    n = n + 1
                    is_outlier = True
                    ind = index[idx]
                    coords[ind] = -1
                else:
                    is_outlier = False

    def add_coords(self, coordinates, coords):
        sub_lanes = []
        for j in range(self.points_nums):
            if coords[j] > 0:
                val = [coords[j], self.src_height - 10 - j * self.y_pixel_gap]
            else:
                val = [-2, self.src_height - 10 - j * self.y_pixel_gap]
            sub_lanes.append(val)
        coordinates.append(sub_lanes)

    def evaluate(self, output, im_path):
        seg_pred = output[0]
        seg_pred = nn.functional.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.numpy()
        self.generate_json_data(seg_pred, im_path)

    def generate_json_data(self, seg_pred, im_path):
        img_path = im_path
        lane_coords_list = self.get_lane_coords(seg_pred)

        for batch in range(len(seg_pred)):
            lane_coords = lane_coords_list[batch]
            json_dict = {}
            json_dict['lanes'] = []
            json_dict['h_sample'] = []
            json_dict['raw_file'] = os.path.join(
                *split_path(img_path[batch])[-4:])
            json_dict['run_time'] = 0
            for l in lane_coords:
                if len(l) == 0:
                    continue
                json_dict['lanes'].append([])
                for (x, y) in l:
                    json_dict['lanes'][-1].append(int(x))
            for (x, y) in lane_coords[0]:
                json_dict['h_sample'].append(y)
            self.dump_to_json.append(json.dumps(json_dict))
            if self.is_view:
                img = cv2.imread(img_path[batch])
                new_img_name = '_'.join(
                    [x for x in split_path(img_path[batch])[-4:]])

                saved_path = os.path.join(self.save_dir, 'visual', new_img_name)
                self.draw(img, lane_coords, saved_path)

    def predict(self, output, im_path):
        seg_pred = output[0]
        seg_pred = nn.functional.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.numpy()
        img_path = im_path
        lane_coords_list = self.get_lane_coords(seg_pred)

        for batch in range(len(seg_pred)):
            lane_coords = lane_coords_list[batch]
            if True:
                img = cv2.imread(img_path)
                im_file = os.path.basename(im_path)
                saved_path = os.path.join(self.save_dir, 'points', im_file)
                self.draw(img, lane_coords, saved_path)

    def calculate_eval(self):
        output_file = os.path.join(self.save_dir, 'predict_test.json')
        if output_file is not None:
            mkdir(output_file)
        with open(output_file, "w+") as f:
            for line in self.dump_to_json:
                print(line, end="\n", file=f)

        eval_rst, acc, fp, fn = LaneEval.bench_one_submit(
            output_file, self.test_gt_json)
        self.dump_to_json = []
        return acc, fp, fn, eval_rst

    def draw(self, img, coords, file_path=None):
        for i, coord in enumerate(coords):
            for x, y in coord:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 4, self.color_map[i % self.num_classes],
                           2)

        if file_path is not None:
            mkdir(file_path)
            cv2.imwrite(file_path, img)
