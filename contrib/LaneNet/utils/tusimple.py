import paddle.nn as nn
import json
import os
import cv2
import numpy as np

from .lane import LaneEval
from .utils import split_path, mkdir

# this code heavily base on
# https://github.com/ZJULearning/resa/blob/main/runner/evaluator/tusimple/tusimple.py
# https://github.com/ZJULearning/resa/blob/main/datasets/tusimple.py


class Tusimple:
    def __init__(self, num_classes=2,
                 cut_height=0,
                 thresh=0.6,
                 is_show=False,
                 test_gt_json=None,
                 save_dir='output/result'):
        super(Tusimple, self).__init__()
        self.num_classes = num_classes
        self.cut_height = cut_height
        self.dump_to_json = []
        self.thresh = thresh
        self.save_dir = save_dir
        self.is_show = False
        self.test_gt_json = test_gt_json
        self.color_map = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (125, 125, 0),
            (0, 125, 125),
            (125, 0, 125),
            (50, 100, 50),
            (100, 50, 100)
        ]

    def evaluate(self, output, im_path):
        seg_pred = output[0]
        seg_pred = nn.functional.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.numpy()
        self.generate_files(seg_pred, im_path)

    def predict(self, output, im_path):
        seg_pred = output[0]
        seg_pred = nn.functional.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.numpy()
        img_path = im_path
        lane_coords_list = self.prob2lines_tusimple(seg_pred)

        for batch in range(len(seg_pred)):
            lane_coords = lane_coords_list[batch]
            if True:
                img = cv2.imread(img_path)
                im_file = os.path.basename(im_path)
                saved_path = os.path.join(self.save_dir, 'points', im_file)
                self.draw(img, lane_coords, saved_path)

    def calculate_eval(self):
        output_file = os.path.join(self.save_dir, 'predict_test.json')
        with open(output_file, "w+") as f:
            for line in self.dump_to_json:
                print(line, end="\n", file=f)

        eval_rst, acc, fp, fn = LaneEval.bench_one_submit(output_file, self.test_gt_json)
        self.dump_to_json = []
        return acc, fp, fn, eval_rst

    def draw(self, img, coords, file_path=None):
        for i, coord in enumerate(coords):
            for x, y in coord:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 4, self.color_map[i % self.num_classes], 2)

        if file_path is not None:
            mkdir(file_path)
            cv2.imwrite(file_path, img)

    def generate_files(self, seg_pred, im_path):
        img_path = im_path
        lane_coords_list = self.prob2lines_tusimple(seg_pred)

        coord_path = os.path.join(self.save_dir, "coord_output")
        for batch in range(len(seg_pred)):
            lane_coords = lane_coords_list[batch]
            path_tree = split_path(img_path[batch])
            save_dir, save_name = path_tree[-3:-1], path_tree[-1]
            save_dir = os.path.join(coord_path, *save_dir)
            save_name = os.path.join(save_dir, save_name[:-3] + "lines.txt")
            mkdir(save_name)
            with open(save_name, "w") as f:
                for l in lane_coords:
                    for (x, y) in l:
                        print("{} {}".format(x, y), end=" ", file=f)
                    print(file=f)

            json_dict = {}
            json_dict['lanes'] = []
            json_dict['h_sample'] = []
            json_dict['raw_file'] = os.path.join(*path_tree[-4:])
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
            if self.is_show:
                img = cv2.imread(img_path[batch])
                new_img_name = '_'.join(
                    [x for x in split_path(img_path[batch])[-4:]])

                saved_path = os.path.join(self.save_dir, 'vis', new_img_name)
                self.draw(img, lane_coords, saved_path)

    def prob2lines_tusimple(self, seg_pred):
        lane_coords_list = []
        for batch in range(len(seg_pred)):
            seg = seg_pred[batch]
            lane_coords = self.probmap2lane(seg, thresh=self.thresh)
            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(
                    lane_coords[i], key=lambda pair: pair[1])
            lane_coords_list.append(lane_coords)
        return lane_coords_list

    def fix_gap(self, coordinate):
        if any(x > 0 for x in coordinate):
            start = [i for i, x in enumerate(coordinate) if x > 0][0]
            end = [i for i, x in reversed(list(enumerate(coordinate))) if x > 0][0]
            lane = coordinate[start:end + 1]
            # The line segment is not continuous
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
                            # line interpolation
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
        H -= self.cut_height

        coords = np.zeros(pts)
        coords[:] = -1.0
        for i in range(pts):
            y = int((H - 10 - i * y_px_gap) * h / H)
            if y < 0:
                break
            line = prob_map[y, :]
            id = np.argmax(line)
            val = line[id]
            if val > thresh:
                coords[i] = int(id / w * W)
        if (coords > 0).sum() < 2:
            coords = np.zeros(pts)
        self.fix_gap(coords)
        # print(coords.shape)

        return coords

    def probmap2lane(self, seg_pred, resize_shape=(720, 1280), smooth=True, y_px_gap=10, pts=56, thresh=0.6):
        """
        Arguments:
        ----------
        seg_pred:      np.array size (5, h, w)
        resize_shape:  reshape size target, (H, W)
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

        for i in range(self.num_classes - 1):
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