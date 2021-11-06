import paddle.nn.functional as F
import json
import os
import cv2
import os.path as osp
import numpy as np

from .lane import LaneEval
from .utils import split_path
from .lane_utils import prob2lines_tusimple

# this code heavily base on:: https://github.com/ZJULearning/resa/blob/main/datasets/tusimple.py


class Tusimple:
    def __init__(self, thresh=0.6):
        super(Tusimple, self).__init__()
        exp_dir = "output"
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        self.out_path = os.path.join(exp_dir, "coord_output")
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        self.view_dir = os.path.join(exp_dir, 'vis')
        if not os.path.exists(self.view_dir):
            os.mkdir(self.view_dir)
        self.dump_to_json = []
        self.thresh = thresh  # 0.6

    def evaluate_pred(self, seg_pred, batch):
        img_path = batch['meta']['full_img_path']
        lane_coords_list = prob2lines_tusimple(seg_pred, self.thresh)
        for b in range(len(seg_pred)):
            lane_coords = lane_coords_list[b]
            self.generateJson(img_path[b], lane_coords)
            self.draw(img_path[b], lane_coords)

    def predict(self, output, batch):
        seg_pred = output[0]
        seg_pred = F.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.detach().cpu().numpy()
        img_path = batch
        lane_coords_list = prob2lines_tusimple(seg_pred, self.thresh)
        for b in range(len(seg_pred)):
            lane_coords = lane_coords_list[b]
            self.draw(img_path, lane_coords)

    def draw(self, img_path, lane_coords):
        img = cv2.imread(img_path)
        new_img_name = '_'.join(
            [x for x in split_path(img_path)[-4:]])
        save_dir = os.path.join(self.view_dir, new_img_name)
        self.view(img, lane_coords, save_dir)

    def generateJson(self, img_path, lane_coords):
        path_tree = split_path(img_path)
        save_dir, save_name = path_tree[-3:-1], path_tree[-1]
        save_dir = os.path.join(self.out_path, *save_dir)
        save_name = save_name[:-3] + "lines.txt"
        save_name = os.path.join(save_dir, save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
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

    def evaluate(self, output, batch):
        seg_pred = output[0]
        seg_pred = F.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.detach().cpu().numpy()
        self.evaluate_pred(seg_pred, batch)

    def summarize(self):
        output_file = os.path.join(self.out_path, 'predict_test.json')
        with open(output_file, "w+") as f:
            for line in self.dump_to_json:
                print(line, end="\n", file=f)

        eval_result, acc, fp, fn = LaneEval.bench_one_submit(
            output_file,
            "/Users/huangshenghui/PP/resa-main/data/tusimple/test_label.json")

        self.dump_to_json = []
        return acc, fp, fn, eval_result

    def view(self, img, coords, file_path=None):
        for coord in coords:
            for x, y in coord:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 4, (255, 0, 0), 2)

        if file_path is not None:
            if not os.path.exists(osp.dirname(file_path)):
                os.makedirs(osp.dirname(file_path))
            cv2.imwrite(file_path, img)