import paddle.nn.functional as F
import json
import os
import os.path as osp
import cv2
import numpy as np

from .lane import LaneEval
from .utils import split_path


class Tusimple:
    def __init__(self, num_classes=7, thresh=0.6):
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
        self.thresh = thresh
        self.num_classes = num_classes

    def evaluate_pred(self, seg_pred, exist_pred, im_path):
        img_path = im_path
        lane_coords_list = self.prob2lines_tusimple(seg_pred, exist_pred)

        for b in range(len(seg_pred)):
            lane_coords = lane_coords_list[b]
            path_tree = split_path(img_path[b])
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
            if True:
                img = cv2.imread(img_path[b])
                new_img_name = '_'.join(
                    [x for x in split_path(img_path[b])[-4:]])
                save_dir = os.path.join(self.view_dir, new_img_name)
                self.view(img, lane_coords, save_dir)

    def prob2lines_tusimple(self, seg_pred, exist_pred):
        lane_coords_list = []
        for b in range(len(seg_pred)):
            seg = seg_pred[b]
            exist = [1 if exist_pred[b, i] >
                          0.5 else 0 for i in range(self.num_classes - 1)]
            lane_coords = self.probmap2lane(seg, exist, thresh=self.thresh)
            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(
                    lane_coords[i], key=lambda pair: pair[1])
            lane_coords_list.append(lane_coords)
        return lane_coords_list

    def evaluate(self, output, im_path):
        seg_pred, exist_pred = output[0], output[1]
        seg_pred = F.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.numpy()
        exist_pred = exist_pred.numpy()
        self.evaluate_pred(seg_pred, exist_pred, im_path)

    def predict(self, output, im_path):
        seg_pred, exist_pred = output[0], output[1]
        seg_pred = F.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.numpy()
        exist_pred = exist_pred.numpy()
        img_name = im_path
        img_path = im_path
        lane_coords_list = self.prob2lines_tusimple(seg_pred, exist_pred)

        for b in range(len(seg_pred)):
            lane_coords = lane_coords_list[b]
            if True:
                img = cv2.imread(img_path)
                new_img_name = img_name.replace('/', '_')
                save_dir = os.path.join(self.view_dir, new_img_name)
                self.view(img, lane_coords, save_dir)


    def summarize(self):
        best_acc = 0
        output_file = os.path.join(self.out_path, 'predict_test.json')
        with open(output_file, "w+") as f:
            for line in self.dump_to_json:
                print(line, end="\n", file=f)

        eval_result, acc, fp, fn = LaneEval.bench_one_submit(output_file,
                                                             "/home/work/resa/data/tusimple/test_label.json")

        self.dump_to_json = []
        best_acc = max(acc, best_acc)
        return best_acc, acc, fp, fn, eval_result

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
        H -= 160  # self.cfg.cut_height

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
