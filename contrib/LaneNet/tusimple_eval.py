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

import argparse
from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, config_check
from paddleseg.utils import logger, progbar
from datasets import *
from models import *

from core import infer
from utils import *
import json
import os.path as ops
import numpy as np
import cv2
import time
import os
from utils.lane import LaneEval
import math


def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # params of prediction
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for prediction',
        type=str,
        default=None)
    parser.add_argument(
        '--root',
        type=str,
        default=None,
        help='The root of the Tusimple test_set dataset')

    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    val_dataset = cfg.val_dataset
    if not val_dataset:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    transforms = val_dataset.transforms
    config_check(cfg, val_dataset=val_dataset)

    evaluation(
        model,
        model_path=args.model_path,
        transforms=transforms,
        root_dir=args.root)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def evaluation(model, model_path, transforms, root_dir=None):
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()

    pred_path = os.path.join(root_dir, 'test_tasks_0627.json')
    pred_save_path = os.path.join(root_dir, 'pred.json')
    gt_path = os.path.join(root_dir, 'test_label.json')

    json_pred = [json.loads(line) for line in open(pred_path).readlines()]
    if nranks > 1:
        json_lists = partition_list(json_pred, nranks)
    else:
        json_lists = [json_pred]

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()
    logger.info("Start to process...")
    progbar_pred = progbar.Progbar(target=len(json_lists[0]), verbose=1)
    with paddle.no_grad():
        all_time_inference = []
        all_time_clustering = []
        for i, sample in enumerate(json_lists[local_rank]):
            h_samples = sample['h_samples']
            raw_file = sample['raw_file']
            im_path = ops.join(root_dir, raw_file)

            im = cv2.imread(im_path)
            org_shape = im.shape
            im, _, _ = transforms(im)
            # For lane tasks, image size remains the post-processed size
            ori_shape = im.shape[1:]

            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            time_start = time.time()
            pred = infer.inference(
                model,
                im,
                ori_shape=ori_shape,
                transforms=transforms.transforms)
            time_end = time.time()

            segLogits = paddle.squeeze(pred[0])
            emLogits = paddle.squeeze(pred[1])

            binary_seg_image = segLogits.squeeze(-1)
            instance_seg_image = emLogits.transpose((1, 2, 0))

            binary_seg_image = binary_seg_image.numpy().astype('int64')
            instance_seg_image = instance_seg_image.numpy()

            clu_start = time.time()
            _, _, cluster_result = postprocessor.get_cluster_result(
                binary_seg_image, instance_seg_image)
            clu_end = time.time()

            cluster_time = clu_end - clu_start
            inference_time = time_end - time_start
            print("\n")
            logger.info("Inference time: {:.3f} cluster time: {:.3f} ".format(
                inference_time * 1000, cluster_time * 1000))

            cluster_result = cv2.resize(
                cluster_result,
                dsize=(org_shape[1], org_shape[0]),
                interpolation=cv2.INTER_NEAREST)
            elements = np.unique(cluster_result)
            for line_idx in elements:
                if line_idx == 0:
                    continue
                else:
                    mask = (cluster_result == line_idx)
                    select_mask = mask[h_samples]
                    row_result = []
                    for row in range(len(h_samples)):  # 竖直方向
                        col_indexes = np.nonzero(select_mask[row])[0]
                        if len(col_indexes) == 0:
                            row_result.append(-2)
                        else:
                            minE = col_indexes.min()
                            maxE = col_indexes.max()
                            # 水平方向
                            ret = int(minE + (maxE - minE) / 2)
                            row_result.append(ret)
                    json_pred[i]['lanes'].append(row_result)
                    json_pred[i]['run_time'] = inference_time
                    all_time_inference.append(inference_time)
                    all_time_clustering.append(cluster_time)
            progbar_pred.update(i + 1)

        inference_avg = np.sum(all_time_inference[500:2500]) / 2000
        cluster_avg = np.sum(all_time_clustering[500:2500]) / 2000

        logger.info(
            "inference_avg: {:.3f} cluster_avg: {:.3f} total time: {:.3f} ".
            format(inference_avg * 1000, cluster_avg * 1000,
                   (cluster_avg + inference_avg) * 1000))

        logger.info("Inference speed: {:.3f} clustering speed: {:.3f} ".format(
            1 / inference_avg, 1 / cluster_avg))

        with open(pred_save_path, 'w') as f:
            for res in json_pred:
                json.dump(res, f)
                f.write('\n')

        result = LaneEval.bench_one_submit(pred_save_path, gt_path)
        logger.info(result)


if __name__ == '__main__':
    args = parse_args()
    main(args)
