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
from tqdm import tqdm
from utils.lane import LaneEval


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


def evaluation(model, model_path, transforms, root_dir=None):
    utils.utils.load_entire_model(model, model_path)
    model.eval()

    pred_path = os.path.join(root_dir, 'test_tasks_0627.json')
    pred_save_path = os.path.join(root_dir, 'pred.json')
    gt_path = os.path.join(root_dir, 'test_label.json')

    json_pred = [json.loads(line) for line in open(pred_path).readlines()]
    postprocessor = lanenet_postprocess.LaneNetPostProcessor()
    logger.info("Start to process...")
    with paddle.no_grad():
        all_time_forward = []
        all_time_clustering = []
        for i, sample in enumerate(tqdm(json_pred)):
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

            inference_time = time_end - time_start
            logger.info("net inference time: {:.3f}ms ".format(
                inference_time * 1000))

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
            logger.info("cluster time: {:.3f}ms ".format(cluster_time * 1000))

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
                    all_time_forward.append(inference_time)
                    all_time_clustering.append(cluster_time)

        forward_avg = np.sum(all_time_forward[500:2000]) / 1500
        cluster_avg = np.sum(all_time_clustering[500:2000]) / 1500

        logger.info('The Inference time for one image is: {:.3f}ms'.format(
            forward_avg * 1000))
        logger.info('The Clustering time for one image is: {:.3f}ms'.format(
            cluster_avg * 1000))
        logger.info('The total time for one image is: {:.3f}ms'.format(
            (cluster_avg + forward_avg) * 1000))

        logger.info('The speed for Inference pass is: {:.3f}fps'.format(
            1 / forward_avg))
        logger.info('The speed for clustering pass is: {:.3f}fps'.format(
            1 / cluster_avg))

        with open(pred_save_path, 'w') as f:
            for res in json_pred:
                json.dump(res, f)
                f.write('\n')

        result = LaneEval.bench_one_submit(pred_save_path, gt_path)
        logger.info(result)


if __name__ == '__main__':
    args = parse_args()
    main(args)
