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

import os
import math

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import cv2
import numpy as np
import paddle

from paddleseg import utils
from . import infer
from utils import lanenet_postprocess
from paddleseg.utils import logger, progbar


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def to_png_fn(fn, name=""):
    """
    Append png as filename postfix
    """
    directory, filename = os.path.split(fn)
    basename, ext = os.path.splitext(filename)

    return basename + name + ".png"


def minmax_scale(input_arr):
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output'):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()
    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            im = cv2.imread(im_path)
            gt_image = im
            im = im.astype('float32')
            im, _, _ = transforms(im)
            # For lane tasks, image size remains the post-processed size
            ori_shape = im.shape[1:]

            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            pred = infer.inference(
                model,
                im,
                ori_shape=ori_shape,
                transforms=transforms.transforms)

            segLogits = paddle.squeeze(pred[0])
            emLogits = paddle.squeeze(pred[1])

            binary_seg_image = segLogits.squeeze(-1)
            instance_seg_image = emLogits.transpose((1, 2, 0))

            binary_seg_image = binary_seg_image.numpy().astype('int64')
            instance_seg_image = instance_seg_image.numpy()

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image,
                instance_seg_result=instance_seg_image,
                source_image=gt_image)

            pred_binary_fn = os.path.join(
                save_dir, to_png_fn(im_path, name='_pred_binary'))
            pred_lane_fn = os.path.join(save_dir,
                                        to_png_fn(im_path, name='_pred_lane'))
            pred_instance_fn = os.path.join(
                save_dir, to_png_fn(im_path, name='_pred_instance'))
            dirname = os.path.dirname(pred_binary_fn)

            makedirs(dirname)
            mask_image = postprocess_result['mask_image']

            for j in range(4):
                instance_seg_image[:, :, j] = minmax_scale(
                    instance_seg_image[:, :, j])
            embedding_image = np.array(instance_seg_image).astype(np.uint8)

            plt.figure('mask_image')
            plt.imshow(mask_image[:, :, (2, 1, 0)])
            plt.figure('src_image')
            plt.imshow(gt_image[:, :, (2, 1, 0)])
            plt.figure('instance_image')
            plt.imshow(embedding_image[:, :, (2, 1, 0)])
            plt.figure('binary_image')
            plt.imshow(binary_seg_image * 255, cmap='gray')
            plt.show()

            cv2.imwrite(pred_binary_fn,
                        np.array(binary_seg_image * 255).astype(np.uint8))
            cv2.imwrite(pred_lane_fn, postprocess_result['source_image'])
            cv2.imwrite(pred_instance_fn, mask_image)
            print(pred_lane_fn, 'saved!')

            progbar_pred.update(i + 1)
