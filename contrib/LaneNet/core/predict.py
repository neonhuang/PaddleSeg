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

import cv2
import numpy as np
import paddle

from paddleseg import utils
from . import infer
from paddleseg.utils import logger, progbar
from utils.utils import minmax_scale, to_png_fn, partition_list, makedirs
from utils import tusimple


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

    postprocessor = tusimple.Tusimple()
    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            im = cv2.imread(im_path).astype('float32')
            im = im[160:, :, :]
            gt_image = im
            im, _ = transforms(im)
            # For lane tasks, image size remains the post-processed size
            ori_shape = im.shape[1:]

            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            pred = infer.inference(
                model,
                im,
                ori_shape=ori_shape,
                transforms=transforms.transforms)

            postprocessor.predict(pred, im_path)

            progbar_pred.update(i + 1)
