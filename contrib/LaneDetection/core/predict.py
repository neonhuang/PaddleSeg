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
from paddleseg.utils import logger, progbar, visualize


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def predict(model,
            model_path,
            val_dataset,
            image_list,
            image_dir=None,
            save_dir='output'):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        val_dataset (paddle.io.Dataset): Used to read and process validation datasets.
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

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
    transforms = val_dataset.transforms
    cut_height = val_dataset.cut_height
    postprocessor = tusimple.Tusimple(num_classes=val_dataset.num_classes,
                                      cut_height=cut_height,
                                      save_dir=save_dir)
    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=None)
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            im = cv2.imread(im_path).astype('float32')
            ori_img = im
            im = im[cut_height:, :, :]
            cut_shape = im.shape[:2]
            im, _ = transforms(im)
            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            pred = infer.inference(
                model,
                im,
                ori_shape=cut_shape,
                transforms=transforms.transforms)

            # get lane points
            postprocessor.predict(pred[1], im_path)

            pred = paddle.squeeze(pred[0])
            pred = pred.numpy().astype('uint8')

            mask = np.zeros(shape=ori_img.shape[:2], dtype=np.uint8)
            mask[cut_height:, :] = pred
            pred = mask

            # get the saved name
            if image_dir is not None:
                im_file = im_path.replace(image_dir, '')
            else:
                im_file = os.path.basename(im_path)
            if im_file[0] == '/' or im_file[0] == '\\':
                im_file = im_file[1:]

            # save added image
            added_image = utils.visualize.visualize(
                im_path, pred, color_map, weight=0.6)
            added_image_path = os.path.join(added_saved_dir, im_file)
            mkdir(added_image_path)
            cv2.imwrite(added_image_path, added_image)

            # save pseudo color prediction
            pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
            pred_saved_path = os.path.join(
                pred_saved_dir,
                os.path.splitext(im_file)[0] + ".png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)

            progbar_pred.update(i + 1)
