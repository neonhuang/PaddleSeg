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

import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class LaneCrossEntropyLoss(nn.Layer):
    def __init__(self, ignore_index=255, data_format='NCHW'):
        super(LaneCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.EPS = 1e-8

    def get_dynamic_weight(self, label):
        label = paddle.reshape(label, [-1])
        paddle.unique(label)
        unique_labels, counts = paddle.unique(label, return_counts=True)

        # fix some case
        if counts.shape[0] == 3:
            counts = counts[paddle.nonzero(counts * (unique_labels != 255))]
        if counts.shape[0] == 1:
            counts = paddle.to_tensor(np.append(counts.numpy(), 0))

        counts = paddle.cast(counts, 'float32')
        weight = 1.0 / paddle.log((counts / paddle.sum(counts) + 1.02))

        return weight

    def softmax_with_loss(self, logit, label, num_classes=2, weight=None):
        ignore_mask = (label != 255).astype('int64')  # [4, 1, 256, 512]
        ignore_mask = paddle.cast(ignore_mask, 'float32')
        label = paddle.minimum(
            label, paddle.assign(np.array([num_classes - 1], dtype=np.int64)))
        logit = paddle.transpose(logit, [0, 2, 3, 1])
        logit = paddle.reshape(logit, [-1, num_classes])
        label = paddle.reshape(label, [-1, 1])
        label = paddle.cast(label, 'int64')
        ignore_mask = paddle.reshape(ignore_mask, [-1, 1])
        if weight is None:
            loss, probs = F.softmax_with_cross_entropy(
                logit,
                label,
                ignore_index=self.ignore_index,
                return_softmax=True)
        else:
            label = paddle.squeeze(label, axis=[-1])
            label_one_hot = F.one_hot(label, num_classes=num_classes)
            if isinstance(weight, list):
                assert len(
                    weight
                ) == num_classes, "weight length must equal num of classes"
                weight = paddle.assign(np.array([weight], dtype='float32'))
            elif isinstance(weight, str):
                assert weight.lower(
                ) == 'dynamic', 'if weight is string, must be dynamic!'
                tmp = []
                total_num = paddle.cast(paddle.shape(label)[0], 'float32')
                for i in range(num_classes):
                    cls_pixel_num = paddle.sum(label_one_hot[:, i])
                    ratio = total_num / (cls_pixel_num + 1)
                    tmp.append(ratio)
                weight = paddle.concat(tmp)
                weight = weight / paddle.sum(weight) * num_classes
            elif isinstance(weight, paddle.Tensor):
                pass
            else:
                raise ValueError(
                    'Expect weight is a list, string or Variable, but receive {}'
                    .format(type(weight)))

            weight = paddle.reshape(weight, [1, num_classes])
            weighted_label_one_hot = label_one_hot * weight
            probs = F.softmax(logit)
            loss = F.cross_entropy(
                probs,
                weighted_label_one_hot,
                soft_label=True,
                ignore_index=-100)
            weighted_label_one_hot.stop_gradient = True

        loss = loss * ignore_mask
        avg_loss = paddle.mean(loss) / (paddle.mean(ignore_mask) + self.EPS)

        label.stop_gradient = True
        ignore_mask.stop_gradient = True
        return avg_loss

    def forward(self, logit, label, semantic_weights=None):
        weight = self.get_dynamic_weight(label)
        seg_loss = self.softmax_with_loss(logit, label, 2, weight)

        return seg_loss
