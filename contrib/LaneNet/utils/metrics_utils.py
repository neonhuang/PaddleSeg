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

EPS = 1e-8


def compute_metric_lane(pred, label):
    pred = paddle.transpose(pred, [0, 2, 3, 1])
    label = paddle.unsqueeze(label, axis=1)
    label = paddle.transpose(label, [0, 2, 3, 1])

    # fix a bug
    shape = pred.shape
    pred = paddle.reshape(pred, [-1])
    unique_labels, counts = paddle.unique(pred, return_counts=True)
    if counts.shape[0] == 1:
        acc = paddle.to_tensor(0)
        fp = paddle.to_tensor(1)
        fn = paddle.to_tensor(1)
        return acc, fp, fn

    pred = paddle.reshape(pred, shape)

    P1_and_G1 = paddle.sum(paddle.gather_nd(label, paddle.nonzero(pred)))
    G1 = paddle.cast(
        paddle.shape(paddle.gather_nd(label, paddle.nonzero(label)))[0],
        'int64')

    accuracy = paddle.cast(P1_and_G1, 'float32') / (G1 + EPS)

    P1 = paddle.cast(
        paddle.shape(paddle.gather_nd(pred, paddle.nonzero(pred)))[0], 'int64')

    false_pred = P1 - P1_and_G1
    fp = paddle.cast(false_pred, 'float32') / (P1 + EPS)

    mis_pred = G1 - P1_and_G1
    fn = paddle.cast(mis_pred, 'float32') / (G1 + EPS)

    accuracy.stop_gradient = True
    fp.stop_gradient = True
    fn.stop_gradient = True
    return accuracy, fp, fn


def compute_metric(pred, label):
    pred = paddle.transpose(pred, [0, 2, 3, 1])
    label = paddle.unsqueeze(label, axis=1)
    label = paddle.transpose(label, [0, 2, 3, 1])

    # fix a bug
    shape = pred.shape
    pred = paddle.reshape(pred, [-1])
    unique_labels, counts = paddle.unique(pred, return_counts=True)
    if counts.shape[0] == 1:
        acc = paddle.to_tensor(0)
        fp = paddle.to_tensor(1)
        fn = paddle.to_tensor(1)
        return acc, fp, fn

    pred = paddle.reshape(pred, shape)

    idx = paddle.nonzero(pred)
    pix_cls_ret = paddle.gather_nd(label, idx)
    correct_num = paddle.sum(paddle.cast(pix_cls_ret, 'float32'))

    gt_num = paddle.cast(
        paddle.shape(paddle.gather_nd(label, paddle.nonzero(label)))[0],
        'int64')
    pred_num = paddle.cast(
        paddle.shape(paddle.gather_nd(pred, idx))[0], 'int64')
    accuracy = correct_num / (gt_num + EPS)

    false_pred = pred_num - correct_num
    fp = paddle.cast(false_pred, 'float32') / (
        paddle.cast(paddle.shape(pix_cls_ret)[0], 'int64') + EPS)

    label_cls_ret = paddle.gather_nd(label, paddle.nonzero(label))
    mis_pred = paddle.cast(paddle.shape(label_cls_ret)[0],
                           'int64') - correct_num
    fn = paddle.cast(mis_pred, 'float32') / (
        paddle.cast(paddle.shape(label_cls_ret)[0], 'int64') + EPS)
    accuracy.stop_gradient = True
    fp.stop_gradient = True
    fn.stop_gradient = True
    return accuracy, fp, fn
