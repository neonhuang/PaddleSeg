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
from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class DiscriminativeLoss(nn.Layer):
    def __init__(self, ignore_index=255, data_format='NCHW'):
        super(DiscriminativeLoss, self).__init__()

    def unsorted_segment_sum(self, data, segment_ids, unique_labels,
                             feature_dims):
        unique_labels_shape = paddle.shape(unique_labels)
        zeros = paddle.zeros(
            shape=[unique_labels_shape[0], feature_dims], dtype='float32')
        segment_ids = paddle.unsqueeze(segment_ids, axis=[1])
        segment_ids.stop_gradient = True
        segment_sum = paddle.scatter_nd_add(zeros, segment_ids, data)
        zeros.stop_gradient = True

        return segment_sum

    def norm(self, x, axis=-1):
        distance = paddle.sum(paddle.abs(x), axis=axis, keepdim=True)
        return distance

    def discriminative_loss_single(self, prediction, correct_label, feature_dim,
                                   label_shape, delta_v, delta_d, param_var,
                                   param_dist, param_reg):
        correct_label = paddle.reshape(correct_label,
                                       [label_shape[1] * label_shape[0]])
        prediction = paddle.transpose(prediction, [1, 2, 0])
        reshaped_pred = paddle.reshape(
            prediction, [label_shape[1] * label_shape[0], feature_dim])

        unique_labels, unique_id, counts = paddle.unique(
            correct_label, return_inverse=True, return_counts=True)

        correct_label.stop_gradient = True
        counts = paddle.cast(counts, 'float32')
        num_instances = paddle.shape(unique_labels)

        segmented_sum = self.unsorted_segment_sum(
            reshaped_pred, unique_id, unique_labels, feature_dims=feature_dim)

        counts_rsp = paddle.reshape(counts, (-1, 1))
        mu = paddle.divide(segmented_sum, counts_rsp)
        counts_rsp.stop_gradient = True
        mu_expand = paddle.gather(mu, unique_id)
        tmp = paddle.subtract(mu_expand, reshaped_pred)

        distance = self.norm(tmp)
        distance = distance - delta_v

        distance_pos = paddle.greater_equal(distance,
                                            paddle.zeros_like(distance))
        distance_pos = paddle.cast(distance_pos, 'float32')
        distance = distance * distance_pos

        distance = paddle.square(distance)

        l_var = self.unsorted_segment_sum(
            distance, unique_id, unique_labels, feature_dims=1)
        l_var = paddle.divide(l_var, counts_rsp)
        l_var = paddle.sum(l_var)
        l_var = l_var / paddle.cast(num_instances * (num_instances - 1),
                                    'float32')

        mu_interleaved_rep = paddle.reshape(
            paddle.expand(
                paddle.flatten(mu),
                shape=[num_instances,
                       paddle.flatten(mu).shape[0]]),
            shape=[mu.shape[0] * num_instances, mu.shape[1]])

        zeros = paddle.zeros(
            shape=[mu.shape[0], mu.shape[1] * num_instances], dtype='float32')
        for i in range(0, mu.shape[1] * num_instances, mu.shape[1]):
            zeros[:, i:mu.shape[1] + i] = mu
        mu_band_rep = zeros
        mu_band_rep = paddle.reshape(
            mu_band_rep, (num_instances * num_instances, feature_dim))

        mu_diff = paddle.subtract(mu_band_rep, mu_interleaved_rep)

        intermediate_tensor = paddle.sum(paddle.abs(mu_diff), axis=1)
        intermediate_tensor.stop_gradient = True
        zero_vector = paddle.zeros([1], 'float32')
        bool_mask = paddle.not_equal(intermediate_tensor, zero_vector)

        temp = paddle.nonzero(bool_mask.astype('int64'))
        mu_diff_bool = paddle.gather(mu_diff, temp)

        mu_norm = self.norm(mu_diff_bool)
        mu_norm = 2. * delta_d - mu_norm
        mu_norm_pos = paddle.greater_equal(mu_norm, paddle.zeros_like(mu_norm))
        mu_norm_pos = paddle.cast(mu_norm_pos, 'float32')
        mu_norm = mu_norm * mu_norm_pos
        mu_norm_pos.stop_gradient = True

        mu_norm = paddle.square(mu_norm)

        l_dist = paddle.mean(mu_norm)

        l_reg = paddle.mean(self.norm(mu, axis=1))

        l_var = param_var * l_var
        l_dist = param_dist * l_dist
        l_reg = param_reg * l_reg
        loss = l_var + l_dist + l_reg
        return loss, l_var, l_dist, l_reg

    def discriminative_loss(self, prediction, correct_label):
        feature_dim = 4
        image_shape = prediction.shape[2:]
        delta_v = 0.5
        delta_d = 3.0
        param_var = 1.0
        param_dist = 1.0
        param_reg = 0.001
        batch_size = prediction.shape[0]
        output_ta_loss = 0.
        output_ta_var = 0.
        output_ta_dist = 0.
        output_ta_reg = 0.
        for i in range(batch_size):
            pred = prediction[i]
            corr = correct_label[i]
            disc_loss_single, l_var_single, l_dist_single, l_reg_single = self.discriminative_loss_single(
                pred, corr, feature_dim, image_shape, delta_v, delta_d,
                param_var, param_dist, param_reg)
            output_ta_loss += disc_loss_single
            output_ta_var += l_var_single
            output_ta_dist += l_dist_single
            output_ta_reg += l_reg_single

        disc_loss = output_ta_loss / batch_size
        l_var = output_ta_var / batch_size
        l_dist = output_ta_dist / batch_size
        l_reg = output_ta_reg / batch_size

        disc_loss = disc_loss + 0.00001 * l_reg
        return disc_loss

    def forward(self, logit, label, semantic_weights=None):
        disc_loss = self.discriminative_loss(logit, label)
        return disc_loss
