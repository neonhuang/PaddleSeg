# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
from paddle.nn import Conv2D
from paddle.nn import MaxPool2D

from paddleseg.cvlibs import manager
from paddleseg.models import layers

# Bottleneck type
REGULAR = 1
DOWNSAMPLING = 2
UPSAMPLING = 3
DILATED = 4
ASYMMETRIC = 5


class bottleneck(nn.Layer):
    def __init__(self,
                 output_depth,
                 filter_size,
                 regularizer_prob,
                 projection_ratio=4,
                 type=REGULAR,
                 seed=0,
                 output_shape=None,
                 dilation_rate=None,
                 decoder=False,
                 name_scope='bottleneck'):
        super(bottleneck, self).__init__()
        self.output_depth = output_depth
        self.filter_size = filter_size
        self.regularizer_prob = regularizer_prob
        self.projection_ratio = projection_ratio
        self.type = type
        self.seed = seed
        self.output_shape = output_shape
        self.dilation_rate = dilation_rate
        self.decoder = decoder
        self.name_scope = name_scope

        self._relu = layers.Activation("relu")
        self._prelu = layers.Activation("prelu")

    def prelu(self, x, decoder=False):
        if decoder:
            return self._relu(x)
        return self._prelu(x)

    def forward(self, inputs):
        # Calculate the depth reduction based on the projection ratio used in 1x1 convolution.
        reduced_depth = int(inputs.shape[1] / self.projection_ratio)
        type = self.type

        # DOWNSAMPLING BOTTLENECK
        if type == DOWNSAMPLING:
            # =============MAIN BRANCH=============
            # Just perform a max pooling
            inputs_shape = inputs.shape
            net_main = Conv2D(
                in_channels=inputs_shape[1],  # input_channels,
                out_channels=inputs_shape[1],
                kernel_size=3,
                stride=2,
                padding="SAME",
                # weight_attr=ParamAttr(name=self.name_scope + "/down_sample/" + "main_max_pool"),
                bias_attr=False)(inputs)

            # First get the difference in depth to pad, then pad with zeros only on the last dimension.
            depth_to_pad = abs(inputs_shape[1] - self.output_depth)
            paddings = [0, 0, 0, depth_to_pad, 0, 0, 0, 0]
            net_main = nn.functional.pad(net_main, paddings)

            # block1
            net = layers.ConvBN(
                inputs.shape[1],
                reduced_depth,
                kernel_size=2,
                stride=2,
                padding="same",
                # weight_attr=ParamAttr(name=self.name_scope + "/down_sample/" + "block1")
            )(inputs)
            net = self.prelu(net, decoder=self.decoder)

            # block2
            net = layers.ConvBN(
                net.shape[1],
                reduced_depth,
                kernel_size=self.filter_size,
                padding="same",
                # weight_attr=ParamAttr(name=self.name_scope + "/down_sample/" + "block2")
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # block3
            net = layers.ConvBN(
                net.shape[1],
                self.output_depth,
                kernel_size=1,
                padding="same",
                # weight_attr=ParamAttr(name=self.name_scope + "/down_sample/" + "block3")
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # Regularizer
            net = nn.Dropout(self.regularizer_prob)(net)

            # Finally, combine the two branches together via an element-wise addition
            net = net + net_main
            net = self.prelu(net, decoder=self.decoder)
            return net, inputs_shape

        # DILATION CONVOLUTION BOTTLENECK
        # Everything is the same as a regular bottleneck except for the dilation rate argument
        elif type == DILATED:
            # Check if dilation rate is given
            if not self.dilation_rate:
                raise ValueError('Dilation rate is not given.')

            # Save the main branch for addition later
            # dilated
            net_main = inputs

            # First projection with 1x1 kernel (dimensionality reduction)
            # block1
            net = layers.ConvBN(
                inputs.shape[1],
                reduced_depth,
                kernel_size=1,
                padding="same",
                # weight_attr=ParamAttr(name=self.name_scope + "/dilated/" + "block1")
            )(inputs)
            net = self.prelu(net, decoder=self.decoder)

            # Second conv block --- apply dilated convolution here
            # block2
            net = layers.ConvBN(
                net.shape[1],
                reduced_depth,
                kernel_size=self.filter_size,
                padding="same",
                dilation=self.dilation_rate,
                # weight_attr=ParamAttr(name=self.name_scope + "/dilated/" + "block2")
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # Final projection with 1x1 kernel (Expansion)
            # block3
            net = layers.ConvBN(
                net.shape[1],
                self.output_depth,
                kernel_size=1,
                # weight_attr=ParamAttr(name=self.name_scope + "/dilated/" + "block3")
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # Regularizer
            net = nn.Dropout(self.regularizer_prob)(net)
            net = self.prelu(net, decoder=self.decoder)

            # Add the main branch
            net = net_main + net
            net = self.prelu(net, decoder=self.decoder)

            return net

        # ASYMMETRIC CONVOLUTION BOTTLENECK
        # Everything is the same as a regular bottleneck except for a [5,5] kernel decomposed into two [5,1] then [1,5]
        elif type == ASYMMETRIC:
            # Save the main branch for addition later
            # asymmetric
            net_main = inputs
            # First projection with 1x1 kernel (dimensionality reduction)
            # block1
            net = layers.ConvBN(
                inputs.shape[1],
                reduced_depth,
                kernel_size=1,
                # weight_attr=ParamAttr(name=self.name_scope + "/asymmetric/" + "block1")
            )(inputs)
            net = self.prelu(net, decoder=self.decoder)

            # Second conv block --- apply asymmetric conv here
            # block2
            net = nn.Conv2D(
                net.shape[1],
                reduced_depth,
                kernel_size=[self.filter_size, 1],
                padding="same",
                # weight_attr=ParamAttr(name=self.name_scope + "/asymmetric/" + "block2/asymmetric_conv2a")
            )(net)

            net = layers.ConvBN(
                net.shape[1],
                reduced_depth,
                kernel_size=[1, self.filter_size],
                padding="same",
                # weight_attr=ParamAttr(
                #     name=self.name_scope + "/asymmetric/" + "block2/asymmetric_conv2b")
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # Final projection with 1x1 kernel
            # block3
            net = layers.ConvBN(
                net.shape[1],
                self.output_depth,
                kernel_size=1,
                padding="same",
                # weight_attr=ParamAttr(name=self.name_scope + "/asymmetric/" + "block3")
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # Regularizer
            net = nn.Dropout(self.regularizer_prob)(net)
            net = self.prelu(net, decoder=self.decoder)

            # Add the main branch
            net = net_main + net
            net = self.prelu(net, decoder=self.decoder)

            return net

        # UPSAMPLING BOTTLENECK
        # Everything is the same as a regular one, except convolution becomes transposed.
        elif type == UPSAMPLING:
            # Check if pooling indices is given

            # Check output_shape given or not
            if self.output_shape is None:
                raise ValueError('Output depth is not given')

            # =======MAIN BRANCH=======
            # Main branch to upsample. output shape must match with the shape of the layer that was pooled initially, in order
            # for the pooling indices to work correctly. However, the initial pooled layer was padded, so need to reduce dimension
            # before unpooling. In the paper, padding is replaced with convolution for this purpose of reducing the depth!

            net_unpool = layers.ConvBN(
                inputs.shape[1],
                self.output_depth,
                kernel_size=1,
                # weight_attr=ParamAttr(name=self.name_scope + "/upsampling/" + "unpool")
            )(inputs)
            net_unpool = nn.functional.interpolate(
                net_unpool, self.output_shape[2:], mode='bilinear')

            # First 1x1 projection to reduce depth
            # block1
            net = layers.ConvBN(
                inputs.shape[1],
                reduced_depth,
                kernel_size=1,
                # weight_attr=ParamAttr(name=self.name_scope + "/upsampling/" + "block1")
            )(inputs)
            net = self.prelu(net, decoder=self.decoder)

            # block2
            net = nn.Conv2DTranspose(
                net.shape[1],
                reduced_depth,
                kernel_size=self.filter_size,
                stride=2,
                padding="SAME")(net)
            net = layers.SyncBatchNorm(reduced_depth, data_format='NCHW')(net)
            net = self.prelu(net, decoder=self.decoder)

            # Final projection with 1x1 kernel
            # block3
            net = layers.ConvBN(
                net.shape[1],
                self.output_depth,
                kernel_size=1,
                # weight_attr=ParamAttr(name=self.name_scope + "/upsampling/" + "block3")
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # Regularizer
            net = nn.Dropout(self.regularizer_prob)(net)
            net = self.prelu(net, decoder=self.decoder)

            # Finally, add the unpooling layer and the sub branch together
            net = net + net_unpool
            net = self.prelu(net, decoder=self.decoder)

            return net

        # REGULAR BOTTLENECK
        else:
            # regular
            net_main = inputs

            # First projection with 1x1 kernel
            # block1
            net = layers.ConvBN(
                inputs.shape[1],
                reduced_depth,
                kernel_size=1,
                # weight_attr=ParamAttr(name=self.name_scope + "/regular/" + "block1")
            )(inputs)
            net = self.prelu(net, decoder=self.decoder)

            # Second conv block
            # block2
            net = layers.ConvBN(
                net.shape[1],
                reduced_depth,
                kernel_size=self.filter_size,
                # weight_attr=ParamAttr(name=self.name_scope + "/regular/" + "block2")
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # Final projection with 1x1 kernel
            # block3
            net = layers.ConvBN(
                net.shape[1],
                self.output_depth,
                kernel_size=1,
                # weight_attr=ParamAttr(name=self.name_scope + "/regular/" + "block3")
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # Regularizer
            net = nn.Dropout(self.regularizer_prob)(net)
            net = self.prelu(net, decoder=self.decoder)

            # Add the main branch
            net = net_main + net
            net = self.prelu(net, decoder=self.decoder)

            return net


class ENet(nn.Layer):
    def __init__(self, pretrained=None):
        super(ENet, self).__init__()
        self.pretrained = pretrained

        self._conv1 = layers.ConvBN(
            3, 13, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self._relu = layers.Activation("relu")
        self._prelu = layers.Activation("prelu")
        self._pool = MaxPool2D(kernel_size=2, stride=2, padding="SAME")

    def prelu(self, x, decoder=False):
        if decoder:
            return self._relu(x)
        return self._prelu(x)

    def initial_block(self, input):
        net_conv = self._conv1(input)
        net_conv = self.prelu(net_conv)
        net_pool = self._pool(input)
        net_concatenated = paddle.concat([net_conv, net_pool], axis=1)
        return net_concatenated

    def ENet_stage1(self, inputs, name_scope='stage1_block'):
        net, inputs_shape_1 = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.01,
            type=DOWNSAMPLING,
            name_scope=name_scope + '/bottleneck1_0')(inputs)

        net = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.01,
            name_scope=name_scope + '/bottleneck1_1')(net)

        net = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.01,
            name_scope=name_scope + '/bottleneck1_2')(net)

        net = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.01,
            name_scope=name_scope + '/bottleneck1_3')(net)

        net = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.01,
            name_scope=name_scope + '/bottleneck1_4')(net)

        return net, inputs_shape_1

    def ENet_stage2(self, inputs, name_scope='stage2_block'):
        net, inputs_shape_2 = bottleneck(
            output_depth=128,
            filter_size=3,
            regularizer_prob=0.1,
            type=DOWNSAMPLING,
            name_scope=name_scope + '/bottleneck2_0')(inputs)

        for i in range(2):
            net = bottleneck(
                output_depth=128,
                filter_size=3,
                regularizer_prob=0.1,
                name_scope=name_scope + '/bottleneck2_{}'.format(
                    str(4 * i + 1)))(net)

            net = bottleneck(
                output_depth=128,
                filter_size=3,
                regularizer_prob=0.1,
                type=DILATED,
                dilation_rate=(2**(2 * i + 1)),
                name_scope=name_scope + '/bottleneck2_{}'.format(
                    str(4 * i + 2)))(net)

            net = bottleneck(
                output_depth=128,
                filter_size=5,
                regularizer_prob=0.1,
                type=ASYMMETRIC,
                name_scope=name_scope + '/bottleneck2_{}'.format(
                    str(4 * i + 3)))(net)

            net = bottleneck(
                output_depth=128,
                filter_size=3,
                regularizer_prob=0.1,
                type=DILATED,
                dilation_rate=(2**(2 * i + 2)),
                name_scope=name_scope + '/bottleneck2_{}'.format(
                    str(4 * i + 4)))(net)

        return net, inputs_shape_2

    def ENet_stage3(self, inputs, name_scope='stage3_block'):
        for i in range(2):
            net = bottleneck(
                output_depth=128,
                filter_size=3,
                regularizer_prob=0.1,
                name_scope=name_scope + '/bottleneck3_{}'.format(
                    str(4 * i + 0)))(inputs)
            net = bottleneck(
                output_depth=128,
                filter_size=3,
                regularizer_prob=0.1,
                type=DILATED,
                dilation_rate=(2**(2 * i + 1)),
                name_scope=name_scope + '/bottleneck3_{}'.format(
                    str(4 * i + 1)))(net)
            net = bottleneck(
                output_depth=128,
                filter_size=5,
                regularizer_prob=0.1,
                type=ASYMMETRIC,
                name_scope=name_scope + '/bottleneck3_{}'.format(
                    str(4 * i + 2)))(net)
            net = bottleneck(
                output_depth=128,
                filter_size=3,
                regularizer_prob=0.1,
                type=DILATED,
                dilation_rate=(2**(2 * i + 2)),
                name_scope=name_scope + '/bottleneck3_{}'.format(
                    str(4 * i + 3)))(net)

            return net

    def ENet_stage4(self,
                    inputs,
                    inputs_shape,
                    connect_tensor,
                    skip_connections=True,
                    name_scope='stage4_block'):
        net = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.1,
            type=UPSAMPLING,
            decoder=True,
            output_shape=inputs_shape,
            name_scope=name_scope + '/bottleneck4_0')(inputs)
        if skip_connections:
            net = net + connect_tensor

        net = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.1,
            decoder=True,
            name_scope=name_scope + '/bottleneck4_1')(net)

        net = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.1,
            decoder=True,
            name_scope=name_scope + '/bottleneck4_2')(net)
        return net

    def ENet_stage5(self,
                    inputs,
                    inputs_shape,
                    connect_tensor,
                    skip_connections=True,
                    name_scope='stage5_block'):
        net = bottleneck(
            output_depth=16,
            filter_size=3,
            regularizer_prob=0.1,
            type=UPSAMPLING,
            decoder=True,
            output_shape=inputs_shape,
            name_scope=name_scope + '/bottleneck5_0')(inputs)

        if skip_connections:
            net = net + connect_tensor

        net = bottleneck(
            output_depth=16,
            filter_size=3,
            regularizer_prob=0.1,
            decoder=True,
            name_scope=name_scope + '/bottleneck5_1')(net)
        return net

    def decoder(self, input, num_classes):
        initial, stage1, stage2, inputs_shape_1, inputs_shape_2 = input
        segStage3 = self.ENet_stage3(stage2, name_scope='LaneNetSeg')
        segStage4 = self.ENet_stage4(
            segStage3, inputs_shape_2, stage1, name_scope='LaneNetSeg')
        segStage5 = self.ENet_stage5(
            segStage4, inputs_shape_1, initial, name_scope='LaneNetSeg')

        segLogits = nn.Conv2DTranspose(
            segStage5.shape[1],
            num_classes,
            kernel_size=2,
            stride=2,
            padding="SAME")(segStage5)

        emStage3 = self.ENet_stage3(stage2, name_scope='LaneNetEm')
        emStage4 = self.ENet_stage4(
            emStage3, inputs_shape_2, stage1, name_scope='LaneNetEm')
        emStage5 = self.ENet_stage5(
            emStage4, inputs_shape_1, initial, name_scope='LaneNetEm')

        emLogits = nn.Conv2DTranspose(
            emStage5.shape[1], 4, kernel_size=2, stride=2,
            padding="SAME")(emStage5)

        return segLogits, emLogits

    def forward(self, inputs):
        initial = self.initial_block(inputs)
        stage1, inputs_shape_1 = self.ENet_stage1(
            initial, name_scope='LaneNetBase')
        stage2, inputs_shape_2 = self.ENet_stage2(
            stage1, name_scope='LaneNetBase')
        output = (initial, stage1, stage2, inputs_shape_1, inputs_shape_2)

        segLogits, emLogits = self.decoder(output, 2)
        return segLogits, emLogits


@manager.BACKBONES.add_component
def EnetBone(**args):
    model = ENet(**args)
    return model
