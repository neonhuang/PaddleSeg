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

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
import paddle
import paddle.nn as nn
from paddle.nn import Conv2D
from paddle.nn import MaxPool2D

from paddleseg.cvlibs import manager
from paddleseg.models import layers


@manager.MODELS.add_component
class LaneEnet(nn.Layer):
    """
    Args:
        num_classes (int): The unique number of target classes.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(
            self,
            num_classes,  # 相互独立的目标类别的数量。
            pretrained=None):  # 预训练模型的url或path。 默认:None
        super().__init__()

        self.enet = ENet(pretrained=pretrained)
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        logit_list = self.enet(x)
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
        else:
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.Conv2D):
                    param_init.kaiming_normal_init(sublayer.weight)
                elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                    param_init.constant_init(sublayer.weight, value=1.0)
                    param_init.constant_init(sublayer.bias, value=0.0)


class ENet(nn.Layer):
    def __init__(self, pretrained=None):
        super(ENet, self).__init__()
        self.pretrained = pretrained
        self._relu = layers.Activation("relu")
        self._prelu = layers.Activation("prelu")
        self._initial_block = Initial_Block()
        self._enet_stage1 = ENet_stage1(name_scope="LaneNetBase")
        self._enet_stage2 = ENet_stage2(name_scope="LaneNetBase")
        self._enetSeg_stage3 = ENet_stage3(name_scope="LaneNetSeg")
        self._enetSeg_stage4 = ENet_stage4(name_scope="LaneNetSeg")
        self._enetSeg_stage5 = ENet_stage5(name_scope="LaneNetSeg")
        self._enetEm_stage3 = ENet_stage3(name_scope="LaneNetEm")
        self._enetEm_stage4 = ENet_stage4(name_scope="LaneNetEm")
        self._enetEm_stage5 = ENet_stage5(name_scope="LaneNetEm")

    def prelu(self, x, decoder=False):
        if decoder:
            return self.relu(x)
        return self.prelu(x)

    def encoder(self, inputs):
        initial = self._initial_block(inputs)
        stage1, inputs_shape_1 = self._enet_stage1(initial)
        stage2, inputs_shape_2 = self._enet_stage2(stage1)
        output = (initial, stage1, stage2, inputs_shape_1, inputs_shape_2)
        return output

    def decoder(self, input):
        initial, stage1, stage2, inputs_shape_1, inputs_shape_2 = input

        segStage3 = self._enetSeg_stage3(stage2)
        # upsampling
        segStage4 = self._enetSeg_stage4(segStage3, inputs_shape_2, stage1)
        segStage5 = self._enetSeg_stage5(segStage4, inputs_shape_1, initial)

        segLogits = nn.Conv2DTranspose(
            segStage5.shape[1], 2, kernel_size=2, stride=2,
            padding="SAME")(segStage5)

        emStage3 = self._enetEm_stage3(stage2)
        # upsampling
        emStage4 = self._enetEm_stage4(emStage3, inputs_shape_2, stage1)
        emStage5 = self._enetEm_stage5(emStage4, inputs_shape_1, initial)

        emLogits = nn.Conv2DTranspose(
            emStage5.shape[1], 4, kernel_size=2, stride=2,
            padding="SAME")(emStage5)

        return segLogits, emLogits

    def forward(self, inputs):
        output = self.encoder(inputs)
        segLogits, emLogits = self.decoder(output)
        return segLogits, emLogits


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

    def forward(self, inputs, output_shape=None):
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
            )(inputs)
            net = self.prelu(net, decoder=self.decoder)

            # block2
            net = layers.ConvBN(
                net.shape[1],
                reduced_depth,
                kernel_size=self.filter_size,
                padding="same",
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # block3
            net = layers.ConvBN(
                net.shape[1],
                self.output_depth,
                kernel_size=1,
                padding="same",
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
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # Final projection with 1x1 kernel (Expansion)
            # block3
            net = layers.ConvBN(
                net.shape[1],
                self.output_depth,
                kernel_size=1,
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
            )(inputs)
            net = self.prelu(net, decoder=self.decoder)

            # Second conv block --- apply asymmetric conv here
            # block2
            net = nn.Conv2D(
                net.shape[1],
                reduced_depth,
                kernel_size=[self.filter_size, 1],
                padding="same",
            )(net)

            net = layers.ConvBN(
                net.shape[1],
                reduced_depth,
                kernel_size=[1, self.filter_size],
                padding="same",
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # Final projection with 1x1 kernel
            # block3
            net = layers.ConvBN(
                net.shape[1],
                self.output_depth,
                kernel_size=1,
                padding="same",
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
            self.output_shape = output_shape
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
            )(inputs)
            net_unpool = nn.functional.interpolate(
                net_unpool, self.output_shape[2:], mode='bilinear')

            # First 1x1 projection to reduce depth
            # block1
            net = layers.ConvBN(
                inputs.shape[1],
                reduced_depth,
                kernel_size=1,
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
            )(inputs)
            net = self.prelu(net, decoder=self.decoder)

            # Second conv block
            # block2
            net = layers.ConvBN(
                net.shape[1],
                reduced_depth,
                kernel_size=self.filter_size,
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # Final projection with 1x1 kernel
            # block3
            net = layers.ConvBN(
                net.shape[1],
                self.output_depth,
                kernel_size=1,
            )(net)
            net = self.prelu(net, decoder=self.decoder)

            # Regularizer
            net = nn.Dropout(self.regularizer_prob)(net)
            net = self.prelu(net, decoder=self.decoder)

            # Add the main branch
            net = net_main + net
            net = self.prelu(net, decoder=self.decoder)

            return net


class Initial_Block(nn.Layer):
    '''
    The initial block for ENet has 2 branches: The convolution branch and MaxPool branch.
    The conv branch has 13 filters, while the maxpool branch gives 3 channels corresponding to the RGB channels.
    Both output layers are then concatenated to give an output of 16 channels.

    :param inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    :return net_concatenated(Tensor): a 4D Tensor of new shape [batch_size, height, width, channels]
    '''

    def __init__(self):
        super().__init__()

        self.conv_3x3_BN = layers.ConvBN(
            3, 13, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.pool = MaxPool2D(kernel_size=2, stride=2, padding="SAME")
        self.prelu = layers.Activation("prelu")

    def forward(self, x):
        # Convolutional branch
        net_conv = self.conv_3x3_BN(x)
        net_conv = self.prelu(net_conv)

        # Max pool branch
        net_pool = self.pool(x)

        # Concatenated output - does it matter max pool comes first or conv comes first? probably not.
        net_concatenated = paddle.concat([net_conv, net_pool], axis=1)
        return net_concatenated


class ENet_stage1(nn.Layer):
    def __init__(self, name_scope='stage1_block'):
        super().__init__()
        self.bottleneck1_0 = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.01,
            type=DOWNSAMPLING,
            name_scope=name_scope + '/bottleneck1_0')
        self.bottleneck1_1 = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.01,
            name_scope=name_scope + '/bottleneck1_1')

        self.bottleneck1_2 = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.01,
            name_scope=name_scope + '/bottleneck1_2')

        self.bottleneck1_3 = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.01,
            name_scope=name_scope + '/bottleneck1_3')

        self.bottleneck1_4 = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.01,
            name_scope=name_scope + '/bottleneck1_4')

    def forward(self, inputs):
        net, inputs_shape_1 = self.bottleneck1_0(inputs)
        net = self.bottleneck1_1(net)
        net = self.bottleneck1_2(net)
        net = self.bottleneck1_3(net)
        net = self.bottleneck1_4(net)
        return net, inputs_shape_1


class ENet_stage2(nn.Layer):
    def __init__(self, name_scope='stage2_block'):
        super().__init__()
        self.bottleneck2_0 = bottleneck(
            output_depth=128,
            filter_size=3,
            regularizer_prob=0.1,
            type=DOWNSAMPLING,
            name_scope=name_scope + '/bottleneck2_0')

        self.block_list = []
        for i in range(2):
            block_name = bottleneck(
                output_depth=128,
                filter_size=3,
                regularizer_prob=0.1,
                name_scope=name_scope + '/bottleneck2_{}'.format(
                    str(4 * i + 1)))
            self.block_list.append(block_name)

            block_name = bottleneck(
                output_depth=128,
                filter_size=3,
                regularizer_prob=0.1,
                type=DILATED,
                dilation_rate=(2**(2 * i + 1)),
                name_scope=name_scope + '/bottleneck2_{}'.format(
                    str(4 * i + 2)))
            self.block_list.append(block_name)

            block_name = bottleneck(
                output_depth=128,
                filter_size=5,
                regularizer_prob=0.1,
                type=ASYMMETRIC,
                name_scope=name_scope + '/bottleneck2_{}'.format(
                    str(4 * i + 3)))
            self.block_list.append(block_name)

            block_name = bottleneck(
                output_depth=128,
                filter_size=3,
                regularizer_prob=0.1,
                type=DILATED,
                dilation_rate=(2**(2 * i + 2)),
                name_scope=name_scope + '/bottleneck2_{}'.format(
                    str(4 * i + 4)))
            self.block_list.append(block_name)

    def forward(self, inputs):
        net, inputs_shape_2 = self.bottleneck2_0(inputs)

        for i in range(2):
            net = self.block_list[4 * i + 0](net)
            net = self.block_list[4 * i + 1](net)
            net = self.block_list[4 * i + 2](net)
            net = self.block_list[4 * i + 3](net)
        return net, inputs_shape_2


class ENet_stage3(nn.Layer):
    def __init__(self, name_scope='stage3_block'):
        super().__init__()

        self.block_list = []
        for i in range(2):
            block_name = bottleneck(
                output_depth=128,
                filter_size=3,
                regularizer_prob=0.1,
                name_scope=name_scope + '/bottleneck3_{}'.format(
                    str(4 * i + 0)))
            self.block_list.append(block_name)

            block_name = bottleneck(
                output_depth=128,
                filter_size=3,
                regularizer_prob=0.1,
                type=DILATED,
                dilation_rate=(2**(2 * i + 1)),
                name_scope=name_scope + '/bottleneck3_{}'.format(
                    str(4 * i + 1)))
            self.block_list.append(block_name)

            block_name = bottleneck(
                output_depth=128,
                filter_size=5,
                regularizer_prob=0.1,
                type=ASYMMETRIC,
                name_scope=name_scope + '/bottleneck3_{}'.format(
                    str(4 * i + 2)))
            self.block_list.append(block_name)

            block_name = bottleneck(
                output_depth=128,
                filter_size=3,
                regularizer_prob=0.1,
                type=DILATED,
                dilation_rate=(2**(2 * i + 2)),
                name_scope=name_scope + '/bottleneck3_{}'.format(
                    str(4 * i + 3)))
            self.block_list.append(block_name)

    def forward(self, inputs):
        net = inputs
        for i in range(2):
            net = self.block_list[4 * i + 0](net)
            net = self.block_list[4 * i + 1](net)
            net = self.block_list[4 * i + 2](net)
            net = self.block_list[4 * i + 3](net)
        return net


class ENet_stage4(nn.Layer):
    def __init__(self,
                 inputs_shape=None,
                 skip_connections=True,
                 name_scope='stage4_block'):
        super().__init__()
        self.skip_connections = skip_connections
        self.bottleneck4_0 = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.1,
            type=UPSAMPLING,
            decoder=True,
            output_shape=inputs_shape,
            name_scope=name_scope + '/bottleneck4_0')

        self.bottleneck4_1 = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.1,
            decoder=True,
            name_scope=name_scope + '/bottleneck4_1')

        self.bottleneck4_2 = bottleneck(
            output_depth=64,
            filter_size=3,
            regularizer_prob=0.1,
            decoder=True,
            name_scope=name_scope + '/bottleneck4_2')

    def forward(self, inputs, inputs_shape, connect_tensor):
        net = self.bottleneck4_0(inputs, output_shape=inputs_shape)
        if self.skip_connections:
            net = net + connect_tensor
        net = self.bottleneck4_1(net)
        net = self.bottleneck4_2(net)

        return net


class ENet_stage5(nn.Layer):
    def __init__(self,
                 inputs_shape=None,
                 skip_connections=True,
                 name_scope='stage5_block'):
        super().__init__()
        self.skip_connections = skip_connections
        self.bottleneck5_0 = bottleneck(
            output_depth=16,
            filter_size=3,
            regularizer_prob=0.1,
            type=UPSAMPLING,
            decoder=True,
            output_shape=inputs_shape,
            name_scope=name_scope + '/bottleneck5_0')

        self.bottleneck5_1 = bottleneck(
            output_depth=16,
            filter_size=3,
            regularizer_prob=0.1,
            decoder=True,
            name_scope=name_scope + '/bottleneck5_1')

    def forward(self, inputs, inputs_shape, connect_tensor):
        net = self.bottleneck5_0(inputs, output_shape=inputs_shape)
        if self.skip_connections:
            net = net + connect_tensor
        net = self.bottleneck5_1(net)
        return net