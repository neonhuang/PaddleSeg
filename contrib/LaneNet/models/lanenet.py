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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.models.bisenet import DetailBranch, SemanticBranch, BGA


@manager.MODELS.add_component
class Lanenet(nn.Layer):
    """
    Args:
        num_classes (int): The unique number of target classes.
        lambd (float, optional): A factor for controlling the size of semantic branch channels. Default: 0.25.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(
            self,
            num_classes,  # 相互独立的目标类别的数量。
            lambd=0.25,  # 控制语义分支通道大小的因素。默认:0.25
            align_corners=False,
            pretrained=None):  # 预训练模型的url或path。 默认:None
        super().__init__()

        C1, C2, C3 = 64, 64, 128
        db_channels = (C1, C2, C3)
        C1, C3, C4, C5 = int(C1 * lambd), int(C3 * lambd), 64, 128
        sb_channels = (C1, C3, C4, C5)
        mid_channels = 128

        self.db = DetailBranch(db_channels)
        self.sb = SemanticBranch(sb_channels)

        self.bga = BGA(mid_channels,
                       align_corners)  # Bilateral Guided Aggregation
        self.binary_seg = SegHead(128, 64, 2)
        self.instance_seg = SegHead(128, 64, 4)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        dfm = self.db(x)
        _, _, _, _, sfm = self.sb(x)
        agr = self.bga(dfm, sfm)
        binary_seg_branch_output = self.binary_seg(agr)
        instance_seg_branch_output = self.instance_seg(agr)

        if not self.training:
            logit_list = [binary_seg_branch_output, instance_seg_branch_output]
        else:
            logit_list = [binary_seg_branch_output, instance_seg_branch_output]

        logit_list = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

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


class SegHead(nn.Layer):
    def __init__(self, in_dim, mid_dim, num_classes):
        super().__init__()

        self.conv_3x3 = nn.Sequential(
            layers.ConvBNReLU(in_dim, mid_dim, 3), nn.Dropout(0.1))

        self.conv_1x1_bn = nn.Sequential(
            layers.ConvBNReLU(mid_dim, in_dim, 1), nn.Dropout(0.1))

        self.conv_1x1 = nn.Conv2D(in_dim, num_classes, 1, 1)

    def forward(self, x):
        conv1 = self.conv_3x3(x)
        conv2 = self.conv_1x1_bn(conv1)
        conv3 = self.conv_1x1(conv2)
        return conv3
