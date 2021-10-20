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

import paddle.nn as nn

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init


@manager.MODELS.add_component
class LaneFCN(nn.Layer):
    """
    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone network,currently support vgg, bisenet, enet
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(
            self,
            num_classes,  # 相互独立的目标类别的数量。
            backbone=None,
            pretrained=None):  # 预训练模型的url或path。 默认:None
        super().__init__()

        self.backbone = backbone
        self.fcn = FCN(backbone, pretrained)

    def forward(self, x):
        logit_list = self.fcn(x)
        return logit_list


class FCN(nn.Layer):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.backbone = backbone
        if backbone is None:
            raise Exception("LaneNet fcn need backbone, but received None")

        self.deconv1 = nn.Conv2DTranspose(
            64, 64, kernel_size=4, stride=2, padding="SAME")
        self.deconv2 = nn.Conv2DTranspose(
            64, 64, kernel_size=16, stride=8, padding="SAME")
        self.conv1x1 = nn.Conv2D(64, 2, 1, 1)
        self.conv1x2 = nn.Conv2D(64, 4, 1, 1)

        self.pretrained = pretrained
        self.init_weight()

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

    def encoder(self, x):
        x3, x4, x5 = self.backbone(x)
        output = {}
        output['pool3'] = x3
        output['pool4'] = x4
        output['pool5'] = x5
        return output

    def decoder(self, input):
        encoder_list = ['pool5', 'pool4', 'pool3']
        # score stage
        input_tensor = input[encoder_list[0]]
        dim = input_tensor.shape[1]
        score = nn.Conv2D(dim, 64, 1, 1)(input_tensor)

        encoder_list = encoder_list[1:]
        for i in range(len(encoder_list)):
            deconv_out = self.deconv1(score)
            input_tensor = input[encoder_list[i]]
            dim = input_tensor.shape[1]
            score = nn.Conv2D(dim, 64, 1, 1)(input_tensor)
            score = deconv_out + score

        emLogits = self.deconv2(score)
        segLogits = self.conv1x1(emLogits)
        emLogits = self.conv1x2(emLogits)
        return segLogits, emLogits

    def forward(self, x):
        output = self.encoder(x)
        segLogits, emLogits = self.decoder(output)
        return segLogits, emLogits
