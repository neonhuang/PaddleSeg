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
class LaneBCELoss(nn.Layer):
    def __init__(self, ignore_index=255, data_format='NCHW'):
        super(LaneBCELoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logit, label, semantic_weights=None):
        label = label.astype('float32')
        exist = logit
        exist_loss = 0.1 * paddle.nn.BCEWithLogitsLoss()(exist, label)

        return exist_loss
