#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function


import unittest
import numpy as np
import struct
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_fusion_gru_op import fusion_gru
from paddle.fluid.tests.unittests.test_fusion_lstm_op import fc, ACTIVATION


def copy_bits_from_float_to_uint16(f):
    return struct.unpack('<I', struct.pack('<f', f))[0] >> 16


def convert_float_to_uint16(float_list):
    new_output = []
    for first in float_list:
        for second in first:
                    new_output.append(np.uint16(copy_bits_from_float_to_uint16(second)))

    return np.reshape(new_output, float_list.shape).view(np.uint16)


class TestFusionGRUBF16MKLDNNOp(OpTest):
    def set_confs(self):
        self.mkldnn_data_type = False

    def setUp(self):
        self.op_type = "fusion_gru"
        self.lod = [[2, 4, 3]]
        self.M = 3
        self.D = 5
        self.is_reverse = False
        self.with_h0 = False
        self.with_bias = True
        self.act_state = 'tanh'
        self.act_gate = 'sigmoid'
        self.origin_mode = False
        self.use_mkldnn = True
        self.force_fp32_output = False
        self.set_confs()

        T = sum(self.lod[0])
        N = len(self.lod[0])
        
        # fp32 X input for reference implementation and
        # corressponding bf16 data as input to GRU oneDNN bf16 kernel
        x_fp32 = np.random.rand(T, self.M).astype('float32')
        x_bf16 = convert_float_to_uint16(x_fp32)

        wx_fp32 = np.random.rand(self.M, 3 * self.D).astype('float32')
        wh_fp32 = np.random.rand(self.D, 3 * self.D).astype('float32')

        # bias is fp32 despite other inputs being in bf16
        bias = np.random.rand(
            1, 3 * self.D).astype('float32') if self.with_bias else np.zeros(
                (1, 3 * self.D), dtype='float32')

        h0_fp32 = np.random.rand(N, self.D).astype('float32') if self.with_h0 else np.zeros(
                (N, self.D), dtype='float32')


        _, _, _, hidden = fusion_gru(
            x_fp32, self.lod, h0_fp32, wx_fp32, wh_fp32, bias, self.is_reverse, self.origin_mode,
            ACTIVATION[self.act_state], ACTIVATION[self.act_gate])

        hidden_bf16 =  convert_float_to_uint16(hidden)

        self.inputs = {'X': (x_bf16, self.lod), 'WeightX': wx_fp32, 'WeightH': wh_fp32}

        if self.with_bias:
            self.inputs['Bias'] = bias

        if self.with_h0:
            self.inputs['H0'] = h0_bf16

        self.error_margin = 1
        h0_bf16 = convert_float_to_uint16(h0_fp32)
        self.outputs = {'Hidden': (hidden_bf16, self.lod)}

        self.attrs = {
            'activation': self.act_state,
            'gate_activation': self.act_gate,
            'is_reverse': self.is_reverse,
            'origin_mode': self.origin_mode,
            'force_fp32_output': self.force_fp32_output,
            'use_mkldnn': self.use_mkldnn
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False, atol=self.error_margin)


class TestFusionGRUINT8MKLDNNOp2(TestFusionGRUBF16MKLDNNOp):
    def set_confs(self):
        self.origin_mode = False


class TestFusionGRUINT8MKLDNNOp3(TestFusionGRUBF16MKLDNNOp):
    def set_confs(self):
        self.with_bias = False

if __name__ == "__main__":
    unittest.main()
