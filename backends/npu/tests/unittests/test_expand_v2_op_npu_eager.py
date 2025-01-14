# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import unittest
import numpy as np
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
import paddle

from npu_utils import check_soc_version, check_run_big_shape_test

np.random.seed(10)


class TestExpandV2OpBfloat(OpTest):
    def setUp(self):
        self.set_npu()
        self.init_data()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "expand_v2"
        middle_inputs = np.random.random(self.ori_shape).astype(np.float32)
        middle_inputs = convert_float_to_uint16(middle_inputs)
        self.inputs = {"X": middle_inputs}
        self.attrs = {"shape": self.shape}
        output = np.tile(convert_uint16_to_float(middle_inputs), self.expand_times)
        self.outputs = {"Out": output}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_data(self):
        self.ori_shape = [2, 4, 20]
        self.shape = [2, 4, 20]
        self.expand_times = [1, 1, 1]

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place, atol=0.004)


@check_run_big_shape_test()
class TestExpandV2OpBfloatRank1(TestExpandV2OpBfloat):
    def init_data(self):
        self.ori_shape = [1, 1, 4096, 4096]
        self.shape = [2, 1, 4096, 4096]
        self.expand_times = [2, 1, 1, 1]


@check_run_big_shape_test()
class TestExpandV2OpBfloatRank2(TestExpandV2OpBfloat):
    def init_data(self):
        self.ori_shape = [2, 1, 1, 4096]
        self.shape = [2, 1, 4096, 4096]
        self.expand_times = [1, 1, 4096, 1]


@check_run_big_shape_test()
class TestExpandV2OpBfloatRank3(TestExpandV2OpBfloat):
    def init_data(self):
        self.ori_shape = [4096]
        self.shape = [2, 4096]
        self.expand_times = [2, 1]


if __name__ == "__main__":
    unittest.main()
