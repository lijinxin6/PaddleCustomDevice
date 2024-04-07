# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at #
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
from numpy import linalg as LA
from tests.op_test import OpTest
import paddle

paddle.enable_static()


class TestL2LossOp(OpTest):
    """Test npu squared_l2_norm"""

    def setUp(self):
        self.set_npu()
        self.init_shape()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "squared_l2_norm"
        self.max_relative_error = 0.05

        X = np.random.uniform(-1, 1, self.shape).astype("float32")
        X[np.abs(X) < self.max_relative_error] = 0.1
        self.inputs = {"X": X}
        self.outputs = {"Out": np.array([np.square(LA.norm(X))])}

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_shape(self):
        self.shape = (13, 19)

    def test_check_output(self):
        self.check_output_with_place(place=self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=self.max_relative_error
        )


class TestL2LossOpRank1(OpTest):
    def init_shape(self):
        self.shape = (4000, 8192)


class TestL2LossOpRank2(OpTest):
    def init_shape(self):
        self.shape = (8192, 1280)


class TestL2LossOpRank3(OpTest):
    def init_shape(self):
        self.shape = (1024, 8192)


class TestL2LossOpRank4(OpTest):
    def init_shape(self):
        self.shape = (8192, 7168)


class TestL2LossOpRank5(OpTest):
    def init_shape(self):
        self.shape = (3584, 8192)


class TestL2LossOpRank6(OpTest):
    def init_shape(self):
        self.shape = 8192


class TestL2LossOpRank7(OpTest):
    def init_shape(self):
        self.shape = (8192, 4000)


class TestL2LossOpFp16(OpTest):
    """Test npu squared_l2_norm"""

    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "squared_l2_norm"
        self.max_relative_error = 0.05

        X = np.random.uniform(-1, 1, (13, 19)).astype("float16")
        X[np.abs(X) < self.max_relative_error] = 0.1
        self.inputs = {"X": X}
        self.outputs = {"Out": np.square(LA.norm(X))}

    def set_npu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(place=self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=self.max_relative_error
        )


if __name__ == "__main__":
    unittest.main()
