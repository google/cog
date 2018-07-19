# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for model/ops.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import tensorflow as tf

from model import ops


class OpsTest(unittest.TestCase):

  def testConv2DByBatch(self):
    tf.reset_default_graph()

    bs = 4
    in_channels = 3
    h, w = 7, 7
    fh, fw = 1, 1
    out_channels = 5

    inputs = tf.placeholder('float', [bs, h, w, in_channels])
    filters = tf.placeholder('float', [bs, fh, fw, in_channels, out_channels])

    outputs = ops.conv2d_by_batch(inputs, filters, (1, 1, 1, 1), 'SAME')

    inputs_ = np.random.randn(bs, h, w, in_channels)
    filters_ = np.random.rand(bs, fh, fw, in_channels, out_channels)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs_ = sess.run(outputs,
                          feed_dict={inputs: inputs_, filters: filters_})

    tmp = list()
    for i in range(bs):
      tmp.append(np.dot(inputs_[i], filters_[i, 0, 0]))
    outputs_2 = np.array(tmp)

    self.assertTrue(np.mean(abs(outputs_-outputs_2)) < 1e-6)


if __name__ == '__main__':
  unittest.main()
