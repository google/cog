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

"""Network ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers


class BaseAttention(object):
  """Base class for applying attention."""

  def __init__(self, attention_size):
    """Initialize attention class.

    Args:
      attention_size: int or tensor, size of attention vector.
    """
    self.attention_size = attention_size

  def __call__(self, inputs_to_attend, attn_gen):
    """Generate and apply attention to given inputs.

    Args:
      inputs_to_attend: tensor [batch_size, num_units], inputs to attend
      attn_gen: tensor [batch_size, num_units], inputs to generate attention

    Returns:
      inputs_attended: tensor, inputs after attention
      attention: tensor or tuple of tensors [batch_size, attention_size]
    """
    return inputs_to_attend, None


class FeatureAttention(BaseAttention):
  """Feature attention."""

  def __init__(self, attention_size, use_mlp=False):
    super(FeatureAttention, self).__init__(attention_size)
    self._use_mlp = use_mlp

  def __call__(self, inputs_to_attend, attn_gen):
    """Generate and apply attention to given inputs.

    Attention is generated using a linear transformation from inputs.
    The shifting is initialized at 0, scaling normalized at 1.

    Args:
      inputs_to_attend: tensor, inputs to attend
          For feature attention, this has to be the output of a conv2d layer
          with shape [batch_size, kernel_size, kernel_size, n_channels] (e.g.
          (32, 14, 14, 64))
      attn_gen: tensor [batch_size, num_units], inputs to generate attention

    Returns:
      inputs_attended: tensor, inputs after attention,
          same shape as inputs_to_attend
      attention: tensor or tuple of tensors [batch_size, attention_size]
    """
    attention_size = inputs_to_attend.get_shape().as_list()[-1]

    if self._use_mlp:
      attn_gen = layers.fully_connected(attn_gen, 128,
                                        scope='feature_attn_gen0')
    shift_and_scale = layers.fully_connected(
        attn_gen,
        attention_size * 2,
        activation_fn=None,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.zeros_initializer(),
        scope='feature_attn_gen')
    shift, scale = tf.split(shift_and_scale, num_or_size_splits=2, axis=1)
    scale = tf.nn.relu(scale + 1.0)

    shift = tf.expand_dims(tf.expand_dims(shift, axis=1), axis=1)
    inputs_to_attend += shift

    scale = tf.expand_dims(tf.expand_dims(scale, axis=1), axis=1)
    inputs_to_attend *= scale

    inputs_to_attend = tf.nn.relu(inputs_to_attend)
    return inputs_to_attend, (shift, scale)


class SpatialAttention(BaseAttention):
  """Spatial attention."""

  def __init__(self, attention_size):
    super(SpatialAttention, self).__init__(attention_size)

  def __call__(self, inputs_to_attend, attn_gen):
    """Generate and apply attention to given inputs.

    Generate a spatial attention with shape [batch_size, kernel_size**2]

    Args:
      inputs_to_attend: tensor, inputs to attend
          For spatial attention, this has to be the output of a conv2d layer
          with shape [batch_size, kernel_size, kernel_size, n_channels]
      attn_gen: tensor [batch_size, num_units], inputs to generate attention

    Returns:
      inputs_attended: tensor, inputs after attention
      attention: tensor or tuple of tensors [batch_size, attention_size]
    """
    _, height, width, _ = inputs_to_attend.get_shape().as_list()
    attention_size = height * width
    # Use a MLP here
    attention = layers.fully_connected(attn_gen, 10, scope='attn_spatial1')
    attention = layers.fully_connected(
        attention,
        attention_size,
        activation_fn=None,
        scope='attn_spatial2')
    attention = tf.nn.softmax(attention)

    # [batch_size, kernel_size, kernel_size, n_channels]
    inputs_shape = inputs_to_attend.get_shape().as_list()
    # reshape to [batch_size, kernel_size, kernel_size]
    attention_shaped = tf.reshape(attention, inputs_shape[:3])
    attention_shaped = tf.expand_dims(attention_shaped, axis=-1)
    inputs_to_attend *= attention_shaped

    return inputs_to_attend, attention


class BaseMemory(object):
  """Base class for (read-only) memory mechanism."""

  def __init__(self, memory):
    """Memory class.

    Args:
      memory: tensor (n_item, batch_size, n_units)
    """
    self._memory = memory
    self.query_size = memory.get_shape().as_list()[2]

  def generate_query(self, inputs):
    """Generate memory query."""
    del inputs
    return None

  def retrieve(self, query=None):
    """Retrieve memory.

    Typically, memory will be retrieved based on similarity with query.

    Args:
      query: tensor, (batch_size, n_units)

    Returns:
      memory_retrieved: tensor (batch, n_units), memory retrieved.
          By default, return the average across items.
      alignment: None
    """
    del query
    return tf.reduce_mean(self._memory, axis=0), None


def conv2d_by_batch(inputs, filters, strides, padding):
  """Batch-dependent 2-D convolution.

  Args:
    inputs: tensor (batch_size, in_height, in_width, in_channels)
    filters: tensor
        (batch_size, filter_height, filter_width, in_channels, out_channels)
    strides: list of ints
    padding: string, 'SAME' or 'VALID'

  Returns:
    outputs: tensor (batch_size, out_height, out_width, out_channels)

  Raises:
    ValueError: if inputs and filters have different batch sizes or number of
        channels
  """
  # Require known batch_size
  bs, h, w, in_channels = inputs.get_shape().as_list()
  bs2, fh, fw, in_channels2, out_channels = filters.get_shape().as_list()
  if bs != bs2:
    raise ValueError(
        'Batch size for inputs and filters need to be the same, ' +
        str(bs) + ' != ' + str(bs2))

  if in_channels != in_channels2:
    raise ValueError(
        'Number of channels for inputs and filters need to be the same, ' +
        str(in_channels) + ' != ' + str(in_channels2))

  inputs = tf.transpose(inputs, [1, 2, 0, 3])  # [h, w, bs, in_channels]
  inputs = tf.reshape(inputs, [1, h, w, bs * in_channels])

  filters = tf.transpose(filters, [1, 2, 0, 3, 4])  # [h, w, bs, in_ch, out_ch]
  filters = tf.reshape(filters, [fh, fw, in_channels * bs, out_channels])

  outputs = tf.nn.depthwise_conv2d(inputs, filters, strides, padding)

  outputs = tf.reshape(outputs, [h, w, bs, in_channels, out_channels])
  outputs = tf.transpose(outputs, [2, 0, 1, 3, 4])  # (bs, h, w, in_ch, out_ch)
  outputs = tf.reduce_sum(outputs, axis=3)  # (bs, h, w, out_ch)

  return outputs


class BaseVisMemory(tf.nn.rnn_cell.RNNCell):
  """Base class for visual memory mechanism.

  Inherit from RNNCell to use the zero_state method.
  """

  def __init__(self, num_units):
    """Visual memory class."""
    super(BaseVisMemory, self).__init__()
    self._num_units = num_units

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def generate_ctrl(self, inputs_ctrl):
    """Generate memory query."""
    del inputs_ctrl
    return None

  def __call__(self, inputs, state, ctrl):
    """Run one step."""
    del inputs
    del ctrl
    return state, state


class VisMemory(tf.nn.rnn_cell.RNNCell):
  """Base class for visual memory mechanism.

  Inherit from RNNCell to use the zero_state method.

  This visual memory cell is based on a convolutional LSTM. However,
  unlike normal LSTMs, the input, forget, and output gates in this network
  are not self-determined. They are provided as external inputs.
  """

  def __init__(self,
               shape,
               in_channels,
               out_channels,
               n_maps,
               forget_bias=1.0,
               activation=None,
               reuse=None,
               name=None):
    """Visual memory class.

    Args:
      shape: tuple, the shape of each visual memory map.
      in_channels: int, the number of input maps or channels
      out_channels: int, the number of output maps to project to
      n_maps: int, number of maps in the visual memory cell
      forget_bias: float, the default value of the forget gate
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
    """
    super(VisMemory, self).__init__(_reuse=reuse, name=name)
    self._shape = shape
    self._in_channels = in_channels
    self._out_channels = out_channels
    self._n_maps = n_maps
    self._num_units_input_gate = in_channels * n_maps
    self._num_units_output_gate = out_channels * n_maps
    self._num_units_forget_gate = n_maps
    self._state_split = (self._num_units_input_gate,
                         self._num_units_forget_gate,
                         self._num_units_output_gate)
    self._num_units = self._n_maps * self._shape[0] * self._shape[1]
    self._forget_bias = forget_bias
    self._activation = activation or tf.tanh

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._out_channels * self._shape[0] * self._shape[1]

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
      filled with zeros
    """

    shape = self.shape
    num_features = self.num_features
    zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * 2])
    return zeros

  def generate_ctrl(self, inputs_ctrl):
    """Generate memory query."""
    ctrls = layers.fully_connected(
        inputs_ctrl,
        sum(self._state_split),
        activation_fn=None,
        biases_initializer=tf.zeros_initializer(),
        scope='vis_mem_ctrl')
    return ctrls

  def __call__(self, inputs, state, ctrls):
    """Run one step.

    Args:
      inputs: tensor (batch_size, in_height, in_width, in_channels)
      state: tensor (batch_size, kernel_size * kernel_size * n_maps)
      ctrls: tensor (batch_size, num_units). Here ctrls can be split into
          three tensors i, f, o that represent the input, forget, and output
          gates.

    Returns:
      output: tensor
    """
    sigmoid = tf.sigmoid
    add = tf.add
    multiply = tf.multiply

    state = tf.reshape(
        state, [-1, self._shape[0], self._shape[1], self._n_maps])

    # i = input_gate, f = forget_gate, o = output_gate
    i, f, o = tf.split(ctrls, num_or_size_splits=self._state_split, axis=1)

    in_gates = tf.reshape(i, [-1, 1, 1, self._in_channels, self._n_maps])
    in_gates = sigmoid(in_gates)
    gated_inputs = conv2d_by_batch(inputs, in_gates, (1, 1, 1, 1), 'SAME')

    forget_bias_tensor = tf.constant(self._forget_bias, dtype=f.dtype)
    forget_gates = tf.reshape(f, (-1, 1, 1, self._n_maps))
    forget_gates = sigmoid(add(forget_gates, forget_bias_tensor))

    gated_states = multiply(state, forget_gates)
    new_state = add(gated_inputs, gated_states)

    out_gates = tf.reshape(o, [-1, 1, 1, self._n_maps, self._out_channels])
    output = self._activation(new_state)
    output = conv2d_by_batch(output, out_gates, (1, 1, 1, 1), 'SAME')

    # Flatten while maintaining batch_size
    output = layers.flatten(output)
    new_state = layers.flatten(new_state)

    return output, new_state
