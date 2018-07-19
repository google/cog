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
"""Networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf

from model import ops

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def tf_repeat(tensor, repeats):
  """Tensorflow function corresponding to numpy repeat.

  This code is not meant to be a general solution.

  Args:
    tensor: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the
      same as the number of dimensions in input

  Returns:
    A Tensor. Has the same type as input. Has the shape of tensor.shape *
    repeats
  """
  with tf.variable_scope('repeat'):
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)

    # This will only work when one dimension is not determined
    assert sum([s is None for s in tensor.get_shape().as_list()]) < 2
    new_shape = []
    for s, r in zip(tensor.get_shape().as_list(), repeats):
      if s is None:
        new_shape += [-1]
      else:
        new_shape += [s * r]

    repeated_tensor = tf.reshape(tiled_tensor, new_shape)
  return repeated_tensor


def get_initial_state(state_size, batch_size, name):
  """Get initial state given state_size.

  Args:
    state_size: int, tuple of ints or tuples, or LSTMStateTuples
    batch_size: int. Necessary to generate the same initial state for all trials
    name: string, base_name of the initialization variables.

  Returns:
    initial_state: tensor or tuple of tensors, (batch_size, state_size)
  """
  if isinstance(state_size, int):
    initial_state = tf.get_variable(name, [1, state_size], dtype=tf.float32)
    initial_state = tf.tile(initial_state, [batch_size, 1])
  else:
    initial_state = tuple(
        get_initial_state(s, batch_size, name + '_' + str(i))
        for i, s in enumerate(state_size))
    if isinstance(state_size, tf.nn.rnn_cell.LSTMStateTuple):
      initial_state = tf.nn.rnn_cell.LSTMStateTuple(*initial_state)
  return initial_state


def simple_vgg_preprocessing(hp, in_imgs):
  if hp.normalize_images:
    # in_imgs shape [n_epoch*batch_size, img_size, img_size, n_channels]
    new_in_imgs = tf.map_fn(
        lambda img: tf.image.per_image_standardization(img), in_imgs)

    return new_in_imgs
  else:
    n_channels = 3
    means = [_R_MEAN, _G_MEAN, _B_MEAN]
    channels = tf.split(in_imgs, n_channels, axis=-1)
    for i in range(n_channels):
      channels[i] -= means[i]
    new_in_imgs = tf.concat(channels, axis=-1)

    return new_in_imgs


def myconv(inputs, filters, shift_and_scale=True):
  """My convolutional layer."""
  net = tf.layers.conv2d(inputs, filters, [3, 3], padding='same')
  net = tf.layers.max_pooling2d(net, [2, 2], 2, padding='valid')

  if shift_and_scale:
    center, scale = True, True
    activation_fn = tf.nn.relu
  else:
    center, scale = False, False
    activation_fn = None

  # If used inside a RNN, set updates_collections=None
  net = tf.contrib.layers.batch_norm(
      net,
      is_training=True,  # TEMPORARILY setting all to True
      center=center,
      scale=scale,
      activation_fn=activation_fn,
      fused=True,
      updates_collections=None)
  return net


def vis_out_process(inputs, n_output):
  """Process output of vision network.

  Args:
    inputs: tensor (batch_size, kernel_size, kernel_size, n_channels),
        this is the output of a convnet before fully connected layers.
    n_output: int. Number of units in the vision network output.

  Returns:
    net: tensor (batch_size, n_output), output of the processing.
  """
  with tf.variable_scope('vis_output'):
    kernel_size = inputs.shape[1]
    # Last layer should be linear
    net = tf.layers.conv2d(
        inputs,
        n_output, [kernel_size, kernel_size],
        padding='valid',
        name='fc1')
    net = tf.squeeze(net, [1, 2], name='fc1/squeezed')

  return net


def question_network(inputs,
                     batch_size,
                     seq_length,
                     vocabulary_size,
                     embedding_size,
                     num_units,
                     cell_type='gru',
                     bidir=False,
                     train_init=False):
  """Question processing network.

  Args:
    inputs: tensor (n_word, batch_size), input to the question network.
    seq_length: int32 tensor (batch_size,), length of each sentence
    vocabulary_size: int, Size of vocabulary
    embedding_size: int. Embedding size of the words.
    num_units: int. Number of rule network units.
    cell_type: string, type of RNN cell.
    bidir: bool. If true, use bidirectional RNN.
    train_init: bool. If true, train initial state.

  Returns:
    in_rule:
    output: tensor (n_word, num_units), outputs at all time points.
  """
  with tf.variable_scope('rnn_rule'):
    word_embeddings = tf.get_variable('word_embeddings',
                                      [vocabulary_size, embedding_size])

    # embedded_rule is (n_words, embedding_size)
    embedded_rule = tf.nn.embedding_lookup(word_embeddings, inputs)

    if cell_type == 'gru':
      cell_rule = tf.nn.rnn_cell.GRUCell(num_units)
    elif cell_type == 'lstm':
      cell_rule = tf.nn.rnn_cell.LSTMCell(num_units)
    else:
      raise NotImplementedError()

    # batch_size = inputs.get_shape().as_list()[1]
    if bidir:
      # Use bi-directional RNN
      if train_init:
        init_rnn_fw = get_initial_state(cell_rule.state_size, batch_size,
                                        'init_rnn_fw')
        init_rnn_bw = get_initial_state(cell_rule.state_size, batch_size,
                                        'init_rnn_bw')
      else:
        init_rnn_fw = None
        init_rnn_bw = None

      outputs, outputs_final = tf.nn.bidirectional_dynamic_rnn(
          cell_rule,
          cell_rule,
          embedded_rule,
          dtype=tf.float32,
          time_major=True,
          sequence_length=seq_length,
          initial_state_fw=init_rnn_fw,
          initial_state_bw=init_rnn_bw,
      )

      # concat along unit axis
      outputs = tf.concat(outputs, axis=-1)

      if cell_type == 'lstm':
        # Only take the m-state of LSTM
        outputs_final = tf.concat([outputs_final[0][1], outputs_final[1][1]],
                                  -1)
      else:
        outputs_final = tf.concat(outputs_final, -1)

    else:
      # Single-directional RNN
      if train_init:
        init_rnn = get_initial_state(cell_rule.state_size, batch_size,
                                     'init_rnn')
      else:
        init_rnn = None

      outputs, outputs_final = tf.nn.dynamic_rnn(
          cell_rule,
          embedded_rule,
          dtype=tf.float32,
          time_major=True,
          sequence_length=seq_length,
          initial_state=init_rnn,
      )

      if cell_type == 'lstm':
        outputs_final = outputs_final[1]

  return outputs, outputs_final


def like_rnncell(cell):
  """Checks that a given object is an RNNCell by using duck typing."""
  conditions = [hasattr(cell, "output_size"), hasattr(cell, "state_size"),
                hasattr(cell, "zero_state"), callable(cell)]
  return all(conditions)


class MyNetwork(tf.contrib.rnn.RNNCell):
  """Customized network supporting visual attention and read-only memory."""

  def __init__(self,
               cell,
               vis_input_shape,
               input_processing=None,
               output_size=None,
               feature_attn_mechanism=None,
               spatial_attn_mechanism=None,
               memory_mechanism=None,
               out_rule_final=None,
               vis_memory_mechanism=None,
               mode='train',
               hp=None):
    """Create a cell with attention and memory.

    Args:
      cell: RNNCell instance, the core network.
      vis_input_shape: a list of 3 ints, the shape of visual inputs
          (kernel_size, kernel_size, num_channels)
      input_processing: callable. Further process input at each time step.
      output_size: None or int, output_size. If None, then output_size of cell.
      feature_attn_mechanism: None or a BaseAttention instance
      spatial_attn_mechanism: None or a BaseAttention instance
      memory_mechanism: None or a BaseMemory instance
      out_rule_final: final output from RNN over the question
      vis_memory_mechanism: None or a VisMemory instance
      mode: string, train or analyze.
      hp: tensorflow hyperparameter isntance.
    """
    super(MyNetwork, self).__init__()
    if not like_rnncell(cell):
      raise TypeError('The parameter cell is not RNNCell.')
    self._state_is_tuple = True
    self._cell = cell

    self._vis_input_shape = vis_input_shape

    self._input_processing = input_processing
    self._output_size = output_size or self._cell.output_size

    assert feature_attn_mechanism is not None
    self._feature_attn_mechanism = feature_attn_mechanism
    self._feature_attention_size = feature_attn_mechanism.attention_size

    assert spatial_attn_mechanism is not None
    self._spatial_attn_mechanism = spatial_attn_mechanism
    self._spatial_attention_size = spatial_attn_mechanism.attention_size

    self._memory_mechanism = memory_mechanism
    self._out_rule_final = out_rule_final

    assert vis_memory_mechanism is not None
    self._vis_memory_mechanism = vis_memory_mechanism

    self._mode = mode

    self.hp = hp

  @property
  def state_size(self):
    return (self._cell.state_size, self._vis_memory_mechanism.state_size)

  @property
  def output_size(self):
    return self._output_size

  def call(self, inputs, state):
    """Perform a step of the RNN.

    Args:
      inputs: Tensor [batch_size, ...], the input at this time step,
              e.g. (32, 12544)
      state: nested tuple.

    Returns:
      A tuple of cell_output and new_state
    """
    state, vis_mem_state = state

    # self._vis_input_shape - (kernel_size, kernel_size, num_channels)
    num_vis_units = (
        self._vis_input_shape[0] * self._vis_input_shape[1] *
        self._vis_input_shape[2])
    num_input_units = inputs.get_shape().as_list()[-1]

    if num_input_units > num_vis_units:
      inputs_vis, inputs_other = tf.split(
          inputs, [num_vis_units, num_input_units - num_vis_units], axis=-1)
    else:
      assert num_input_units == num_vis_units
      inputs_vis = inputs

    # Reshape to [None, kernel_size, kernel_size, num_channels]
    # e.g. (32, 14, 14, 64)
    inputs_vis = tf.reshape(inputs_vis, [-1] + list(self._vis_input_shape))

    # Flatten and concatenate the state tuple
    # (batch_size, n_rnn) (e.g. (32, 512))
    state_cat = tf.concat(tf.contrib.framework.nest.flatten(state), 1)

    if self.hp.verbal_attention:
      # Retrieve memory
      # alignment [batch_size, n_items]
      # Newest version of TF return a tuple from attention mechanisms
      # (alignment, next_state). Ignore the next_state (which is equal
      # to alignment anyway) for now.
      alignment = self._memory_mechanism(state_cat, None)[0]
      # Weighted-sum over all items
      alignment_exp = tf.expand_dims(alignment, axis=-1)
      memory_retrieved = tf.multiply(self._memory_mechanism.values, alignment_exp)
      memory_retrieved = tf.reduce_sum(memory_retrieved, 1)
      # print("SHAPE OF memory_retrieved: " + str(memory_retrieved.shape))
    else:
      memory_retrieved = self._out_rule_final
      # print("SHAPE OF out_rule_final: " + str(memory_retrieved.shape))

    # Apply feature-based visual attention
    if self.hp.state_dep_feature_attention:
      feature_attn_gen = tf.concat([state_cat, memory_retrieved], 1)
    else:
      feature_attn_gen = memory_retrieved
    with tf.variable_scope('feature_attn1'):
      inputs_attended, feature_attn = self._feature_attn_mechanism(
          inputs_vis, feature_attn_gen)

    if self.hp.feature_attend_to_2conv:
      # Do another round of conv followed by feature attention
      inputs_attended = myconv(inputs_attended, 128, shift_and_scale=False)
      with tf.variable_scope('feature_attn2'):
        inputs_attended, feature_attn = self._feature_attn_mechanism(
            inputs_attended, feature_attn_gen)

    if self.hp.memory_dep_spatial_attention:
      spatial_attn_gen = tf.concat([state_cat, memory_retrieved], 1)
    else:
      spatial_attn_gen = state_cat
    # Apply spatial attention (follows feature attention)
    inputs_attended, spatial_attn = self._spatial_attn_mechanism(
        inputs_attended, spatial_attn_gen)

    # inputs_ctrl = memory_retrieved
    # inputs_ctrl = state_cat
    inputs_ctrl = tf.concat([state_cat, memory_retrieved], 1)
    vis_mem_ctrl = self._vis_memory_mechanism.generate_ctrl(inputs_ctrl)
    # vis_memory_state [batch_size, num_units]
    new_vis_mem_out, new_vis_mem_state = self._vis_memory_mechanism(
        inputs_attended, vis_mem_state, vis_mem_ctrl)

    # Process through a fully connected layer
    in_core_cell = self._input_processing(inputs_attended)

    in_core_cell = tf.concat([in_core_cell, vis_mem_state, memory_retrieved], 1)
    # in_core_cell = tf.concat([in_core_cell, memory_retrieved], 1)
    if num_input_units > num_vis_units:
      in_core_cell = tf.concat([in_core_cell, inputs_other], 1)

    if self.hp.feed_spatial_attn_back:
      # Feed spatial attention back to core
      in_core_cell = tf.concat([in_core_cell, spatial_attn], 1)

    if self.hp.feed_space_sum_to_core:
      # Feed a spatially-summed visual output to core
      inputs_space_sum = tf.reduce_sum(inputs_attended, [1, 2])
      in_core_cell = tf.concat([in_core_cell, inputs_space_sum], 1)

    # Run the rnn cell one step with the attended inputs
    cell_output, new_state = self._cell(in_core_cell, state)

    cell_output = tf.concat([cell_output, new_vis_mem_out], axis=1)

    (clipped_new_state, clipped_new_vis_mem_state), _ = tf.clip_by_global_norm(
        [new_state, new_vis_mem_state], clip_norm=self.hp.rnn_state_norm_clip)

    return cell_output, (clipped_new_state, clipped_new_vis_mem_state)


class Model(object):
  """The model."""

  def __init__(self, hparams, config, mode='train'):
    """The model.

    Args:
      hparams: tensorflow HParams instance of tunable hyperparameters.
      config: dictionary of constant values, not tunable.
      mode: 'train' or 'val'
    """
    self.hp = hparams
    self.config = config
    self.mode = mode

  def build(self, data, batch_size, is_training=True):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
      self._build(data, batch_size, is_training)

  def _build(self, data, batch_size, is_training=True):
    """Build the whole network.

    Args:
      data: dictionary of tensors
      batch_size: int, batch_size.
          batch size currently needs to be explicitly provided.
      is_training: boolean, True if in training mode

    Raises:
      ValueError: if batch size not a multiple of QUESTIONS_PER_IMAGE
    """

    # Network structure constants
    # img_size = get_img_size(self.hp.vis_type)

    n_rnn = self.hp.n_rnn
    n_out_word = self.config['out_voc_size']
    n_out_pnt = self.config['grid_size']**2
    if self.config['pnt_net']:
      n_out = n_out_pnt + n_out_word
    else:
      n_out = n_out_word

    # Question network
    out_rule, out_rule_final = question_network(
        data['question'], batch_size, data['seq_len'],
        self.config['in_voc_size'], self.hp.embedding_size, self.hp.n_rnn_rule,
        self.hp.rnn_rule_type, self.hp.rnn_rule_bidir, self.hp.train_init)

    # Vision network
    img_size, n_channels = data['image'].get_shape().as_list()[-2:]
    # image input [n_epoch, batch_size, img_size, img_size, n_channels]
    if self.config['dataset'] == 'clevr':
      if batch_size % self.config['q_per_image'] != 0:
        raise ValueError('Batch size has to be multiple of QUESTIONS_PER_IMAGE')
      batch_size_img = int(batch_size / self.config['q_per_image'])

    if not self.hp.use_vgg_pretrain:
      # Reshape it to [n_epoch*batch_size, img_size, img_size, n_channels]
      in_imgs_flat = tf.reshape(data['image'],
                                [-1, img_size, img_size, n_channels])

      # Common preprocessing
      in_imgs_preprocessed = simple_vgg_preprocessing(self.hp, in_imgs_flat)
      # vision_network = myconv
      with tf.variable_scope('vision'):
        out_vis = myconv(in_imgs_preprocessed, 32, shift_and_scale=True)
        out_vis = myconv(out_vis, 64, shift_and_scale=True)
        if self.hp.feature_attend_to_2conv:
          out_vis = myconv(out_vis, 64, shift_and_scale=False)
        else:
          out_vis = myconv(out_vis, 64, shift_and_scale=True)
          out_vis = myconv(out_vis, 128, shift_and_scale=False)
    else:
      out_vis = data['image']

    # (kernel_size, kernel_size, n_ch)
    out_vis_shape = out_vis.get_shape().as_list()[-3:]
    kernel_size = out_vis_shape[-2]
    out_vis_channels = out_vis_shape[-1]

    # Reshape to [n_epoch, batch_size, kernel_size, kernel_size, n_channels]
    if self.config['dataset'] == 'clevr':
      # Repeat the images for all questions for this image
      out_vis = tf.reshape(out_vis, [-1, batch_size_img] + out_vis_shape)
      out_vis = tf_repeat(out_vis, [1, self.config['q_per_image'], 1, 1, 1])
    else:
      out_vis = tf.reshape(out_vis, [-1, batch_size] + out_vis_shape)

    # Optional repeat in time
    if self.hp.n_time_repeat:
      in_rnn = tf_repeat(out_vis, [self.hp.n_time_repeat, 1, 1, 1, 1])

      in_new_epoch = [1] + [0] * (self.hp.n_time_repeat - 1)
      in_new_epoch = tf.constant(in_new_epoch, dtype=tf.float32)
      in_new_epoch = tf.expand_dims(in_new_epoch, 1)
      in_new_epoch = tf.tile(in_new_epoch, tf.shape(out_vis)[:2])
      in_new_epoch = tf.expand_dims(in_new_epoch, -1)
    else:
      in_rnn = out_vis

    in_rnn = tf.reshape(in_rnn,
                        [-1, batch_size, (kernel_size**2) * out_vis_shape[-1]])

    if self.hp.signal_new_epoch and self.hp.n_time_repeat > 1:
      in_rnn = tf.concat([in_rnn, in_new_epoch], axis=-1)

    # Core network
    with tf.variable_scope('core'):
      if self.hp.rnn_type == 'lstm':
        core_cell = tf.nn.rnn_cell.LSTMCell(n_rnn)
      elif self.hp.rnn_type == 'gru':
        # so far works a bit better than LSTM
        core_cell = tf.nn.rnn_cell.GRUCell(n_rnn)
      elif self.hp.rnn_type == 'rnn':
        flin = lambda x: x
        core_cell = tf.nn.rnn_cell.BasicRNNCell(n_rnn, activation=flin)
      else:
        raise NotImplementedError('Unknown rnn_type.')

      if self.hp.feature_attention:
        # Feature attention size = n_channels in last layer of vision net
        attention_size = out_vis_channels
        # TODO(gryang): change the no longer used API
        feature_attn_mechanism = ops.FeatureAttention(
            attention_size, use_mlp=self.hp.feature_attention_use_mlp)
      else:
        attention_size = 0
        feature_attn_mechanism = ops.BaseAttention(attention_size)

      if self.hp.spatial_attention:
        attention_size = kernel_size**2
        spatial_attn_mechanism = ops.SpatialAttention(attention_size)
      else:
        attention_size = 0
        spatial_attn_mechanism = ops.BaseAttention(attention_size)

      if not self.hp.verbal_attention:
        memory_mechanism = None
      else:
        # out_rule [max_word, batch_size, num_units]
        # memory [batch_size, max_word, num_units], e.g. (32, 25, 256)
        memory = tf.transpose(out_rule, [1, 0, 2])
        memory_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            self.hp.memory_query_size, memory)

      if self.hp.vis_memory_maps == 0:
        vis_memory_mechanism = ops.BaseVisMemory(0)
      else:
        vis_memory_mechanism = ops.VisMemory(
            (7, 7),  # TODO(gryang): infer this value
            in_channels=128,
            out_channels=1,
            n_maps=self.hp.vis_memory_maps,
            forget_bias=self.hp.controller_gru_init_factor
        )

      output_size = n_rnn + vis_memory_mechanism.output_size

      output_size_exp = output_size

      cell = MyNetwork(
          core_cell,
          out_vis_shape,
          input_processing=lambda x: vis_out_process(x, self.hp.n_out_vis),
          output_size=output_size_exp,
          feature_attn_mechanism=feature_attn_mechanism,
          spatial_attn_mechanism=spatial_attn_mechanism,
          memory_mechanism=memory_mechanism,
          out_rule_final=out_rule_final,
          vis_memory_mechanism=vis_memory_mechanism,
          mode=self.mode,
          hp=self.hp)

      if self.hp.train_init:
        initial_state = get_initial_state(cell.state_size, batch_size,
                                          'init_rnn')
      else:
        initial_state = None

      # time major has to be True
      # out_net [n_time, batch_size, output_size_exp]
      out_net, out_state = tf.nn.dynamic_rnn(
          cell,
          in_rnn,
          initial_state=initial_state,
          dtype=tf.float32,
          time_major=True)

      state_norm = tf.global_norm(tf.contrib.framework.nest.flatten(out_state))
      tf.summary.scalar('final_ctrl_state_norm', state_norm)

    if self.hp.n_time_repeat and self.mode != 'analyze':
      # Reshape to [n_epoch, n_time_repeat, batch_size, n_rnn]
      # Only keep the last time of each epoch
      out_net = tf.reshape(out_net,
                           (-1, self.hp.n_time_repeat, batch_size, output_size))
      out_net = out_net[:, -1, :, :]

    # Reshape to (n_epoch * batch_size, n_rnn)
    out_net = tf.reshape(out_net, (-1, output_size))

    if self.hp.vis_memory_maps > 0:
      out_net, out_pnt_from_memory = tf.split(
          out_net, [n_rnn, vis_memory_mechanism.output_size], axis=1)

    # Fully connected linear readout
    with tf.variable_scope('output'):
      if self.hp.final_mlp:
        out_net = tf.contrib.layers.fully_connected(out_net, 256, scope='mlp1')
        out_net = tf.contrib.layers.fully_connected(out_net, 256, scope='mlp2')
        # TODO(gryang): allow setting is_training
        out_net = tf.contrib.layers.dropout(
            out_net, keep_prob=0.5, is_training=is_training)
        out_net = tf.contrib.layers.fully_connected(
            out_net, n_out, activation_fn=None, scope='mlp3')
      else:
        out_net = tf.contrib.layers.fully_connected(out_net, n_out, activation_fn=None)

    if self.config['pnt_net']:
      out_pnt_net, out_word_net = tf.split(out_net, [n_out_pnt, n_out_word], 1)
    else:
      out_word_net = out_net

    if self.config['pnt_net']:
      if self.hp.vis_memory_maps > 0:
        if self.hp.only_vis_to_pnt:
          out_pnt_net = out_pnt_from_memory
        else:
          out_pnt_net += out_pnt_from_memory

      # Population vector readout
      out_pnt_xy_net = tf.matmul(
          tf.nn.softmax(out_pnt_net), self.config['prefs'])
      self.out_pnt_xy_net = out_pnt_xy_net

    self.img_size = img_size

    if self.config['pnt_net']:
      self.out_pnt_net = out_pnt_net
    self.out_word_net = out_word_net

    ################### Get loss of the network ###############################
    if self.config['pnt_net']:
      # Loss of pointing
      n_out_pnt = self.out_pnt_net.get_shape().as_list()[1]
      # out_pnt shape (batch_size*n_epoch, n_out_pnt)
      # mask_pnt shape (batch_size*n_epoch)

      loss_pnt = tf.nn.softmax_cross_entropy_with_logits(
          labels=data['point'], logits=self.out_pnt_net)
      loss_pnt = tf.reduce_mean(tf.multiply(loss_pnt, data['mask_point']))

      # Accuracy
      diff_xy = tf.reduce_sum(
          tf.square(self.out_pnt_xy_net - data['point_xy']), axis=-1)
      diff_xy = tf.boolean_mask(diff_xy, tf.cast(data['mask_point'], tf.bool))
      diff_threshold = 0.15**2
      acc_pnt = tf.less(diff_xy, diff_threshold)

    # Loss of word output
    # n_out_word = self.out_word_net.get_shape().as_list()[1]
    # out_word shape (batch_size*n_epoch, n_out_word)
    # mask_word shape
    acc_word = tf.equal(data['answer'], tf.argmax(self.out_word_net, -1))
    loss_word = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=data['answer'], logits=self.out_word_net)

    if self.config['pnt_net']:
      acc_word = tf.boolean_mask(acc_word, tf.cast(data['mask_answer'],
                                                   tf.bool))
      loss_word = tf.multiply(loss_word, data['mask_answer'])

    loss_word = tf.reduce_mean(loss_word)

    # Combine word and pnt
    if self.config['pnt_net']:
      loss = loss_pnt + loss_word
      # Concat along time axis
      acc = tf.concat([acc_pnt, acc_word], axis=0)
    else:
      loss = loss_word
      acc = acc_word

    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    # Apply L2 regularization
    if self.hp.l2_weight > 0:
      regularizer = tf.contrib.layers.l2_regularizer(self.hp.l2_weight)
      # Use a hack to apply to all weights
      weights = [
          v for v in tf.trainable_variables()
          if 'weight' in v.name or 'kernel' in v.name
      ]
      loss += tf.contrib.layers.apply_regularization(regularizer, weights)

    self.loss = loss
    self.acc = acc
    tf.summary.scalar('summary/train/loss_avg', loss)
    tf.summary.scalar('summary/train/accuracy_avg', acc)

    ######################### Get one training step ###########################
    learning_rate = tf.train.exponential_decay(
        self.hp.learning_rate, tf.train.get_global_step(),
        decay_steps=1000000.0/batch_size, decay_rate=self.hp.lr_decay,
        staircase=False)
    tf.summary.scalar('summary/train/learning_rate', learning_rate)
    if self.hp.optimizer == 'adam':
      opt = tf.train.AdamOptimizer(
          learning_rate=learning_rate,
          beta1=(1 - self.hp.adam_beta1),
          beta2=(1 - self.hp.adam_beta2),
          epsilon=self.hp.adam_epsilon
      )
    elif self.hp.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer
      opt = optimizer(learning_rate=learning_rate)
    elif self.hp.optimizer == 'momentum':
      opt = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=self.hp.momentum,
          use_nesterov=self.hp.nesterov)
    else:
      raise NotImplementedError()

    vars_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # Always exclude vgg if exist
    self.var_pretrained = [v for v in vars_train if 'vgg' in v.name]
    if self.var_pretrained:
      print('Exclude training:')
      print(self.var_pretrained)
    vars_train = [v for v in vars_train if v not in self.var_pretrained]

    # Compute the gradients for a list of variables.
    grads_and_vars = opt.compute_gradients(self.loss, vars_train)

    print('Warning: following variables have None gradients.')

    all_grads = [g for g, _ in grads_and_vars]

    self.grads_and_vars = list()
    self.grads_and_vars_test = list()
    capped_grads_and_vars = list()
    capped_all_grads, global_norm = tf.clip_by_global_norm(
        all_grads, clip_norm=self.hp.grad_clip)
    tf.summary.scalar('grad_global_norm', global_norm)
    for i, (grad, var) in enumerate(grads_and_vars):
      if grad is None:
        print(var)
      else:
        capped_grads_and_vars.append((capped_all_grads[i], var))

    # Ask the optimizer to apply the capped gradients.
    self.train_step = opt.apply_gradients(
        capped_grads_and_vars, global_step=tf.train.get_global_step())

    return self.train_step, self.acc

  def initialize(self, sess):
    """Network initialization."""
    sess.run(tf.global_variables_initializer())  # initialize all variables
