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

"""Data generator for CLEVR data.

Constructs a queue that fills with example of images and randomly pulled
questions and answers. Each time an image is chosen, a random question is picked
from all possible questions for that image.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import json
import os
import re
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread
import tensorflow as tf

import clevr.constants as const


def dataset_from_tfrecord(tfrecord_dir, data_type, batch_size,
                          is_training=True):
  """Generate a tensorflow dataset from TFRecord files."""
  if is_training:
    int64_keys = ['answer', 'seq_len']
  else:
    int64_keys = ['answer', 'seq_len', 'image_index', 'question_index',
                  'question_family_index']

  def _parser(record):
    """Parse the TFRecord."""
    keys_to_features = {
        'image': tf.FixedLenFeature([1], tf.string),
        'question': tf.FixedLenFeature(
            [const.QUESTIONS_PER_IMAGE*const.MAXSEQLENGTH], tf.int64),
    }
    for key in int64_keys:
      keys_to_features[key] = tf.FixedLenFeature(
          [const.QUESTIONS_PER_IMAGE], tf.int64)

    parsed = tf.parse_single_example(record, keys_to_features)
    # The decoding type must match the encoding type in convert_to_tfrecord
    parsed['image'] = tf.decode_raw(parsed['image'], tf.float32)
    parsed['image'] = tf.reshape(parsed['image'], [128, 128, 3])
    parsed['question'] = tf.reshape(
        parsed['question'], [const.QUESTIONS_PER_IMAGE, const.MAXSEQLENGTH])
    return parsed

  if data_type == 'train':
    n_shard = 14
  elif data_type in ('val', 'test'):
    n_shard = 3
    
  # Use for local testing without all data
  # n_shard = 1

  filenames = list()
  for i in range(n_shard):
    # This name must match the saving name in convert_to_tfrecord
    filename = 'CLEVR_' + data_type + '_{:05d}'.format(i) + '.tfrecords'
    filenames.append(os.path.join(tfrecord_dir, filename))

  # create TensorFlow Dataset objects
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_parser)
  if is_training:
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=2)

  return dataset


def preprocess_data(data, batch_size_img, crop_mode=None, is_training=True):
  """Preprocessing to proper shapes."""
  # data['image'] had shape [batch_size_img, 128, 128, 3]
  if is_training:
    if crop_mode == '112':
      data['image'] = tf.random_crop(data['image'],
                                     [batch_size_img, 112, 112, 3])
    elif crop_mode == '128':
      # Note that relation_net used resize_image_with_crop_or_pad
      data['image'] = tf.image.resize_images(data['image'], [136, 136])
      data['image'] = tf.random_crop(data['image'],
                                     [batch_size_img, 128, 128, 3])

    int64_keys = ['answer', 'seq_len']
  else:
    if crop_mode == '112':
      data['image'] = tf.image.resize_images(data['image'], [112, 112])

    int64_keys = ['answer', 'seq_len', 'image_index', 'question_index',
                  'question_family_index']

  # Reshape to [max_seq_length, batch_size]
  data['question'] = tf.reshape(data['question'], (-1, const.MAXSEQLENGTH))
  data['question'] = tf.transpose(data['question'], perm=[1, 0])
  for key in int64_keys:
    data[key] = tf.reshape(data[key], [-1])  # Reshape to [batch_size]
  return data


def data_from_tfrecord(tfrecord_dir, data_type, batch_size, hparams,
                       is_training=True):
  """Generate a data iterator from TFRecord files."""
  # TODO(gryang): get rid of the use of hparams
  batch_size_img = int(batch_size/const.QUESTIONS_PER_IMAGE)
  dataset = dataset_from_tfrecord(tfrecord_dir, data_type, batch_size_img,
                                  is_training=is_training)

  if hparams.use_vgg_pretrain:
    crop_mode = None
  else:
    if hparams.use_img_size_128:
      crop_mode = '128'
    else:
      crop_mode = '112'

  iterator = dataset.make_one_shot_iterator()
  data = iterator.get_next()
  data = preprocess_data(data, batch_size_img, crop_mode=crop_mode,
                         is_training=is_training)
  return data
