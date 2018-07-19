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

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread
import tensorflow as tf
import time

import clevr.constants as const


def load_questions(data_dir, data_type):
  """Load questions information from directory.

  Args:
    data_dir: str, data directory.
    data_type: str, one of 'train', 'val', 'test'

  Returns:
    questions: list of dictionaries. Each dictionary contains information
    about a question, its corresponding image and answer.
  """
  filename = data_dir + '/questions/CLEVR_' + data_type + '_questions.json'
  with open(filename) as data_file:
    return json.load(data_file)['questions']


def process_question(question):
  """Process question.

  Keep punctuations and use lower cases.

  Args:
    question: a string of a sentence

  Returns:
    question_words: a list of words in this sentence
  """
  question_words = re.findall(r"[\w']+|[.,!?;]", question)
  return [w.encode('utf-8').lower() for w in question_words]


def convert_to_smaller_images(data_dir, new_data_dir, data_type):
  """Convert original CLEVR images to smaller images in tensorflow."""
  img_dir = os.path.join(data_dir, 'images', data_type)
  new_img_dir = os.path.join(new_data_dir, 'images', data_type)
  tf.gfile.MakeDirs(new_img_dir)

  # Process images in tensorflow
  input_image = tf.placeholder(tf.uint8, shape=(320, 460, 3))
  image = tf.cast(input_image, tf.float32)
  image = tf.reshape(image, [1, 320, 460, 3])
  image = tf.image.resize_area(image, [112, 112])

  with tf.Session() as sess:
    for i, image_file in enumerate(os.listdir(img_dir)):
      if i % 1000 == 0:
        print('Processing image ' + str(i))
      input_image0_ = imread(os.path.join(img_dir, image_file))
      input_image_ = input_image0_[:, 10:-10, :3]

      image_ = sess.run(image, feed_dict={input_image: input_image_})
      cv2.imwrite(os.path.join(new_img_dir, image_file), image_[0])

  # Example comparison
  plt.figure()
  plt.imshow(input_image0_.astype(np.uint8))
  plt.grid(False)

  plt.figure()
  plt.imshow(image_[0].astype(np.uint8))
  plt.grid(False)


def convert_to_tfrecord(data_dir, tfrecord_dir, data_type,
                        questions=None):
  """Convert images to TFRecord format.

  Convert the image file name, question, answer triplet into TFRecord files.

  Args:
    data_dir: str, original data directory.
    tfrecord_dir: str, new data directory.
    data_type: str, one of 'train', 'val', 'test'
    questions: dictionary of questions and other info, if not None
  """
  tf.reset_default_graph()

  q_per_img = const.QUESTIONS_PER_IMAGE
  img_dir = os.path.join(data_dir, 'images', data_type)

  tf.gfile.MakeDirs(tfrecord_dir)

  if questions is None:
    questions = load_questions(data_dir, data_type)

  # Create a dictionary with image_filename as keys
  image_dict = defaultdict(lambda: defaultdict(list))

  permutation = np.random.permutation(len(questions))
  for ind in permutation:
    q = questions[ind]

    question_words = process_question(q['question'])
    seq_len = len(question_words)
    question_words = [const.INPUTVOCABULARY.index(w) for w in question_words]
    question_words += [0] * (const.MAXSEQLENGTH - seq_len)  # pad

    dict_tmp = image_dict[q['image_filename']]

    if data_type != 'test':
      answer = const.OUTPUTVOCABULARY.index(q['answer'])
      dict_tmp['answer'].append(answer)
      dict_tmp['question_family_index'].append(q['question_family_index'])
    else:
      # Put some garbage since it is not going to be used.
      answer = const.OUTPUTVOCABULARY.index('yes')
      dict_tmp['answer'].append(answer)
      dict_tmp['question_family_index'].append(43)

    dict_tmp['question'].extend(question_words)
    dict_tmp['seq_len'].append(seq_len)
    dict_tmp['image_index'].append(q['image_index'])
    dict_tmp['question_index'].append(q['question_index'])

  # Set up the image processing here
  input_image = tf.placeholder(tf.uint8, shape=(320, 460, 3))
  output_image = tf.cast(input_image, tf.float32)
  output_image = tf.reshape(output_image, [1, 320, 460, 3])

  output_image = tf.image.resize_area(output_image, [128, 128])

  # TFRecord writer
  shard_size = 5000

  # Variables stored as integers
  int64_keys = ['question', 'answer', 'seq_len', 'image_index',
                'question_index', 'question_family_index']

  with tf.Session() as sess:
    counter = 0
    writer = None

    for image_filename, qa_dict in image_dict.iteritems():
      if counter % 1000 == 0:
        print('Processing image {:d}'.format(counter))
      if counter % shard_size == 0:
        if counter > 0:
          writer.close()
        shard_name = '{:05d}'.format(int(counter / shard_size))
        record_name = 'CLEVR_' + data_type + '_' + shard_name + '.tfrecords'
        record_name = os.path.join(tfrecord_dir, record_name)
        writer = tf.python_io.TFRecordWriter(record_name)

      # Load image and convert to smaller images
      image = imread(os.path.join(img_dir, image_filename))
      image = image[:, 10:-10, :3]
      image = sess.run(output_image, feed_dict={input_image: image})
      image = image[0]

      if len(qa_dict['answer']) != 10:
        print(str(len(qa_dict['answer'])) + "  " + str(qa_dict))

      # For a small number of images, the number of question is less than
      # QUESTIONS_PER_IMAGE. Then we repeat the previous questions.
      if len(qa_dict['answer']) < q_per_img:
        for key in int64_keys:
          qa_dict[key] *= q_per_img  # repeat the list

      # Then we keep QUESTIONS_PER_IMAGE
      for key in int64_keys:
        if key == 'question':
          qa_dict[key] = qa_dict[key][:const.MAXSEQLENGTH * q_per_img]
        else:
          qa_dict[key] = qa_dict[key][:q_per_img]

      # Convert data to tf.train format
      feature = dict()
      image_data = tf.train.BytesList(value=[image.tobytes()])
      feature['image'] = tf.train.Feature(bytes_list=image_data)
      for key in int64_keys:
        int64_data = tf.train.Int64List(value=qa_dict[key])
        feature[key] = tf.train.Feature(int64_list=int64_data)

      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())

      counter += 1

    writer.close()


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


def visualize_dataset(tfrecord_dir, data_type):
  """Visually inspect the dataset."""
  dataset = dataset_from_tfrecord(tfrecord_dir, data_type, 10)
  iterator = dataset.make_one_shot_iterator()
  data = iterator.get_next()

  with tf.Session() as sess:
    # get each element of the training dataset until the end is reached
    start_time = time.time()
    data_value = sess.run(data)
    print(time.time()-start_time)

  image = data_value['image'][0]
  plt.figure()
  plt.imshow(image.astype(np.uint8))
  plt.grid(False)

  for i in range(const.QUESTIONS_PER_IMAGE):
    question = data_value['question'][0, i]
    seq_len = data_value['seq_len'][0, i]
    question = ' '.join(
        [const.INPUTVOCABULARY[question[j]] for j in range(seq_len)])
    print(question)
    print(const.OUTPUTVOCABULARY[data_value['answer'][0, i]])


def main(_):
  data_dir = '/tmp/clevr/CLEVR_v1.0'
  tfrecord_dir = '/tmp/clevr/tf_record'
  data_type = 'test'
  convert_to_tfrecord(data_dir, tfrecord_dir, data_type)
  # visualize_dataset(tfrecord_dir, data_type)


if __name__ == '__main__':
  tf.app.run(main)
