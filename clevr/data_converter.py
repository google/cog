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
import clevr.data_generator as dg

tf.app.flags.DEFINE_string(
    'command', 'convert',
    'Whether to "convert" or "visualize" the clevr data. '
    'Visualization shows the first image and prints its questions.')
tf.app.flags.DEFINE_string(
    'raw_clevr_dir', None,
    'Directory that contains the unzipped clevr data, CLEVR_V1.0.')
tf.app.flags.DEFINE_string(
    'tfrecord_dir', None,
    'Directory to write tf records containing CLEVR data to.')
tf.app.flags.DEFINE_string(
    'data_type', 'all',
    'The type of data to convert. One of "train", "val", "test", "all".')


FLAGS = tf.app.flags.FLAGS


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
    print("Loading questions")
    questions = load_questions(data_dir, data_type)

  # Create a dictionary with image_filename as keys
  image_dict = defaultdict(lambda: defaultdict(list))

  print("Processing questions")
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

      # A few images have less than 10 questions. No images have more.
      #if len(qa_dict['answer']) != 10:
      #  print(str(len(qa_dict['answer'])) + "  " + str(qa_dict))

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


def visualize_dataset(tfrecord_dir, data_type):
  """Visually inspect the dataset."""
  dataset = dg.dataset_from_tfrecord(tfrecord_dir, data_type, 10)
  iterator = dataset.make_one_shot_iterator()
  data = iterator.get_next()

  with tf.Session() as sess:
    start_time = time.time()
    data_value = sess.run(data)
    print(time.time() - start_time)

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

  plt.show()


def main(_):
  tfrecord_dir = FLAGS.tfrecord_dir
  assert tfrecord_dir is not None, "Please specify --tfrecord_dir"
  data_type = FLAGS.data_type
  assert data_type in ['train', 'val', 'test', 'all'], (
      "--data_type must be one of: 'train', 'val', 'test', 'all'")
  command = FLAGS.command
  assert command in ['convert', 'visualize'], (
      "--command must be one of: 'convert', 'visualize'")

  if command == 'convert':
    data_dir = FLAGS.raw_clevr_dir
    assert data_dir is not None, "Please specify --raw_clevr_dir"
    convert_to_tfrecord(data_dir, tfrecord_dir, data_type)
    if data_type == 'all':
      convert_to_tfrecord(data_dir, tfrecord_dir, 'train')
      convert_to_tfrecord(data_dir, tfrecord_dir, 'val')
      convert_to_tfrecord(data_dir, tfrecord_dir, 'test')
  elif command == 'visualize':
    visualize_dataset(tfrecord_dir, data_type)


if __name__ == '__main__':
  tf.app.run(main)
