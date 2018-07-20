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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import itertools
import os
import random
import sys
import time
import tensorflow as tf

from tensorflow.core.framework import summary_pb2
import clevr.data_generator as clevr
import clevr.constants as constants
import model.network as network

tf.app.flags.DEFINE_string(
    'hparams', '', 'Comma separated list of name=value hyperparameter pairs.')

tf.app.flags.DEFINE_string(
    'clevr_test_output', None,
    'If set to a path, generates and write test answers for CLEVR dataset '
    'into <clevr_test_output>.txt and <clevr_test_output>_with_ids.txt. '
    'Test tf records are expected to be in <data_dir>. '
    'Latest checkpoint is loaded from <train_dir>/checkpoints.')

# Training parameters
# task_family flag inherited from task_bank.py
tf.app.flags.DEFINE_integer('num_steps', 100000, 'number of training steps')
tf.app.flags.DEFINE_integer('display_step', 10, 'display every # steps')
tf.app.flags.DEFINE_integer('summary_step', 500, 'log summaries every # steps')
tf.app.flags.DEFINE_integer('batch_size', 250, 'batch size for training')

# Logistics
tf.app.flags.DEFINE_string('data_dir', '/tmp/clevr/tfrecord',
                           'Directory of training and validation data.')
tf.app.flags.DEFINE_string('train_dir', '/tmp/clevr/train',
                           'Directory to put the training logs.')
tf.app.flags.DEFINE_boolean('report_param_stat', False,
                            'If true, report parameter statistics')
FLAGS = tf.app.flags.FLAGS


def get_default_hparams_dict():
  return dict(
      # learning rate decay: lr multiplier per 1M examples
      # value of 0.966 tranlates to 0.5 per 20M examples
      lr_decay=1.0,
      # learning rate
      learning_rate=0.0005,
      # gradient clipping
      grad_clip=80.,
      # clipping value for rnn state norm
      rnn_state_norm_clip=5000.,
      # number of core recurrent units
      n_rnn=512,
      # type of core rnn
      rnn_type='gru',
      # whether to use 128 as input image size
      use_img_size_128=False,
      # number of vision network output
      n_out_vis=128,
      # type of visual network
      use_vgg_pretrain=False,
      # number of units for question network
      n_rnn_rule=128,
      # type of rule rnn
      rnn_rule_type='lstm',
      # embedding size for question words
      embedding_size=64,
      # train initial state or not
      train_init=True,
      # beta1 for AdamOptimizer
      adam_beta1=0.1,
      # beta2 for AdamOptimizer
      adam_beta2=0.0001,
      # epsilon for AdamOptimizer
      adam_epsilon=1e-8,
      # rule network bidirectional or not
      rnn_rule_bidir=True,
      # number of time point to repeat for each epoch
      n_time_repeat=8,
      # build network with visual feature attention or not
      feature_attention=True,
      # state-dependent attention?
      state_dep_feature_attention=False,
      # whether use a MLP to generation feature attention
      feature_attention_use_mlp=False,
      # whether apply feature attention to the second-to-last conv layer
      feature_attend_to_2conv=True,
      # whether to feed a spatially-summed visual input to core
      feed_space_sum_to_core=True,
      # build network with visual spatial attention or not
      spatial_attention=True,
      # whether spatial attention depends on retrieved memory
      memory_dep_spatial_attention=False,
      # whether spatial attention is fed back to controller
      feed_spatial_attn_back=True,
      # how are rule outputs used as memory
      verbal_attention=True,
      # size of the query for rule memory
      memory_query_size=128,
      # number of maps in visual spatial memory
      vis_memory_maps=0,
      # only use visual memory to point short-cut
      only_vis_to_pnt=True,
      # optimizer to use
      optimizer='adam',
      # number of epochs each trial
      n_epoch=1,
      # signal new epoch
      signal_new_epoch=False,
      # final readout using a MLP
      final_mlp=False,
      # L2 regularization, consider a value between 1e-4 and 1e-5
      l2_weight=2*1e-5,

      # value 'factor' param to variance_scaling_initializer used as
      # controller GRU kernel initializer
      controller_gru_init_factor=1.0,

      # normalize images mean 0/std 1
      normalize_images=False,
  )


def run_test(train_dir, test_output_dir, hparams):
  print("\nRUNNING MODEL ON THE TEST SET\n")

  tf.reset_default_graph()

  ######################### Read the data ################################
  data = clevr.data_from_tfrecord(FLAGS.data_dir, 'test', batch_size=250,
                                  hparams=hparams, is_training=False)

  ######################### Build the network ################################
  tf.train.get_or_create_global_step()
  model = network.Model(hparams, constants.config)
  model.build(data, batch_size=FLAGS.batch_size, is_training=True)
  model_answers = tf.argmax(model.out_word_net, -1)
  true_answers = data['answer']
  i_idx = data['image_index']
  q_idx = data['question_index']

  saver = tf.train.Saver()

  ######################### Build the network ################################
  checkpoint_path = train_dir + '/checkpoints'
  with tf.Session() as sess:
    cpkt_path = tf.train.latest_checkpoint(checkpoint_path)
    if cpkt_path is not None:
      print("Restoring model from: " + cpkt_path)
      saver.restore(sess, cpkt_path)
      print("Done restoring model")
    else:
      raise RuntimeError("Failed to find latest checkpoint in: " + 
                         checkpoint_path)

    global_step = sess.run(tf.train.get_global_step())
    print("Global step value loaded from checkpoint: " + str(global_step))
    ans = {}
    for i in itertools.count():
      if i % 100 == 0:
        print('Processing batch', i)

      try:
        m_ans, q_idx_ = sess.run([model_answers, q_idx])
        for m, q in zip(m_ans, q_idx_):
          ans[q] = constants.OUTPUTVOCABULARY[m]
      except tf.errors.OutOfRangeError as e:
        print("Done processing test dataset. Saving results")
        break

    items = ans.items()
    items.sort()

    with_ids = os.path.join(test_output_dir, 'clevr_test_with_ids.txt')
    without_ids = os.path.join(test_output_dir, 'clevr_test.txt')

    with tf.gfile.FastGFile(with_ids, 'w') as f:
      f.write('\n'.join(map(lambda x: '%d %s' % x, items)))

    with tf.gfile.FastGFile(without_ids, 'w') as f:
      f.write('\n'.join(map(lambda x: str(x[1]), items)))

    print("Results written to " + with_ids + " and " + without_ids)


def evaluate(sess, model_val, n_batches, test_writer, global_step, trial,
             tuner, acc_train):
  t_start = time.time()
  acc_val_ = 0
  loss_val_ = 0
  for _ in range(n_batches):
    acc_tmp, loss_tmp, = sess.run([model_val.acc, model_val.loss])
    acc_val_ += acc_tmp
    loss_val_ += loss_tmp
  acc_val_ /= n_batches
  loss_val_ /= n_batches

  # Write summaries
  vals = [summary_pb2.Summary.Value(tag='summary/accuracy_val',
                                    simple_value=acc_val_),
          summary_pb2.Summary.Value(tag='summary/loss_val',
                                    simple_value=loss_val_)]
  test_writer.add_summary(summary_pb2.Summary(value=vals), trial)

  print('Step {:d} Trial {:d}'.format(global_step, trial))
  print('Evaluation took: {:0.2f}s'.format(time.time() - t_start))
  print('Accuracy training: {:0.4f}'.format(acc_train))
  print('Accuracy validation: {:0.4f}'.format(acc_val_))
  sys.stdout.flush()

  # Report the test set precision as the measure
  if tuner:
    tuner.report_measure(acc_val_, trial)

  return acc_val_


def run_training(hparams, train_dir, tuner):
  """Train.

  Args:
    hparams: A HParam object with the hyperparameters to use.
    train_dir: Path of a directory where to log training events.
    tuner: Optional hyperparameter Tuner object.  TODO(iga): Remove
  """
  if not FLAGS.train_dir:
    raise ValueError('traning directory is not provided.')
  if not FLAGS.data_dir:
    raise ValueError('data directory is not provided.')

  if not tf.gfile.Exists(train_dir):
    tf.gfile.MakeDirs(train_dir)

  print('Hyperparameters:')
  for key, val in sorted(hparams.values().iteritems()):
    print(key, val)

  with tf.gfile.FastGFile(os.path.join(train_dir, 'hparams'), 'w') as f:
    json.dump(hparams.to_json(), f)

  
  tf.reset_default_graph()

  ######################### Tasks to train ###################################
  data_train = clevr.data_from_tfrecord(
      FLAGS.data_dir, 'train', FLAGS.batch_size, hparams)
  data_val = clevr.data_from_tfrecord(
      FLAGS.data_dir, 'val', FLAGS.batch_size, hparams)

  ######################### Build the network ################################
  tf.train.get_or_create_global_step()
  model_train = network.Model(hparams, constants.config)
  model_train.build(data_train, FLAGS.batch_size, is_training=True)

  merged = tf.summary.merge_all()
  test_writer = tf.summary.FileWriter(train_dir + '/tb', flush_secs=120)

  # Report parameter statistics
  if FLAGS.report_param_stat:
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
        TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    print('total_params: {:d}'.format(param_stats.total_parameters))

    PARAM_LIMIT = 2905217
    if param_stats.total_parameters > 1.1 * PARAM_LIMIT:
      raise tf.errors.ResourceExhaustedError(None, None,
          "Hyperparams resulted in too many params: %d" %
          param_stats.total_parameters)

  model_val = network.Model(hparams, constants.config)
  # TODO(gryang): Setting is_training=False doesn't seem to work correctly
  model_val.build(data_val, FLAGS.batch_size, is_training=True)
  saver = tf.train.Saver()


  ########################## Train the network ###############################
  print("Train dir: " + train_dir)
  checkpoint_path = train_dir + '/checkpoints'
  acc_train_ = 0
  best_acc_val_ = 0
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    cpkt_path = tf.train.latest_checkpoint(checkpoint_path)
    if cpkt_path is not None:
      print("Restoring model from: " + cpkt_path)
      saver.restore(sess, cpkt_path)
      print("Done restoring model")
    else:
      print("Did not find checkpoint at: " + checkpoint_path)

    global_step = sess.run(tf.train.get_global_step())
    print("Initial global step value: " + str(global_step))
    print("Running until global step is: " + str(FLAGS.num_steps))
    sys.stdout.flush()
    trial = global_step * FLAGS.batch_size
    while global_step <= FLAGS.num_steps:
      trial = global_step * FLAGS.batch_size
      try:
        # Evaluation
        if global_step > 0 and global_step % FLAGS.display_step == 0:
          acc_val_ = evaluate(sess, model_val, n_batches=300,
                              test_writer=test_writer,
                              global_step=global_step,
                              trial=trial, tuner=tuner,
                              acc_train=acc_train_)

          if acc_val_ > best_acc_val_:
            best_acc_val_ = acc_val_
            save_path = saver.save(sess, train_dir + '/checkpoints/model.ckpt')
            print('Model saved in file {:s}'.format(save_path))

        if global_step > 0 and global_step % FLAGS.summary_step == 0:
          global_step, summary, _, acc_train_ = sess.run(
              [tf.train.get_global_step(),
               merged,
               model_train.train_step,
               model_train.acc])
          test_writer.add_summary(summary, trial)
        else:
          # Training
          global_step, _, acc_train_ = sess.run(
              [tf.train.get_global_step(),
               model_train.train_step,
               model_train.acc])

      except KeyboardInterrupt:
        print('Training interrupted by user.')
        break

    print("Stopping at global step value: " + str(global_step))

    # Test the final accuracy and record it as the last summary point
    # 15k is the number of images in validation set
    n_batches = 15000 // (FLAGS.batch_size // constants.QUESTIONS_PER_IMAGE)
    print("Running final validation step over %d batches" % n_batches)
    final_trial = trial + FLAGS.display_step * FLAGS.batch_size
    print("Logging this eval under trial ", final_trial)
    evaluate(sess, model_val, n_batches=n_batches, test_writer=test_writer,
             global_step=global_step,
             trial=final_trial,
             tuner=tuner,
             acc_train=acc_train_)
    test_writer.close()

  # Run the best network on the test set
  run_test(train_dir, train_dir, hparams)


def main(_):
  hparams_dict = get_default_hparams_dict()
  hparams = tf.contrib.training.HParams(**hparams_dict)
  hparams = hparams.parse(FLAGS.hparams)  # Overwritten by FLAGS.hparams

  if FLAGS.clevr_test_output:
    run_test(FLAGS.train_dir, FLAGS.clevr_test_output, hparams)
  else:
    run_training(hparams, FLAGS.train_dir, None)


if __name__ == '__main__':
  tf.app.run(main)
