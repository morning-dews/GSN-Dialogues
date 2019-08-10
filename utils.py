import random
import tensorflow as tf

from collections import namedtuple

FLAGS = tf.app.flags.FLAGS

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)

def get_datapath(): # The dataset path
  FLAGS.data_path = FLAGS.data_pre + 'train_*.json'
  FLAGS.eval_data_path = FLAGS.data_pre + 'eval_*.json'
  FLAGS.test_data_path = FLAGS.data_pre + 'test_*.json'
  FLAGS.vocab_path = FLAGS.data_pre + 'vocab'

def get_steps(): # setting steps according data_size
  FLAGS.save_step = int(FLAGS.data_size / FLAGS.branch_batch_size / 2)
  FLAGS.eval_step = int(FLAGS.data_size / FLAGS.branch_batch_size / 5)
  FLAGS.decay_steps = int(FLAGS.data_size / FLAGS.branch_batch_size) * FLAGS.decay_epoch

def make_hps():
  print('\n\n##### hyperparameters #####')
  hps_dict = {}
  for key, val in FLAGS.__flags.iteritems():
    print('{}: {}'.format(key, val))  # print all hyperparameters to the screen
    hps_dict[key] = val
  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
  print('##### hyperparameters #####\n\n')

  return hps

def avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
  """Calculate the running average loss via exponential decay.
  """
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss

  running_avg_loss = min(running_avg_loss, 12)
  loss_sum = tf.Summary()
  summary_writer.add_summary(loss_sum, step)

  return running_avg_loss

def concat_conv(args, output_size, bias, bias_start=0.0, scope=None, is_reuse=None):
  # concat the variable in args and make the output as output_size
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2 or not shape[1]:
      assert False, 'Check the inputs of concat_conv function !'
    else:
      total_arg_size += shape[1]

  with tf.variable_scope(scope or "Linear", reuse=is_reuse):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])

    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)

    if bias:
        bias_variable = tf.get_variable('Bias', [output_size])
        res += bias_variable 
  return res