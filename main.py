import os
import sys
import args
import time
import random
import numpy as np
import tensorflow as tf

from data import Vocab
from batcher import Batcher
from model import GSNModel
from decode import BeamSearchDecoder
from train import train, evaluate
from utils import get_datapath, get_steps, set_random_seeds, make_hps

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device


def main(unused_argv):
  set_random_seeds()

  get_datapath() # The dataset path
  get_steps() # setting steps according data_size

  tf.logging.set_verbosity(tf.logging.INFO)
  print('Now the mode of this mode is {} !'.format(FLAGS.mode))

  # if log_dir is not exited, create it.
  if not os.path.exists(FLAGS.log_dir): os.makedirs(FLAGS.log_dir)

  if FLAGS.mode == 'decode':
    FLAGS.branch_batch_size = FLAGS.beam_size  # for beam search
    FLAGS.TS_mode = False

  hps = make_hps() # make a hps namedtuple

  # Vocabulary
  vocab = Vocab(hps.vocab_path, hps.vocab_size)
  # Train or Inference
  if hps.mode == 'train':
    batcher = Batcher(hps.data_path, vocab, hps)
    eval_hps = hps._replace(mode='eval')
    eval_batcher = Batcher(hps.eval_data_path, vocab, eval_hps)

    model = GSNModel(hps, vocab)
    train(model, batcher, eval_batcher, vocab, hps)
  elif hps.mode == 'decode':
    decode_mdl_hps = hps._replace(max_dec_steps=1)
    batcher = Batcher(hps.test_data_path, vocab, decode_mdl_hps)  # for test

    model = GSNModel(decode_mdl_hps, vocab)
    decoder = BeamSearchDecoder(model, batcher, vocab)
    decoder._decode()

if __name__ == '__main__':
  tf.app.run()
