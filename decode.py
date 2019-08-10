import os
import time
import json
import logging
import numpy as np
import tensorflow as tf

import beam_search
import data

import time
import random

from tqdm import tqdm
from glob import glob

FLAGS = tf.app.flags.FLAGS

max_eval_num = FLAGS.max_eval_num
first_ckpt = FLAGS.first_ckpt
slot_size = FLAGS.slot_size

class BeamSearchDecoder(object):
  def __init__(self, model, batcher, vocab):
    self.model = model
    self.model._build_graph()
    self.saver = tf.train.Saver()
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    self.sess = tf.Session(config=config)
    self.batcher = batcher
    self.vocab = vocab
    
    train_dir = FLAGS.log_dir + '/ckpt'
    self.finish_set = set([slot_size * (i + 1) for i in range(first_ckpt / slot_size  - 1)])

    while True:
      try:
        ckpt_state = tf.train.get_checkpoint_state(train_dir)
        self.checkpoint_pre = ckpt_state.model_checkpoint_path.split('-')[0] + '-'
        # self.checkpoint_pre = '-'.join(ckpt_state.model_checkpoint_path.split('-')[:-1]) + '-'
        if FLAGS.cpkt_idx == -1:
          self.checkpoint_index = int(ckpt_state.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
          self.checkpoint_index = FLAGS.cpkt_idx
        break
      except:
        print('ERROR: Failed to restore checkpoint: {}, sleep for {} secs'.format(train_dir, 10))
        time.sleep(600)
        continue

    self.decode_dir = os.path.join(FLAGS.log_dir, "decode_continue")
    if not os.path.exists(self.decode_dir): os.makedirs(self.decode_dir)

    self.generate_pre = self.decode_dir + '/generate_'
    self.response_pre = self.decode_dir + '/response_'

    self.local_path = os.getcwd()


  def _decode(self):
    while True:
      self.checkpoint_index = self.get_ckpt_idx(self.checkpoint_pre)

      if self.checkpoint_index is None:
        print('wait for new checkpoint...')
        time.sleep(600)
        continue

      self.saver.restore(self.sess, self.checkpoint_pre + str(self.checkpoint_index))
      print('INFO: Loading checkpoint {}'.format(self.checkpoint_pre + str(self.checkpoint_index)))

      self.g_f = open(self.generate_pre + str(self.checkpoint_index) + '.txt', 'w')
      self.r_f = open(self.response_pre + str(self.checkpoint_index) + '.txt', 'w')

      self.bleu_f = open(self.decode_dir + '/bleu.txt', 'a')

      for _ in tqdm(xrange(max_eval_num)):
        batch = self.batcher._next_batch()
        if batch is None: 
          return

        best_hyp = beam_search.run_beam_search(self.sess, self.model, self.vocab, batch)

        output_ids = [int(t) for t in best_hyp.tokens[1:]]
        decoded_words = data.outputids2words(output_ids, self.vocab)

        try:
          fst_stop_idx = decoded_words.index(data.DECODING_END)
          decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
          decoded_words = decoded_words

        decoded_out = ' '.join(decoded_words)
        response_out = batch.response[0].strip()[:-1]

        self.g_f.write(decoded_out + '\n')
        self.r_f.write(response_out + '\n')

      self.g_f.close()
      self.r_f.close()

      f_decode = self.generate_pre + str(self.checkpoint_index) + '.txt'
      f_ground = self.response_pre + str(self.checkpoint_index) + '.txt'
      os.chdir(self.local_path + '/../cal_bleu/')
      try:
        eval_result = os.popen('python eval.py -out=' + f_decode + ' -src=' + f_ground + ' -tgt=' + f_ground)
        eval_result = eval_result.read()
        os.chdir(self.local_path)

        self.bleu_result = '===' + str(self.checkpoint_index) + '===\n' + eval_result + '\n'
        self.bleu_f.write(self.bleu_result)
        self.bleu_f.close()
      except:
        print 'Get automatic metric score failed!'
        os.chdir(self.local_path)

      self.finish_set.add(self.checkpoint_index)



  def get_ckpt_idx(self, path):
    file_list = glob(path + '*.index')
    f_idx_list = []
    for f in file_list:
      f_idx_list.append(int(f.split('-')[-1].split('.')[0]))

    f_idx_list = list(set(f_idx_list).difference(self.finish_set))
    f_idx_list.sort()


    return f_idx_list[0] if len(f_idx_list) != 0 else None
