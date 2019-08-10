import glob
import time
import Queue
import struct
import numpy as np
import tensorflow as tf

import random
from random import shuffle
from threading import Thread
from tensorflow.core.example import example_pb2

import json
import data
import pickle as pkl

FLAGS = tf.app.flags.FLAGS

class Batch(object):

  def __init__(self, tfexamples, hps, vocab, struct_dist):

    self.hps = hps

    self.enc_batch = np.zeros((hps.branch_batch_size, 
                               hps.sen_batch_size, 
                               hps.max_enc_steps), 
                              dtype=np.int32)
    self.enc_lens = np.zeros((hps.branch_batch_size, 
                              hps.sen_batch_size), 
                             dtype=np.int32)
    self.attn_mask = -1e10 * np.ones((hps.branch_batch_size, 
                                      hps.sen_batch_size, 
                                      hps.max_enc_steps), 
                                     dtype=np.float32)    # attention mask batch
    self.branch_lens_mask = np.zeros((hps.branch_batch_size, 
                                      hps.sen_batch_size, 
                                      hps.sen_batch_size), 
                                     dtype=np.float32)
    
    self.dec_batch = np.zeros((hps.branch_batch_size, 
                               hps.max_dec_steps), 
                              dtype=np.int32)      # decoder input
    self.target_batch = np.zeros((hps.branch_batch_size, 
                                  hps.max_dec_steps), 
                                 dtype=np.int32)   # target sequence index batch
    self.padding_mark = np.zeros((hps.branch_batch_size, 
                                  hps.max_dec_steps), 
                                 dtype=np.float32) # target mask batch
    # self.tgt_batch_len = np.zeros((hps.branch_batch_size), dtype=np.int32)                     # target batch length
 
    self.state_matrix = np.zeros((hps.branch_batch_size, 
                                  hps.sen_batch_size, 
                                  hps.sen_batch_size), 
                                 dtype=np.int32)
    self.struct_conv = np.zeros((hps.branch_batch_size, 
                                 hps.sen_batch_size, 
                                 hps.sen_batch_size), 
                                dtype=np.int32)
    self.struct_dist = np.zeros((hps.branch_batch_size, 
                                 hps.sen_batch_size, 
                                 hps.sen_batch_size), 
                                dtype=np.float32)

    self.relate_user = np.zeros((hps.branch_batch_size, 
                                 hps.sen_batch_size, 
                                 hps.sen_batch_size), 
                                dtype=np.int32)

    self.mask_emb = np.zeros((hps.branch_batch_size, 
                              hps.sen_batch_size, 
                              hps.sen_batch_size, 
                              hps.sen_hidden_dim * 2), 
                             dtype=np.float32)
    self.mask_user = np.zeros((hps.branch_batch_size, 
                               hps.sen_batch_size, 
                               hps.sen_batch_size, 
                               hps.sen_hidden_dim * 2), 
                              dtype=np.float32)
    mask_tool = np.ones((hps.sen_hidden_dim * 2), 
                        dtype=np.float32)

    # self.tgt_index = np.zeros((hps.branch_batch_size, hps.sen_batch_size), dtype=np.int32)
    self.tgt_index = np.zeros((hps.branch_batch_size), 
                              dtype=np.int32)

    self.context = []
    self.response = []

    enc_lens_mid = []

    # self.small_or_large = []

    for i, ex in enumerate(tfexamples):
      # self.small_or_large.append(ex.small_large)

      for j, branch in enumerate(ex.enc_input):
        self.enc_batch[i, j, :] = branch[:]

      for enc_idx, enc_len in enumerate(ex.enc_len):
        self.enc_lens[i][enc_idx] = enc_len
        if enc_len != 0:
          self.state_matrix[i][enc_idx][enc_idx] = hps.sen_batch_size * i + enc_idx + 1
        for j in range(enc_len):
          self.attn_mask[i][enc_idx][j] = 0

      # the relaton of sentence
      for pair_struct in ex.context_struct:
        # struct_conv represent the relation of sentence A@B
        self.struct_conv[i][pair_struct[1]][pair_struct[0]] = 1
        self.mask_emb[i][pair_struct[1]][pair_struct[0]][:] = mask_tool
      # the relation of same user
      for pair_relat in ex.relation_pair:
        self.relate_user[i][pair_relat[1]][pair_relat[0]] = 1
        self.mask_user[i][pair_relat[1]][pair_relat[0]][:] = mask_tool

      for j in range(ex.branch_len):
        # self.struct_dist[i, :, :] = struct_dist[j]
        for k in range(ex.branch_len):
          self.branch_lens_mask[i][j][k] = 1
      
      # decoder input
      self.dec_batch[i, :] = ex.dec_input
      
      # train target
      self.target_batch[i, :] = ex.dec_target
      
      # decoder padding
      for j in xrange(ex.dec_len):
        self.padding_mark[i][j] = 1

      # response idx
      self.tgt_index[i] = ex.tgt_idx + i * hps.sen_batch_size

      # TODO: add prediction
      self.struct_dist[i, :, :] = 0

      self.context.append(ex.original_context)
      self.response.append(ex.original_response)

    self.enc_lens = np.reshape(self.enc_lens, (hps.branch_batch_size * hps.sen_batch_size))
    # self.enc_lens[:] = enc_lens_mid
    


class Batcher(object):
  """A class to generate minibatches of data.
  """
  BATCH_QUEUE_MAX = 5
  def __init__(self, data_path, vocab, hps):
    """
    """
    print data_path
    self.hps = hps

    self.data_path = data_path
    self.vocab = vocab
    self.hps = hps

    self.batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
    self.input_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self.hps.branch_batch_size)

    # with open('/'.join(data_path.split('/')[:-1]) + '/' + 'pred_struct_dist.pkl', 'r') as f_pred:
      # self.struct_dist = pkl.load(f_pred)
    self.struct_dist = None

    if hps.mode == 'eval':
      self.eval_num = 0

    self.num_input_threads = 1
    self.num_batch_threads = 1
    self.cache_size = 5

    self.input_threads = []
    for _ in xrange(self.num_input_threads):
      self.input_threads.append(Thread(target=self._fill_input_queue))
      self.input_threads[-1].daemon = True
      self.input_threads[-1].start()
      
    self.batch_threads = []
    for _ in xrange(self.num_batch_threads):
      self.batch_threads.append(Thread(target=self._fill_batch_queue))
      self.batch_threads[-1].daemon = True
      self.batch_threads[-1].start()

    self.watch_thread = Thread(target=self._watch_threads)
    self.watch_thread.daemon = True
    self.watch_thread.start()

  def _next_batch(self):
    """Return a Batch from the batch queue.
    """
    if self.hps.mode == 'eval':
      if self.eval_num > 5000 / self.hps.branch_batch_size:
        self.eval_num = 0
        return None
      else:
        self.eval_num += 1

    batch = self.batch_queue.get()
    return batch

  def _fill_input_queue(self):
    """Reads data from file and put into input queue
    """
    while True:
      filelist = glob.glob(self.data_path)
      if self.hps.mode == 'decode':
        filelist = sorted(filelist)
      else:
        shuffle(filelist)
        
      # for f in filelist:
      #   with open(f, 'rb') as reader:
      #     while True:
      #       len_bytes = reader.read(8)
      #       if not len_bytes: break
      #       str_len = struct.unpack('q', len_bytes)[0]
      #       example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
      #       e = example_pb2.Example.FromString(example_str)
            
      #       try:
      #         record = e.features.feature['record'].bytes_list.value[0]
      #       except ValueError:
      #         print('WARNING: Failed to get article or abstract from example: {}'.format(text_format.MessageToString(e)))
      #         continue

      #       record = record_maker(record, self.vocab, self.hps)
      #       self.input_queue.put(record)

      for f in filelist:
        with open(f, 'rb') as reader:
          for record in reader:
            record = record_maker(record, self.vocab, self.hps)
            self.input_queue.put(record)

  def _fill_batch_queue(self):
    """Get data from input queue and put into batch queue
    """
    while True:
      if self.hps.mode == 'decode':
        ex = self.input_queue.get()
        b = [ex for _ in xrange(self.hps.branch_batch_size)]
        self.batch_queue.put(Batch(b, self.hps, self.vocab, self.struct_dist))
      else:
        inputs = []
        for _ in xrange(self.hps.branch_batch_size * self.cache_size):
          inputs.append(self.input_queue.get())

        batches = []
        for i in xrange(0, len(inputs), self.hps.branch_batch_size):
          batches.append(inputs[i:i+self.hps.branch_batch_size])
        if not self.hps.mode in ['eval', 'decode']:
          shuffle(batches)
        for b in batches:
          self.batch_queue.put(Batch(b, self.hps, self.vocab, self.struct_dist))

  def _watch_threads(self):
    """Watch input queue and batch queue threads and restart if dead."""
    while True:
      time.sleep(60)
      for idx,t in enumerate(self.input_threads):
        if not t.is_alive():
          tf.logging.error('Found input queue thread dead. Restarting.')
          new_t = Thread(target=self._fill_input_queue)
          self.input_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self.batch_threads):
        if not t.is_alive():
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self._fill_batch_queue)
          self.batch_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
         
          
class record_maker(object):
  def __init__(self, record, vocab, hps):
    self.hps = hps

    start_id = vocab._word2id(data.DECODING_START)
    end_id = vocab._word2id(data.DECODING_END)
    self.pad_id = vocab._word2id(data.PAD_TOKEN)

    ### load data from the json string
    record = json.loads(record)
    context_list = record['context']              # the context
    response = record['answer']                   # the answer
    self.tgt_idx = record['ans_idx']              # The index of the context sentence corresponding to the answer
    self.context_struct = record['relation_at']   # the relation structure of the context sentence
    self.relation_pair = record['relation_user']  # the relation structure of the user (speakers)

    ### encoder
    context_words = []
    for context in context_list:
      words = context.strip().split()[:hps.max_enc_steps]
      context_words.append(words)

    self.branch_len = len(context_words)
    self.enc_len = []
    self.enc_input = []
    for words in context_words:
      self.enc_len.append(len(words))
      self.enc_input.append([vocab._word2id(w) for w in words] + \
                            [self.pad_id]*(hps.max_enc_steps-len(words)))

    self.pad_sent = [self.pad_id for _  in range(hps.max_enc_steps)] # the sentence which only have 'pad_id'
    while len(self.enc_input) < hps.sen_batch_size:
      self.enc_len.append(0)
      self.enc_input.append(self.pad_sent)

    ### decoder
    response_words = response.strip().split()
    dec_ids = [vocab._word2id(w) for w in response_words]
    # dec_ids lens
    self.dec_len = len(dec_ids) + 1 if (len(dec_ids) + 1) < hps.max_dec_steps else hps.max_dec_steps
    # decoder input
    self.dec_input = [start_id] + dec_ids[:hps.max_dec_steps - 1] + \
                     [self.pad_id]*(hps.max_dec_steps-len(dec_ids)-1)
    # decoder target
    self.dec_target = dec_ids[:hps.max_dec_steps - 1] + [end_id] + \
                      [self.pad_id]*(hps.max_dec_steps-len(dec_ids)-1)

    self.original_context = ' '.join(context_list)
    self.original_response = response
