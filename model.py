import os
import math
import numpy as np
import tensorflow as tf

from attention_modules import attention_decoder, attention_struct

FLAGS = tf.app.flags.FLAGS

class GSNModel(object):
  def __init__(self, hps, vocab):
    self.hps = hps
    self.vocab = vocab

  # add the placeholder for the model
  def _add_placeholder(self, hps, vsize):
    self.enc_batch = tf.placeholder(tf.int32, [hps.branch_batch_size, hps.sen_batch_size, hps.max_enc_steps], name='enc_batch')
    self.enc_lens = tf.placeholder(tf.int32, [hps.branch_batch_size * hps.sen_batch_size], name='enc_lens')

    self.dec_batch = tf.placeholder(tf.int32, [hps.branch_batch_size, hps.max_dec_steps], name='dec_batch')
    self.target_batch = tf.placeholder(tf.int32, [hps.branch_batch_size, hps.max_dec_steps], name='target_batch')

    self.branch_lens_mask = tf.placeholder(tf.float32, [hps.branch_batch_size, hps.sen_batch_size, hps.sen_batch_size], name='branch_lens_mask')
    self.attn_mask = tf.placeholder(tf.float32, [hps.branch_batch_size, hps.sen_batch_size, hps.max_enc_steps], name='attn_mask')
    self.padding_mark = tf.placeholder(tf.float32, [hps.branch_batch_size, hps.max_dec_steps], name='padding_mark')
    self.mask_emb = tf.placeholder(tf.float32, [hps.branch_batch_size, hps.sen_batch_size, hps.sen_batch_size, hps.sen_hidden_dim * 2], name='mask_emb')
    self.mask_user = tf.placeholder(tf.float32, [hps.branch_batch_size, hps.sen_batch_size, hps.sen_batch_size, hps.sen_hidden_dim * 2], name='mask_user')

    self.state_matrix = tf.placeholder(tf.int32, [hps.branch_batch_size, hps.sen_batch_size, hps.sen_batch_size], name='state_matrix') 
    self.struct_conv = tf.placeholder(tf.int32, [hps.branch_batch_size, hps.sen_batch_size, hps.sen_batch_size], name='struct_conv')
    self.struct_dist = tf.placeholder(tf.float32, [hps.branch_batch_size, hps.sen_batch_size, hps.sen_batch_size], name='struct_conv')

    self.relate_user = tf.placeholder(tf.int32, [hps.branch_batch_size, hps.sen_batch_size, hps.sen_batch_size], name='relate_user')
    self.tgt_index = tf.placeholder(tf.int32, [hps.branch_batch_size], name='tgt_index')
    self.zero_emb = tf.zeros([1 , hps.sen_hidden_dim * 2], dtype=tf.float32, name='zero_hidden')

  # build the random initializer
  def _add_rand_initializer(self, hps):
    self.norm_trunc = tf.truncated_normal_initializer(stddev=hps.norm_trunc)
    self.norm_uinf = tf.random_uniform_initializer(-hps.norm_unif, hps.norm_unif, 
                                                   seed=self.hps.random_seed)


  def _train(self, sess, batch):  # train model for one step
    feed_dict={
        self.enc_batch: batch.enc_batch,
        self.enc_lens: batch.enc_lens,
        self.attn_mask: batch.attn_mask, 
        self.branch_lens_mask: batch.branch_lens_mask,
        self.dec_batch: batch.dec_batch,
        self.target_batch: batch.target_batch,
        self.padding_mark: batch.padding_mark,
        self.tgt_index: batch.tgt_index,
        self.struct_conv: batch.struct_conv,
        self.struct_dist: batch.struct_dist,
        self.state_matrix: batch.state_matrix,
        self.relate_user: batch.relate_user, 
        self.mask_emb: batch.mask_emb,
        self.mask_user: batch.mask_user
    }
    sess_return = {
        'train_op': self.train_op,
        'summaries': self.summaries,
        'loss': self.loss,
        'global_step': self.global_step,
        'seq_loss': self.seq_loss,
        'struct_loss': self.struct_loss,
    }
    return sess.run(sess_return, feed_dict)


  def _eval(self, sess, batch):  # eval model for one step
    feed_dict={
        self.enc_batch: batch.enc_batch,
        self.enc_lens: batch.enc_lens,
        self.attn_mask: batch.attn_mask, 
        self.branch_lens_mask: batch.branch_lens_mask,
        self.dec_batch: batch.dec_batch,
        self.target_batch: batch.target_batch,
        self.padding_mark: batch.padding_mark, 
        self.tgt_index: batch.tgt_index,
        self.struct_conv: batch.struct_conv,
        self.struct_dist: batch.struct_dist,
        self.state_matrix: batch.state_matrix,
        self.relate_user: batch.relate_user, 
        self.mask_emb: batch.mask_emb, 
        self.mask_user: batch.mask_user
    }
      
    sess_return = {
        'summaries': self.summaries,
        'loss': self.loss,
        'global_step': self.global_step,
        'output': self.output_eval, 
        'seq_loss': self.seq_loss,
        'struct_loss': self.struct_loss,
    }

    return sess.run(sess_return, feed_dict)

  ### Model train/inference process
  def _encode(self, sess, batch):  # encode for beam search
    feed_dict={
        self.enc_batch: batch.enc_batch,
        self.enc_lens: batch.enc_lens,
        self.branch_lens_mask: batch.branch_lens_mask, 
        self.struct_conv: batch.struct_conv,
        self.struct_dist: batch.struct_dist,
        self.state_matrix: batch.state_matrix,
        self.relate_user: batch.relate_user,
        self.mask_emb: batch.mask_emb,
        self.mask_user: batch.mask_user,
        self.tgt_index: batch.tgt_index,
    }

    # (sen_enc_states ,dec_in_state, global_step, pat1, pat2) = sess.run([self.sen_enc_states ,self.dec_state, self.global_step, self.mse_loss_b, self.bacws], feed_dict)
    sess_return = {
          'sen_state': self.sen_enc_states, 
          'dec_state': self.dec_state, 
          'global_step': self.global_step, # only for run the global step function
    }

    results = sess.run(sess_return, feed_dict)
    
    sen_enc_states = results['sen_state']
    dec_in_state  = results['dec_state']
    # global_step = results['global_step']

    return sen_enc_states, dec_in_state[0]

  def _decode(self, sess, batch, latest_tokens, sen_enc_states, dec_init_states):  # beam search decoder
    beam_size = len(dec_init_states)

    dec_batch = latest_tokens
    dec_batch = np.array(dec_batch)
    dec_batch = np.reshape(dec_batch, (-1, self.hps.max_dec_steps))

    feed = {
        self.sen_enc_states: sen_enc_states, 
        self.attn_mask: batch.attn_mask, 
        self.dec_state: dec_init_states,
        self.dec_batch: dec_batch,
        self.tgt_index: batch.tgt_index,
    }

    sess_return = {
      "ids": self.topk_ids,
      "probs": self.topk_log_probs,
      "states": self.dec_out_state,
      "attn_dists": self.attn_dists
    }

    results = sess.run(sess_return, feed_dict=feed)

    new_states = np.array(results['states']).reshape((self.hps.branch_batch_size, self.hps.sen_hidden_dim))
    
    top_k_ids = np.array(results['ids']).reshape(((self.hps.branch_batch_size, self.hps.branch_batch_size * 2)))
    top_k_probs = np.array(results['probs']).reshape(((self.hps.branch_batch_size, self.hps.branch_batch_size * 2)))

    assert len(results['attn_dists'])==1
    attn_dists = np.array(results['attn_dists'][0]).reshape((self.hps.branch_batch_size, -1)).tolist()
    
    return top_k_ids, top_k_probs, new_states, attn_dists

  def _get_position_encoding(self, length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    ''' add the position encoding
    '''
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) /
                               (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    
    return signal

  def _build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph
    """
    print('INFO: Building graph...')
    hps = self.hps              # the hyper-parameter setting
    vsize = self.vocab._size()  # the size of vocabulary

    # add placeholder
    self._add_placeholder(hps, vsize)

    with tf.variable_scope('seq2seq'):
      # random_uniform_initializer
      self._add_rand_initializer(hps)

      # embedding for encoder-decoder framework
      with tf.variable_scope('embedding'):
        # build a word embedding
        embedding = tf.get_variable('embedding', 
                                    [vsize, hps.emb_dim], 
                                    initializer=self.norm_trunc, 
                                    dtype=tf.float32)

        # embed the input variable
        emb_enc_inputs = tf.nn.embedding_lookup(embedding, 
                                                self.enc_batch)
        emb_dec_inputs = tf.nn.embedding_lookup(embedding, 
                                                self.dec_batch)

      with tf.variable_scope('sent_encoder'):
        # build the two LSTM Cells for bi-sentence encoder
        cell_fw = tf.contrib.rnn.LSTMCell(hps.sen_hidden_dim, 
                                          initializer=self.norm_uinf, 
                                          state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(hps.sen_hidden_dim, 
                                          initializer=self.norm_uinf, 
                                          state_is_tuple=True)

        # add the dropout layer for RNN encoder layer
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                output_keep_prob=hps.dropout)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                output_keep_prob=hps.dropout)
        
        emb_enc_inputs = tf.reshape(emb_enc_inputs,   # reshape for computing the sentence hidden variable
                                    [hps.branch_batch_size * hps.sen_batch_size, 
                                     hps.max_enc_steps, 
                                     hps.emb_dim])
        # encode all sentences in the dialogue session
        (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, 
                                                                            emb_enc_inputs, 
                                                                            sequence_length=self.enc_lens, 
                                                                            swap_memory=True, 
                                                                            dtype=tf.float32)
        encoder_outputs = tf.reshape(tf.concat(axis=2, values=encoder_outputs),  # for attention
                                     [hps.branch_batch_size * hps.sen_batch_size, 
                                      hps.max_enc_steps, 
                                      hps.sen_hidden_dim * 2])
        
        # postion enbedding, learning from transformer
        if hps.positional_enc:
          max_length = encoder_outputs.get_shape().as_list()[1]
          positional_encoding = self._get_position_encoding(max_length, 
                                                            hps.positional_enc_dim)
          encoder_outputs = tf.concat([tf.tile(tf.expand_dims(positional_encoding, 0), 
                                        [hps.branch_batch_size * hps.sen_batch_size, 
                                         1, 
                                         1]), 
                                       encoder_outputs], 
                                       -1)

        self.sen_enc_states = encoder_outputs

        with tf.variable_scope('fw_reduce'):
          enc_states_h = tf.concat((fw_st.h, bw_st.h), 1)
          enc_states_c = tf.concat((fw_st.c, bw_st.c), 1)
  
          # Weight and Bias for transfer the hidden state
          c_v_reduce = tf.get_variable('c_v_reduce', 
                                       [self.hps.sen_hidden_dim * 2, self.hps.sen_hidden_dim], 
                                       initializer=self.norm_trunc, 
                                       dtype=tf.float32)
          c_b_reduce = tf.get_variable('c_b_reduce', 
                                       [self.hps.sen_hidden_dim], 
                                       initializer=self.norm_trunc, 
                                       dtype=tf.float32)
          h_v_reduce = tf.get_variable('h_v_reduce', 
                                       [self.hps.sen_hidden_dim * 2, self.hps.sen_hidden_dim], 
                                       initializer=self.norm_trunc, 
                                       dtype=tf.float32)
          h_b_reduce = tf.get_variable('h_b_reduce', 
                                       [self.hps.sen_hidden_dim], 
                                       initializer=self.norm_trunc, 
                                       dtype=tf.float32)

          enc_states_c = tf.nn.relu(tf.matmul(enc_states_c, c_v_reduce) + c_b_reduce)
          enc_states_h = tf.nn.relu(tf.matmul(enc_states_h, h_v_reduce) + h_b_reduce)

          # concat the c and h of the LSTM
          enc_states = tf.concat((enc_states_c, enc_states_h), 1)

          ''' concat a zero embedding at the frist dimension in the enc_states
              because we want use it as the variable of all padding sentences
              when we use the GSN computing method.
          '''
          hidden_state_list = tf.concat([self.zero_emb, enc_states], 0)

        if hps.pred_struct:
          ### use attention for predict the structure in Dialogue
          attn_enc_states = tf.reshape(mid_enc_states, 
                                       [hps.branch_batch_size, 
                                        hps.sen_batch_size, 
                                        hps.sen_hidden_dim * 2])
          # use attention to predict the structure of the dialogue session
          self.forws, self.bacws, self.struct_mask = attention_struct(attn_enc_states)
          self.bacws = tf.transpose(self.bacws, perm=[0, 2, 1])
          # child structure
          # self.forws = tf.sigmoid(self.forws) * self.struct_mask * self.branch_lens_mask
          if hps.filt_fake:
            self.mask_sent_attn = tf.cast(tf.cast(self.branch_lens_mask - 1, tf.bool), tf.float32) * -1e8
            self.bacws = tf.nn.softmax(self.bacws + self.mask_sent_attn, dim=2)
            self.bacws = self.bacws * self.struct_mask * self.branch_lens_mask
          else:
            self.bacws = tf.nn.softmax(self.bacws, dim=2) * self.struct_mask * self.branch_lens_mask
            self.forws = self.bacws
        
        ### use GRU as a GATE to update the hidden-state
        # the gate for information which is from children to parents
        cell_c_p = tf.contrib.rnn.GRUCell(self.hps.sen_hidden_dim * 2, 
                                          kernel_initializer=self.norm_uinf, 
                                          bias_initializer=self.norm_trunc)
        # the gate for information which is from parents to children
        cell_p_c = tf.contrib.rnn.GRUCell(self.hps.sen_hidden_dim * 2, 
                                          kernel_initializer=self.norm_uinf, 
                                          bias_initializer=self.norm_trunc)

        # add the dropout layer for the gate
        cell_c_p = tf.contrib.rnn.DropoutWrapper(cell_c_p, 
                                                 output_keep_prob=hps.dropout)
        cell_p_c = tf.contrib.rnn.DropoutWrapper(cell_p_c, 
                                                 output_keep_prob=hps.dropout)
        
        ### use GRU as a GATE to update the same user's utterance (which is called user link)
        cell_user_c_p = tf.contrib.rnn.GRUCell(self.hps.sen_hidden_dim * 2, 
                                               kernel_initializer=self.norm_uinf, 
                                               bias_initializer=self.norm_trunc)
        cell_user_p_c = tf.contrib.rnn.GRUCell(self.hps.sen_hidden_dim * 2, 
                                               kernel_initializer=self.norm_uinf, 
                                               bias_initializer=self.norm_trunc)

        cell_user_c_p = tf.contrib.rnn.DropoutWrapper(cell_user_c_p, 
                                                      output_keep_prob=hps.dropout)
        cell_user_p_c = tf.contrib.rnn.DropoutWrapper(cell_user_p_c, 
                                                      output_keep_prob=hps.dropout)

        if not hps.pred_struct:
          struct_child = tf.matmul(self.state_matrix, self.struct_conv)
          struct_parent = tf.matmul(self.struct_conv, self.state_matrix)
        else:
          struct_child = tf.cast(tf.cast(self.forws, tf.bool), tf.int32)
          struct_parent = tf.cast(tf.cast(self.bacws, tf.bool), tf.int32)
          # struct_child = tf.cast(self.forws, tf.int32)
          # struct_parent = tf.cast(self.bacws, tf.int32)

          struct_child = tf.matmul(self.state_matrix, struct_child)
          struct_parent = tf.matmul(struct_parent, self.state_matrix)
        
        relate_user_child  = tf.matmul(self.relate_user, self.struct_conv)
        relate_user_parent = tf.matmul(self.struct_conv, self.relate_user)

      # transfer the information from children to parents
      with tf.variable_scope('update_c_p'):
        for _ in xrange(hps.n_gram):
          with tf.variable_scope('update_c_p_state'):
            emb_enc_p = tf.nn.embedding_lookup(hidden_state_list, 
                                               struct_parent, 
                                               partition_strategy='dev')
            emb_enc_c = tf.nn.embedding_lookup(hidden_state_list, 
                                               struct_child, 
                                               partition_strategy='dev')

            emb_enc_p = tf.reshape(emb_enc_p, 
                                   [hps.branch_batch_size * hps.sen_batch_size * hps.sen_batch_size, 
                                    hps.sen_hidden_dim * 2])
            emb_enc_c = tf.reshape(emb_enc_c, 
                                   [hps.branch_batch_size * hps.sen_batch_size * hps.sen_batch_size, 
                                    hps.sen_hidden_dim * 2])

            (enc_p_change, _) = cell_c_p(inputs=emb_enc_c, state=emb_enc_p)

            enc_p_change = tf.reshape(enc_p_change, 
                                      [hps.branch_batch_size, 
                                       hps.sen_batch_size, 
                                       hps.sen_batch_size, 
                                       hps.sen_hidden_dim * 2])

            if not hps.pred_struct:
              enc_c_change = enc_p_change * self.mask_emb
            else:
              self.bacws = self.bacws * self.struct_dist
              self.bacws = tf.nn.softmax(self.bacws, dim=2)
              if hps.TS_mode:
                weight_p = (1 - hps.ts_ground) * self.bacws + hps.ts_ground * tf.cast(tf.cast(self.struct_conv, tf.bool), tf.float32)
              else:
                weight_p = self.bacws

              weight_p_tmp = weight_p
              weight_p = tf.expand_dims(weight_p, -1)
              weight_p = tf.tile(weight_p, [1, 1, 1, hps.sen_hidden_dim * 2])

              enc_p_change = enc_p_change * weight_p

            # struct_tgt = tf.cast(self.struct_conv, tf.float32)
            self.mse_loss_b = 0 #tf.losses.mean_squared_error(self.bacws, struct_tgt)

            enc_p_change = tf.reduce_sum(enc_p_change, 1)
            enc_p_change = tf.reshape(enc_p_change, 
                                      [hps.branch_batch_size * hps.sen_batch_size, 
                                       hps.sen_hidden_dim * 2])
            enc_p_change = tf.concat([self.zero_emb, enc_p_change], 0)

            # use the norm to control the information fusion
            if hps.use_norm:
              vlid_norm_sent = tf.norm(enc_p_change, axis=1)
              vild_norm_sent = tf.square(vlid_norm_sent)
              dict_norm_sent = tf.div(vlid_norm_sent + self.hps.norm_alpha, 
                                      vlid_norm_sent + 1)
              dict_norm_sent = tf.reshape(dict_norm_sent, 
                                          [self.hps.branch_batch_size * self.hps.sen_batch_size + 1, 
                                           1])
              dict_norm_sent = tf.stop_gradient(dict_norm_sent)

            if not hps.user_struct and hps.use_norm:
              hidden_state_list += enc_p_change * dict_norm_sent
            elif not hps.user_struct and not hps.use_norm:
              hidden_state_list += enc_p_change

          ### to update the relate info
          if hps.user_struct:
            with tf.variable_scope('update_c_p_user'):
              emb_relate_p = tf.nn.embedding_lookup(hidden_state_list, 
                                                    relate_user_parent, 
                                                    partition_strategy='dev')
              emb_relate_c = tf.nn.embedding_lookup(hidden_state_list, 
                                                    relate_user_child, 
                                                    partition_strategy='dev')

              emb_relate_p = tf.reshape(emb_relate_p, 
                                        [hps.branch_batch_size * hps.sen_batch_size * hps.sen_batch_size, 
                                         hps.sen_hidden_dim * 2])
              emb_relate_c = tf.reshape(emb_relate_c, 
                                        [hps.branch_batch_size * hps.sen_batch_size * hps.sen_batch_size, 
                                         hps.sen_hidden_dim * 2])

              (enc_user_p_change, _) = cell_user_c_p(inputs=emb_relate_c, state=emb_relate_p)

              enc_user_p_change = tf.reshape(enc_user_p_change, 
                                             [hps.branch_batch_size, 
                                              hps.sen_batch_size, 
                                              hps.sen_batch_size, 
                                              hps.sen_hidden_dim * 2])
              enc_user_p_change = enc_user_p_change * self.mask_user

              enc_user_p_change = tf.reduce_sum(enc_user_p_change, 1)
              enc_user_p_change = tf.reshape(enc_user_p_change, 
                                             [hps.branch_batch_size * hps.sen_batch_size, 
                                              hps.sen_hidden_dim * 2])
              enc_user_p_change = tf.concat([self.zero_emb, enc_user_p_change], 0)

              if hps.use_norm:
                vlid_norm_user = tf.norm(enc_user_p_change, axis=1)
                vild_norm_user = tf.square(vlid_norm_user)
                dict_norm_user = tf.div(vlid_norm_user + self.hps.norm_alpha, 
                                        vlid_norm_user + 1)
                dict_norm_user = tf.reshape(dict_norm_user, 
                                            [self.hps.branch_batch_size * self.hps.sen_batch_size + 1, 
                                             1])
                dict_norm_user = tf.stop_gradient(dict_norm_user)
                
                hidden_state_list = hidden_state_list + enc_user_p_change * dict_norm_user + enc_p_change * dict_norm_sent
              else:
                hidden_state_list = hidden_state_list + enc_user_p_change + enc_p_change

      with tf.variable_scope('update_p_c'):
        for _ in xrange(hps.n_gram):
          with tf.variable_scope('update_p_c_state'):
            emb_enc_p = tf.nn.embedding_lookup(hidden_state_list, 
                                               struct_parent, 
                                               partition_strategy='dev')
            emb_enc_c = tf.nn.embedding_lookup(hidden_state_list, 
                                               struct_child, 
                                               partition_strategy='dev')


            emb_enc_p = tf.reshape(emb_enc_p, 
                                   [hps.branch_batch_size * hps.sen_batch_size * hps.sen_batch_size, 
                                    hps.sen_hidden_dim * 2])
            emb_enc_c = tf.reshape(emb_enc_c, 
                                   [hps.branch_batch_size * hps.sen_batch_size * hps.sen_batch_size, 
                                    hps.sen_hidden_dim * 2])

            (enc_c_change, _) = cell_p_c(inputs=emb_enc_p, state=emb_enc_c)

            enc_c_change = tf.reshape(enc_c_change, 
                                      [hps.branch_batch_size, 
                                       hps.sen_batch_size, 
                                       hps.sen_batch_size, 
                                       hps.sen_hidden_dim * 2])
            if not hps.pred_struct:
              enc_c_change = enc_c_change * self.mask_emb
            else:
              self.forws = self.forws * self.struct_dist
              self.forws = tf.nn.softmax(self.forws, dim=2)
              if hps.TS_mode:
                weight_c = (1 - hps.ts_ground) * self.forws + hps.ts_ground * tf.cast(self.struct_conv, tf.float32)
              else:
                weight_c = self.forws

              weight_c = tf.expand_dims(weight_c, -1)
              weight_c = tf.tile(weight_c, [1, 1, 1, hps.sen_hidden_dim * 2])

              ### multiply the weight
              enc_c_change = enc_c_change * weight_c

            enc_c_change = tf.reduce_sum(enc_c_change, 2)
            enc_c_change = tf.reshape(enc_c_change, 
                                      [hps.branch_batch_size * hps.sen_batch_size, 
                                       hps.sen_hidden_dim * 2])
            enc_c_change = tf.concat([self.zero_emb, enc_c_change], 0)

            if hps.use_norm:
              vlid_norm_sent = tf.norm(enc_c_change, axis=1)
              vild_norm_sent = tf.square(vlid_norm_sent)
              dict_norm_sent = tf.div(vlid_norm_sent + self.hps.norm_alpha, 
                                      vlid_norm_sent + 1)
              dict_norm_sent = tf.reshape(dict_norm_sent, 
                                          [self.hps.branch_batch_size * self.hps.sen_batch_size + 1, 
                                           1])
              dict_norm_sent = tf.stop_gradient(dict_norm_sent)

            if not hps.user_struct and hps.use_norm:
              hidden_state_list += enc_c_change * dict_norm_sent
            elif not hps.user_struct and not hps.use_norm:
              hidden_state_list += enc_c_change


          if hps.user_struct:
            ### to update the relate info
            with tf.variable_scope('update_p_c_user'):
              emb_relate_p = tf.nn.embedding_lookup(hidden_state_list, 
                                                    relate_user_parent, 
                                                    partition_strategy='dev')
              emb_relate_c = tf.nn.embedding_lookup(hidden_state_list, 
                                                    relate_user_child, 
                                                    partition_strategy='dev')

              emb_relate_p = tf.reshape(emb_relate_p, 
                                        [hps.branch_batch_size * hps.sen_batch_size * hps.sen_batch_size, 
                                         hps.sen_hidden_dim * 2])
              emb_relate_c = tf.reshape(emb_relate_c, 
                                        [hps.branch_batch_size * hps.sen_batch_size * hps.sen_batch_size, 
                                         hps.sen_hidden_dim * 2])

              (enc_user_c_change, _) = cell_user_p_c(inputs=emb_relate_p, state=emb_relate_c)

              enc_user_c_change = tf.reshape(enc_user_c_change, 
                                             [hps.branch_batch_size, 
                                              hps.sen_batch_size, 
                                              hps.sen_batch_size, 
                                              hps.sen_hidden_dim * 2])
              enc_user_c_change = enc_user_c_change * self.mask_user

              enc_user_c_change = tf.reduce_sum(enc_user_c_change, 2)
              enc_user_c_change = tf.reshape(enc_user_c_change, 
                                             [hps.branch_batch_size * hps.sen_batch_size, 
                                              hps.sen_hidden_dim * 2])
              enc_user_c_change = tf.concat([self.zero_emb, enc_user_c_change], 0)


              if hps.use_norm:
                vlid_norm_user = tf.norm(enc_user_c_change, axis=1)
                vild_norm_user = tf.square(vlid_norm_user)
                dict_norm_user = tf.div(vlid_norm_user + self.hps.norm_alpha, 
                                        vlid_norm_user + 1)
                dict_norm_user = tf.reshape(dict_norm_user, 
                                            [self.hps.branch_batch_size * self.hps.sen_batch_size + 1, 
                                             1])
                dict_norm_user = tf.stop_gradient(dict_norm_user)
                hidden_state_list = hidden_state_list + enc_user_c_change * dict_norm_user + enc_c_change * dict_norm_sent
              else:
                hidden_state_list = hidden_state_list + enc_user_c_change + enc_c_change
              

      with tf.variable_scope('reduce'):
        v_reduce = tf.get_variable('v_reduce', 
                                   [self.hps.sen_hidden_dim * 2, 
                                    self.hps.sen_hidden_dim], 
                                   initializer=self.norm_trunc, 
                                   dtype=tf.float32)
        vb_reduce = tf.get_variable('vb_reduce', 
                                    [self.hps.sen_hidden_dim], 
                                    initializer=self.norm_trunc, 
                                    dtype=tf.float32)
        new_hidden_state = tf.nn.relu(tf.matmul(hidden_state_list, v_reduce) + vb_reduce)

      # get variable for decoder
      dec_hidden_state_init = tf.gather(new_hidden_state, self.tgt_index + 1)
      if not hps.long_attn:
        enc_state = tf.gather(self.sen_enc_states, self.tgt_index)
        attn_mask = tf.reshape(self.attn_mask, [-1, self.hps.max_enc_steps])
        attn_mask = tf.gather(attn_mask, self.tgt_index)
      else:
        enc_state = tf.reshape(self.sen_enc_states, 
                               [self.hps.branch_batch_size, 
                                self.hps.sen_batch_size*self.hps.max_enc_steps, 
                                -1])
        attn_mask = tf.reshape(self.attn_mask, 
                               [self.hps.branch_batch_size, 
                                self.hps.sen_batch_size*self.hps.max_enc_steps])

      # self.dec_state = new_hidden_state[1:]
      self.dec_state = dec_hidden_state_init
      self.sen_enc_states = enc_state

      with tf.variable_scope('decoder'):
        # build a GRU cell for decoder
        cell = tf.contrib.rnn.GRUCell(self.hps.sen_hidden_dim, 
                                      kernel_initializer=self.norm_uinf, 
                                      bias_initializer=self.norm_trunc)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=hps.dropout)

        ### change for multi-decode
        # if self.hps.mode != 'decode':
        dec_out, self.dec_out_state, self.attn_dists = attention_decoder(emb_dec_inputs, 
                                                                         dec_hidden_state_init, 
                                                                         enc_state, 
                                                                         cell, 
                                                                         attn_mask, 
                                                                         hps.mode=="decode")
        # else:
        #   print 'INFO: decoding...'
        #   dec_out, self.dec_out_state, self.attn_dists = attention_decoder(emb_dec_inputs, self.dec_state, 
        #                                                                    self.sen_enc_states, cell, hps.mode=="decode")

      with tf.variable_scope('output_projection'):
        w = tf.get_variable('w', 
                            [hps.sen_hidden_dim, vsize], 
                            initializer=self.norm_trunc, 
                            dtype=tf.float32)
        w_t = tf.transpose(w)
        v = tf.get_variable('v', [vsize], 
                            initializer=self.norm_trunc, 
                            dtype=tf.float32)

        vocab_scores = []
        for i,output in enumerate(dec_out):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          vocab_scores.append(tf.nn.xw_plus_b(output, w, v))

        vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]

      log_vocab_dists = [tf.log(dist + 1e-10) for dist in vocab_dists]

      if hps.mode in ['train', 'eval']:
        with tf.variable_scope('loss'):
          self.output_eval = tf.nn.softmax(tf.stack(vocab_scores, axis=1))
          ### sequence loss
          self.seq_loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), 
                                                           self.target_batch, 
                                                           self.padding_mark)

          ### structure loss
          if hps.pred_struct:
            struct_tgt = tf.cast(self.struct_conv, tf.float32)
            self.struct_loss = (-tf.reduce_sum(struct_tgt * tf.log(self.forws + 1e-8)) - tf.reduce_sum(struct_tgt * tf.log(self.bacws + 1e-8)))
            self.struct_loss /= (hps.branch_batch_size * (hps.sen_batch_size **2 - hps.sen_batch_size) / 2)
          else:
            self.struct_loss = tf.constant(0, dtype=tf.float32)

          self.loss = self.seq_loss + hps.delta * self.struct_loss

          tf.summary.scalar(hps.mode + '_loss', self.loss) 
          tf.summary.scalar(hps.mode + '_seq_loss', self.seq_loss) 
          tf.summary.scalar(hps.mode + '_struct_loss', self.struct_loss)

    if hps.mode == "decode":
      assert len(log_vocab_dists)==1, 'Check the log_vocab_dists!'
      log_vocab_dists = log_vocab_dists[0]
      self.topk_log_probs, self.topk_ids = tf.nn.top_k(log_vocab_dists, hps.branch_batch_size * 2)
    
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self.hps.mode == 'train':
      gradients = tf.gradients(self.loss, 
                               tf.trainable_variables(), 
                               aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
      grads, global_norm = tf.clip_by_global_norm(gradients, self.hps.norm_grad) 
      # grads = []
      # for g, v in zip(gradients, tf.trainable_variables()):
      #   grads.append(tf.clip_by_value(g, -self.hps.norm_grad, self.hps.norm_grad))

      # optimizer
      if hps.lr_decay:
        self.learning_rate = tf.train.polynomial_decay(self.hps.lr, self.global_step,
                                                       hps.decay_steps, hps.end_learning_rate,
                                                       power=hps.power, cycle=hps.cycle)
      else:
        self.learning_rate = self.hps.lr
    
      if self.hps.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
      elif self.hps.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(self.learning_rate, 
                                              initial_accumulator_value=self.hps.adagrad_acc)
      else:
        assert False, 'Check the optimizer setting!'
      
      self.train_op = optimizer.apply_gradients(zip(grads, tf.trainable_variables()), 
                                                global_step=self.global_step, 
                                                name='train_step')
      tf.summary.scalar(hps.mode + '_lr', self.learning_rate)

    self.summaries = tf.summary.merge_all()

