import tensorflow as tf

from tensorflow.python.ops import nn_ops
from utils import concat_conv

# decoder with attention mechanism
def attention_decoder(decoder_inputs, 
                      initial_state, 
                      encoder_states, 
                      cell, 
                      attn_mask, 
                      initial_state_attention=False, 
                      pointer=False):
  with tf.variable_scope("attention_decoder") as scope:
    batch_size = encoder_states.get_shape()[0].value
    attn_size = encoder_states.get_shape()[2].value

    encoder_states = tf.expand_dims(encoder_states, axis=2)

    W_h = tf.get_variable("W_h", [1, 1, attn_size, attn_size])
    encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")
    v = tf.get_variable("v", [attn_size])

    def attention(decoder_state):
      """Calculate the context vector and attention distribution from the decoder state.
      """
      with tf.variable_scope("Attention"):
        decoder_features = concat_conv([decoder_state], attn_size, True)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)
        e = tf.reduce_sum(v * tf.tanh(encoder_features + decoder_features), [2, 3])
        e += attn_mask
        attn_dist = nn_ops.softmax(e)
        context_vector = tf.reduce_sum(tf.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [1, 2])
        context_vector = tf.reshape(context_vector, [-1, attn_size])

      return context_vector, attn_dist

    outputs, attn_dists = [], []
    state = initial_state
    context_vector = tf.zeros([batch_size, attn_size])
    context_vector.set_shape([None, attn_size])
    if initial_state_attention:
      context_vector, _ = attention(initial_state)
    
    decoder_inputs = tf.unstack(decoder_inputs, axis=1)

    print('INFO: Adding attention_decoder of {} timesteps...'.format(len(decoder_inputs)))
    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        tf.get_variable_scope().reuse_variables()

      input_size = inp.get_shape().with_rank(2)[1]

      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)

      x = concat_conv([inp] + [context_vector], input_size, True)

      cell_output, state = cell(x, state)

      if i == 0 and initial_state_attention:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          context_vector, attn_dist= attention(state)
      else:
        context_vector, attn_dist = attention(state)
      attn_dists.append(attn_dist)

      with tf.variable_scope("AttnOutputProjection"):
        output = concat_conv([cell_output] + [context_vector], cell.output_size, True)
      outputs.append(output)

    return outputs, state, attn_dists

# predict the struct in dialogue session
def attention_struct(sents_state):
  with tf.variable_scope("attention_struct") as scope:
    inf_conv = tf.constant([-1e8 for _ in range(FLAGS.branch_batch_size)], dtype=tf.float32, name='inf_conv')
    zero_conv = tf.constant([0 for _ in range(FLAGS.branch_batch_size)], dtype=tf.float32, name='zero_conv')

    batch_size = sents_state.get_shape()[0].value
    sent_size = sents_state.get_shape()[1].value
    attn_size = sents_state.get_shape()[2].value

    sents_state = tf.expand_dims(sents_state, axis=2)

    W_h = tf.get_variable("W_h", [1, 1, attn_size, attn_size])
    v = tf.get_variable("v", [attn_size])

    def attention(nod_in, sents_in):
      """Calculate the context vector and attention distribution from the decoder state.
      """
      with tf.variable_scope("Attention"):
        nod_features = concat_conv([nod_in], attn_size, True)
        nod_features = tf.expand_dims(tf.expand_dims(nod_features, 1), 1)

        sents_feature = nn_ops.conv2d(sents_in, W_h, [1, 1, 1, 1], "SAME")

        e = tf.reduce_sum(v * tf.tanh(sents_feature + nod_features), [2, 3])
        # attn_dist = e
        attn_dist = nn_ops.softmax(e)
      return attn_dist

    sents_state = tf.unstack(sents_state, axis=1)
    forws, bacws, mask = [], [], []

    for idx in range(sent_size):
      if idx > 0:
        tf.get_variable_scope().reuse_variables()
      nod_out = sents_state[idx]
      nod_out = tf.squeeze(nod_out, [1])

      sents_out = sents_state[:idx] + sents_state[idx + 1:]
      sents_out = tf.stack(sents_out, axis=1)

      dist = attention(nod_out, sents_out)

      dist = tf.unstack(dist, axis=1)
      forw = [inf_conv for _ in range(idx + 1)] + dist[idx:]
      bacw = dist[:idx] + [inf_conv for _ in range(sent_size - idx)]

      forw = tf.stack(forw, axis=1)
      bacw = tf.stack(bacw, axis=1)

      mask_line = [1 for _ in range(idx)] + [0 for _ in range(sent_size - idx)]

      forws.append(forw)
      bacws.append(bacw)
      mask.append(mask_line)

    forws = tf.stack(forws, axis=2)
    bacws = tf.stack(bacws, axis=2)

    return forws, bacws, mask