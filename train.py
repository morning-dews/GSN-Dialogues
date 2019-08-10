import os
import time
import numpy as np
import tensorflow as tf

from utils import avg_loss

def train(model, batcher, eval_batcher, vocab, hps): # train the model
  train_dir = os.path.join(hps.log_dir, "train")
  cpkt_dir = os.path.join(hps.log_dir, "ckpt")

  if not os.path.exists(train_dir): os.makedirs(train_dir)
  if not os.path.exists(cpkt_dir): os.makedirs(cpkt_dir)

  cpkt_dir = os.path.join(cpkt_dir, 'train')

  model._build_graph()
  saver = tf.train.Saver(max_to_keep=100)
  summary_writer = tf.summary.FileWriter(train_dir)

  try:
    print('INFO: starting training')
    train_step, loss_avg = 0.0, 0.0
    t0=time.time()
    gpuConfig = tf.ConfigProto()
    gpuConfig.gpu_options.allow_growth = True
    with tf.Session(config = gpuConfig) as sess:
      sess.run(tf.global_variables_initializer())
      if hps.is_continue:
        ckpt = tf.train.get_checkpoint_state(hps.log_dir)
        if ckpt and ckpt.model_checkpoint_path:  
            saver.restore(sess, ckpt.model_checkpoint_path)

      while hps.epoch == -1 or (train_step <= hps.epoch * (hps.data_size / hps.branch_batch_size)):
        batch = batcher._next_batch()
        if batch is None: break
        results = model._train(sess, batch)

        summaries = results['summaries']
        train_step = results['global_step']

        loss = results['loss']
        seq_loss = results['seq_loss']
        struct_loss = results['struct_loss']

        loss_avg = avg_loss(loss, loss_avg, summary_writer, train_step, decay=0.99)

        summary_writer.add_summary(summaries, train_step)
        if train_step % hps.print_step == 0:
          t1=time.time()
          # print('Training step {}: {:>3}h {:>2}m {:>2}s'
          #        .format(train_step, int(t1-t0)//3600,(int(t1-t0)%3600)//60, int(t1-t0)%60))
          # print('\tloss: {}, seq_loss: {}, struct_loss: {}, loss_avg: {}\n'
          #        .format(loss, seq_loss, struct_loss, loss_avg))
          print('Training step {}: {:>3}h {:>2}m {:>2}s'
                 .format(train_step, int(t1-t0)//3600,(int(t1-t0)%3600)//60, int(t1-t0)%60))
          print('\tloss: {}, loss_avg: {}\n'
                 .format(loss, loss_avg))

        if train_step % hps.summary_step == 0:
          summary_writer.flush()

        if train_step % hps.save_step == 0:
          saver.save(sess, cpkt_dir, global_step=model.global_step)

        if train_step % hps.eval_step == 0:
          evaluate(sess, model, eval_batcher, summary_writer, vocab)

  except KeyboardInterrupt:
    print('Saving checkpoint ...')
    saver.save(sess, cpkt_dir, global_step=model.global_step)
    print('Checkpoint Saved ...')

# eval this model
def evaluate(sess, model, batcher, summary_writer, vocab):
  eval_step, running_avg_loss = 0, 0
  while True:
    batch = batcher._next_batch()
    if batch is None: break

    t0=time.time()
    results = model._eval(sess, batch)
    t1=time.time()

    loss = results['loss']
    seq_loss = results['seq_loss']
    struct_loss = results['struct_loss']
    train_step = results['global_step']

    running_avg_loss += loss
    eval_step += 1

    # show the decode sentence every 1000 steps
    if eval_step % 70 == 0:
      for line in results['output'][:5]:
        for word in line:
          word_id = word.tolist().index(max(word))
          print vocab._id2word(word_id),
        print ''
      print '\n'

  running_avg_loss /= eval_step

  print('Info: eval average loss is {}'.format(running_avg_loss))

  summary_eval_loss = tf.Summary(value=[tf.Summary.Value(tag='eval_loss', simple_value=np.asscalar(running_avg_loss))])

  summary_writer.add_summary(summary_eval_loss, train_step)
  summary_writer.flush()
