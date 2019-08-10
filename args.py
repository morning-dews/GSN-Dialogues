import tensorflow as tf

# hyperparameters
tf.app.flags.DEFINE_string('device', '0', 'device to run tensorflow')
tf.app.flags.DEFINE_integer('random_seed', 123, 'random seed')
tf.app.flags.DEFINE_string('mode', 'train', '[train|decode]')
tf.app.flags.DEFINE_boolean('is_continue', True, 'continue train from a checkpoint')

tf.app.flags.DEFINE_integer('branch_batch_size', 100, 'branch_batch size (mini_batch size)')
tf.app.flags.DEFINE_integer('sen_batch_size', 9, 'the num of sentence in every branch')
tf.app.flags.DEFINE_integer('beam_size', 3, 'beam size')
tf.app.flags.DEFINE_integer('vocab_size', 30000, 'Size of vocabulary. set to 0 to read all')
tf.app.flags.DEFINE_integer('emb_dim', 300, 'word embedding size')
tf.app.flags.DEFINE_integer('sen_hidden_dim', 300, 'dimension of sentence RNN hidden states')
tf.app.flags.DEFINE_integer('branch_hidden_dim', 300, 'dimension of branch RNN hidden states')
tf.app.flags.DEFINE_integer('max_enc_steps', 50, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 50, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('min_dec_steps', 5, 'min length in beam search')

tf.app.flags.DEFINE_integer('data_size', 345000, 'data size')
tf.app.flags.DEFINE_integer('print_step', 50, 'print loss every X step')
tf.app.flags.DEFINE_integer('save_step', None, 'save ckpt every X step')
tf.app.flags.DEFINE_integer('eval_step', None, 'eval every X step')
tf.app.flags.DEFINE_integer('summary_step', 500, 'tensorboard')

tf.app.flags.DEFINE_float('norm_unif', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('norm_trunc', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('norm_grad', 3.0, 'gradient clipping')
tf.app.flags.DEFINE_float('dropout', 1.0, 'the keep rate in dropout')

# dir path
tf.app.flags.DEFINE_string('data_pre', '../data/bin_6to10/', 'data file path')
tf.app.flags.DEFINE_string('data_path', None, 'train data path')
tf.app.flags.DEFINE_string('eval_data_path', None, 'eval data path')
tf.app.flags.DEFINE_string('test_data_path', None, 'test data path')
tf.app.flags.DEFINE_string('vocab_path', None, 'vocabulary file path')

tf.app.flags.DEFINE_string('log_dir', '../logs/gram3', 'Path for logs.')

### optimizer
tf.app.flags.DEFINE_string('optimizer', 'Adam', '[Adam|Adagrad]')
tf.app.flags.DEFINE_float('lr', 1e-4, 'learning rate')
tf.app.flags.DEFINE_boolean('lr_decay', False, 'learn decay')
tf.app.flags.DEFINE_integer('decay_steps', None, 'decay every x step')
tf.app.flags.DEFINE_integer('decay_epoch', 10, 'decay every x epoch')
tf.app.flags.DEFINE_float('end_learning_rate', 1e-4, 'min lr')
tf.app.flags.DEFINE_float('power', 4, 'power in learn decay')
tf.app.flags.DEFINE_boolean('cycle', False, 'cycle in learn decay')
tf.app.flags.DEFINE_float('adagrad_acc', 0.1, 'acc for Adagrad')

### others
tf.app.flags.DEFINE_integer('n_gram', 3, 'number of n_gram')
tf.app.flags.DEFINE_boolean('use_norm', True, 'use norm')
tf.app.flags.DEFINE_float('norm_alpha', 0.25, 'norm_alpha')
# tf.app.flags.DEFINE_boolean('use_all_update', True, 'use norm')
tf.app.flags.DEFINE_boolean('user_struct', True, 'use the struct of user relation')
tf.app.flags.DEFINE_boolean('long_attn', False, 'use the struct of all sent attn')
tf.app.flags.DEFINE_boolean('positional_enc', True, 'use the positional encoding tricks')
tf.app.flags.DEFINE_integer('positional_enc_dim', 64, 'dimension of word embeddings')

tf.app.flags.DEFINE_integer('max_eval_num', 5000, 'the size of test data')
tf.app.flags.DEFINE_integer('first_ckpt', 0, 'the ckpt you want decode when you set the mode to decode')
tf.app.flags.DEFINE_integer('slot_size', None, 'the step size between the ckpt-x')

# Important settings
tf.app.flags.DEFINE_integer('epoch', -1, 'max number of epoch, -1 is infinite.')
tf.app.flags.DEFINE_boolean('use_glove', False, 'If True, use pre_train glove. If False, use embedding net to train.')
tf.app.flags.DEFINE_boolean('use_context_attn', False, 'use context attn lead the dists of the sent_vector')
tf.app.flags.DEFINE_string('glove_path', '../data/glove.840B.300d.txt', 'glove embedding path')

# tf.app.flags.DEFINE_string('file_mid', 'decode_new_3', 'ckpt dir path')
tf.app.flags.DEFINE_integer('cpkt_idx', -1, 'the index of ckpt for decoding, -1 for the newest one')

# a failed attempt, please ignore all code about these hyperparameters
tf.app.flags.DEFINE_boolean('pred_struct', False, 'predict the relation structure in dialogue')
tf.app.flags.DEFINE_boolean('TS_mode', True, 'Use Teacher-Forcing in structure prediction')
tf.app.flags.DEFINE_float('ts_ground', 0.5, 'the weight for ground-truth in Teacher-Forcing for structure prediction')
tf.app.flags.DEFINE_boolean('filt_fake', True, 'filt the fake sentence')
tf.app.flags.DEFINE_float('delta', 1, 'weight for structure learning loss')