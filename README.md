# GSN-Dialogues

## Requirements
- TensorFlow 1.2
- Python 2.7

## Explanations
### Training

```bash
python -u main.py --mode=train \
                  --branch_batch_size=70 \
                  --sen_batch_size=9 \
                  --n_gram=$NUMBER_OF_N_GRAM$ \
                  --branch_hidden_dim=300 \
                  --sen_hidden_dim=300 \
                  --use_norm=True \
                  --norm_alpha=0.25 \
                  --user_struct=$USE_USER_FLOW$ \
                  --long_attn=False \
                  --lr=5e-4 \
                  --end_learning_rate=1e-5 \
                  --lr_decay=True \
                  --decay_epoch=10 \
                  --positional_enc=True \
                  --positional_enc_dim=64 \
                  --data_pre=$DATA_DIR$ \
                  --log_dir=$PATH OF CHECKPOINTS AND RESULTS$ \
                  --data_size=$DATA_SIZE$ \
                  --device=$GPU_ID$
```

### Inference
```bash
python -u main.py --mode=decode \
                  --beam_size=3 \
                  --sen_batch_size=9 \
                  --n_gram=$NUMBER_OF_N_GRAM$ \
                  --branch_hidden_dim=300 \
                  --sen_hidden_dim=300 \
                  --use_norm=True \
                  --norm_alpha=0.25 \
                  --user_struct=$USE_USER_FLOW$ \
                  --long_attn=False \
                  --positional_enc=True \
                  --positional_enc_dim=64 \
                  --data_pre=$DATA_DIR$ \
                  --log_dir=$PATH OF CHECKPOINTS AND RESULTS$ \
                  --data_size=$DATA_SIZE$ \
                  --first_ckpt=$THE FIRSE CHECKPOINT$ \
                  --slot_size=$THE_SIZE_BETWEEN_CHECKPOINTS$ \
                  --device=$GPU_ID$
```

### Data Format
You must make data as the following format.

You can split big data corpus as many smaller files. such as ``train_\*.json''  for faster data reading (*batcher.py* will get all file names by glob()).

```
Every line in the data file is a JSON string which can be loaded as a dict.
Keys:
- "context": string. A list of context sentences.
- "answer": string. The response sentences.
- "ans_idx": int. The index of sentence which is followed by the answer in context list.
- "relation_at": list, a list of tuples. Every tuple is an information flow. For example, [1, 0] represents sentence 1 is following sentence 0 in context. 
- "relation_user": list, a list of tuples. Every tuple is users' information flows. For example, [2, 1, 0] represents there will be three user information flows. e.g. [2, 1], [2, 0] and [1, 0].

For example:
{
    "context": ["you do n't get a window with gears ?", "i do get glxgears EMOJI but no display of fps EMOJI", "should show in the terminal", "make that , yes there are EMOJI", "no reason to kick 'em .", "google for ur question theres a site about that somewhere , u need wine and mozilla-controls", "what do you need exactly ?"], 
    "answer": "depends - this is quite a large chan - and apokryphos did n't ban , just remoed", 
    "ans_idx": 4, 
    "relation_at": [[1, 0], [2, 1], [3, 2], [4, 2], [5, 4], [6, 5]], 
    "relation_user": [[2, 0]]
}
```
You can find more information in *batcher.py* or our paper.

<br>

This repository contains an implementation of the *GSN: A Graph-Structured Network for Multi-Party Dialogues*. If you use this code in your work, please cite the following

```
@article{hu2019gsn,
  title={GSN: A Graph-Structured Network for Multi-Party Dialogues},
  author={Hu, Wenpeng and Chan, Zhangming and Liu, Bing and Zhao, Dongyan and Ma, Jinwen and Yan, Rui},
  journal={arXiv preprint arXiv:1905.13637},
  year={2019}
}
```
