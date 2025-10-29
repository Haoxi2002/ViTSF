#!/bin/bash

nohup python -u run.py \
  --data_dir ./data/ \
  --file_name ECW.csv \
  --model N2V \
  --seq_len 48 \
  --pred_len 24 \
  --h 48 \
  --hidden_dim 16 \
  --patch_size 8 8 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 128 \
  --n_blocks 4 \
  --dropout 0.05 \
  --learning_rate 0.003 \
  --use_multi_gpu \
  --batch_size 4 > ECW_N2V.txt 2>&1 &

