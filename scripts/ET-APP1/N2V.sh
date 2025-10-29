#!/bin/bash

nohup python -u run.py \
  --data_dir ./data/ \
  --file_name ET-APP1.csv \
  --model N2V \
  --target mps \
  --seq_len 288 \
  --pred_len 288 \
  --h 288 \
  --hidden_dim 4 \
  --patch_size 36 36 \
  --token_mlp_dim 128 \
  --channel_mlp_dim 8 \
  --n_blocks 2 \
  --dropout 0 \
  --learning_rate 0.002 \
  --use_multi_gpu \
  --batch_size 128 > APP1_N2V.txt 2>&1 &

