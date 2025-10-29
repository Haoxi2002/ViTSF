#!/bin/bash

nohup python -u run.py \
  --data_dir ./data/ \
  --file_name ECW.csv \
  --model LSTM \
  --seq_len 48 \
  --pred_len 24 \
  --d_model 128 \
  --e_layers 2 \
  --enc_in 797 \
  --use_multi_gpu \
  --batch_size 4 > ECW_LSTM.txt 2>&1 &


