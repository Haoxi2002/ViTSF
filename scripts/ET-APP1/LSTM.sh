#!/bin/bash

nohup python -u run.py \
  --data_dir ./data/ \
  --file_name ET-APP1.csv \
  --model LSTM \
  --target mps \
  --seq_len 288 \
  --pred_len 288 \
  --d_model 128 \
  --e_layers 2 \
  --enc_in 1 \
  --use_multi_gpu \
  --batch_size 128 > APP1_LSTM.txt 2>&1 &

