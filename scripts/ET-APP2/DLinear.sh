#!/bin/bash

nohup python -u run.py \
  --data_dir ./data/ \
  --file_name ET-APP2.csv \
  --model DLinear \
  --target mps \
  --seq_len 288 \
  --label_len 144 \
  --pred_len 288 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 1 \
  --use_multi_gpu \
  --batch_size 128 > APP2_DLinear.txt 2>&1 &

