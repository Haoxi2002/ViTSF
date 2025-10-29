#!/bin/bash

nohup python -u run.py \
  --data_dir ./data/ \
  --file_name ET-APP1.csv \
  --model MV_DTSF \
  --target mps \
  --seq_len 288 \
  --pred_len 288 \
  --h 288 \
  --enc_in 1 \
  --dropout 0.05 \
  --use_multi_gpu \
  --batch_size 128 > APP1_MV_DTSF.txt 2>&1 &


