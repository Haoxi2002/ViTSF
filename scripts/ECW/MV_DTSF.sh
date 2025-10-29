#!/bin/bash

nohup python -u run.py \
  --data_dir ./data/ \
  --file_name ECW.csv \
  --model MV_DTSF \
  --seq_len 48 \
  --pred_len 24 \
  --h 48 \
  --enc_in 797 \
  --dropout 0.05 \
  --use_multi_gpu \
  --batch_size 4 > ECW_MV_DTSF.txt 2>&1 &


