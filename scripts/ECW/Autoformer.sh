#!/bin/bash

nohup python -u run.py \
  --data_dir ./data/ \
  --file_name ECW.csv \
  --model Autoformer \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 24 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 2048 \
  --moving_avg 25 \
  --factor 3 \
  --enc_in 797 \
  --dec_in 797 \
  --c_out 797 \
  --dropout 0.05 \
  --use_multi_gpu \
  --batch_size 4 > ECW_Autoformer.txt 2>&1 &


