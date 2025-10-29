#!/bin/bash

nohup python -u run.py \
  --data_dir ./data/ \
  --file_name ET-APP1.csv \
  --model KAE_Informer \
  --target mps \
  --seq_len 288 \
  --label_len 144 \
  --pred_len 288 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 2048 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --dropout 0.05 \
  --use_multi_gpu \
  --batch_size 128 > APP1_KAE_Informer.txt 2>&1 &


