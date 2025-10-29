#!/bin/bash

nohup python -u run.py \
  --data_dir ./data/ \
  --file_name ECW.csv \
  --model PatchTST \
  --seq_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 797 \
  --use_multi_gpu \
  --batch_size 4 > ECW_PatchTST.txt 2>&1 &

