export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --data_dir ./data/ETT-small/ \
  --file_name ETTh1.csv \
  --model PatchTST \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --data_dir ./data/ETT-small/ \
  --file_name ETTh1.csv \
  --model PatchTST \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --data_dir ./data/ETT-small/ \
  --file_name ETTh1.csv \
  --model PatchTST \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --data_dir ./data/ETT-small/ \
  --file_name ETTh1.csv \
  --model PatchTST \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64