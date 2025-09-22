export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --data_dir ./data/traffic/ \
  --file_name traffic.csv \
  --model PatchTST \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 862 \
  --batch_size 8

python -u run.py \
  --data_dir ./data/traffic/ \
  --file_name traffic.csv \
  --model PatchTST \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 862 \
  --batch_size 8

python -u run.py \
  --data_dir ./data/traffic/ \
  --file_name traffic.csv \
  --model PatchTST \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 862 \
  --batch_size 8

python -u run.py \
  --data_dir ./data/traffic/ \
  --file_name traffic.csv \
  --model PatchTST \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 862 \
  --batch_size 4