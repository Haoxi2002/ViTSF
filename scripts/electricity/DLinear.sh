export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --data_dir ./data/electricity/ \
  --file_name electricity.csv \
  --model DLinear \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --data_dir ./data/electricity/ \
  --file_name electricity.csv \
  --model DLinear \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --data_dir ./data/electricity/ \
  --file_name electricity.csv \
  --model DLinear \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --data_dir ./data/electricity/ \
  --file_name electricity.csv \
  --model DLinear \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64