export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --data_dir ./data/ \
  --file_name ET-APP1.csv \
  --model DLinear \
  --seq_len 288 \
  --pred_len 288 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

