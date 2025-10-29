export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --data_dir ./data/ \
  --file_name ET-APP1.csv \
  --model LSTM \
  --target mps \
  --seq_len 288 \
  --pred_len 288 \
  --d_model 128 \
  --e_layers 2 \
  --enc_in 1 \
  --batch_size 128


