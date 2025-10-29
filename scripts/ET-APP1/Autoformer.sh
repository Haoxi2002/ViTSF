export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --data_dir ./data/ \
  --file_name ET-APP1.csv \
  --model Autoformer \
  --target mps \
  --seq_len 288 \
  --label_len 144 \
  --pred_len 288 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 2048 \
  --moving_avg 25 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --dropout 0.05 \
  --batch_size 128


