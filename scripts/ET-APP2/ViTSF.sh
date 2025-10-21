export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --data_dir ./data/ \
  --file_name ET-APP2.csv \
  --model ViTSF \
  --seq_len 288 \
  --pred_len 288 \
  --h 144 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 8 \
  --dropout 0.1 \
  --learning_rate 0.002

