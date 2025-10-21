export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --data_dir ./data/ \
  --file_name ECW.csv \
  --model ViTSF \
  --seq_len 48 \
  --pred_len 24 \
  --h 24 \
  --hidden_dim 8 \
  --patch_size 6 6 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 8 \
  --dropout 0.1 \
  --learning_rate 0.002 \
  --batch_size 4

