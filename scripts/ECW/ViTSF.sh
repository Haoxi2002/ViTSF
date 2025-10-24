export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --data_dir ./data/ \
  --file_name ECW.csv \
  --model ViTSF \
  --seq_len 48 \
  --pred_len 24 \
  --h 48 \
  --hidden_dim 16 \
  --patch_size 8 8 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 128 \
  --n_blocks 4 \
  --dropout 0.05 \
  --learning_rate 0.003 \
  --batch_size 4

