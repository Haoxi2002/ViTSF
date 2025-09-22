export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --data_dir ./data/ETT-small/ \
  --file_name ETTh1.csv \
  --model ViTSF \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --h 96 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 8 \
  --dropout 0.1 \
  --learning_rate 0.002

python -u run.py \
  --data_dir ./data/ETT-small/ \
  --file_name ETTh1.csv \
  --model ViTSF \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --h 96 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 6 \
  --dropout 0 \
  --learning_rate 0.002

python -u run.py \
  --data_dir ./data/ETT-small/ \
  --file_name ETTh1.csv \
  --model ViTSF \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --h 96 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 6 \
  --dropout 0 \
  --learning_rate 0.002

python -u run.py \
  --data_dir ./data/ETT-small/ \
  --file_name ETTh1.csv \
  --model ViTSF \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --h 96 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 6 \
  --dropout 0 \
  --learning_rate 0.002