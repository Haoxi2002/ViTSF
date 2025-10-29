export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --data_dir ./data/ \
  --file_name ET-APP3.csv \
  --model ViTSF \
  --target mps \
  --seq_len 288 \
  --pred_len 288 \
  --h 144 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 4 \
  --dropout 0.1 \
  --learning_rate 0.002 \
  --use_multi_gpu \
  --batch_size 128 > APP3_ViTSF.txt 2>&1 &


