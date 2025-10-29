export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --data_dir ./data/ \
  --file_name ECW.csv \
  --model MV_DTSF \
  --seq_len 48 \
  --pred_len 24 \
  --h 48 \
  --enc_in 797 \
  --dropout 0.05 \
  --batch_size 4


