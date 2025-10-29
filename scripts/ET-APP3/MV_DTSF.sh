export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --data_dir ./data/ \
  --file_name ET-APP3.csv \
  --model MV_DTSF \
  --seq_len 288 \
  --pred_len 288 \
  --h 288 \
  --enc_in 1 \
  --dropout 0.05 \
  --batch_size 128


