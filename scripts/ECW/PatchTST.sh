export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --data_dir ./data/ \
  --file_name ECW.csv \
  --model PatchTST \
  --seq_len 48 \
  --pred_len 24 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

