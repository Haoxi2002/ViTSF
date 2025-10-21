export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --data_dir ./data/ \
  --file_name ET-APP3.csv \
  --model PatchTST \
  --seq_len 288 \
  --pred_len 288 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

