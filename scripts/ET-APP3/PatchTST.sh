export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --data_dir ./data/ \
  --file_name ET-APP3.csv \
  --model PatchTST \
  --target mps \
  --seq_len 288 \
  --label_len 144 \
  --pred_len 288 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --batch_size 128

