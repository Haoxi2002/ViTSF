export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_id Alibaba2020_PatchTST_cpu_72 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model PatchTST \
  --seq_len 72 \
  --pred_len 72 \
  --target plan_cpu \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_PatchTST_cpu_144 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model PatchTST \
  --seq_len 144 \
  --pred_len 144 \
  --target plan_cpu \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_PatchTST_cpu_288 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model PatchTST \
  --seq_len 288 \
  --pred_len 288 \
  --target plan_cpu \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_PatchTST_mem_72 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model PatchTST \
  --seq_len 72 \
  --pred_len 72 \
  --target plan_mem \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_PatchTST_mem_144 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model PatchTST \
  --seq_len 144 \
  --pred_len 144 \
  --target plan_mem \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_PatchTST_mem_288 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model PatchTST \
  --seq_len 288 \
  --pred_len 288 \
  --target plan_mem \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_PatchTST_gpu_72 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model PatchTST \
  --seq_len 72 \
  --pred_len 72 \
  --target plan_gpu \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_PatchTST_gpu_144 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model PatchTST \
  --seq_len 144 \
  --pred_len 144 \
  --target plan_gpu \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_PatchTST_gpu_288 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model PatchTST \
  --seq_len 288 \
  --pred_len 288 \
  --target plan_gpu \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --batch_size 64