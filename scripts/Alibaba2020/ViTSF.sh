export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --task_id Alibaba2020_ViTSF_cpu_72 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model ViTSF \
  --seq_len 72 \
  --pred_len 72 \
  --target plan_cpu \
  --h 72 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 8 \
  --dropout 0.1 \
  --learning_rate 0.002 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_ViTSF_cpu_144 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model ViTSF \
  --seq_len 144 \
  --pred_len 144 \
  --target plan_cpu \
  --h 144 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 8 \
  --dropout 0.1 \
  --learning_rate 0.002 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_ViTSF_cpu_288 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model ViTSF \
  --seq_len 288 \
  --pred_len 288 \
  --target plan_cpu \
  --h 288 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 8 \
  --dropout 0.1 \
  --learning_rate 0.002 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_ViTSF_mem_72 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model ViTSF \
  --seq_len 72 \
  --pred_len 72 \
  --target plan_mem \
  --h 72 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 8 \
  --dropout 0.1 \
  --learning_rate 0.002 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_ViTSF_mem_144 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model ViTSF \
  --seq_len 144 \
  --pred_len 144 \
  --target plan_mem \
  --h 144 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 8 \
  --dropout 0.1 \
  --learning_rate 0.002 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_ViTSF_mem_288 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model ViTSF \
  --seq_len 288 \
  --pred_len 288 \
  --target plan_mem \
  --h 288 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 8 \
  --dropout 0.1 \
  --learning_rate 0.002 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_ViTSF_gpu_72 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model ViTSF \
  --seq_len 72 \
  --pred_len 72 \
  --target plan_gpu \
  --h 72 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 8 \
  --dropout 0.1 \
  --learning_rate 0.002 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_ViTSF_gpu_144 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model ViTSF \
  --seq_len 144 \
  --pred_len 144 \
  --target plan_gpu \
  --h 144 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 8 \
  --dropout 0.1 \
  --learning_rate 0.002 \
  --batch_size 64

python -u run.py \
  --task_id Alibaba2020_ViTSF_gpu_288 \
  --data_dir ./data/ \
  --file_name Alibaba2020.csv \
  --model ViTSF \
  --seq_len 288 \
  --pred_len 288 \
  --target plan_gpu \
  --h 288 \
  --hidden_dim 8 \
  --patch_size 12 12 \
  --token_mlp_dim 512 \
  --channel_mlp_dim 64 \
  --n_blocks 8 \
  --dropout 0.1 \
  --learning_rate 0.002 \
  --batch_size 64