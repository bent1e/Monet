#####################################################################
# AVT SFT
#####################################################################
# old, non-parallel 
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m src.main \
    --epochs "10" \
    --bsz 2 \
    --grad_accum_steps 8 \
    --task "mm-reasoning" \
    --stage "avt_sft" \
    --data_path "./new/created_dataset/filtered_data/CoF/filtered_train.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train.json" \
    "./new/created_dataset/filtered_data/VTS_1/filtered_train.json" \
    --log_file "./log.txt" \
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" 


# nparallel, w/o deepspeed
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
  --model "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
  --epochs 10 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_sft" \
  --data_path "./new/created_dataset/filtered_data/CoF/filtered_train.json" \
  --log_file "./log.txt" \
  --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
  --sft_analysis_ratio 0.1 \
  --sft_analysis_categories non_observation_poss observation_poss 


# parallel, w deepspeed
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
  --epochs 10 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_sft" \
  --data_path \
    "./new/created_dataset/filtered_data/CoF/filtered_train.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train.json" \
    "./new/created_dataset/filtered_data/VTS_1/filtered_train.json" \
  --shuffle_train \
  --log_file "./log.txt" \
  --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --ce_emphasize_factor 3.0 \
  --ce_emphasize_warmup_steps 100 


export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
  --model "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
  --epochs 10 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_sft" \
  --data_path \
    "./new/created_dataset/filtered_data/CoF/filtered_train.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train.json" \
    "./new/created_dataset/filtered_data/VTS_1/filtered_train.json" \
  --log_file "./log.txt" \
  --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
  --deepspeed ./deepspeed/ds_zero2_gpu.json 


#####################################################################
# AVT stage1
#####################################################################
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
    --epochs 2 \
    --bsz 1 \
    --grad_accum_steps 16 \
    --task "mm-reasoning" \
    --stage "avt_stage1" \
    --data_path \
    "./new/created_dataset/filtered_data/CoF/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata.json" \
    "./new/created_dataset/filtered_data/VTS_1/filtered_train_short3000_w_metadata.json" \
    --log_file "./log.txt" \
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct-0812-avt_sft-shuffle" \
    --min_latent_size 6 \
    --min_latent_compress_factor 30 \
    --max_latent_compress_factor 40 \
    --alignment_weight 2.0 \
    --ce_emphasize_factor 3.0 \
    --alignment "observation_all" \
    --deepspeed ./deepspeed/ds_zero2_gpu.json


#####################################################################
# AVT v2 stage1
#####################################################################
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
    --epochs 2 \
    --bsz 1 \
    --grad_accum_steps 16 \
    --task "mm-reasoning" \
    --stage "avt_v2_stage1" \
    --data_path \
    "./new/created_dataset/filtered_data/CoF/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata.json" \
    "./new/created_dataset/filtered_data/VTS_1/filtered_train_short3000_w_metadata.json" \
    --log_file "./log.txt" \
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
    --latent_size 10 \
    --ce_emphasize_factor 1.0 \
    --deepspeed ./deepspeed/ds_zero2_gpu.json

#####################################################################
# AVT v2 stage1 — MULTI-NODE (example: 2 nodes, 4 GPUs each)
#####################################################################
# On ALL nodes, set these variables consistently.
# On node0, set NODE_RANK=0; on node1, set NODE_RANK=1.
export MASTER_ADDR="172.16.0.9"
export MASTER_PORT=29501
export NNODES=1
export NODE_RANK=0          # change to 1 on the second node
export GPUS_PER_NODE=4      # GPUs per node

# Optional NCCL/network tuning (adjust to your cluster fabric)
export NCCL_DEBUG=WARN
# export NCCL_SOCKET_IFNAME=eth0    # uncomment and set to your NIC if needed
# export NCCL_IB_DISABLE=1          # uncomment if no InfiniBand is available
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nnodes $NNODES --node-rank $NODE_RANK \
  --nproc-per-node $GPUS_PER_NODE \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  -m src.main \
    --epochs 2 \
    --bsz 1 \
    --grad_accum_steps 16 \
    --task "mm-reasoning" \
    --stage "avt_v2_stage1" \
    --data_path \
    "./new/created_dataset/filtered_data/CoF/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata.json" \
    "./new/created_dataset/filtered_data/VTS_1/filtered_train_short3000_w_metadata.json" \
    --log_file "./log.txt" \
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
    --latent_size 10 \
    --ce_emphasize_factor 1.0 \
    --deepspeed ./deepspeed/ds_zero2_gpu.json

#####################################################################
# AVT v2 stage2
#####################################################################
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
    --epochs 2 \
    --bsz 1 \
    --grad_accum_steps 16 \
    --task "mm-reasoning" \
    --stage "avt_v2_stage2" \
    --data_path \
    "./new/created_dataset/filtered_data/CoF/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata.json" \
    "./new/created_dataset/filtered_data/VTS_1/filtered_train_short3000_w_metadata.json" \
    --log_file "./log.txt" \
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
    --latent_size 6 \
    --ce_emphasize_factor 1.0 \
    --alignment_weight 1.0 \
    --deepspeed ./deepspeed/ds_zero2_gpu.json



export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
    --epochs 15 \
    --lr 0.00001 \
    --bsz 1 \
    --grad_accum_steps 16 \
    --task "mm-reasoning" \
    --stage "avt_v2_stage2" \
    --data_path \
    "./new/created_dataset/filtered_data/CoF/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata.json" \
    "./new/created_dataset/filtered_data/VTS_1/filtered_train_short3000_w_metadata.json" \
    --log_file "./log.txt" \
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct-08_20-avt_v2_stage1-latent6-ce_factor1.0" \
    --latent_size 6 \
    --ce_emphasize_factor 1.0 \
    --alignment_weight 2.0 \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --teacher_latent_dir /home/dids/shiyang/codes/abstract-visual-token/new/precomputed_teacher_latents/teacher_latents
    