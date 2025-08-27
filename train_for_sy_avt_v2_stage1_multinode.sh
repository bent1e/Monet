#!/usr/bin/env bash
# Multi-node version of train_for_sy_avt_v2_stage1.sh
# Usage hints:
#   - On node0 (master): set NODE_RANK=0 and MASTER_ADDR to node0's IP/hostname
#   - On node1..N-1: set NODE_RANK accordingly and use the same MASTER_ADDR/PORT/NNODES
#   - Ensure the same code, conda env, dataset/model paths are visible on all nodes

# NCCL/IB settings (kept from the single-node script; adjust to your cluster)
export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

# Multi-node parameters (override with env vars or edit here)
export MASTER_ADDR=${MASTER_ADDR:-"node0.example.com"}
export MASTER_PORT=${MASTER_PORT:-29501}
export NNODES=${NNODES:-2}
export NODE_RANK=${NODE_RANK:-0}          # set per node
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}  # GPUs per node

# Optional debug/tuning
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# export NCCL_SOCKET_IFNAME=eth0   # uncomment and set to your NIC if needed
# export NCCL_IB_DISABLE=1         # uncomment if no InfiniBand is available
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

LATENT_SIZE=6
CE_EMPHASIZE_FACTOR=1.0
SAVE_CKPT=08_26-avt_v2_stage1-latent${LATENT_SIZE}-ce_factor${CE_EMPHASIZE_FACTOR}

# Activate environment and cd to project
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/Latent_Think/abstract-visual-token
export TOKENIZERS_PARALLELISM=false

# Multi-node torchrun
torchrun \
  --nnodes $NNODES --node-rank $NODE_RANK \
  --nproc-per-node $GPUS_PER_NODE \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  -m src.main \
  --epochs 10 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_v2_stage1" \
  --data_path "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoF/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoM_w_MathVista/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/PixelReasoner/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/ReFocus/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_count/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_visual_search/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_geometry/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_maze/filtered_train_short3000_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/VTS_1/filtered_train_short3000_w_metadata.json" \
  --log_file "./log.txt" \
  --load_model_path "/ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen2.5-VL-7B-Instruct" \
  --save_model_path /mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v2_stage1/${SAVE_CKPT} \
  --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name ${SAVE_CKPT}
