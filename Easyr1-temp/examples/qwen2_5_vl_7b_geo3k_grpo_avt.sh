#!/bin/bash
export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export LD_PRELOAD=/share/mayanqi/libnccl.so.2.27.5.ubuntu-cuda12.fix6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -x

pwd/python -c "import os; print(os.getcwd())"

latent_size=8
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=0bc4e54288b51054cdf77a4a48e49880984a6ca7
export ABS_VIS_LATENT_SIZE=${latent_size}
export AVT_DEBUG=0
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate sy-easyr1
cd /mmu_vcg_ssd/shiyang06-temp/Latent_Think/Easyr1-temp


#data.train_files=/home/dids/shiyang/datasets/geometry3k/data/train-00000-of-00001.parquet \
#data.val_files=/home/dids/shiyang/datasets/geometry3k/data/validation-00000-of-00001.parquet \
MODEL_PATH=/mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/avt_v5/10.16_v5_s2_rw_pt-sft3884_s1-pt-sft3884_teacher-al2.0-1000_latent8_ce4.0_align2.0/checkpoint-1000
ROLLOUT_N=8
TEMPERATURE=0.5
GPU_UTILIZATION=0.85
SELECT_ACC_THRESHOLD=0.6
KL_COEF=0.01
ORI_BSZ=128 # 32
ONLINE_ACCUM_SIZE=512 # 128
TRAIN_MAX_SAMPLES=-1 # 512
VAL_MAX_SAMPLES=-1 # 16
N_GPUS_PER_NODE=8
AVT_RL_SIGMA=10.0
LENGTH_PENALTY_COEF=0.001
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=4096
python -m verl.trainer.main \
    config=examples/config_avt.yaml \
    data.train_files=/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Thyme-RL/data \
    data.val_files=/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Thyme-RL/data/val \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=AVTv5_latent${latent_size}_temp${TEMPERATURE}_tacc${SELECT_ACC_THRESHOLD}_rlsgm${AVT_RL_SIGMA}_APIjudge_lenPenalty${LENGTH_PENALTY_COEF} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.temperature=${TEMPERATURE} \
    worker.rollout.gpu_memory_utilization=${GPU_UTILIZATION} \
    worker.rollout.sampling_strategy=avt \
    worker.rollout.avt.select_acc_threshold=${SELECT_ACC_THRESHOLD} \
    worker.rollout.online_difficulty_sampling=true \
    worker.reward.reward_function=./examples/reward_function/avt_reward_function.py:compute_score_w_prev_correctness \
    worker.rule_based_judge.judge_function=./examples/reward_function/avt_reward_function.py:rule_then_api_batch_judge \
    worker.rule_based_judge.api_name="gemini-2.5-pro" \
    worker.rule_based_judge.api_key="sk-a62bc7c0899a47dba605e3d3ab332e37" \
    worker.actor.avt_rl_sigma=${AVT_RL_SIGMA} \
    worker.ref.avt_rl_sigma=${AVT_RL_SIGMA} \
    algorithm.kl_coef=${KL_COEF} \
    data.rollout_batch_size=${ORI_BSZ} \
    data.online_accum_size=${ONLINE_ACCUM_SIZE} \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH}

