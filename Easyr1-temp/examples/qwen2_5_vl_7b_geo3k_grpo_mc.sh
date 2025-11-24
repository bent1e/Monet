#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4,5,6,7 #
MODEL_PATH=/data1/qxwang/checkpoints/Qwen2.5-VL-7B-Instruct

export NCCL_P2P_DISABLE=1 #  without this, will stuck at "(WorkerDict pid=3614951) NCCL version 2.21.5+cuda12.4"
export WANDB_API_KEY=0bc4e54288b51054cdf77a4a48e49880984a6ca7
export RAY_TMPDIR=/data1/tmp/ray

SELECT_ACC_THRESHOLD=0.6
KL_COEF=0.01
ORI_BSZ=256
PR_BSZ=512
MAX_SAMPLES=-1
SHUFFLE_GROUPS=true
RMV_ALL_CORRECT=true
DROP_LAST_STEPS=1
python3 -m verl.trainer.main \
    config=examples/config_process_reward.yaml \
    data.train_files=/data1/qxwang/datasets/multimodal/geometry3k/data/train-00000-of-00001.parquet \
    data.val_files=/data1/qxwang/datasets/multimodal/geometry3k/data/validation-00000-of-00001.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo_mc_prbsz${PR_BSZ}_oribsz${ORI_BSZ}_tsim0.85_tacc${SELECT_ACC_THRESHOLD}_droplast${DROP_LAST_STEPS}_easyr1judgeV1_klcoef${KL_COEF}_0.9rwd_sort_shuffle${SHUFFLE_GROUPS}_rmvAC${RMV_ALL_CORRECT} \
    trainer.n_gpus_per_node=4 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.sampling_strategy=mc \
    worker.rollout.mc.select_acc_threshold=${SELECT_ACC_THRESHOLD} \
    worker.rollout.mc.shuffle_groups_across_samples=${SHUFFLE_GROUPS} \
    worker.rollout.mc.remove_all_correct_groups=${RMV_ALL_CORRECT} \
    worker.rollout.mc.drop_last_steps=${DROP_LAST_STEPS} \
    algorithm.kl_coef=${KL_COEF} \
    data.rollout_batch_size=${ORI_BSZ} \
    data.pr_batch_size=${PR_BSZ} \
    data.max_samples=${MAX_SAMPLES} 

