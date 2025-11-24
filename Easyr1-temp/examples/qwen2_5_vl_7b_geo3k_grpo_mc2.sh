#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

#export NCCL_P2P_DISABLE=1 #  without this, will stuck at "(WorkerDict pid=3614951) NCCL version 2.21.5+cuda12.4"
export WANDB_API_KEY=0bc4e54288b51054cdf77a4a48e49880984a6ca7
#export RAY_TMPDIR=/data1/tmp/ray
# qwen2_5_vl_7b_geo_grpo_mc2_prbsz${PR_BSZ}_oribsz${ORI_BSZ}_tsim${T_SIM}_tacc${SELECT_ACC_THRESHOLD}_easyr1judgeV1_formatRWDV2_sort_shuffle${SHUFFLE_GROUPS}_rmvAC${RMV_ALL_CORRECT}_normstepadv${NORM_STEP_ADV}_ablPR${ABLATION_PR}
# qwen2_5_vl_7b_geo_grpo_mc2_oribsz${ORI_BSZ}_tacc${SELECT_ACC_THRESHOLD}_ablPR${ABLATION_PR}

# formatRWD # \\boxed
# formatRWDV2: \\boxed, penalize '###' without subsequent ' Step i:'
# default (if not marked): kl=0.01, t_sim=0.95, select_acc_threshold=0.6, pr_bsz=512, ori_bsz=256, sort, shuffle_groups=true, rmv_all_correct=true, drop_last_steps=1, norm_step_adv=true, ablation_pr=true
MODEL_PATH=/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct
GPU_UTILIZATION=0.7
T_SIM=0.95
SELECT_ACC_THRESHOLD=0.6
CORRECT_CLUSTER_THRESHOLD=0.25
KL_COEF=0.01
ORI_BSZ=8 #128
PR_BSZ=32 #512
MAX_SAMPLES=64 #-1
SHUFFLE_GROUPS=true
RMV_ALL_CORRECT=true
DROP_LAST_STEPS=1
NORM_STEP_ADV=true
ABLATION_PR=true
#EMB_MODEL_DEVICE="cuda:4"
EMB_MODEL_PATH="/home/dids/shiyang/checkpoints/Qwen3-Embedding-4B"
#CKPT_PATH="/data1/qxwang/codes/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_geo_grpo_mc2_oribsz256_tacc0.6_ablPRtrue/global_step_40"
CKPT_PATH="/home/dids/shiyang/codes/Easyr1-temp/checkpoints/easy_r1/qwen2_5_vl_7b_geo_grpo_mc2_prbsz512_oribsz128_tsim0.95_tcrtcls0.25_tacc0.6_easyr1judgeV1_0.1formatRWDV3_0.001lenPenalty_ablPRtrue/global_step_85"
#  trainer.load_checkpoint_path=${CKPT_PATH} \
python3 -m verl.trainer.main \
    config=examples/config_process_reward.yaml \
    data.train_files=/home/dids/shiyang/datasets/geometry3k/data/train-00000-of-00001.parquet \
    data.val_files=/home/dids/shiyang/datasets/geometry3k/data/validation-00000-of-00001.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo_mc2_prbsz${PR_BSZ}_oribsz${ORI_BSZ}_tsim${T_SIM}_tcrtcls${CORRECT_CLUSTER_THRESHOLD}_tacc${SELECT_ACC_THRESHOLD}_easyr1judgeV1_0.1formatRWDV3_0.001lenPenalty_ablPR${ABLATION_PR} \
    trainer.n_gpus_per_node=4 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.gpu_memory_utilization=${GPU_UTILIZATION} \
    worker.rollout.sampling_strategy=mc2 \
    worker.rollout.mc.step_hash_threshold=${T_SIM} \
    worker.rollout.mc.correct_cluster_threshold=${CORRECT_CLUSTER_THRESHOLD} \
    worker.rollout.mc.select_acc_threshold=${SELECT_ACC_THRESHOLD} \
    worker.rollout.mc.shuffle_groups_across_samples=${SHUFFLE_GROUPS} \
    worker.rollout.mc.remove_all_correct_groups=${RMV_ALL_CORRECT} \
    worker.rollout.mc.drop_last_steps=${DROP_LAST_STEPS} \
    worker.rollout.mc.normalize_step_wise_adv=${NORM_STEP_ADV} \
    worker.rollout.mc.ablation_process_reward=${ABLATION_PR} \
    worker.rollout.mc.embedding_model_path=${EMB_MODEL_PATH} \
    algorithm.kl_coef=${KL_COEF} \
    data.rollout_batch_size=${ORI_BSZ} \
    data.pr_batch_size=${PR_BSZ} \
    data.max_samples=${MAX_SAMPLES}
    

