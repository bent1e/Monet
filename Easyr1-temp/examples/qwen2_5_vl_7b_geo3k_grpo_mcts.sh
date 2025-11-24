#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=6,7,8,9  #
MODEL_PATH=/data1/qxwang/checkpoints/Qwen2.5-VL-7B-Instruct

export NCCL_P2P_DISABLE=1 #  without this, will stuck at "(WorkerDict pid=3614951) NCCL version 2.21.5+cuda12.4"
export WANDB_API_KEY=0bc4e54288b51054cdf77a4a48e49880984a6ca7

python3 -m verl.trainer.main \
    config=examples/config_process_reward.yaml \
    data.train_files=/data1/qxwang/datasets/multimodal/geometry3k/data/train-00000-of-00001.parquet \
    data.val_files=/data1/qxwang/datasets/multimodal/geometry3k/data/validation-00000-of-00001.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo_mcts_bsz256_r5_d10_mind7_t1.0_wSWFCA \
    trainer.n_gpus_per_node=4 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.sampling_strategy=mctc
