
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -x


export PYTHONUNBUFFERED=1
export WANDB_API_KEY=0bc4e54288b51054cdf77a4a48e49880984a6ca7
export DEEPSEEK_API_KEY=sk-a62bc7c0899a47dba605e3d3ab332e37
export AVT_DEBUG=0

unset LD_PRELOAD
unset NCCL_TOPO_FILE
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# 可选：拉长 worker 注册超时，避免误判为超时
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=120
export VLLM_NO_USAGE_STATS=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DASHBOARD=1
export RAY_DASHBOARD_ENABLED=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_NUM_CPUS=16
export RAY_NUM_GPUS=8
# 强制本地 Ray，避免误连到已有集群
export USE_RAY_LOCAL=1
export RAY_ADDRESS=local
# 让 Prometheus 监控端口自动选择空闲端口，避免 Address already in use
export RAY_METRICS_EXPORT_PORT=0
# 将 Ray 日志输出到 stderr，便于在控制台定位问题
export RAY_LOG_TO_STDERR=0
# 可选：将 Ray 切换到单进程 local_mode 以排查（置 1 开启）
export RAY_LOCAL_MODE=0
# Ray 在 OOM 时让任务退出并提供更详细日志
export RAY_task_exit_on_oom=1
# Ray 本地缓存与对象溢出到磁盘，减少 /dev/shm 压力（使用短路径，避免 AF_UNIX 107 字节限制）
export RAY_SPILL_DIR=/tmp/ray_spill
export RAY_TMPDIR=/tmp/ray_tmp
mkdir -p /tmp/ray_spill /tmp/ray_tmp
# 限制对象存储内存，进一步降低 /dev/shm 压力（单位：字节）
export RAY_OBJECT_STORE_MEMORY=134217728
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=300
source /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Lib/miniconda3/bin/activate
conda activate sy-easyr1
cd /mmu_vcg_ssd/shiyang06-temp/Latent_Think/Easyr1-temp

MODEL_PATH=/mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/avt_v5/10.21_v5_s2_imgrsz3000_mix_pt-sft1250_s1-pt-sft1250_teacher-al2.0-emph2.0-2250_latent8_ce4.0_align2.0/checkpoint-1250
latent_size=10
export ABS_VIS_LATENT_SIZE=${latent_size}
ROLLOUT_N=8
TEMPERATURE=0.5
GPU_UTILIZATION=0.85
SELECT_ACC_THRESHOLD=0.6
KL_COEF=0.01
ORI_BSZ=64 # 32
ONLINE_ACCUM_SIZE=256 # 128
TRAIN_MAX_SAMPLES=-1 # 512
VAL_MAX_SAMPLES=-1 # 16
N_GPUS_PER_NODE=8
TENSOR_PARALLEL_SIZE=1
AVT_RL_SIGMA=10.0
LENGTH_PENALTY_COEF=0.001
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=4096

python -m verl.trainer.main \
    config=examples/config_avt.yaml \
    data.train_files=/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Thyme-RL/data \
    data.val_files=/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Thyme-RL/data/val \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=AVTv5_mix-1250-2250-1250_latent${latent_size}_temp${TEMPERATURE}_tacc${SELECT_ACC_THRESHOLD}_rlsgm${AVT_RL_SIGMA}_APIjudge_lenPenalty${LENGTH_PENALTY_COEF}_use-lat-rwd \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.temperature=${TEMPERATURE} \
    worker.rollout.gpu_memory_utilization=${GPU_UTILIZATION} \
    worker.rollout.enable_chunked_prefill=true \
    worker.rollout.sampling_strategy=avt \
    worker.rollout.max_num_seqs=128 \
    worker.rollout.avt.select_acc_threshold=${SELECT_ACC_THRESHOLD} \
    worker.rollout.online_difficulty_sampling=true \
    worker.reward.reward_function=./examples/reward_function/avt_reward_function.py:compute_score_w_prev_correctness \
    worker.rule_based_judge.judge_function=./examples/reward_function/avt_reward_function.py:rule_then_api_batch_judge \
    worker.rule_based_judge.api_name="gemini-2.5-pro" \
    worker.actor.avt_rl_sigma=${AVT_RL_SIGMA} \
    worker.ref.avt_rl_sigma=${AVT_RL_SIGMA} \
    algorithm.kl_coef=${KL_COEF} \
    data.rollout_batch_size=${ORI_BSZ} \
    data.online_accum_size=${ONLINE_ACCUM_SIZE} \
    data.dataloader_num_workers=8 \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    

