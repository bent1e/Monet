export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# precompute teacher representations
# teacher info
LATENT_SIZE=10
CE_EMPHASIZE_FACTOR=4.0
ALIGNMENT_WEIGHT=2.0
EMPHASIZE_LATENT_WEIGHT=2.0

TEACHER=10.15_v5_s1_rw_pt-sft_gc_latent10_ce4.0_align-wt2.0_emph-wt2.0/checkpoint-1000
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06-temp/Latent_Think/abstract-visual-token
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=8 --master-port=29502 -m src.precompute_teacher_latents \
  --bsz 1 \
  --task "mm-reasoning" \
  --data_path \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Visual_CoT/supp.json" \
  --log_file "./log.txt" \
  --load_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/avt_v5/${TEACHER} \
  --save_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/precomputed_teacher_reps/${TEACHER} \
  --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size ${LATENT_SIZE} \
  --output_hidden_states