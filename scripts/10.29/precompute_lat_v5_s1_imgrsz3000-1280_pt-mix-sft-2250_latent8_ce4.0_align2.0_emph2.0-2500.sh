export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06-temp/Latent_Think/abstract-visual-token
LATENT_SIZE=8
TEACHER_EMPHASIZE_LATENT_WEIGHT=2.0
TEACHER_ALIGN_WEIGHT=2.0
TEACHER=10.26_v5_s1-imgrsz3000-1280_mix_pt-sft-2250_latent8_ce4.0_align-wt${TEACHER_ALIGN_WEIGHT}_emph-wt${TEACHER_EMPHASIZE_LATENT_WEIGHT}/checkpoint-2500

torchrun --nproc-per-node=8 --master-port=29501 -m src.precompute_teacher_latents \
  --bsz 1 \
  --task "mm-reasoning" \
  --data_path \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Visual_CoT/filtered_train_10.7_1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoM_w_MathVista/filtered_train_w_metadata_9.1_max_seq_len2500.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/ReFocus/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_visual_search/filtered_train_w_metadata_9.24_further_washed_max_seq_len3000.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_geometry/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_count/filtered_train_w_metadata_9.25_max_seq_len4096_max_seq_len3000.json" \
  --log_file "./log.txt" \
  --load_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/avt_v5/${TEACHER} \
  --save_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/precomputed_teacher_reps/${TEACHER} \
  --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size ${LATENT_SIZE} \
  --output_hidden_states \
  --resume



export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06-temp/Latent_Think/abstract-visual-token
LATENT_SIZE=8
TEACHER_EMPHASIZE_LATENT_WEIGHT=2.0
TEACHER_ALIGN_WEIGHT=2.0
TEACHER=10.26_v5_s1-imgrsz3000-1280_mix_pt-sft-2250_latent8_ce4.0_align-wt${TEACHER_ALIGN_WEIGHT}_emph-wt${TEACHER_EMPHASIZE_LATENT_WEIGHT}/checkpoint-2500

torchrun --nproc-per-node=8 --master-port=29502 -m src.precompute_teacher_latents \
  --bsz 1 \
  --task "mm-reasoning" \
  --data_path \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Visual_CoT/filtered_train_10.7_2.json" \
  --log_file "./log.txt" \
  --load_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/avt_v5/${TEACHER} \
  --save_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/precomputed_teacher_reps/${TEACHER} \
  --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size ${LATENT_SIZE} \
  --output_hidden_states \
  --resume



export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06-temp/Latent_Think/abstract-visual-token
LATENT_SIZE=8
TEACHER_EMPHASIZE_LATENT_WEIGHT=2.0
TEACHER_ALIGN_WEIGHT=2.0
TEACHER=10.26_v5_s1-imgrsz3000-1280_mix_pt-sft-2250_latent8_ce4.0_align-wt${TEACHER_ALIGN_WEIGHT}_emph-wt${TEACHER_EMPHASIZE_LATENT_WEIGHT}/checkpoint-2500

torchrun --nproc-per-node=8 --master-port=29503 -m src.precompute_teacher_latents \
  --bsz 1 \
  --task "mm-reasoning" \
  --data_path \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Visual_CoT/filtered_train_10.7_3.json" \
  --log_file "./log.txt" \
  --load_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/avt_v5/${TEACHER} \
  --save_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/precomputed_teacher_reps/${TEACHER} \
  --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size ${LATENT_SIZE} \
  --output_hidden_states \
  --resume




export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06-temp/Latent_Think/abstract-visual-token
LATENT_SIZE=8
TEACHER_EMPHASIZE_LATENT_WEIGHT=2.0
TEACHER_ALIGN_WEIGHT=2.0
TEACHER=10.26_v5_s1-imgrsz3000-1280_mix_pt-sft-2250_latent8_ce4.0_align-wt${TEACHER_ALIGN_WEIGHT}_emph-wt${TEACHER_EMPHASIZE_LATENT_WEIGHT}/checkpoint-2500

torchrun --nproc-per-node=8 --master-port=29504 -m src.precompute_teacher_latents \
  --bsz 1 \
  --task "mm-reasoning" \
  --data_path \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Visual_CoT/filtered_train_10.7_4.json" \
  --log_file "./log.txt" \
  --load_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/avt_v5/${TEACHER} \
  --save_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/precomputed_teacher_reps/${TEACHER} \
  --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size ${LATENT_SIZE} \
  --output_hidden_states \
  --resume



export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06-temp/Latent_Think/abstract-visual-token
LATENT_SIZE=8
TEACHER_EMPHASIZE_LATENT_WEIGHT=2.0
TEACHER_ALIGN_WEIGHT=2.0
TEACHER=10.26_v5_s1-imgrsz3000-1280_mix_pt-sft-2250_latent8_ce4.0_align-wt${TEACHER_ALIGN_WEIGHT}_emph-wt${TEACHER_EMPHASIZE_LATENT_WEIGHT}/checkpoint-2500

torchrun --nproc-per-node=8 --master-port=29505 -m src.precompute_teacher_latents \
  --bsz 1 \
  --task "mm-reasoning" \
  --data_path \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_arc_agi/raw_train_w_obs_w_metadata_swap.json" \
  "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_checkers/raw_train_w_obs_w_metadata_swap.json" \
  "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_connect_four/raw_train_w_obs_w_metadata_swap.json" \
  "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_rpm/raw_train_w_obs_w_metadata_swap.json" \
  "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_tetris/raw_train_w_obs_w_metadata_swap.json" \
  /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_maze/filtered_train_short3000_w_metadata_9.25_further_washed.json \
  --log_file "./log.txt" \
  --load_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/avt_v5/${TEACHER} \
  --save_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/precomputed_teacher_reps/${TEACHER} \
  --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size ${LATENT_SIZE} \
  --output_hidden_states \
  --resume



