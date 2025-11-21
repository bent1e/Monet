export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7



LATENT_SIZE=8
CE_EMPHASIZE_FACTOR=4.0
ALIGNMENT_WEIGHT=2.0
TEACHER_ALIGN_WEIGHT=2.0
TEACHER_EMPHASIZE_LATENT_WEIGHT=2.0
SAVE_CKPT=10.21_aux_womme_v5_s2_imgrsz3000_mix_pt-sft1250_s1-pt-sft1250_teacher-al${TEACHER_ALIGN_WEIGHT}-emph${TEACHER_EMPHASIZE_LATENT_WEIGHT}-2250_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_align${ALIGNMENT_WEIGHT}
LOAD_CKPT=10.21_v5_s2_imgrsz3000_mix_pt-sft1250_s1-pt-sft1250_teacher-al${TEACHER_ALIGN_WEIGHT}-emph${TEACHER_EMPHASIZE_LATENT_WEIGHT}-2250_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_align${ALIGNMENT_WEIGHT}/checkpoint-1250
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06-temp/Latent_Think/abstract-visual-token
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=8 --master-port=29501 -m src.main \
  --epochs 5 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_sft" \
  --data_path \
      "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/aux_data/aux_LogicVista.json" \
      "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/aux_data/aux_MathVision.json" \
  --log_file "./log.txt" \
  --load_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/avt_v5/${LOAD_CKPT} \
  --save_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/avt_v5/${SAVE_CKPT} \
  --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name ${SAVE_CKPT} \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR} \
  --save_freq 2 \
  --allow_no_observation \
  --log_freq 2