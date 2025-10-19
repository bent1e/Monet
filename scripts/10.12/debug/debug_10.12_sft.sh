export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

# count has been cleaned
CE_EMPHASIZE_FACTOR=2.0
SAVE_CKPT=10.12_sft_game_ce${CE_EMPHASIZE_FACTOR}
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06-temp/Latent_Think/abstract-visual-token
export TOKENIZERS_PARALLELISM=false
python -m src.main \
  --epochs 4 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_sft" \
  --data_path \
  "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_arc_agi/raw_train_w_obs_w_metadata_swap.json" \
  --log_file "./log.txt" \
  --load_model_path /ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen2.5-VL-7B-Instruct \
  --save_model_path /mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/avt_sft/${SAVE_CKPT} \
  --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR}

  