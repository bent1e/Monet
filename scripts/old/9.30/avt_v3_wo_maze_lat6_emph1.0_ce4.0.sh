export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

LATENT_SIZE=6
CE_EMPHASIZE_FACTOR=4.0
ALIGN_VISION_LATENT_LOSS_WEIGHT=0.0001
EMPHASIZE_LATENT_WEIGHT=1.0
SAVE_CKPT=9.30_avt_v3_wo_maze_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_align-wt${ALIGN_VISION_LATENT_LOSS_WEIGHT}_emph-wt${EMPHASIZE_LATENT_WEIGHT}
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/Latent_Think/abstract-visual-token
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=8 --master-port=29501 -m src.main \
  --epochs 6 \
  --bsz 1 \
  --grad_accum_steps 8 \
  --task "mm-reasoning" \
  --stage "avt_v2_stage1" \
  --data_path "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoM_w_MathVista/filtered_train_w_metadata_9.1_max_seq_len2500.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/ReFocus/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_visual_search/filtered_train_w_metadata_9.24_further_washed_max_seq_len3000.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_geometry/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_count/filtered_train_w_metadata_9.25_max_seq_len4096_max_seq_len3000.json" \
  --log_file "./log.txt" \
  --load_model_path /ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen2.5-VL-7B-Instruct \
  --save_model_path /mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/${SAVE_CKPT} \
  --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name ${SAVE_CKPT} \
  --latent_size ${LATENT_SIZE} \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR} \
  --use_align_vision_latent_loss_projector \
  --align_vision_latent_loss_weight ${ALIGN_VISION_LATENT_LOSS_WEIGHT} \
  --emphasize_latent_weight ${EMPHASIZE_LATENT_WEIGHT}

src=/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/${SAVE_CKPT}
output=/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/${SAVE_CKPT}.zip
include_subdir=False

if [ -z "$src" ] || [ -z "$output" ] || [ -z "$include_subdir" ]; then
    echo "用法: $0 <source_folder> <output_zip> <include_subdir:True|False>"
    exit 1
fi

if [ "$include_subdir" = "True" ]; then
    # 压缩整个目录（包含子目录），只保留最后一级目录名
    (cd "$(dirname "$src")" && zip -r "$output" "$(basename "$src")")
else
    # 只压缩目录下的文件，不包含子目录，保留最后一级目录名
    (cd "$(dirname "$src")" && zip "$output" "$(basename "$src")"/*)
fi

python /mmu_vcg_ssd/shiyang06/Tool/huggingface.py --item $output