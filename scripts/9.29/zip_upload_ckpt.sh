#!/usr/bin/env bash
set -e

LATENT_SIZE=4
CE_EMPHASIZE_FACTOR=2.0
ALIGN_VISION_LATENT_LOSS_WEIGHT=0.0001
EMPHASIZE_LATENT_WEIGHT=1.0
SAVE_CKPT="9.29_avt_v3_pt_wo_maze_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_align-wt${ALIGN_VISION_LATENT_LOSS_WEIGHT}_emph-wt${EMPHASIZE_LATENT_WEIGHT}"
BASE_DIR="/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3"

# 使用方式： ./zip_upload_ckpt.sh 100 200 300

for num in "$@"; do
    src="${BASE_DIR}/${SAVE_CKPT}/checkpoint-${num}"
    out="${BASE_DIR}/${SAVE_CKPT}_checkpoint-${num}.zip"
    if [ -d "$src" ]; then
        echo "打包 $src -> $out"
        (cd "$(dirname "$src")" && zip -r -9 "$out" "$(basename "$src")" >/dev/null)
        echo "上传 $out"
        python "/mmu_vcg_ssd/shiyang06/Tool/huggingface.py" --item "$out"
    else
        echo "跳过: $src 不存在"
    fi
done
