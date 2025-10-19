#!/usr/bin/env bash
set -e

BASE_DIR="/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3"
HF_UPLOADER="/mmu_vcg_ssd/shiyang06/Tool/huggingface.py"

if [ $# -lt 2 ]; then
    echo "用法: $0 <SAVE_CKPT> <NUM1> [NUM2 ...]"
    exit 1
fi

SAVE_CKPT="$1"
shift   # 剩下的参数就是 NUM 列表

for num in "$@"; do
    src="${BASE_DIR}/${SAVE_CKPT}/checkpoint-${num}"
    out="${BASE_DIR}/${SAVE_CKPT}_checkpoint-${num}.zip"
    if [ -d "$src" ]; then
        echo "打包 $src -> $out"
        (cd "$(dirname "$src")" && zip -r -9 "$out" "$(basename "$src")" >/dev/null)
        echo "上传 $out"
        python "$HF_UPLOADER" --item "$out"
    else
        echo "跳过: $src 不存在"
    fi
done
