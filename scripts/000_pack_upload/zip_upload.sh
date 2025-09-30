#!/bin/bash

# 定义多个任务
src_list=(
    "/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/9.29_avt_v3_pt_wo_maze_wo_maze_latent6_ce2.0_align-wt0.0001_emph-wt1.0/checkpoint-100"
    "/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/9.29_avt_v3_pt_wo_maze_wo_maze_latent6_ce2.0_align-wt0.0001_emph-wt1.0/checkpoint-200"
    "/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/9.29_avt_v3_pt_wo_maze_latent6_ce2.0_align-wt0.0001_emph-wt1.0/checkpoint-100"
    
)
output_list=(
    "/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/9.29_avt_v3_pt_wo_maze_wo_maze_latent6_ce2.0_align-wt0.0001_emph-wt1.0-checkpoint-100.zip"
    "/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/9.29_avt_v3_pt_wo_maze_wo_maze_latent6_ce2.0_align-wt0.0001_emph-wt1.0-checkpoint-200.zip"
    "/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/9.29_avt_v3_pt_wo_maze_latent6_ce2.0_align-wt0.0001_emph-wt1.0-checkpoint-100.zip"
)
include_subdir_list=(
    "False"
    "False"
    "False"
)

# 定义压缩函数
compress_and_upload() {
    local src="$1"
    local output="$2"
    local include_subdir="$3"

    echo "开始处理: $src -> $output"

    if [ -z "$src" ] || [ -z "$output" ] || [ -z "$include_subdir" ]; then
        echo "用法: $0 <source_folder> <output_zip> <include_subdir:True|False>"
        return
    fi

    if [ "$include_subdir" = "True" ]; then
        # 压缩整个目录（包含子目录）
        (cd "$(dirname "$src")" && zip -r "$output" "$(basename "$src")")
    else
        # 只压缩目录下的文件，不包含子目录
        (cd "$(dirname "$src")" && zip "$output" "$(basename "$src")"/*)
    fi

    # 上传到 Huggingface
    python /mmu_vcg_ssd/shiyang06/Tool/huggingface.py --item "$output"
}

# 并行执行
for i in "${!src_list[@]}"; do
    compress_and_upload "${src_list[i]}" "${output_list[i]}" "${include_subdir_list[i]}" &
done

# 等待所有子任务完成
wait
echo "所有任务已完成 ✅"