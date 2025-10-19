export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/Latent_Think/abstract-visual-token/new

python -m eval_code.eval \
    --dataset-name Visual_CoT \
    --dataset_path /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Visual-CoT/viscot_363k_1.json \
    --dataset_images_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Visual-CoT/cot_images_tar_split \
    --out-root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/ \
    --policy-model-path /ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen2.5-VL-7B-Instruct \
    --devices 0,1,2,3,4,5,6,7 \
    --policy_mllm_tensor_parallel_size 1 \
    --start-id 0 \
    --judge_mode data_spec api \
    --api_name gemini-2.5-pro \
    --api_max_workers 64 \
    --batch 1024 \
    --policy_batch 256 \
    --groups_per_gpu 2 \
    --gpu_memory_utilization 0.48