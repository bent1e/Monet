conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
    --dataset-name Visual_CoT \
    --dataset_path /home/dids/shiyang/datasets/Visual-CoT/viscot_363k_1.json \
    --dataset_images_root /home/dids/shiyang/datasets/Visual-CoT/cot_images_tar_split \
    --out-root ./created_dataset/filtered_data/ \
    --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
    --devices 0,1,2,3 \
    --policy_mllm_tensor_parallel_size 1 \
    --start-id 0 \
    --limit 10000 \
    --judge_mode data_spec api \
    --api_name deepseek-chat \
    --api_max_workers 64 \
    --policy_batch 256 \
    --groups_per_gpu 2 \
    --gpu_memory_utilization 0.48