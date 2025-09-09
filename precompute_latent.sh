export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
CKPT=9.1_ablation_avt_v2_stage1_latent24_ce5.0_mask-q-img_align-vis-lat-proj-0.0001/checkpoint-2300
torchrun --nproc-per-node=4 --master-port=29501 -m src.precompute_teacher_latent \
    --bsz 1 \
    --task "mm-reasoning" \
    --stage "avt_v2_precompute_latent" \
    --data_path \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_9.1.json" \
    --log_file "./log.txt" \
    --load_model_path /home/dids/shiyang/checkpoints/avt_v2_stage1/${CKPT} \
    --save_model_path ./new/precomputed_teacher_latents/${CKPT} \
    --latent_size 24 \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --output_hidden_states