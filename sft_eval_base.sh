export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
LATENT_SIZE=6
TEACHER_LATENT=08_24-avt_v2_stage1-latent${LATENT_SIZE}-ce_factor1.0-step500
CKPT=Qwen2.5-VL-7B-Instruct
python -m src.sft_eval_non_dist \
    --bsz 1 \
    --task "mm-reasoning" \
    --stage "avt_v2_precompute_latent" \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --data_path \
    "./new/created_dataset/filtered_data/CoF/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata.json" \
    --log_file "./log.txt" \
    --load_model_path /home/dids/shiyang/checkpoints/${CKPT} \
    --latent_size ${LATENT_SIZE} \
    --eval_on_observation_token \
    --mask_latent \
    --observation_tokens_only_see_latent_tokens \
    --teacher_latent_dir /home/dids/shiyang/codes/abstract-visual-token/new/precomputed_teacher_latents/${TEACHER_LATENT}


export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
LATENT_SIZE=10
TEACHER_LATENT=08_24-avt_v2_stage1-latent${LATENT_SIZE}-ce_factor1.0-step500
CKPT=Qwen2.5-VL-7B-Instruct
python -m src.sft_eval_non_dist \
    --bsz 1 \
    --task "mm-reasoning" \
    --stage "avt_v2_precompute_latent" \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --data_path \
    "./new/created_dataset/filtered_data/CoF/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata.json" \
    --log_file "./log.txt" \
    --load_model_path /home/dids/shiyang/checkpoints/${CKPT} \
    --latent_size ${LATENT_SIZE} \
    --eval_on_observation_token \
    --observation_tokens_only_see_latent_tokens \
    --teacher_latent_dir /home/dids/shiyang/codes/abstract-visual-token/new/precomputed_teacher_latents/${TEACHER_LATENT}


export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
CKPT=Qwen2.5-VL-7B-Instruct
LATENT_SIZE=6
TEACHER_LATENT=08_24-avt_v2_stage1-latent${LATENT_SIZE}-ce_factor1.0-step500
CKPT=Qwen2.5-VL-7B-Instruct
python -m src.sft_eval_non_dist \
    --bsz 1 \
    --task "mm-reasoning" \
    --stage "avt_v2_precompute_latent" \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --data_path \
    "./new/created_dataset/filtered_data/CoF/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata.json" \
    --log_file "./log.txt" \
    --load_model_path /home/dids/shiyang/checkpoints/${CKPT} \
    --latent_size ${LATENT_SIZE} \
    --ce_emphasize_factor 1.0 \
    --eval_on_teacher_sequence \
    --eval_on_observation_tokens \
    --observation_tokens_only_see_image_tokens \
    --teacher_latent_dir /home/dids/shiyang/codes/abstract-visual-token/new/precomputed_teacher_latents/${TEACHER_LATENT}













conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python logs/plot_token_errors.py \
  --jsonl /home/dids/shiyang/codes/abstract-visual-token/logs/token_errors/token_errors_Qwen2.5-VL-7B-Instruct-08_20-avt_v2_stage1-latent6-ce_factor1.0-mask_latent-CoF.jsonl \
  --out ./logs/token_error_figs \
  --num 5 \
  --tokenizer /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct/