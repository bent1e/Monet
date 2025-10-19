export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
CKPT=after9.28/Qwen2.5-VL-7B-Instruct-9.27_avt_v2_stage1_latent12_ce5.0_align-wt0.0001_emph-wt1.0
torchrun --nproc-per-node=2 --master-port=29501 -m src.precompute_teacher_latents \
    --bsz 1 \
    --task "mm-reasoning" \
    --data_path \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
  "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
    --log_file "./log.txt" \
    --load_model_path /home/dids/shiyang/checkpoints/${CKPT} \
    --save_model_path ./new/precomputed_teacher_latents/${CKPT} \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --output_hidden_states \
    --latent_size 12 \
    --alignment_layer all_layers
