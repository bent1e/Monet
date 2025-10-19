proxy_on
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
LATENT_SIZE=12
CE_EMPHASIZE_FACTOR=5.0
ALIGN_VISION_LATENT_LOSS_WEIGHT=0.0001
EMPHASIZE_LATENT_WEIGHT=2.0
SAVE_CKPT=9.28_tiny_avt_v2_stage2_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_align-wt${ALIGN_VISION_LATENT_LOSS_WEIGHT}_emph-wt${EMPHASIZE_LATENT_WEIGHT}
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
  --epochs 8 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_v2_stage2" \
  --data_path \
  "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
  "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
  --log_file "./log.txt" \
  --load_model_path  /home/dids/shiyang/checkpoints/after9.28/Qwen2.5-VL-7B-Instruct-9.27_avt_v2_stage1_latent12_ce5.0_align-wt0.0001_emph-wt1.0\
  --save_model_path /home/dids/shiyang/checkpoints/avt_v2_stage2/${SAVE_CKPT} \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size ${LATENT_SIZE} \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR} \
  --teacher_latent_dir /home/dids/shiyang/codes/abstract-visual-token/new/precomputed_teacher_latents/after9.28/Qwen2.5-VL-7B-Instruct-9.27_avt_v2_stage1_latent12_ce5.0_align-wt0.0001_emph-wt1.0 \
  --alignment_layer all_layers
  