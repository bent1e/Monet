export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /data1/qxwang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

LATENT_SIZE=8
TEACHER=10.5_v5_stage1_pt_womaze_10ep_latent8_ce4.0_align-wt2.0_emph-wt2.0
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=4 --master-port=29501 -m src.precompute_teacher_latents \
  --bsz 1 \
  --task "mm-reasoning" \
  --data_path "./new/created_dataset/filtered_data/Zebra_CoT_arc_agi/raw_train_w_obs_w_metadata.json" \
  --log_file "./log.txt" \
  --load_model_path /data1/qxwang/checkpoints/after10.9/${TEACHER} \
  --save_model_path /data1/qxwang/checkpoints/precomputed_teacher_reps/${TEACHER} \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size ${LATENT_SIZE} \
  --output_hidden_states

export CUDA_VISIBLE_DEVICES=0
LATENT_SIZE=8
CE_EMPHASIZE_FACTOR=4.0
ALIGNMENT_WEIGHT=2.0
torchrun --nproc-per-node=1 --master-port=29501 -m src.main\
  --epochs 5 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --stage "avt_v5_stage2" \
  --task "mm-reasoning" \
  --data_path "./new/created_dataset/filtered_data/Zebra_CoT_arc_agi/raw_train_w_obs_w_metadata.json" \
  --log_file "./log.txt" \
  --load_model_path /data1/qxwang/checkpoints/after10.9/${TEACHER} \
  --save_model_path /data1/qxwang/checkpoints/precomputed_teacher_reps/${TEACHER} \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size ${LATENT_SIZE} \
  --alignment_weight ${ALIGNMENT_WEIGHT} \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR} \
  --teacher_reps_dir /data1/qxwang/checkpoints/precomputed_teacher_reps/${TEACHER} \
  --alignment_layer all_layers