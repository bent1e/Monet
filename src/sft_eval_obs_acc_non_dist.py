'''
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
python -m src.sft_eval_obs_acc_non_dist \
    --bsz 1 \
    --task "mm-reasoning" \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --data_path \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_9.24_further_washed.json" \
  "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
  "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
  "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train_w_metadata_9.25_max_seq_len4096_max_seq_len3000.json" \
    --log_file "./log.txt" \
    --load_model_path /home/dids/shiyang/checkpoints/after9.28/10.1_avt_sft_wo_maze_ce2.0/10.1_avt_sft_wo_maze_ce2.0

'''

import os as _early_os
# Disable parallelism in HuggingFace tokenizers to avoid fork-related warnings/deadlocks
_early_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import os
import shutil
from functools import partial
import torch
import torch.distributed as dist
import json
from pathlib import Path
import re
from new.avt_qwen_model import apply_qwen2_5_avt
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoTokenizer, AutoProcessor
from PIL import Image
import logging
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info
from src.utils import *
from src.task import *
from src.trainer import *
import random
import wandb

args = get_args()
config = Qwen2_5_VLConfig.from_pretrained(args.load_model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.load_model_path,
    config=config,
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(args.load_model_path, use_fast=False)

dataset_names = ""
for data_path in args.data_path:
    dataset_name = data_path.split("/")[-2]
    dataset_names += f"-{dataset_name}"
preprocess_function = task_preporcess_config[args.task]
all_train_dataset = []
for data_path in args.data_path:
    if data_path.endswith('.jsonl'):
        train_dataset = load_jsonl_dataset(data_path)
    elif data_path.endswith('.json'):
        train_dataset = load_json_dataset(data_path)
    all_train_dataset.extend(train_dataset[:])

if args.shuffle_train:
    random.seed(42)
    random.shuffle(all_train_dataset)

train_dataset = []
cur_max = -1
for i, sample in tqdm(enumerate(all_train_dataset[:]),
                      desc="Collecting training data and length check...",
                      total=len(all_train_dataset)):
    if 'avt' in args.stage:
        processed = preprocess_function(sample, dataset_root=args.dataset_root)
    else:
        processed = preprocess_function(sample)
    if processed is not None:
        train_dataset.append(processed)

# ================= Prepare tokenizer special ids and misc =================
processor.tokenizer.add_tokens("<abs_vis_token_pad>", special_tokens=True)
processor.tokenizer.add_tokens("<abs_vis_token>", special_tokens=True)
processor.tokenizer.add_tokens("</abs_vis_token>", special_tokens=True)
processor.tokenizer.add_tokens("<observation>", special_tokens=True)
processor.tokenizer.add_tokens("</observation>", special_tokens=True)

latent_start_idx = processor.tokenizer("<abs_vis_token>", return_tensors="pt")["input_ids"][0]
latent_end_idx = processor.tokenizer("</abs_vis_token>", return_tensors="pt")["input_ids"][0]
latent_pad_idx = processor.tokenizer("<abs_vis_token_pad>", return_tensors="pt")["input_ids"][0]
observation_start_idx = processor.tokenizer("<observation>", return_tensors="pt")["input_ids"][0]
observation_end_idx = processor.tokenizer("</observation>", return_tensors="pt")["input_ids"][0]
end_pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]
answer_start_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
img_start_idx = processor.tokenizer("<|vision_start|>", return_tensors="pt")["input_ids"][0]
img_end_idx = processor.tokenizer("<|vision_end|>", return_tensors="pt")["input_ids"][0]
img_pad_idx = processor.tokenizer("<|image_pad|>", return_tensors="pt")["input_ids"][0]

SPECIAL_id = {
    "v_start": img_start_idx,
    "v_end": img_end_idx,
    "img_pad": img_pad_idx,
    "abs_start": latent_start_idx,
    "abs_end": latent_end_idx,
    "abs_pad": latent_pad_idx,
    "obs_start": observation_start_idx,
    "obs_end": observation_end_idx,
}

# Resize embeddings to include newly added tokens if needed
try:
    new_vocab_size = len(processor.tokenizer)
    model.resize_token_embeddings(new_vocab_size)
    model.config.vocab_size = new_vocab_size
except Exception:
    pass

# Configure latent ids on model for downstream logic
model.config.latent_token_id = int(latent_pad_idx)
model.config.latent_start_id = int(latent_start_idx)
model.config.latent_end_id = int(latent_end_idx)
model.config.answer_start_pattern = answer_start_pattern.tolist()

# Freeze visual to match training behavior and eval-only run
for p in model.visual.parameters():
    p.requires_grad = False

model.eval()
try:
    model.gradient_checkpointing_disable()
except Exception:
    pass


def _init_distributed_if_needed():
    """Initialize torch.distributed if launched with torchrun.

    Returns (rank, world_size, local_rank, is_main).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank, (rank == 0)


def _device(local_rank: int | None = None) -> torch.device:
    """Return the CUDA device matching local_rank if available, else CPU."""
    if torch.cuda.is_available():
        if local_rank is None:
            # Fallback to env var if present
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")

def collate_fn_w_or_wo_helper_img(examples, alignment="boxed_start"):
    # Support wrapped examples providing sample_id
    batch = {}
    batch['metadata'] = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]
    batch_assistant_img_cnts = [sum(1 for step in examples[i][2]['content'] if step["type"] == "image") for i in range(len(examples))]
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]

    # replace <abs_vis_token></abs_vis_token> with <|vision_start|><|image_pad|><|vision_end|> for each <|im_start|>assistant content
    texts = [place_output_image_avt(text) for text in texts]
    
    ################################################
    # w helper img
    ################################################
    image_inputs, _ = process_vision_info(examples)
    image_inputs, new_sizes = resize_by_token_budget(image_inputs)
    teacher_texts = add_abs_vis_token_after_helper_img(texts, args.latent_size, "<abs_vis_token_pad>")
    teacher_batch = processor(text=teacher_texts, images=image_inputs, return_tensors="pt", padding=True)
    total_image_pads = 0
    for txt in texts:
        total_image_pads += txt.count("<|image_pad|>")
    assert total_image_pads == len(image_inputs)
    batch['teacher_pixel_values'] = teacher_batch['pixel_values']
    batch['teacher_image_grid_thw'] = teacher_batch['image_grid_thw']
    batch['teacher_input_ids'] = teacher_batch['input_ids']
    batch['teacher_attention_mask'] = teacher_batch['attention_mask']
    
    ################################################
    # wo helper img
    ################################################
    # replace <|vision_start|><|image_pad|><|vision_end|> with <abs_vis_token><abs_vis_token_pad>...</abs_vis_token> for each <|im_start|>assistant content
    student_texts = remove_helper_images(texts, args.latent_size, "<abs_vis_token_pad>")
    user_examples = remove_assistant_images(examples)
    user_image_inputs, _ = process_vision_info(user_examples)
    resize_ptr = 0
    if new_sizes is not None:
        for i, img in enumerate(user_image_inputs):
            img = img.resize(new_sizes[resize_ptr], Image.BICUBIC)
            user_image_inputs[i] = img
            resize_ptr += batch_assistant_img_cnts[i] + 1 # user_image_inputs only contain question images of each batch sample, so we need to skip the helper images in the new_sizes by adding batch_assistant_img_cnts[i]
    student_batch = processor(text=student_texts, images=user_image_inputs, return_tensors="pt", padding=True)
    total_image_pads = 0
    for txt in student_texts:
        total_image_pads += txt.count("<|image_pad|>")
    assert total_image_pads == len(user_image_inputs)
    batch['student_pixel_values'] = student_batch['pixel_values']
    batch['student_image_grid_thw'] = student_batch['image_grid_thw']
    batch["student_input_ids"] = student_batch["input_ids"]
    batch["student_attention_mask"] = student_batch["attention_mask"]

    observation_start_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, observation_start_idx)
    observation_end_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, observation_end_idx)
    batch["teacher_observation_poss"] = []
    assert len(observation_start_poss) == len(observation_end_poss)
    for start_poss, end_poss in zip(observation_start_poss, observation_end_poss):
        poss_of_a_sample = []
        if len(start_poss) > 0 and len(end_poss) > 0:
            assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
            for start, end in zip(start_poss, end_poss):
                poss_of_a_sample.extend(list(range(start, end)))
        batch["teacher_observation_poss"].append(poss_of_a_sample)

    observation_start_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, observation_start_idx)
    observation_end_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, observation_end_idx)
    batch["student_observation_poss"] = []
    assert len(observation_start_poss) == len(observation_end_poss)
    for start_poss, end_poss in zip(observation_start_poss, observation_end_poss):
        poss_of_a_sample = []
        if len(start_poss) > 0 and len(end_poss) > 0:
            assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
            for start, end in zip(start_poss, end_poss):
                poss_of_a_sample.extend(list(range(start, end)))
        batch["student_observation_poss"].append(poss_of_a_sample)

    batch["teacher_labels"] = generate_labels_after_multi_token_start(batch["teacher_input_ids"], answer_start_pattern, ignore_ids=[end_pad_token_idx, latent_pad_idx, observation_start_idx, observation_end_idx, latent_end_idx])
    batch["student_labels"] = generate_labels_after_multi_token_start(batch["student_input_ids"], answer_start_pattern, ignore_ids=[end_pad_token_idx, latent_pad_idx, observation_start_idx, observation_end_idx, latent_end_idx])

    return batch

def main():
    # Distributed init and device setup
    rank, world_size, local_rank, is_main = _init_distributed_if_needed()
    device = _device(local_rank)
    model.to(device)

    # Iterate data and precompute
    bs = max(1, int(getattr(args, 'bsz', 1)))  # Note: per-process batch size under torchrun
    total = len(train_dataset)

    # Build per-rank shard of indices (strided sharding keeps balance without extra copies)
    indices = list(range(total))
    shard = indices[rank::world_size] if (world_size > 1) else indices

    # ================= Resume support (conditional via --resume) =================
    resume_enabled = getattr(args, 'resume', False)
    resume_path = None
    processed_ids = set()
    w_helper_img_acc_cum_resume = 0.0
    wo_helper_img_acc_cum_resume = 0.0
    valid_cnt_resume = 0
    if resume_enabled:
        # Unified storage directory for progress JSON files
        unified_dir = Path('/mmu_vcg_ssd/shiyang06-temp/Latent_Think/abstract-visual-token/logs/obs_acc_analysis')
        if is_main:
            try:
                unified_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f'[RESUME] Failed to create directory {unified_dir}: {e}')
        if dist.is_initialized():
            dist.barrier()
        # Derive a safe model name from load_model_path
        raw_name = os.path.basename(str(args.load_model_path).rstrip('/')) or 'model'
        safe_model_name = re.sub(r'[^A-Za-z0-9_.-]', '_', raw_name)
        resume_path = unified_dir / f'{safe_model_name}_rank{rank}.json'
        if resume_path.exists():
            try:
                with resume_path.open('r', encoding='utf-8') as f:
                    saved = json.load(f)
                if saved.get('shard') == shard:
                    processed_ids = set(saved.get('processed_ids', []))
                    w_helper_img_acc_cum_resume = float(saved.get('w_sum', 0.0))
                    wo_helper_img_acc_cum_resume = float(saved.get('wo_sum', 0.0))
                    valid_cnt_resume = int(saved.get('cnt', 0))
                    if is_main:
                        print(f'[RESUME] Loaded resume state from {resume_path}, processed {len(processed_ids)} batches previously.')
                else:
                    if is_main:
                        print(f'[RESUME] Shard mismatch – ignoring existing resume file {resume_path}.')
            except Exception as e:
                if is_main:
                    print(f'[RESUME] Failed to load resume file {resume_path}: {e}')

    with torch.inference_mode():
        rng = range(0, len(shard), bs)
        pbar = tqdm(rng, desc="eval sft token acc", disable=not is_main)
        # Initialize with resume sums (0 if disabled)
        w_helper_img_acc_cum = w_helper_img_acc_cum_resume if resume_enabled else 0.0
        wo_helper_img_acc_cum = wo_helper_img_acc_cum_resume if resume_enabled else 0.0
        valid_cnt = valid_cnt_resume if resume_enabled else 0
        for i in pbar:
            cur_ids = shard[i:i + bs]
            if len(cur_ids) == 0:
                continue
            # Use starting sample id as batch identifier for resume
            batch_token = cur_ids[0]
            if resume_enabled and batch_token in processed_ids:
                continue  # already processed this batch earlier

            examples = [train_dataset[j] for j in cur_ids]
            batch = collate_fn_w_or_wo_helper_img(examples)

            # w helper img
            inputs = {
                "input_ids": batch['teacher_input_ids'].to(device),
                "attention_mask": batch['teacher_attention_mask'].to(device),
                "pixel_values": batch['teacher_pixel_values'].to(device),
                "image_grid_thw": batch['teacher_image_grid_thw'].to(device),
                "latent_mode": False,
                'labels': batch['teacher_labels'],
                'loss_type': ['ce'],
                'compute_emphasize_acc': True,
                'ce_emphasize_poss': batch['teacher_observation_poss']
            }
            outputs = model(**inputs)
            w_helper_img_acc_cum += float(outputs.mean_emphasize_acc)

            # wo helper img
            inputs = {
                "input_ids": batch['student_input_ids'].to(device),
                "attention_mask": batch['student_attention_mask'].to(device),
                "pixel_values": batch['student_pixel_values'].to(device),
                "image_grid_thw": batch['student_image_grid_thw'].to(device),
                "latent_mode": True,
                'labels': batch['student_labels'],
                'loss_type': ['ce'],
                'compute_emphasize_acc': True,
                'ce_emphasize_poss': batch['student_observation_poss']
            }
            outputs = model(**inputs)
            wo_helper_img_acc_cum += float(outputs.mean_emphasize_acc)
            valid_cnt += 1

            # Persist resume state (per rank) after each successful batch
            if resume_enabled and resume_path is not None:
                processed_ids.add(batch_token)
                try:
                    tmp_path = resume_path.with_suffix('.tmp')
                    with tmp_path.open('w', encoding='utf-8') as f:
                        json.dump({
                            'shard': shard,
                            'processed_ids': list(processed_ids),
                            'w_sum': w_helper_img_acc_cum,
                            'wo_sum': wo_helper_img_acc_cum,
                            'cnt': valid_cnt,
                        }, f)
                    tmp_path.replace(resume_path)
                except Exception as e:
                    if is_main:
                        print(f'[RESUME] Failed to write resume file {resume_path}: {e}')

        # Aggregate across ranks if distributed
        if dist.is_initialized():
            t = torch.tensor([w_helper_img_acc_cum, wo_helper_img_acc_cum, float(valid_cnt)], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            w_helper_img_acc_cum, wo_helper_img_acc_cum, valid_cnt = t.tolist()

        if is_main:
            w_acc = (w_helper_img_acc_cum / max(1.0, valid_cnt)) if valid_cnt else 0.0
            wo_acc = (wo_helper_img_acc_cum / max(1.0, valid_cnt)) if valid_cnt else 0.0
            print(f"w helper img avg token acc: {w_acc}")
            print(f"wo helper img avg token acc: {wo_acc}")
            print(f"difference (w - wo): {w_acc - wo_acc}")
            # Keep resume JSON for record; user can delete manually if desired.
            if resume_enabled and resume_path is not None:
                print(f"[RESUME] Progress file retained at {resume_path}")

    # Clean up distributed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
