import os as _early_os
# Disable parallelism in HuggingFace tokenizers to avoid fork-related warnings/deadlocks
_early_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import shutil
from functools import partial
import torch
from new.avt_qwen_model import apply_qwen2_5_avt
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoTokenizer, AutoProcessor
from PIL import Image
import logging
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info
import torch.distributed as dist
from src.utils import *
from src.task import *
from src.trainer import *
import random
import wandb
args=get_args()
config = Qwen2_5_VLConfig.from_pretrained(args.load_model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.load_model_path,
    config=config,
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(args.load_model_path, use_fast=False)

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
for i, sample in tqdm(enumerate(all_train_dataset[:]), desc="Collecting training data and length check...", total=len(all_train_dataset)):
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




def _device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def collate_fn_avt_v2_stage2(examples, alignment="boxed_start"):
    # Support wrapped examples providing sample_id
    batch = {}
    batch['metadata'] = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]
    batch_assistant_img_cnts = [sum(1 for step in examples[i][2]['content'] if step["type"] == "image") for i in range(len(examples))]
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]

    # replace <abs_vis_token></abs_vis_token> with <|vision_start|><|image_pad|><|vision_end|> for each <|im_start|>assistant content
    texts = [place_output_image_avt(text) for text in texts]
    
    ################################################
    # teacher
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
    # student
    ################################################
    # replace <|vision_start|><|image_pad|><|vision_end|> with <abs_vis_token><abs_vis_token_pad>...</abs_vis_token> for each <|im_start|>assistant content
    student_texts = replace_visual_spectial_tokens_avt(texts, args.latent_size, "<abs_vis_token_pad>")
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

    batch["attention_mask_4d"] = {"full_attention": build_4d_attn(
        input_ids=batch["student_input_ids"],
        pad_mask=batch["student_attention_mask"],
        token_ids=SPECIAL_id,
        large_neg=-1e5,
        mask_latent=getattr(args, 'mask_latent', False),
    ) }

    batch["student_alignment_poss"] = find_ids_poss(batch["student_input_ids"], answer_start_pattern, latent_pad_idx)
    batch["teacher_alignment_poss"] = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, latent_pad_idx)

    latent_start_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, latent_start_idx)
    latent_end_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, latent_end_idx)
    batch["ce_emphasize_poss"] = []
    for start_poss, end_poss in zip(latent_start_poss, latent_end_poss):
        poss_of_a_sample = []
        if len(start_poss) > 0 and len(end_poss) > 0:
            assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
            poss_of_a_sample.extend(start_poss)
            poss_of_a_sample.extend(end_poss)
        batch["ce_emphasize_poss"].append(poss_of_a_sample)


    # mask tokens of '<|im_start|>assistant', '<|endoftext|>', and '<abs_vis_token_pad>' 
    batch["student_labels"] = generate_labels_after_multi_token_start(batch["student_input_ids"], answer_start_pattern, ignore_ids=[end_pad_token_idx, latent_pad_idx])

    return batch

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


    return batch


def main():

    # ---- token-level error logging config ----
    # Only rank-0 writes jsonl to avoid duplication
    exp_name = args.load_model_path.split('/')[-1]
    token_error_log_interval = 1
    token_error_max_records = 20
    token_error_written = 0
    token_err_dir = os.path.join('./logs', 'token_errors')
    os.makedirs(token_err_dir, exist_ok=True)
    token_error_path = os.path.join(
        token_err_dir,
        f"token_errors_{exp_name}.jsonl",
    )

    # Initialize distributed if requested
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1
    if is_dist and not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))

    device = _device()
    model.to(device)

    # Iterate data and precompute
    bs = max(1, int(getattr(args, 'bsz', 1)))
    total = len(train_dataset)

    # Build index shards per rank
    indices = list(range(total))
    if is_dist:
        per = (total + world_size - 1) // world_size
        start_idx = rank * per
        end_idx = min(total, (rank + 1) * per)
        shard = indices[start_idx:end_idx]
    else:
        shard = indices

    if is_dist:
        dist.barrier()

    with torch.inference_mode():
        rng = range(0, len(shard), bs)
        pbar = tqdm(rng, desc=f"[rank {rank}] precompute", disable=(rank != 0))
        for i in pbar:
            cur_ids = shard[i:i+bs]
            examples = [train_dataset[j] for j in cur_ids]
            batch = collate_fn_avt_v2_stage2(examples)
            
            # Try to load precomputed teacher latents
            latents=None
            batch_metadata = batch.get('metadata', None)
            if batch_metadata is not None and args.teacher_latent_dir and os.path.isdir(args.teacher_latent_dir):
                latents_list = []
                for metadata in batch_metadata:
                    dataset_name = metadata['dataset_name']
                    sample_id = metadata['sample_id']
                    metadata_info = f"{dataset_name}_{sample_id}"
                    path = os.path.join(args.teacher_latent_dir, f"latent_{metadata_info}.pt")
                    if not os.path.isfile(path):
                        latents_list = []
                        break
                    data = torch.load(path, map_location='cpu')
                    latents_list.append(data['latent'])
                if batch_metadata is not None and len(latents_list) == len(batch_metadata):
                    latents = torch.stack([t if t.dim() == 2 else t.squeeze(0) for t in latents_list], dim=0).to(model.device)


            batch_size = args.bsz
            ce_patch_pos = []   # List[List[int]]
            for b in range(batch_size):
                latent_poss = torch.where(batch['student_input_ids'][b]==latent_pad_idx)[0].tolist()
                ce_patch_pos.append(latent_poss)
                assert len(latent_poss) == latents[b].shape[0]


            inputs = {
                'stage': 'avt_v2_stage1',               # produce latent_embeds as trainer_v2_stage2 expects
                'latent_mode': True,
                'input_ids': batch['student_input_ids'].to(device),
                'attention_mask': batch['student_attention_mask'].to(device),
                'pixel_values': batch['student_pixel_values'].to(device),
                'image_grid_thw': batch['student_image_grid_thw'].to(device),
                'labels': batch['student_labels'],
                'alignment_poss': batch['teacher_alignment_poss'],
                'loss_type': [],
                'output_latent_embeds': False,
                'ce_patch_pos': ce_patch_pos,
                'ce_patch_vec': latents
            }
            outputs = model(**inputs, return_dict=True)

            # ------- token-level error dump (rank-0, sampled by interval) -------

            if rank == 0 and (i % max(1, token_error_log_interval) == 0) and token_error_written < token_error_max_records:
                logits = getattr(outputs, 'logits', None)
                labels = inputs.get('labels', None)
                input_ids = inputs.get('input_ids', None)
                if logits is not None and labels is not None and input_ids is not None:
                    with torch.no_grad():
                        # Align with CE: compare logits[:, :-1] vs labels[:, 1:]
                        preds = torch.argmax(logits, dim=-1)
                        preds_shift = preds[:, :-1]
                        labels_shift = labels[:, 1:]
                        mask = labels_shift.ne(-100)
                        # Take the first sample in batch for logging to keep file small
                        b = 0
                        if preds_shift.size(0) > 0:
                            ps = preds_shift[b].detach().cpu().tolist()
                            ls = labels_shift[b].detach().cpu().tolist()
                            ms = mask[b].detach().cpu().tolist()
                            inp = input_ids[b].detach().cpu().tolist()
                            # Keep token strings for visualization (safe access)
                            tok = None
                            try:
                                tok = processor.tokenizer.convert_ids_to_tokens(inp)
                            except Exception:
                                tok = None
                            # Build record at full sequence length (unshifted indices)
                            # We store also the aligned region indices for convenience
                            record = {
                                'global_step': i,
                                'sample_index': 0,
                                'input_ids': inp,
                                'token_strs': tok,
                                'pred_ids_shift': ps,
                                'label_ids_shift': ls,
                                'mask_shift': ms,
                                'aligned_offset': 1,  # labels/logits alignment offset
                                'exp_name': exp_name,
                            }
                            # Optional: attach sample_id if provided by collator
                            if 'sample_id' in inputs:
                                try:
                                    record['sample_id'] = int(inputs['sample_id'][b])
                                except Exception:
                                    pass
                            # Write JSONL append
                            import json
                            with open(token_error_path, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            token_error_written += 1



    if is_dist:
        dist.barrier()



if __name__ == "__main__":
    main()