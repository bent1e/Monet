import os
import json
import copy
from typing import List, Tuple, Optional, Any

import torch
from src.utils import *
from src.utils import get_args
from transformers import AutoProcessor
from tqdm import tqdm

# --- batched wrapper for vision info (calls original single-sample API) ---
from qwen_vl_utils import process_vision_info as _process_vision_info_single

def process_vision_info_batched(conversations_batch: List[dict]) -> Tuple[List[Any], List[Any]]:
    """
    Batched wrapper for process_vision_info.
    It calls the original single-sample API per item, then aggregates results.
    Returns:
        all_image_inputs: List[image_inputs_per_sample]
        all_extra_info:   List[extra_info_per_sample]
    """
    all_image_inputs, all_extra_info = [], []
    for conv in conversations_batch:
        image_inputs, extra_info = _process_vision_info_single([conv])
        all_image_inputs.append(image_inputs)
        all_extra_info.append(extra_info)
    return all_image_inputs, all_extra_info


# ------------------------------
# Prepare a single sample (text + conversations only; images later in batch)
# ------------------------------
def _prepare_single_sample_for_batch(
    sample: dict,
    dataset_root: str,
    processor,
) -> Optional[Tuple[dict, dict, str]]:
    """
    Prepare a single sample to be used later in a batched pass:
    - normalize conversations
    - build text prompt string
    - DO NOT decode/resize images here (will be done in batch)
    Returns:
        (sample_copy, conversations, text) if valid; otherwise None.
    """
    sample_copy = copy.deepcopy(sample)
    conversations = sample_copy["data"]

    # Normalize the conversation & basic validation
    for i, step in enumerate(conversations):
        new_step = step.copy()
        if step["role"] == "system":
            new_step["content"][0]["text"] = "You are a helpful assistant."
        for j, content in enumerate(new_step["content"]):
            if content["type"] == "image":
                # Build absolute image path; keep your special replacement rule for kling_mm
                img_file_name = content.pop("image_file_name")
                if "kling_mm" in dataset_root:
                    img_file_name = img_file_name.replace("created_dataset/filtered_data/", "")
                content["image"] = os.path.join(dataset_root, img_file_name)

                # Enforce that assistant text immediately before an image contains <abs_vis_token></abs_vis_token>
                if j > 0 and new_step["content"][j - 1]["type"] == "text" and step["role"] == "assistant":
                    if "<abs_vis_token></abs_vis_token>" not in new_step["content"][j - 1]["text"]:
                        return None  # invalid as per your rule
            new_step["content"][j] = content
        conversations[i] = new_step

    # Build text string for this sample
    text = processor.apply_chat_template(conversations, tokenize=False)
    text = place_output_image_avt(text)
    return sample_copy, conversations, text


def _tensor_seq_lengths(attention_mask: torch.Tensor) -> List[int]:
    """
    Compute per-sample effective sequence lengths from attention_mask.
    """
    # attention_mask: (B, L) of dtype long/bool/int
    return attention_mask.long().sum(dim=1).tolist()


# ----------------------------------------
# Batched filter with per-sample resizing; GPU-accelerated length checks
# ----------------------------------------
@torch.inference_mode()
def filter_invalid_samples_in_json(
    json_path: str,
    dataset_root: str,
    processor,
    cur_max: int = -1,
    id: int = -1,
    max_seq_len: int = 4096,
    bsz: int = 4,
    device: Optional[torch.device] = None,
):
    """
    device: torch.device('cuda:0') or CPU if CUDA unavailable.
    Only length computation and masks are moved to GPU to accelerate filtering.
    """
    save_name = None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_data: List[dict] = []
    removed_count = 0
    dataset_name = os.path.basename(os.path.dirname(json_path))

    # Batch buffers (we keep conversations; images are built in a batched pass)
    batch_samples: List[dict] = []
    batch_conversations: List[dict] = []
    batch_texts: List[str] = []

    def _flush_batch():
        nonlocal filtered_data, removed_count, cur_max
        if not batch_samples:
            return

        # 1) Batched vision parse (CPU): turns conversations -> raw image_inputs per sample
        all_image_inputs_raw, _ = process_vision_info_batched(batch_conversations)

        # 2) resize_by_token_budget per sample (must be bsz==1 for this op)
        resized_images_list = []
        for image_inputs_single in all_image_inputs_raw:
            resized_images, _ = resize_by_token_budget(image_inputs_single)  # only supports single sample
            resized_images_list.append(resized_images)

        # 3) Tokenize as a batch (CPU), then move only masks to GPU for fast length calc
        batch = processor(
            text=batch_texts,
            images=resized_images_list,
            return_tensors="pt",
            padding=True,
        )

        # Move attention_mask to device for fast reduction
        attn = batch["attention_mask"]
        if device is not None:
            attn = attn.to(device, non_blocking=True)

        # 4) Per-sample length and filtering (on GPU if available)
        seq_lens = _tensor_seq_lengths(attn)
        if seq_lens:
            cur_max = max(cur_max, max(seq_lens))

        for sample_obj, seq_len in zip(batch_samples, seq_lens):
            if seq_len <= max_seq_len:
                filtered_data.append(sample_obj)
            else:
                # This message remains on CPU; only the reduction ran on GPU
                print("Found too long data:", dataset_name, seq_len)
                removed_count += 1

        # 5) Clear buffers
        batch_samples.clear()
        batch_conversations.clear()
        batch_texts.clear()

        # Optional: free GPU cache for long runs
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
            # Do not call empty_cache too frequently; uncomment if memory pressure appears.
            # torch.cuda.empty_cache()

    # Iterate and build batches
    for idx, sample in tqdm(enumerate(data), desc=json_path, total=len(data)):
        prepared = _prepare_single_sample_for_batch(sample, dataset_root=dataset_root, processor=processor)
        if prepared is None:
            removed_count += 1
            continue

        sample_copy, conversations, text = prepared
        batch_samples.append(sample_copy)
        batch_conversations.append(conversations)
        batch_texts.append(text)

        if len(batch_samples) >= bsz:
            _flush_batch()

    # Flush the last incomplete batch
    _flush_batch()

    # Save only if something was removed
    if removed_count > 0:
        save_name = json_path.replace(".json", f"_max_seq_len{max_seq_len}.json")
        with open(save_name, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    return removed_count, cur_max, save_name


def main():
    cur_max = -1
    args = get_args()
    processor = AutoProcessor.from_pretrained(args.load_model_path, use_fast=False)

    # Resolve device: prefer CUDA if available, else CPU
    device_str = getattr(args, "device", None)
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Determine batch size (fallback if not provided in args)
    bsz = getattr(args, "bsz", 4)

    total_id = 0
    for json_path in args.data_path:
        if os.path.isfile(json_path):
            removed_count, cur_max, save_name = filter_invalid_samples_in_json(
                json_path=json_path,
                dataset_root=args.dataset_root,
                processor=processor,
                cur_max=cur_max,
                id=total_id,
                max_seq_len=args.max_seq_len,
                bsz=bsz,
                device=device,
            )
            total_id += 1
            if removed_count > 0:
                print(
                    f"Saved to {save_name}, removed {removed_count} samples due to exceeding max_seq_len {args.max_seq_len}."
                )


if __name__ == "__main__":
    main()

"""
Example usages (GPU):

# Use specific GPU
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python remove_too_long_gpu.py \
    --max_seq_len 2048 \
    --bsz 128 \
    --device cuda:0 \
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
    --data_path \
    "/home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_9.1.json"

# Another env with GPU
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/Latent_Think/abstract-visual-token
python remove_too_long.py \
    --max_seq_len 3000 \
    --bsz 16 \
    --device cuda:0 \
    --load_model_path /ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen2.5-VL-7B-Instruct \
    --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
    --data_path "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/ReFocus/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_count/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_geometry/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_maze/filtered_train_short3000_w_metadata_9.1.json"
"""
