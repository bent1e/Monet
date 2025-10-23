import os
import json
import copy
import pdb
from typing import List, Tuple, Optional, Any

from src.utils import get_args, place_output_image_avt, resize_by_token_budget
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from qwen_vl_utils import process_vision_info as _process_vision_info_single


def process_vision_info_batched(
    conversations_batch: List[dict],
) -> Tuple[List[Any], List[Any]]:
    """
    Batched wrapper for process_vision_info.
    For each sample in conversations_batch, call the original single-sample API
    and gather results into two lists (one per sample).

    Returns:
        all_image_inputs: List[image_inputs_per_sample]
        all_extra_info:   List[extra_info_per_sample]  # keep shape/type as original
    """
    all_image_inputs = []
    all_extra_info = []
    for conv in conversations_batch:
        # Original API expects a list of conversations for a single sample.
        image_inputs, extra_info = _process_vision_info_single([conv])
        all_image_inputs.append(image_inputs)
        all_extra_info.append(extra_info)
    return all_image_inputs, all_extra_info


def _prepare_single_sample_for_batch(
    sample: dict,
    dataset_root: str,
    processor,
) -> Optional[Tuple[dict, str, list, str]]:
    """
    Prepare a single sample for later batched tokenization.
    - Do single-sample vision loading and resizing (resize_by_token_budget supports bsz==1 only).
    - Return (sample_copy, text_prompt, resized_images, dataset_name) if valid; otherwise None.

    Notes:
    * All checks that depend on the raw conversation (e.g., <abs_vis_token> presence)
      are done here, so invalid samples are filtered out early.
    """
    sample_copy = copy.deepcopy(sample)
    conversations = sample_copy['data']
    if 'image_file_name' in conversations[1]['content'][0]:
        img_key = 'image_file_name'
    else:
        img_key = 'image'
    dataset_name = conversations[1]['content'][0][img_key].split('/')[-3]
    seen_observation = False
    n_img_pad = 0
    n_img = 0
    # Normalize the conversation & basic validation
    for i, step in enumerate(conversations):
        new_step = step.copy()
        if step["role"] == "system":
            new_step["content"][0]["text"] = "You are a helpful assistant."
        seen_assistant_image = False if step["role"] == "assistant" else None
        for j, content in enumerate(new_step["content"]):
            if content["type"] == "image":
                if "image_file_name" in content:
                    img_file_name = content.pop("image_file_name")
                else:
                    img_file_name = content.pop("image")
                if "kling_mm" in dataset_root:
                    img_file_name = img_file_name.replace("created_dataset/filtered_data/", "")
                content["image"] = os.path.join(dataset_root, img_file_name)
                if j > 0 and new_step["content"][j - 1]["type"] == "text" and step["role"] == "assistant":
                    if "<abs_vis_token></abs_vis_token>" not in new_step["content"][j - 1]["text"]:
                        return None  # invalid per your rule
                if step["role"] == "assistant":
                    n_img += 1
                    seen_assistant_image = True
            elif content["type"] == "text":
                if step["role"] == "assistant":
                    n_img_pad += content['text'].count('<abs_vis_token></abs_vis_token>')
                    # Validate that any observation text must be preceded by an assistant image within the same step
                    if "<observation>" in content.get("text", "") and not seen_assistant_image:
                        content['text'] = content['text'].replace("<observation>", "").replace("</observation>", "")
                    if "<observation>" in content.get("text", ""):
                        seen_observation = True

                elif step["role"] == "user":
                    img_key = "image_file_name" if "image_file_name" in new_step["content"][0] else "image"
                    if 'Zebra_CoT_visual_search' not in new_step["content"][0][img_key] and 'Zebra_CoT_count' not in new_step["content"][0][img_key]: # keep boxed instructions for Zebra_CoT_visual_search
                        content["text"] = content["text"].replace("\nPut your final answer within \\boxed{}.", "")
            new_step["content"][j] = content
        conversations[i] = new_step

    if n_img != n_img_pad:
        print(f"n_img ({n_img}) != num of <abs_vis_token></abs_vis_token> ({n_img_pad}), discard this sample")
        return None
    if not seen_observation:
        #print("[Preprocess] No observation found in assistant responses. Discard this sample")
        return None

    # Build text (string) for this sample
    text = processor.apply_chat_template(conversations, tokenize=False)
    text = place_output_image_avt(text)

    # Build single-sample image inputs and resize (must be bsz==1 here)
    image_inputs, _ = process_vision_info([conversations])
    image_inputs, _ = resize_by_token_budget(
        image_inputs,
        global_max_pixels=1500*28*28,
        per_img_max_pixels=1280*28*28,
    )  # only supports single sample

    # remove the dataset root prefix
    for i, step in enumerate(conversations):
        new_step = step.copy()
        for j, content in enumerate(new_step["content"]):
            if content["type"] == "image":
                img_file_name = content.pop("image")
                content["image"] = img_file_name.replace(f"{dataset_root}/","")
            new_step["content"][j] = content
        conversations[i] = new_step

    return sample_copy, text, image_inputs, dataset_name


def _tensor_seq_lengths(attention_mask) -> List[int]:
    """
    Compute per-sample effective sequence lengths from attention_mask.
    """
    # attention_mask: (B, L)
    return attention_mask.long().sum(dim=1).tolist()


# ----------------------------------------
# Batched filter with per-sample resizing (parallelized I/O/resize)
# ----------------------------------------

def filter_invalid_samples_in_json(
    json_path: str,
    dataset_root: str,
    processor,
    cur_max: int = -1,
    id: int = -1,
    max_seq_len: int = 4096,
    bsz: int = 4,
    workers: Optional[int] = None,  # new: parallel threads
):
    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 1)

    save_name = None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_data: List[dict] = []
    removed_count = 0
    dataset_name = os.path.basename(os.path.dirname(json_path))

    # Buffers for a batch
    batch_samples: List[dict] = []
    batch_texts: List[str] = []
    batch_images: List[list] = []  # list of image_inputs (already resized per-sample)

    def _flush_batch():
        nonlocal filtered_data, removed_count, cur_max
        if not batch_samples:
            return
        # Batched tokenize after per-sample resizing & text building

        total_image_pads = 0
        for txt in batch_texts:
            total_image_pads += txt.count("<|vision_start|><|image_pad|>")
        total_images = sum([len(b) for b in batch_images])
        if total_image_pads != total_images:
            pdb.set_trace()

        batch = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True)
        seq_lens = _tensor_seq_lengths(batch["attention_mask"])

        # Update cur_max by observed max within this batch
        if len(seq_lens) > 0:
            cur_max = max(cur_max, max(seq_lens))

        # Keep samples whose seq_len <= max_seq_len
        for sample_obj, seq_len in zip(batch_samples, seq_lens):
            if seq_len <= max_seq_len:
                filtered_data.append(sample_obj)
            else:
                removed_count += 1

        # Clear buffers
        batch_samples.clear()
        batch_texts.clear()
        batch_images.clear()

    # Parallel preprocessing: image load/resize + chat template
    def _prep(sample):
        return _prepare_single_sample_for_batch(sample, dataset_root=dataset_root, processor=processor)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # tqdm wraps the generator to preserve order and show progress
        for prepared in tqdm(ex.map(_prep, data, chunksize=8), desc=json_path, total=len(data)):
            if prepared is None:
                removed_count += 1
                continue

            sample_copy, text, images, _ = prepared
            batch_samples.append(sample_copy)
            batch_texts.append(text)
            batch_images.append(images)

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

    # Determine batch size (fallback if not provided in args)
    bsz = getattr(args, "bsz", 4)

    # Parallel threads: prefer CLI value if exists; else CPU cores - 1
    workers = getattr(args, "remove_too_long_workers", None)
    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 1)

    total_id = 0
    for json_path in args.data_path:
        if os.path.isfile(json_path):
            removed_count, cur_max, save_name = filter_invalid_samples_in_json(
                json_path,
                args.dataset_root,
                processor,
                cur_max=cur_max,
                id=total_id,
                max_seq_len=args.max_seq_len,
                bsz=bsz,
                workers=workers,
            )
            total_id += 1
            if removed_count > 0:
                print(
                    f"Saved to {save_name}, removed {removed_count} samples due to exceeding max_seq_len {args.max_seq_len}."
                )


if __name__ == "__main__":
    main()

"""

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06-temp/Latent_Think/abstract-visual-token
python remove_too_long_parallel.py \
    --remove_too_long_workers 16 \
    --max_seq_len 3000 \
    --bsz 16 \
    --load_model_path /ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen2.5-VL-7B-Instruct \
    --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
    --data_path "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_arc_agi/raw_train_w_obs_w_metadata_swap.json" \
  "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_checkers/raw_train_w_obs_w_metadata_swap.json" \
  "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_connect_four/raw_train_w_obs_w_metadata_swap.json" \
  "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_rpm/raw_train_w_obs_w_metadata_swap.json" \
  "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_tetris/raw_train_w_obs_w_metadata_swap.json" \
  /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_maze/filtered_train_short3000_w_metadata_9.25_further_washed.json

Notes:
- This parallel version runs per-sample I/O and resizing using a thread pool (CPU-bound/IO-bound). GPU isn't used since no model forward is executed here.
- You can pass --workers in your own get_args() if supported; otherwise it defaults to CPU cores - 1.
- Increase --bsz to improve throughput if memory allows.
"""