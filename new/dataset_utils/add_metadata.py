#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transform dataset samples to {"metadata": {...}, "data": original_item} format.
- "metadata.sample_id" is a per-file global id starting from START_INDEX.
- "metadata.dataset_name" is inferred from the input file path (segment after "filtered_data/").
- Output file is written as "<orig_stem>_w_metadata.json" in the same directory.
- Supports JSON array files and JSONL (one JSON object/array per line) files.

Additionally, this script normalizes image file paths in user/assistant messages:
- For each content part where type == "image", if "image_file_name" does not contain "images" and
    contains exactly one '/', the single '/' is replaced with '/images/'.
    Example: 'Zebra_CoT_arc_agi/0_0_0.jpg' -> 'Zebra_CoT_arc_agi/images/0_0_0.jpg'.

CLI usage:
    python add_metadata.py file1.json file2.json ...
    python add_metadata.py -i path/pattern1.json path/pattern2.json ...
If no files are provided, DEFAULT_FILES is used.
"""

import os
import glob
import json
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# --------- Configuration ---------
START_INDEX = 0                  # Change to 1 if you prefer 1-based indexing
METADATA_KEY = "metadata"        # Change to "meta_data" if needed
DATA_KEY = "data"

# Default input files if no CLI args are provided
DEFAULT_FILES = [

    "/data1/qxwang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_arc_agi/raw_train_w_obs.json",
]

'''DEFAULT_FILES = [
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoF/filtered_train.json",
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoM_w_MathVista/filtered_train.json",
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/PixelReasoner/filtered_train.json",
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/ReFocus/filtered_train.json",
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_count/filtered_train.json",
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_visual_search/filtered_train.json",
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_geometry/filtered_train.json",
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_maze/filtered_train_short3000.json",
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/VTS_1/filtered_train_short3000.json",
]'''

def infer_dataset_name(p: Path) -> str:
    """
    Infer dataset_name from the path. Prefer the path segment after "filtered_data/".
    Fallback to the immediate parent directory name.
    """
    parts = p.resolve().parts
    if "filtered_data" in parts:
        idx = parts.index("filtered_data")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # Fallback
    return p.parent.name


def detect_format_and_load(p: Path):
    """
    Detect whether the file is JSON array or JSONL, then load accordingly.
    Returns a tuple: (records, fmt), where fmt in {"json", "jsonl"}.
    - For "json": records is a Python list.
    - For "jsonl": records is a Python list of line-parsed JSON objects/arrays.
    """
    text = p.read_text(encoding="utf-8")
    # Detect by first non-whitespace char
    first_non_ws = next((ch for ch in text.lstrip()[:1]), "")
    if first_non_ws == "[":
        # JSON array
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"File looks like JSON but not a list: {p}")
        return data, "json"
    else:
        # Assume JSONL
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records, "jsonl"


def _maybe_fix_image_path(path: str) -> Tuple[str, bool]:
    """
    If path doesn't contain "images" and has exactly one '/', replace that '/' with '/images/'.
    Returns (new_path, changed: bool)
    Example: 'Zebra_CoT_arc_agi/0_0_0.jpg' -> 'Zebra_CoT_arc_agi/images/0_0_0.jpg'
    """
    try:
        if isinstance(path, str) and ("images" not in path) and (path.count("/") == 1):
            return path.replace("/", "/images/", 1), True
    except Exception:
        pass
    return path, False


def _fix_images_in_message(msg: dict) -> int:
    """
    Fix image_file_name fields inside a single message dict if role is user/assistant.
    Returns the number of replacements applied.
    """
    if not isinstance(msg, dict):
        return 0
    role = msg.get("role")
    if role not in {"user", "assistant"}:
        return 0
    content = msg.get("content")
    if not isinstance(content, list):
        return 0
    changed = 0
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image":
            if "image_file_name" in part and isinstance(part["image_file_name"], str):
                new_path, ok = _maybe_fix_image_path(part["image_file_name"])
                if ok:
                    part["image_file_name"] = new_path
                    changed += 1
    return changed


def _fix_images_in_item(item) -> int:
    """
    Apply image_file_name fixes to a conversation item (usually a list of messages).
    Returns the number of replacements applied within this item.
    """
    total = 0
    # Typical structure: item is a list[message]
    if isinstance(item, list):
        for msg in item:
            total += _fix_images_in_message(msg)
        return total
    # Sometimes a dict (single message or wrapped conv)
    if isinstance(item, dict):
        # If it's one message
        if "role" in item and "content" in item:
            return _fix_images_in_message(item)
        # If it might contain a list of messages under a key
        for v in item.values():
            if isinstance(v, list):
                for msg in v:
                    total += _fix_images_in_message(msg)
        return total
    return 0


def write_output(p_out: Path, records, fmt: str):
    """
    Write records to p_out according to fmt.
    - "json": pretty JSON array
    - "jsonl": one JSON object per line
    """
    if fmt == "json":
        p_out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "jsonl":
        with p_out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unknown format: {fmt}")


def transform_one_file(p: Path):
    """
    Transform a single file and write output with suffix "_w_metadata.json" in the same directory.
    """
    if not p.exists():
        print(f"[WARN] File not found: {p}")
        return

    dataset_name = infer_dataset_name(p)
    records, fmt = detect_format_and_load(p)

    out_records = []
    total_rewrites = 0
    for idx, item in enumerate(records, start=START_INDEX):
        # Fix image_file_name paths in-place per requirement
        total_rewrites += _fix_images_in_item(item)
        # Wrap each original item (usually a list dialog) into {"metadata": {...}, "data": original}
        out_records.append({
            METADATA_KEY: {
                "sample_id": idx,
                "dataset_name": dataset_name,
            },
            DATA_KEY: item
        })

    p_out = p.with_name(p.stem + "_w_metadata.json")
    write_output(p_out, out_records, "json" if fmt == "json" else "jsonl")

    print(f"[OK] {p.name} -> {p_out.name} | dataset_name={dataset_name} | n={len(out_records)} | image_paths_rewritten={total_rewrites}")


def expand_inputs(argv: List[str]) -> List[Path]:
    """
    Expand CLI args or default list into concrete file paths.
    Supports:
      - Relative paths and absolute paths
      - Glob patterns for both (e.g., './**/filtered_train*.json' or '/abs/path/**/filtered_*.json')
      - Home expansion with '~'
    """
    def has_glob_chars(s: str) -> bool:
        # Detect common glob chars; '**' is also covered by '*'
        return any(ch in s for ch in ['*', '?', '['])

    paths: List[Path] = []
    raw_list = argv if argv else DEFAULT_FILES

    for spec in raw_list:
        # Expand '~' to user home
        spec = os.path.expanduser(spec)

        # If it's a glob (relative or absolute), expand with glob.glob(recursive=True)
        if has_glob_chars(spec):
            matches = glob.glob(spec, recursive=True)
            if not matches:
                # No match; keep as literal path (may raise later if not found)
                paths.append(Path(spec))
            else:
                paths.extend(Path(m) for m in matches)
        else:
            # Literal path (relative or absolute)
            paths.append(Path(spec))

    # Deduplicate while preserving order; resolve to absolute real paths
    seen = set()
    uniq: List[Path] = []
    for p in paths:
        try:
            rp = p.resolve()
        except FileNotFoundError:
            # If path does not exist, keep as-is (transform_one_file will warn)
            rp = p.absolute()
        if rp not in seen:
            uniq.append(rp)
            seen.add(rp)
    return uniq



def main():
    parser = argparse.ArgumentParser(description="Wrap records with metadata and fix image_file_name paths.")
    parser.add_argument(
        "files",
        nargs="*",
        help="JSON/JSONL files or glob patterns to process (space-separated). If omitted, uses DEFAULT_FILES.",
    )
    parser.add_argument(
        "-i", "--inputs",
        nargs="+",
        help="Alternate way to pass one or more files/patterns to process.",
    )
    args = parser.parse_args()

    cli_list = []
    if args.inputs:
        cli_list.extend(args.inputs)
    if args.files:
        cli_list.extend(args.files)

    inputs = expand_inputs(cli_list)
    if not inputs:
        print("[ERROR] No input files provided.")
        sys.exit(1)

    for p in inputs:
        try:
            transform_one_file(p)
        except Exception as e:
            print(f"[ERROR] Failed to transform {p}: {e}")


if __name__ == "__main__":
    main()
