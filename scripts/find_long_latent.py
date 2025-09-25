#!/usr/bin/env python
"""
python scripts/find_long_latent.py \
  --root /home/dids/shiyang/codes/abstract-visual-token/new/precomputed_teacher_reps/avt_sft/9.24_debug_avt_sft_com_refocus_search-fwsh_ce5.0/checkpoint-180 \
  --threshold 30 \
  --output long_latent_files.txt \
  --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Iterable, List


def iter_files(root: str, exts: Iterable[str]) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not exts:  # empty list means accept all
                yield os.path.join(dirpath, fname)
            else:
                lower = fname.lower()
                if any(lower.endswith(ext) for ext in exts):
                    yield os.path.join(dirpath, fname)


def get_latent_length(latent) -> int | None:
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - torch not installed case
        torch = None  # type: ignore

    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    # Sequence types
    if isinstance(latent, (list, tuple)):
        return len(latent)
    # Torch tensor
    if torch is not None and hasattr(torch, "Tensor") and isinstance(latent, torch.Tensor):
        return latent.shape[0]
    # Numpy array
    if np is not None and hasattr(np, "ndarray") and isinstance(latent, np.ndarray):
        return latent.shape[0]
    # Fallback: try len()
    try:
        return len(latent)  # type: ignore[arg-type]
    except Exception:
        return None


def safe_load(path: str, map_location: str = "cpu"):
    """Attempt to load a file as a torch object or JSON; return object or None."""
    # Try torch
    try:
        import torch  # type: ignore
        return torch.load(path, map_location=map_location)
    except Exception:
        pass
    # Try JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Find files whose data['latent'] length exceeds a threshold.")
    parser.add_argument("--root", required=True, help="Root directory to traverse.")
    parser.add_argument("--threshold", type=int, required=True, help="Minimum latent length to flag.")
    parser.add_argument("--output", default="long_latent_files.txt", help="Output text file to store matching file paths.")
    parser.add_argument("--exts", nargs="*", default=[".pt", ".pth"], help="File extensions to consider (empty => all).")
    parser.add_argument("--verbose", action="store_true", help="Print skipped / error info.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of files to process (0=all).")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"[ERROR] Root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    matches: List[str] = []
    processed = 0
    errors = 0
    total_files = 0

    for total_files, fpath in enumerate(iter_files(root, args.exts), start=1):
        if args.limit and processed >= args.limit:
            break
        obj = safe_load(fpath)
        if not isinstance(obj, dict):
            if args.verbose:
                print(f"[SKIP] Not a dict: {fpath}")
            continue
        if 'latent' not in obj:
            if args.verbose:
                print(f"[SKIP] No 'latent': {fpath}")
            continue
        length = obj['latent'].shape[1]

        if length is None:
            if args.verbose:
                print(f"[SKIP] Could not determine length: {fpath}")
            continue
        if length > args.threshold:
            matches.append(fpath)
            if args.verbose:
                print(f"[MATCH] {fpath} latent_len={length}")
        processed += 1
    
    # Write output
    with open(args.output, "w", encoding="utf-8") as out:
        for m in matches:
            out.write(m + "\n")

    print("=== Summary ===")
    print(f"Root directory : {root}")
    print(f"Extensions     : {args.exts if args.exts else 'ALL'}")
    print(f"Threshold      : {args.threshold}")
    print(f"Scanned files  : {total_files}")
    print(f"Processed dicts: {processed}")
    print(f"Matches        : {len(matches)}")
    print(f"Output written : {os.path.abspath(args.output)}")
    if errors:
        print(f"Errors         : {errors}")


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
