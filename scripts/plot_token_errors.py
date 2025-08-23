#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot token-level predictions vs labels and highlight errors.
Input: a JSONL file created by CustomTrainerSFT token error logging.
Each line contains fields like:
{
  'global_step': int,
  'epoch': float|None,
  'sample_index': int,
  'input_ids': List[int],
  'token_strs': List[str]|None,
  'pred_ids_shift': List[int],
  'label_ids_shift': List[int],
  'mask_shift': List[int],
  'aligned_offset': 1,
  'exp_name': str,
  'sample_id': int|optional
}

Usage:
  python scripts/plot_token_errors.py --jsonl logs/token_errors/token_errors_xxx.jsonl --out out_dir --num 5 --tokenizer <model_or_tokenizer_path>

Notes:
- If token_strs is None in the JSONL, pass --tokenizer to decode tokens.
- We visualize up to N tokens; long sequences will be wrapped in multiple rows.
"""
import argparse
import json
import os
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


def decode_tokens(input_ids: List[int], token_strs: Optional[List[str]], tokenizer_path: Optional[str]):
    if token_strs is not None:
        return token_strs
    if tokenizer_path is None:
        # fallback: show ids as strings
        return [str(i) for i in input_ids]
    if AutoTokenizer is None:
        return [str(i) for i in input_ids]
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        return tok.convert_ids_to_tokens(input_ids)
    except Exception:
        return [str(i) for i in input_ids]


def render_figure(tokens: List[str], pred_ids: List[int], label_ids: List[int], mask: List[int], tokenizer_path: Optional[str], out_path: str, title: str = ""):
    # Compute error flags aligned to shifted region length
    n = min(len(pred_ids), len(label_ids), len(mask))
    pred_ids = pred_ids[:n]
    label_ids = label_ids[:n]
    mask = mask[:n]

    # For tokens, we show the segment tokens[1:1+n] to reflect the shift alignment
    if len(tokens) >= n + 1:
        shown_tokens = tokens[1:1+n]
    else:
        shown_tokens = tokens[:n]

    errors = [int(mask[i] == 1 and pred_ids[i] != label_ids[i]) for i in range(n)]

    # Plot as colored rectangles with text labels
    cols = min(64, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, 1, figsize=(cols * 0.25 + 2, rows * 0.8 + 1), dpi=160)
    if rows == 1:
        axes = [axes]

    idx = 0
    for r in range(rows):
        ax = axes[r]
        ax.set_axis_off()
        for c in range(cols):
            if idx >= n:
                break
            token = shown_tokens[idx] if idx < len(shown_tokens) else ""
            is_err = errors[idx] == 1
            color = "#ffcccc" if is_err else "#ccffcc"
            # draw rectangle
            ax.add_patch(plt.Rectangle((c, 0), 0.95, 0.5, color=color, ec="#666666", lw=0.5))
            # text: token (newline) pred->label
            txt = token
            txt2 = f"{pred_ids[idx]}→{label_ids[idx]}" if mask[idx] == 1 else "(ignored)"
            ax.text(c + 0.02, 0.35, txt, fontsize=6, va='center', ha='left', family='monospace')
            ax.text(c + 0.02, 0.10, txt2, fontsize=6, va='center', ha='left', color='#333333')
            idx += 1
        ax.set_xlim(0, cols)
        ax.set_ylim(-0.1, 0.6)

    if title:
        fig.suptitle(title, fontsize=10)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True, help='token error JSONL file path')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--num', type=int, default=5, help='number of records to plot')
    ap.add_argument('--tokenizer', type=str, default=None, help='tokenizer path if token_strs missing')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    count = 0
    with open(args.jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= args.num:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            input_ids = rec.get('input_ids', [])
            token_strs = rec.get('token_strs', None)
            pred_ids = rec.get('pred_ids_shift', [])
            label_ids = rec.get('label_ids_shift', [])
            mask = rec.get('mask_shift', [])
            tokens = decode_tokens(input_ids, token_strs, args.tokenizer)

            sid = rec.get('sample_id', 'NA')
            gs = rec.get('global_step', 'NA')
            title = f"sample={sid} step={gs}"
            out_path = os.path.join(args.out, f"token_err_step{gs}_sid{sid}_{count}.png")
            render_figure(tokens, pred_ids, label_ids, mask, args.tokenizer, out_path, title)
            count += 1

    print(f"Saved {count} figures to {args.out}")


if __name__ == '__main__':
    main()
