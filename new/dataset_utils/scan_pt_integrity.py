#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan .pt files by attempting torch.load only.
A file is considered corrupted if torch.load raises an exception.
No size or structural heuristics are used.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import torch lazily; we fail fast if it's unavailable
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def safe_torch_load(path: str):
    """Load a checkpoint on CPU; use weights_only if available (PyTorch >= 2.0)."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch-not-available")
    kwargs = {"map_location": "cpu"}
    # Use weights_only if supported to reduce risk and speed up load
    if "weights_only" in torch.load.__code__.co_varnames:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def scan_one(path: str) -> dict:
    """Return dict with status based solely on torch.load success/failure."""
    rec = {
        "path": path,
        "status": "ok",            # "ok" | "corrupted"
        "reason": "load-ok",
        "size_bytes": None,
        "exception": "",
    }
    try:
        rec["size_bytes"] = os.path.getsize(path)
    except Exception as e:
        # Size is not used for judgment; record for debugging only
        rec["size_bytes"] = None

    try:
        _ = safe_torch_load(path)
        rec["status"] = "ok"
        rec["reason"] = "load-ok"
    except Exception as e:
        rec["status"] = "corrupted"
        rec["reason"] = "torch-load-failed"
        rec["exception"] = f"{type(e).__name__}: {e}"

    return rec


def iter_pt_files(root: str):
    """Yield .pt file paths under root recursively."""
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".pt"):
                yield os.path.join(dp, fn)


def main():
    parser = argparse.ArgumentParser(description="Scan .pt files using torch.load only.")
    parser.add_argument("--root", required=True, help="Root directory to scan recursively.")
    parser.add_argument("--out", default="pt_scan_report.ndjson", help="NDJSON report output path.")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes.")
    parser.add_argument("--print_corrupted_only", action="store_true",
                        help="Print corrupted file paths to stdout.")
    args = parser.parse_args()

    files = list(iter_pt_files(args.root))
    total = len(files)
    print(f"[info] Found {total} .pt files under: {args.root}")
    if total == 0:
        print("[info] Nothing to do.")
        return

    ok_cnt = 0
    bad_cnt = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut2p = {ex.submit(scan_one, p): p for p in files}
            done = 0
            for fut in as_completed(fut2p):
                try:
                    rec = fut.result()
                except Exception as e:
                    p = fut2p[fut]
                    rec = {
                        "path": p,
                        "status": "corrupted",
                        "reason": "worker-exception",
                        "size_bytes": None,
                        "exception": f"{type(e).__name__}: {e}",
                    }

                if rec["status"] == "ok":
                    ok_cnt += 1
                else:
                    bad_cnt += 1
                    if args.print_corrupted_only:
                        print(rec["path"])

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                done += 1
                if done % 200 == 0 or done == total:
                    print(f"[progress] {done}/{total} processed (ok={ok_cnt}, corrupted={bad_cnt})")

    print(f"[summary] ok={ok_cnt}, corrupted={bad_cnt}, report={args.out}")
    if not TORCH_AVAILABLE:
        print("[warn] torch is not available; all files were marked corrupted by loader.")


if __name__ == "__main__":
    main()
