# -*- coding: utf-8 -*-
"""
FatFormer-XLA: Kiem Tra & Xac Thuc Tinh Toan Ven, Nguon Goc (Provenance)
Cua Tep Trong So Rut Gon fatformer_crossdomain_weights_only.pth
Doi chieu 1-1 voi Checkpoint Goc M4 (crossdomain_model_best_m4.pth)
"""

import os
import sys
import json
import time
import hashlib
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_ONLY_PATH = os.path.join(PROJECT_ROOT, "DATASET", "checkpoints", "fatformer_crossdomain_weights_only.pth")
M4_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "DATASET", "checkpoints", "m4", "crossdomain_model_best_m4.pth")
REPORT_PATH = os.path.join(PROJECT_ROOT, "DATASET", "logs", "weights_only_verification_report.json")


def compute_sha256(filepath, chunk_size=65536):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()


def main():
    print("=" * 88)
    print("KIEM CHUAN XAC THUC NGUON GOC (PROVENANCE) & TINH TOAN VEN TRONG SO RUT GON")
    print("=" * 88)

    if not os.path.exists(WEIGHTS_ONLY_PATH):
        raise FileNotFoundError(f"Khong tim thay tep weights_only: {WEIGHTS_ONLY_PATH}")
    if not os.path.exists(M4_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Khong tim thay checkpoint M4: {M4_CHECKPOINT_PATH}")

    # 1. Do kich thuoc va ma bam SHA-256
    print(f"[*] Dang tinh SHA-256 cho {os.path.basename(WEIGHTS_ONLY_PATH)}...")
    w_size = os.path.getsize(WEIGHTS_ONLY_PATH)
    w_hash = compute_sha256(WEIGHTS_ONLY_PATH)
    print(f"    - Kich thuoc: {w_size:,} bytes ({w_size / (1024*1024):.2f} MB)")
    print(f"    - SHA-256   : {w_hash}")

    print(f"[*] Dang tinh SHA-256 cho {os.path.basename(M4_CHECKPOINT_PATH)}...")
    m4_size = os.path.getsize(M4_CHECKPOINT_PATH)
    m4_hash = compute_sha256(M4_CHECKPOINT_PATH)
    print(f"    - Kich thuoc: {m4_size:,} bytes ({m4_size / (1024*1024):.2f} MB)")
    print(f"    - SHA-256   : {m4_hash}")

    # 2. Nap va so sanh state_dict
    print("\n[*] Dang nap state_dict de doi chieu tham so...")
    w_dict = torch.load(WEIGHTS_ONLY_PATH, map_location="cpu")
    m4_obj = torch.load(M4_CHECKPOINT_PATH, map_location="cpu")
    m4_dict = m4_obj.get("model_state", m4_obj)

    w_keys = list(w_dict.keys())
    m4_keys = list(m4_dict.keys())

    print(f"    - So luong tham so weights_only : {len(w_keys):,} keys")
    print(f"    - So luong tham so M4 model_state: {len(m4_keys):,} keys")

    missing_in_weights = [k for k in m4_keys if k not in w_dict]
    unexpected_in_weights = [k for k in w_keys if k not in m4_dict]

    is_identical = True
    max_abs_diff = 0.0

    if missing_in_weights or unexpected_in_weights:
        is_identical = False
        print(f"[!] Canh bao lech key: missing={len(missing_in_weights)}, unexpected={len(unexpected_in_weights)}")
    else:
        # Kiem tra do sai lech gia tri tensor
        for k in m4_keys:
            t_w = w_dict[k]
            t_m4 = m4_dict[k]
            if t_w.shape != t_m4.shape:
                is_identical = False
                print(f"[!] Lech shape tai key: {k} ({t_w.shape} vs {t_m4.shape})")
                break
            if t_w.dtype != t_m4.dtype:
                is_identical = False
                print(f"[!] Lech dtype tai key: {k} ({t_w.dtype} vs {t_m4.dtype})")
                break
            diff = (t_w - t_m4).abs().max().item()
            if diff > max_abs_diff:
                max_abs_diff = diff

    print(f"    - Max Absolute Difference: {max_abs_diff:.8e}")
    match_status = "IDENTICAL_BIT_PERFECT" if (is_identical and max_abs_diff == 0.0) else "MISMATCH"
    print(f"    - Ket luan doi chieu      : {match_status}")

    # 3. Xuat bao cao JSON
    report = {
        "metadata": {
            "task": "Task 6.4 Weights Provenance & Integrity Verification",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "weights_only_file": os.path.basename(WEIGHTS_ONLY_PATH),
            "m4_source_file": os.path.basename(M4_CHECKPOINT_PATH),
            "verification_status": match_status,
        },
        "weights_only_info": {
            "path": WEIGHTS_ONLY_PATH,
            "size_bytes": w_size,
            "sha256": w_hash,
            "key_count": len(w_keys),
        },
        "m4_source_info": {
            "path": M4_CHECKPOINT_PATH,
            "size_bytes": m4_size,
            "sha256": m4_hash,
            "key_count": len(m4_keys),
            "contains_optimizer": "optimizer_state" in m4_obj,
            "epoch": m4_obj.get("epoch", None),
            "best_acc": m4_obj.get("best_acc", None),
        },
        "comparison_results": {
            "keys_match_exact": (len(missing_in_weights) == 0 and len(unexpected_in_weights) == 0),
            "missing_keys_count": len(missing_in_weights),
            "unexpected_keys_count": len(unexpected_in_weights),
            "max_abs_diff": max_abs_diff,
            "is_bit_perfect": (is_identical and max_abs_diff == 0.0),
        },
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Da ghi nhan artifact xac thuc nguon goc ra: {REPORT_PATH}")
    print("=" * 88)
    return report


if __name__ == "__main__":
    main()
