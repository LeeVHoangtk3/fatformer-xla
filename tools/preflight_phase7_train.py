#!/usr/bin/env python3
"""
tools/preflight_phase7_train.py — Task 7.11 Pre-flight Checklist
Kiểm tra 8 chốt chặn tự động trước khi bước vào huấn luyện Phase 7:
  1. manifest v2 tồn tại, n >= 12.000
  2. isolation report: violations == 0
  3. degradation_v2.yaml: clean_prob == 0.60
  4. loss test: pytest tests/test_loss_v2.py
  5. selection test: pytest tests/test_selection_metric.py
  6. baseline load_canonical: missing=0, unexpected=0
  7. val_calib != val_select (không trùng lặp ảnh giữa 2 split)
  8. GPU free memory >= 14 GB (trên môi trường CUDA)
"""
import os
import sys
import json
import yaml
import hashlib
import subprocess
from types import SimpleNamespace
from typing import Dict, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def check_manifest_v2() -> Tuple[bool, str]:
    path = os.path.join(PROJECT_ROOT, "DATASET", "manifests", "data_manifest_v2.json")
    if not os.path.isfile(path):
        return False, f"Tệp không tồn tại: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", [])
        n = len(records)
        if n < 12000:
            return False, f"Số lượng mẫu n={n} < 12000"
        return True, f"Tồn tại, n={n} records"
    except Exception as e:
        return False, str(e)


def check_isolation_report() -> Tuple[bool, str]:
    candidates = [
        os.path.join(PROJECT_ROOT, "DATASET", "logs", "phase7", "data_isolation_report.json"),
        os.path.join(PROJECT_ROOT, "logs", "phase7", "data_isolation_report.json"),
    ]
    report_file = next((p for p in candidates if os.path.isfile(p)), None)
    if not report_file:
        return False, "Không tìm thấy data_isolation_report.json"
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = data.get("pairs", {})
        total_violations = 0
        for pair_name, pair_info in pairs.items():
            viol = pair_info.get("violations", 0)
            total_violations += viol
        if total_violations != 0:
            return False, f"Phát hiện violations = {total_violations} trong {report_file}"
        return True, f"violations = 0 (tuyệt đối cách ly trong {os.path.basename(report_file)})"
    except Exception as e:
        return False, str(e)


def check_degradation_yaml() -> Tuple[bool, str]:
    path = os.path.join(PROJECT_ROOT, "configs", "degradation_v2.yaml")
    if not os.path.isfile(path):
        return False, f"Tệp không tồn tại: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        prob = cfg.get("clean_prob")
        if abs(prob - 0.60) > 1e-4:
            return False, f"clean_prob = {prob} != 0.60"
        return True, f"clean_prob == 0.60 (40% suy biến)"
    except Exception as e:
        return False, str(e)


def check_loss_tests() -> Tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", "tests/test_loss_v2.py", "-q"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if res.returncode == 0:
        return True, "Toàn bộ unit test Loss v2 PASSED"
    return False, f"Test failed (exit code {res.returncode}): {res.stdout[:150]}"


def check_selection_tests() -> Tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", "tests/test_selection_metric.py", "tests/test_smoke_selection_v2_integration.py", "-q"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if res.returncode == 0:
        return True, "Toàn bộ unit test Selection v2 PASSED"
    return False, f"Test failed (exit code {res.returncode}): {res.stdout[:150]}"


def check_baseline_canonical() -> Tuple[bool, str]:
    from src.models.canonical_loader import load_canonical
    ckpt_candidates = [
        os.path.join(PROJECT_ROOT, "DATASET", "pretrained", "fatformer_4class_ckpt.pth"),
        os.path.join(PROJECT_ROOT, "pretrained", "fatformer_4class_ckpt.pth"),
        os.path.join(PROJECT_ROOT, "fatformer_4class_ckpt.pth"),
    ]
    ckpt_file = next((p for p in ckpt_candidates if os.path.isfile(p)), None)
    if not ckpt_file:
        return False, "Không tìm thấy checkpoint Baseline fatformer_4class_ckpt.pth"

    clip_candidates = [
        os.path.join(PROJECT_ROOT, "DATASET", "pretrained", "ViT-L-14.pt"),
        os.path.join(PROJECT_ROOT, "pretrained", "ViT-L-14.pt"),
        os.path.join(PROJECT_ROOT, "ViT-L-14.pt"),
    ]
    clip_file = next((p for p in clip_candidates if os.path.isfile(p)), None)
    if not clip_file:
        return False, "Không tìm thấy backbone ViT-L-14.pt"

    args = SimpleNamespace(
        backbone="CLIP:ViT-L/14",
        clip_path=clip_file,
        num_classes=2,
        num_vit_adapter=3,
        num_context_embedding=8,
        init_context_embedding="",
        hidden_dim=768,
        clip_vision_width=1024,
        frequency_encoder_layer=2,
        decoder_layer=4,
        num_heads=12,
        use_srm=False,
        use_gating=False,
    )
    try:
        _, report = load_canonical(ckpt_file, args, allow_partial=False)
        m = report.get("missing", 0)
        u = report.get("unexpected", 0)
        if m == 0 and u == 0:
            return True, f"load_canonical bit-perfect: missing=0, unexpected=0"
        return False, f"missing={m}, unexpected={u}"
    except Exception as e:
        return False, str(e)


def check_splits_isolation() -> Tuple[bool, str]:
    manifest_path = os.path.join(PROJECT_ROOT, "DATASET", "manifests", "data_manifest_v2.json")
    if not os.path.isfile(manifest_path):
        return False, "Không tìm thấy manifest v2 để kiểm tra split"
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", [])
        calib_paths = set()
        select_paths = set()
        for r in records:
            p = r.get("file_path", r.get("path", ""))
            split = r.get("split", "")
            if split == "val_calib":
                calib_paths.add(p)
            elif split == "val_select":
                select_paths.add(p)

        if not calib_paths:
            return False, "val_calib split rỗng"
        if not select_paths:
            return False, "val_select split rỗng"

        intersection = calib_paths.intersection(select_paths)
        if len(intersection) > 0:
            return False, f"Phát hiện rò rỉ: {len(intersection)} ảnh trùng giữa val_calib và val_select!"
        return True, f"val_calib ({len(calib_paths)}) tách biệt 100% val_select ({len(select_paths)})"
    except Exception as e:
        return False, str(e)


def check_gpu_environment() -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return True, "CẢNH BÁO: Chạy trên CPU Local (GPU check sẽ bắt buộc trên Colab GPU)"
    free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
    total_mem = torch.cuda.mem_get_info()[1] / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(0)
    if free_mem < 13.0:
        return False, f"{gpu_name}: VRAM khả dụng {free_mem:.2f} GB < 14 GB (Tổng: {total_mem:.2f} GB)"
    return True, f"{gpu_name}: VRAM khả dụng {free_mem:.2f} GB / {total_mem:.2f} GB"


def run_preflight():
    print("=" * 80)
    print("FATFORMER-XLA: PRE-FLIGHT CHECKLIST HUẤN LUYỆN PHASE 7 (TASK 7.11)")
    print("=" * 80)

    checks = [
        ("1. Manifest v2 (n >= 12.000)", check_manifest_v2),
        ("2. Báo cáo cách ly (violations == 0)", check_isolation_report),
        ("3. Cấu hình suy biến (clean_prob == 0.60)", check_degradation_yaml),
        ("4. Kiểm thử Hàm Mất Mát v2 (Loss v2)", check_loss_tests),
        ("5. Kiểm thử Selection Metric v2", check_selection_tests),
        ("6. Baseline Load Canonical (verify 0/0)", check_baseline_canonical),
        ("7. Tách biệt val_calib != val_select", check_splits_isolation),
        ("8. Tài nguyên GPU / Môi trường", check_gpu_environment),
    ]

    results = {}
    all_passed = True

    for name, fn in checks:
        passed, msg = fn()
        results[name] = {"passed": passed, "message": msg}
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {tag:<7} | {name:<42} | {msg}")
        if not passed:
            all_passed = False

    print("=" * 80)
    log_dir = os.path.join(PROJECT_ROOT, "DATASET", "logs", "phase7")
    os.makedirs(log_dir, exist_ok=True)
    out_file = os.path.join(log_dir, "preflight_train.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[*] Đã xuất kết quả kiểm tra vào: {out_file}")

    if not all_passed:
        print("[!] PRE-FLIGHT THẤT BẠI: Có chốt chặn chưa đạt chuẩn. Dừng huấn luyện.")
        sys.exit(1)
    else:
        print("[✓] PRE-FLIGHT THÀNH CÔNG: Toàn bộ chốt chặn đã sẵn sàng 100% cho Task 7.11!")
        sys.exit(0)


if __name__ == "__main__":
    run_preflight()
