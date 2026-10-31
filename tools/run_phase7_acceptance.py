# -*- coding: utf-8 -*-
"""
FatFormer-XLA: Bộ Điều Phối Nghiệm Thu Toàn Trình GPU Phase 7 (Tasks 7.3 - 7.6 & Cổng G7.1)
========================================================================================
Vận hành chuỗi thực nghiệm khép kín theo đúng khế ước đo lường:
  0. Preflight Fail-Fast: Kiểm tra CUDA, Git commit sạch, checksum ViT-L-14 và các checkpoint.
  1. Eval 7.3 cho Baseline và Anchored trên Clean & Q30 -> Xuất JSON v3 và NPZ logits.
  2. Calibration 7.4 độc lập -> Xuất T, T* có ràng buộc FPR 5% và ECE.
  3. WiSE-FT Gen 2: Quét lưới sinh 12 ứng viên kèm --verify 0/0 keys.
  4. Sweep 12 ứng viên thật trên Clean & Q30 -> Tuyển chọn Pareto vô địch (selected=True duy nhất).
  5. Eval 7.3 cho WiSE-FT Gen 2 Best -> Xuất JSON v3 và NPZ logits.
  6. Re-baseline 7.6 ma trận 4 cấu hình x 2 điều kiện từ số đo thật (evidence_level=measured).
  7. Validator Cổng G7.1: Phán quyết chất lượng học thuật chính thức (PASS/FAIL).
"""

import os
import sys
import json
import time
import shutil
import hashlib
import argparse
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import yaml
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def compute_sha256(file_path: str) -> str:
    """Tính mã băm SHA-256 của tệp tin."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> Tuple[str, str, bool]:
    """Lấy thông tin commit, branch và trạng thái dirty của Git."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        branch = "unknown"

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        commit = "unknown"

    try:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)DATASET/**", ":(exclude)*.ipynb", ":(exclude)*.ipynb_checkpoints/**"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        is_dirty = len(status_out) > 0
    except Exception:
        is_dirty = False

    return branch, commit, is_dirty


def run_preflight_checks(cfg: Dict[str, Any], allow_cpu: bool = False, allow_dirty: bool = False):
    """
    Kiểm tra tiền điều kiện (Preflight Validation) — Dừng ngay nếu không đạt chuẩn.
    """
    print("\n" + "=" * 88)
    print("FATFORMER-XLA: BẮT ĐẦU KIỂM TRA TIỀN ĐIỀU KIỆN (PREFLIGHT VALIDATION)")
    print("=" * 88)

    # 1. Kiểm tra phần cứng CUDA
    require_cuda = bool(cfg.get("environment", {}).get("require_cuda", True))
    if require_cuda and not allow_cpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[PREFLIGHT FAIL] Không tìm thấy GPU CUDA! Pipeline đo lường thật bắt buộc chạy trên GPU "
                "(Colab/Kaggle). Dùng cờ --allow_cpu chỉ khi debug luồng logic."
            )
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  [✓] GPU CUDA        : {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        print("  [!] CẢNH BÁO        : Đang chạy ở chế độ CPU debug (--allow_cpu)")

    # 2. Kiểm tra Git branch & status
    branch, commit, is_dirty = get_git_info()
    expected_branch = cfg.get("environment", {}).get("expected_branch", "hoang")
    print(f"  [✓] Git Branch      : {branch} (Kỳ vọng: {expected_branch})")
    print(f"  [✓] Git Commit      : {commit[:16]}...")
    if branch != expected_branch and branch != "HEAD" and expected_branch != "any":
        raise RuntimeError(f"[PREFLIGHT FAIL] Đang ở nhánh '{branch}', yêu cầu phải ở nhánh '{expected_branch}'.")
    require_clean_git = bool(cfg.get("environment", {}).get("require_clean_git", True))
    if require_clean_git and is_dirty and not allow_dirty:
        raise RuntimeError(
            "[PREFLIGHT FAIL] Git working tree bị dirty! Vui lòng commit hoặc stash trước khi chạy nghiệm thu."
        )

    # 3. Kiểm tra trọng số backbone CLIP
    clip_path = os.path.join(PROJECT_ROOT, cfg["checkpoints"]["clip_vit_l14"]["path"])
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(
            f"[PREFLIGHT FAIL] Không tìm thấy backbone CLIP tại: {clip_path}\n"
            f"Vui lòng tải hoặc mount file ViT-L-14.pt vào thư mục DATASET/pretrained/"
        )
    clip_sha = compute_sha256(clip_path)
    expected_clip_sha = cfg["checkpoints"]["clip_vit_l14"].get("expected_sha256")
    if expected_clip_sha and clip_sha != expected_clip_sha:
        raise ValueError(
            f"[PREFLIGHT FAIL] Sai lệch SHA-256 CLIP ViT-L-14: Nhận {clip_sha}, kỳ vọng {expected_clip_sha}"
        )
    print(f"  [✓] CLIP Backbone   : {clip_path} (SHA: {clip_sha[:16]}...) [KHỚP]")

    # 4. Kiểm tra Checkpoint Baseline
    base_info = cfg["checkpoints"]["baseline"]
    base_path = os.path.join(PROJECT_ROOT, base_info["path"])
    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"[PREFLIGHT FAIL] Không tìm thấy Baseline checkpoint: {base_path}")
    base_sha = compute_sha256(base_path)
    if base_info.get("expected_sha256") and base_sha != base_info["expected_sha256"]:
        raise ValueError(
            f"[PREFLIGHT FAIL] Sai lệch SHA-256 Baseline: Nhận {base_sha}, kỳ vọng {base_info['expected_sha256']}"
        )
    print(f"  [✓] Baseline Ckpt   : {base_path} (SHA: {base_sha[:16]}...) [KHỚP]")

    # 5. Kiểm tra Checkpoint Anchored
    anch_info = cfg["checkpoints"]["anchored"]
    anch_path = os.path.join(PROJECT_ROOT, anch_info["path"])
    if not os.path.isfile(anch_path):
        raise FileNotFoundError(f"[PREFLIGHT FAIL] Không tìm thấy Anchored checkpoint: {anch_path}")
    anch_sha = compute_sha256(anch_path)
    if anch_info.get("expected_sha256") and anch_sha != anch_info["expected_sha256"]:
        raise ValueError(
            f"[PREFLIGHT FAIL] Sai lệch SHA-256 Anchored: Nhận {anch_sha}, kỳ vọng {anch_info['expected_sha256']}"
        )
    print(f"  [✓] Anchored Ckpt   : {anch_path} (SHA: {anch_sha[:16]}...) [KHỚP]")

    # 6. Kiểm tra Dataset CFE (Ép buộc đủ 5 subsets, đủ cả 2 nhãn, khớp đúng số ảnh)
    cfe_root = os.path.join(PROJECT_ROOT, cfg["data"]["cfe_root"])
    if not os.path.isdir(cfe_root):
        raise FileNotFoundError(
            f"[PREFLIGHT FAIL] Không tìm thấy thư mục CFE dataset tại: {cfe_root}\n"
            f"Vui lòng giải nén hoặc liên kết thư mục test_communityforensics/"
        )
    total_imgs = 0
    expected_imgs = cfg["data"].get("expected_samples", 2000)
    for sub in cfg["data"]["subsets"]:
        sub_dir = os.path.join(cfe_root, sub)
        if not os.path.isdir(sub_dir):
            raise FileNotFoundError(f"[PREFLIGHT FAIL] Thiếu tập con CFE: {sub_dir}")
        for lbl in ["0_real", "1_fake"]:
            f_dir = os.path.join(sub_dir, lbl)
            if not os.path.isdir(f_dir):
                raise FileNotFoundError(f"[PREFLIGHT FAIL] Thiếu thư mục nhãn '{lbl}' trong tập con {sub}: {f_dir}")
            imgs_in_folder = [f for f in os.listdir(f_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
            if len(imgs_in_folder) == 0:
                raise ValueError(f"[PREFLIGHT FAIL] Thư mục nhãn rỗng không có ảnh: {f_dir}")
            total_imgs += len(imgs_in_folder)
    print(f"  [✓] CFE Dataset     : {cfe_root} ({total_imgs} ảnh quét được, kỳ vọng đúng {expected_imgs})")
    if total_imgs != expected_imgs:
        raise ValueError(
            f"[PREFLIGHT FAIL] Tổng số ảnh CFE ({total_imgs}) không khớp expected_samples ({expected_imgs})! "
            f"Dataset bị thiếu hoặc sai lệch cấu trúc; dừng ngay để bảo vệ tính toàn vẹn đo lường."
        )

    # 7. Kiểm tra tính tương thích nạp Canonical (0 missing, 0 unexpected)
    from src.models.canonical_loader import load_canonical
    from types import SimpleNamespace
    print("  [*] Đang thử nghiệm strict load canonical hai checkpoint nguồn...")
    for name, c_path, use_mod in [("Baseline", base_path, False), ("Anchored", anch_path, True)]:
        args = SimpleNamespace(
            backbone="CLIP:ViT-L/14", clip_path=clip_path, num_classes=2,
            use_srm=use_mod, use_gating=use_mod, srm_out_dim=1024, num_context_embedding=8
        )
        _, rep = load_canonical(c_path, args)
        if rep.get("missing", 0) != 0 or rep.get("unexpected", 0) != 0:
            raise RuntimeError(f"[PREFLIGHT FAIL] {name} vi phạm 0/0 keys: Missing={rep.get('missing')}, Unexpected={rep.get('unexpected')}")
        print(f"    • {name:<10}: strict load OK (0/0 keys, {rep.get('n_keys')} keys)")

    print("=" * 88)
    print("🏆 TOÀN BỘ TIỀN ĐIỀU KIỆN PREFLIGHT ĐÃ ĐẠT CHUẨN 100%! BẮT ĐẦU QUY TRÌNH!")
    print("=" * 88 + "\n")
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def execute_pipeline_step(cmd_args: List[str], desc: str):
    """Thực thi một lệnh trong pipeline và bắt lỗi fail-fast."""
    cmd_str = " ".join(cmd_args)
    print(f"\n>>> [BƯỚC] {desc}")
    print(f"    Lệnh: {cmd_str}\n")
    start_t = time.time()
    ret = subprocess.run(cmd_args, cwd=PROJECT_ROOT)
    elapsed = time.time() - start_t
    if ret.returncode != 0:
        raise RuntimeError(f"[LỖI PIPELINE] Lệnh thất bại (mã thoát {ret.returncode}): {cmd_str}")
    print(f"\n[✓] Hoàn thành trong {elapsed:.1f}s: {desc}\n")


def validate_measured_calibration_artifact(
    calibration_json: str,
    *,
    expected_model: str,
    calibration_npz: str,
    expected_checkpoint_sha256: Optional[str] = None,
    expected_fpr_target: float = 0.05,
) -> Tuple[bool, str]:
    """Kiểm tra nội dung calibration trước khi cho phép ``--resume`` tái sử dụng."""
    if not os.path.isfile(calibration_json):
        return False, "không tồn tại"
    if not os.path.isfile(calibration_npz):
        return False, f"thiếu calibration split {calibration_npz}"
    try:
        with open(calibration_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema") != "calibration/v1":
            return False, f"schema={payload.get('schema')!r}, yêu cầu 'calibration/v1'"
        if payload.get("evidence_level") != "measured":
            return False, f"evidence_level={payload.get('evidence_level')!r}, yêu cầu 'measured'"
        if payload.get("model") != expected_model:
            return False, f"model={payload.get('model')!r}, yêu cầu {expected_model!r}"

        temperature = float(payload["T"])
        if not np.isfinite(temperature) or temperature <= 0.0:
            return False, f"T không hợp lệ: {temperature}"
        raw_threshold = payload.get("T_star_default", payload.get("T_star"))
        threshold = float(raw_threshold.get("0.05")) if isinstance(raw_threshold, dict) else float(raw_threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            return False, f"T* không hợp lệ: {threshold}"
        if abs(float(payload.get("fpr_target")) - float(expected_fpr_target)) > 1e-12:
            return False, f"fpr_target={payload.get('fpr_target')}, yêu cầu {expected_fpr_target}"

        source_sha = compute_sha256(calibration_npz)
        if payload.get("calibration_source_sha256") != source_sha:
            return False, "SHA-256 calibration split không khớp"
        with np.load(calibration_npz, allow_pickle=False) as split_payload:
            if "split" not in split_payload or str(split_payload["split"]) != "calibration":
                return False, "NPZ nguồn không mang split=calibration"
            n_calib = int(len(split_payload["targets"]))
        if int(payload.get("n_calib", -1)) != n_calib:
            return False, f"n_calib={payload.get('n_calib')}, yêu cầu {n_calib}"

        checkpoint_info = payload.get("checkpoint", {})
        if expected_checkpoint_sha256 and checkpoint_info.get("sha256") != expected_checkpoint_sha256:
            return False, "SHA-256 checkpoint trong calibration không khớp config"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return False, f"không đọc/kiểm tra được: {exc}"
    return True, "measured artifact hợp lệ"


def validate_candidate_eval_artifact(
    sidecar_json: str,
    *,
    checkpoint_path: str,
    calibration_json: str,
    expected_n_samples: int,
) -> Tuple[bool, str]:
    """Chỉ cho resume sidecar evaluation-only v2 gắn với calibration hiện hành."""
    if not os.path.isfile(sidecar_json):
        return False, "không tồn tại"
    try:
        with open(sidecar_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema") != "candidate_eval/v2":
            return False, f"schema={payload.get('schema')!r}, yêu cầu 'candidate_eval/v2'"
        if payload.get("evidence_level") != "measured" or payload.get("split") != "evaluation":
            return False, "sidecar không phải measured evaluation split"
        if int(payload.get("n_samples", -1)) != int(expected_n_samples):
            return False, f"n_samples={payload.get('n_samples')}, yêu cầu {expected_n_samples}"
        if os.path.basename(str(payload.get("checkpoint", ""))) != os.path.basename(checkpoint_path):
            return False, "checkpoint trong sidecar không khớp candidate"
        expected_calibration_sha = compute_sha256(calibration_json)
        if payload.get("calibration_sha256") != expected_calibration_sha:
            return False, "sidecar không gắn với calibration measured hiện hành"
        required_metrics = {
            "real_acc_clean", "fake_at_real90_clean", "auc_clean",
            "real_acc_q30", "fake_at_real90_q30", "auc_q30",
        }
        metrics = payload.get("metrics", {})
        if set(metrics) != required_metrics or not all(np.isfinite(float(metrics[key])) for key in required_metrics):
            return False, "metrics thiếu hoặc không hữu hạn"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return False, f"không đọc/kiểm tra được: {exc}"
    return True, "candidate evaluation artifact hợp lệ"


def validate_wise_candidate_artifact(
    checkpoint_path: str,
    *,
    alpha: float,
    beta: float,
    baseline_sha256: str,
    anchored_sha256: str,
) -> Tuple[bool, str]:
    """Xác nhận checkpoint WiSE-FT đã hoàn tất ghi và verify 0/0 keys."""
    metadata_path = checkpoint_path.replace(".pth", "_metadata.json")
    if not os.path.isfile(checkpoint_path) or not os.path.isfile(metadata_path):
        return False, "thiếu checkpoint hoặc metadata"
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if abs(float(payload.get("alpha")) - float(alpha)) > 1e-12:
            return False, "alpha không khớp"
        if abs(float(payload.get("beta_srm")) - float(beta)) > 1e-12:
            return False, "beta_srm không khớp"
        verification = payload.get("verification", {})
        if verification.get("status") != "ok" or int(verification.get("missing", -1)) != 0 or int(verification.get("unexpected", -1)) != 0:
            return False, "verification không đạt 0/0 keys"
        sources = payload.get("sources", {})
        if sources.get("baseline", {}).get("sha256") != baseline_sha256:
            return False, "baseline source SHA-256 không khớp"
        if sources.get("anchored", {}).get("sha256") != anchored_sha256:
            return False, "anchored source SHA-256 không khớp"
        expected_size = float(payload.get("file_size_mb", 0.0))
        actual_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        if expected_size <= 0.0 or abs(actual_size - expected_size) > 0.02:
            return False, "kích thước checkpoint không khớp metadata"
        expected_sha = payload.get("sha256")
        if not expected_sha or compute_sha256(checkpoint_path) != expected_sha:
            return False, "SHA-256 checkpoint candidate không khớp metadata"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return False, f"không đọc/kiểm tra được: {exc}"
    return True, "checkpoint candidate hợp lệ"


def validate_logits_split_pair(
    source_npz: str,
    calibration_npz: str,
    evaluation_npz: str,
    *,
    calib_ratio: float,
) -> Tuple[bool, str]:
    """Xác nhận hai split là một partition nguyên vẹn của logits nguồn."""
    if not all(os.path.isfile(path) for path in (source_npz, calibration_npz, evaluation_npz)):
        return False, "thiếu source/calibration/evaluation NPZ"
    try:
        with np.load(source_npz, allow_pickle=False) as source:
            source_logits = source["logits"]
            source_targets = source["targets"]
            source_ids = source["sample_ids"]
            n_source = len(source_targets)
            expected_calib_per_label = {
                label: max(1, min(int(np.sum(source_targets == label)) - 1, int(round(np.sum(source_targets == label) * calib_ratio))))
                for label in np.unique(source_targets)
            }

            split_indices: Dict[str, np.ndarray] = {}
            for expected_split, split_path in (
                ("calibration", calibration_npz),
                ("evaluation", evaluation_npz),
            ):
                with np.load(split_path, allow_pickle=False) as split_payload:
                    if "split" not in split_payload or str(split_payload["split"]) != expected_split:
                        return False, f"{split_path} không mang split={expected_split}"
                    indices = split_payload["source_indices"].astype(np.int64, copy=False)
                    if np.any(indices < 0) or np.any(indices >= n_source) or len(np.unique(indices)) != len(indices):
                        return False, f"source_indices không hợp lệ trong {split_path}"
                    if not np.array_equal(split_payload["logits"], source_logits[indices]):
                        return False, f"logits split không khớp nguồn trong {split_path}"
                    if not np.array_equal(split_payload["targets"], source_targets[indices]):
                        return False, f"targets split không khớp nguồn trong {split_path}"
                    if not np.array_equal(split_payload["sample_ids"], source_ids[indices]):
                        return False, f"sample_ids split không khớp nguồn trong {split_path}"
                    split_indices[expected_split] = indices

        calib_indices = split_indices["calibration"]
        eval_indices = split_indices["evaluation"]
        combined = np.concatenate([calib_indices, eval_indices])
        if len(combined) != n_source or not np.array_equal(np.sort(combined), np.arange(n_source)):
            return False, "calibration/evaluation không tạo thành partition đầy đủ"
        calibration_targets = source_targets[calib_indices]
        for label, expected_count in expected_calib_per_label.items():
            actual_count = int(np.sum(calibration_targets == label))
            if actual_count != expected_count:
                return False, f"calibration label {label} có {actual_count} mẫu, yêu cầu {expected_count}"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, f"không đọc/kiểm tra được split: {exc}"
    return True, "partition calibration/evaluation hợp lệ"


def split_calibration_and_eval_logits(
    logits_npz: str,
    calib_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[str, str]:
    """
    Tách tập logits thành 2 phân vùng hoàn toàn độc lập (Zero Data Leakage):
      - *_calib_split.npz (20% mẫu): Chỉ dùng cho Task 7.4 (fit T và T*).
      - *_eval_split.npz (80% mẫu): Dùng cho Sweep Task 7.5, Rebaseline Task 7.6 và Cổng G7.1.
    """
    if not 0.0 < calib_ratio < 1.0:
        raise ValueError(f"calib_ratio phải nằm trong (0, 1), nhận {calib_ratio}")

    with np.load(logits_npz, allow_pickle=False) as data:
        if "logits" not in data or "targets" not in data:
            raise ValueError(f"NPZ thiếu logits/targets bắt buộc: {logits_npz}")
        logits = data["logits"].copy()
        targets = data["targets"].copy()
        sample_ids = data["sample_ids"].copy() if "sample_ids" in data else np.asarray(
            [f"row:{idx}" for idx in range(len(targets))]
        )
    if logits.ndim != 2 or logits.shape[1] != 2 or len(logits) != len(targets):
        raise ValueError(f"Shape logits/targets không hợp lệ trong {logits_npz}")
    if len(sample_ids) != len(targets):
        raise ValueError(f"sample_ids không cùng số hàng với targets trong {logits_npz}")

    # Tách phân tầng theo nhãn để cả calibration và evaluation luôn có đủ hai lớp.
    rng = np.random.default_rng(seed)
    calib_parts: List[np.ndarray] = []
    eval_parts: List[np.ndarray] = []
    for label in np.unique(targets):
        label_idx = np.flatnonzero(targets == label)
        if len(label_idx) < 2:
            raise ValueError(f"Nhãn {label} có quá ít mẫu để tách độc lập: {len(label_idx)}")
        label_idx = rng.permutation(label_idx)
        n_label_calib = max(1, min(len(label_idx) - 1, int(round(len(label_idx) * calib_ratio))))
        calib_parts.append(label_idx[:n_label_calib])
        eval_parts.append(label_idx[n_label_calib:])

    calib_idx = np.sort(np.concatenate(calib_parts))
    eval_idx = np.sort(np.concatenate(eval_parts))
    if np.intersect1d(calib_idx, eval_idx).size != 0 or len(calib_idx) + len(eval_idx) != len(targets):
        raise RuntimeError("[DATA LEAKAGE ERROR] Phân vùng calibration/evaluation không tạo thành partition hợp lệ")

    calib_npz = logits_npz.replace(".npz", "_calib_split.npz")
    eval_npz = logits_npz.replace(".npz", "_eval_split.npz")

    np.savez_compressed(
        calib_npz,
        logits=logits[calib_idx],
        targets=targets[calib_idx],
        sample_ids=sample_ids[calib_idx],
        source_indices=calib_idx,
        source=logits_npz,
        split="calibration",
        n_samples=len(calib_idx),
    )
    np.savez_compressed(
        eval_npz,
        logits=logits[eval_idx],
        targets=targets[eval_idx],
        sample_ids=sample_ids[eval_idx],
        source_indices=eval_idx,
        source=logits_npz,
        split="evaluation",
        n_samples=len(eval_idx),
    )
    print(f"  [✓] Phân vùng độc lập: Calib={calib_npz} ({len(calib_idx)} mẫu) | Eval={eval_npz} ({len(eval_idx)} mẫu)")
    return calib_npz, eval_npz


def write_candidate_eval_sidecar_from_splits(
    clean_eval_npz: str,
    q30_eval_npz: str,
    calibration_json: str,
    sidecar_path: str,
    checkpoint_path: str,
    reference_eval_npz: Optional[str] = None,
) -> None:
    """Ghi sidecar tuyển chọn chỉ từ evaluation split, không dùng calibration rows."""
    from tools.eval_communityforensics import calculate_metrics_v3

    with open(calibration_json, "r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    if calibration.get("evidence_level") != "measured":
        raise ValueError(f"Calibration không phải measured evidence: {calibration_json}")
    temperature = float(calibration["T"])
    raw_threshold = calibration.get("T_star_default", calibration.get("T_star"))
    threshold = float(raw_threshold.get("0.05")) if isinstance(raw_threshold, dict) else float(raw_threshold)

    metrics_by_condition: Dict[str, Dict[str, Any]] = {}
    reference_ids: Optional[np.ndarray] = None
    for condition, npz_path in (("clean", clean_eval_npz), ("q30", q30_eval_npz)):
        with np.load(npz_path, allow_pickle=False) as payload:
            split_name = str(payload["split"]) if "split" in payload else ""
            if split_name != "evaluation":
                raise ValueError(f"Candidate split không được đánh dấu evaluation: {npz_path}")
            logits = payload["logits"].copy()
            targets = payload["targets"].copy()
            sample_ids = payload["sample_ids"].copy()
        if reference_ids is None:
            reference_ids = sample_ids
        elif not np.array_equal(reference_ids, sample_ids):
            raise ValueError("Clean/Q30 candidate splits không cùng sample_ids")
        scaled = logits / max(temperature, 1e-4)
        exp_logits = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        fake_probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[:, 1]
        metrics_by_condition[condition] = calculate_metrics_v3(fake_probs, targets, threshold=threshold)

    clean_m = metrics_by_condition["clean"]
    q30_m = metrics_by_condition["q30"]
    candidate_ids = reference_ids
    if reference_eval_npz:
        with np.load(reference_eval_npz, allow_pickle=False) as reference:
            baseline_ids = reference["sample_ids"]
            if not np.array_equal(baseline_ids, candidate_ids):
                raise ValueError("Candidate evaluation split không khớp baseline reference split")
    sidecar = {
        "schema": "candidate_eval/v2",
        "evidence_level": "measured",
        "split": "evaluation",
        "n_samples": int(len(reference_ids)),
        "candidate": os.path.basename(checkpoint_path),
        "checkpoint": checkpoint_path,
        "calibration": calibration_json,
        "calibration_sha256": compute_sha256(calibration_json),
        "metrics": {
            "real_acc_clean": float(clean_m["real_acc"]),
            "fake_at_real90_clean": float(clean_m["fake_at_real90"]),
            "auc_clean": float(clean_m["auc"]),
            "real_acc_q30": float(q30_m["real_acc"]),
            "fake_at_real90_q30": float(q30_m["fake_at_real90"]),
            "auc_q30": float(q30_m["auc"]),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(sidecar_path)), exist_ok=True)
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, ensure_ascii=False)
    print(f"  [✓] Sidecar evaluation-only: {sidecar_path} ({sidecar['n_samples']} mẫu)")


def run_phase7_acceptance(cfg_path: str, allow_cpu: bool = False, allow_dirty: bool = False, resume: bool = True):
    """
    Hàm thực thi toàn bộ 7 bước nghiệm thu GPU.
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Bước 0: Preflight
    run_preflight_checks(cfg, allow_cpu=allow_cpu, allow_dirty=allow_dirty)

    data_dir = cfg["data"]["cfe_root"]
    clip_path = cfg["checkpoints"]["clip_vit_l14"]["path"]
    base_ckpt = cfg["checkpoints"]["baseline"]["path"]
    anch_ckpt = cfg["checkpoints"]["anchored"]["path"]
    batch_size = str(cfg["runtime"].get("batch_size", 32))
    num_workers = str(cfg["runtime"].get("num_workers", 2))
    seed = int(cfg.get("seed", 42))
    device = "cpu" if allow_cpu else "cuda"
    out_logs_dir = cfg["rebaseline"]["measured_artifacts_dir"]
    os.makedirs(out_logs_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # BƯỚC 1: Đánh giá Eval 7.3 cho Baseline & Anchored trên Clean & Q30
    # -------------------------------------------------------------------------
    for m_type, m_name, c_path in [
        ("baseline", "baseline", base_ckpt),
        ("anchored", "anchored", anch_ckpt),
    ]:
        for cond in ["clean", "q30"]:
            npz_check = os.path.join(out_logs_dir, f"logits_{m_name}_{cond}.npz")
            json_check = os.path.join(out_logs_dir, f"eval_v3_{m_name}_{cond}.json")
            if resume and os.path.isfile(npz_check) and os.path.isfile(json_check):
                print(f"[RESUME] Bỏ qua Eval {m_name.upper()} ({cond.upper()}): Đã có kết quả đo lường tại {npz_check}")
                continue

            execute_pipeline_step(
                [
                    sys.executable, "tools/eval_communityforensics.py",
                    "--data_dir", data_dir,
                    "--clip_path", clip_path,
                    "--checkpoint", c_path,
                    "--model_type", m_type,
                    "--model_name", m_name,
                    "--condition", cond,
                    "--save_logits_npz",
                    "--batch_size", batch_size,
                    "--num_workers", num_workers,
                    "--device", device,
                    "--seed", str(seed),
                    "--output_dir", out_logs_dir,
                ],
                f"Eval 7.3 cho mô hình {m_name.upper()} trên điều kiện {cond.upper()}"
            )

    # -------------------------------------------------------------------------
    # BƯỚC 2: Hiệu chuẩn nhiệt độ & ngưỡng ràng buộc FPR 5% (Task 7.4)
    # Tách phân vùng calibration 20% và evaluation 80% (Zero Data Leakage)
    # -------------------------------------------------------------------------
    calib_ratio = cfg["data"].get("calib_split_ratio", 0.20)
    base_clean_npz = os.path.join(out_logs_dir, "logits_baseline_clean.npz")
    anch_clean_npz = os.path.join(out_logs_dir, "logits_anchored_clean.npz")
    base_q30_npz = os.path.join(out_logs_dir, "logits_baseline_q30.npz")
    anch_q30_npz = os.path.join(out_logs_dir, "logits_anchored_q30.npz")

    split_paths: Dict[str, Tuple[str, str]] = {}
    for source_npz in (base_clean_npz, anch_clean_npz, base_q30_npz, anch_q30_npz):
        calibration_npz = source_npz.replace(".npz", "_calib_split.npz")
        evaluation_npz = source_npz.replace(".npz", "_eval_split.npz")
        split_valid, split_reason = validate_logits_split_pair(
            source_npz,
            calibration_npz,
            evaluation_npz,
            calib_ratio=calib_ratio,
        )
        if resume and split_valid:
            print(f"[RESUME] Bỏ qua tách {os.path.basename(source_npz)}: {split_reason}.")
        else:
            if resume and (os.path.isfile(calibration_npz) or os.path.isfile(evaluation_npz)):
                print(f"[RESUME] Loại split cũ của {os.path.basename(source_npz)}: {split_reason}.")
            calibration_npz, evaluation_npz = split_calibration_and_eval_logits(
                source_npz, calib_ratio=calib_ratio, seed=seed
            )
        split_paths[source_npz] = (calibration_npz, evaluation_npz)

    base_calib_npz, base_clean_eval_npz = split_paths[base_clean_npz]
    anch_calib_npz, _ = split_paths[anch_clean_npz]

    calib_out_dir = cfg["calibration"]["output_dir"]
    os.makedirs(calib_out_dir, exist_ok=True)

    base_calib_json = os.path.join(calib_out_dir, "baseline.json")
    base_calib_valid, base_calib_reason = validate_measured_calibration_artifact(
        base_calib_json,
        expected_model="baseline",
        calibration_npz=base_calib_npz,
        expected_checkpoint_sha256=cfg["checkpoints"]["baseline"].get("expected_sha256"),
        expected_fpr_target=float(cfg["calibration"].get("target_fpr", 0.05)),
    )
    if not (resume and base_calib_valid):
        if resume and os.path.isfile(base_calib_json):
            print(f"[RESUME] Loại Calibration Baseline cũ: {base_calib_reason}. Sẽ fit lại từ measured split.")
        execute_pipeline_step(
            [
                sys.executable, "tools/calibrate_and_select.py",
                "--logits_npz", base_calib_npz,
                "--model_name", "baseline",
                "--checkpoint", base_ckpt,
                "--output_dir", calib_out_dir,
                "--fpr_target", str(cfg["calibration"].get("target_fpr", 0.05)),
            ],
            "Hiệu chuẩn nhiệt độ & ngưỡng FPR 5% cho Baseline (Task 7.4)"
        )
    else:
        print(f"[RESUME] Bỏ qua Calibration Baseline: {base_calib_reason} tại {base_calib_json}")

    anch_calib_json = os.path.join(calib_out_dir, "anchored.json")
    anch_calib_valid, anch_calib_reason = validate_measured_calibration_artifact(
        anch_calib_json,
        expected_model="anchored",
        calibration_npz=anch_calib_npz,
        expected_checkpoint_sha256=cfg["checkpoints"]["anchored"].get("expected_sha256"),
        expected_fpr_target=float(cfg["calibration"].get("target_fpr", 0.05)),
    )
    if not (resume and anch_calib_valid):
        if resume and os.path.isfile(anch_calib_json):
            print(f"[RESUME] Loại Calibration Anchored cũ: {anch_calib_reason}. Sẽ fit lại từ measured split.")
        execute_pipeline_step(
            [
                sys.executable, "tools/calibrate_and_select.py",
                "--logits_npz", anch_calib_npz,
                "--model_name", "anchored",
                "--checkpoint", anch_ckpt,
                "--output_dir", calib_out_dir,
                "--fpr_target", str(cfg["calibration"].get("target_fpr", 0.05)),
            ],
            "Hiệu chuẩn nhiệt độ & ngưỡng FPR 5% cho Anchored (Task 7.4)"
        )
    else:
        print(f"[RESUME] Bỏ qua Calibration Anchored: {anch_calib_reason} tại {anch_calib_json}")

    # -------------------------------------------------------------------------
    # BƯỚC 3: Sinh lưới 12 ứng viên WiSE-FT Gen 2 (Task 7.5)
    # -------------------------------------------------------------------------
    cands_dir = cfg["wise_ft_gen2"]["candidates_dir"]
    alpha_str = ",".join(str(a) for a in cfg["wise_ft_gen2"]["alpha_grid"])
    beta_str = ",".join(str(b) for b in cfg["wise_ft_gen2"]["beta_grid"])

    expected_candidates = [
        (alpha, beta, os.path.join(cands_dir, f"wise_ft_v2_a{int(alpha * 100):02d}_b{int(beta * 100):02d}.pth"))
        for alpha in cfg["wise_ft_gen2"]["alpha_grid"]
        for beta in cfg["wise_ft_gen2"]["beta_grid"]
    ]
    candidate_checks = [
        validate_wise_candidate_artifact(
            path,
            alpha=alpha,
            beta=beta,
            baseline_sha256=cfg["checkpoints"]["baseline"]["expected_sha256"],
            anchored_sha256=cfg["checkpoints"]["anchored"]["expected_sha256"],
        )
        for alpha, beta, path in expected_candidates
    ]
    if resume and all(valid for valid, _ in candidate_checks):
        print(f"[RESUME] Bỏ qua sinh lưới: cả {len(expected_candidates)} candidate đã verify 0/0 keys và khớp nguồn.")
    else:
        execute_pipeline_step(
            [
                sys.executable, "tools/wise_ft_merge.py",
                "--orig_checkpoint", base_ckpt,
                "--target_checkpoint", anch_ckpt,
                "--grid_search",
                "--alpha_grid", alpha_str,
                "--beta_grid", beta_str,
                "--output_dir", cands_dir,
                "--verify",
            ],
            "Sinh lưới 12 ứng viên WiSE-FT Gen 2 kèm kiểm tra nghiêm ngặt verify 0/0 keys"
        )

    # -------------------------------------------------------------------------
    # BƯỚC 4: Đo đạc 12 ứng viên trên CFE & Tuyển chọn Pareto Best (Task 7.5)
    # -------------------------------------------------------------------------
    cand_files = [
        f"wise_ft_v2_a{int(alpha * 100):02d}_b{int(beta * 100):02d}.pth"
        for alpha in cfg["wise_ft_gen2"]["alpha_grid"]
        for beta in cfg["wise_ft_gen2"]["beta_grid"]
    ]
    missing_candidates = [name for name in cand_files if not os.path.isfile(os.path.join(cands_dir, name))]
    if missing_candidates:
        raise FileNotFoundError(f"Thiếu candidate vừa sinh: {missing_candidates}")
    print(f"\n[*] Bắt đầu đo đạc đánh giá {len(cand_files)} ứng viên trên CFE Clean & Q30...")

    base_calib_json = cfg["calibration"]["baseline_json"]
    expected_eval_samples = int(round(cfg["data"]["expected_samples"] * (1.0 - calib_ratio)))
    for cand_f in sorted(cand_files):
        c_path = os.path.join(cands_dir, cand_f)
        cand_name = cand_f.replace(".pth", "")
        cand_eval_json = c_path.replace(".pth", "_eval.json")
        cand_clean_npz = os.path.join(out_logs_dir, f"logits_{cand_name}_clean.npz")
        cand_q30_npz = os.path.join(out_logs_dir, f"logits_{cand_name}_q30.npz")
        candidate_valid, candidate_reason = validate_candidate_eval_artifact(
            cand_eval_json,
            checkpoint_path=c_path,
            calibration_json=base_calib_json,
            expected_n_samples=expected_eval_samples,
        )
        if resume and candidate_valid and os.path.isfile(cand_clean_npz) and os.path.isfile(cand_q30_npz):
            print(f"[RESUME] Bỏ qua Đo đạc Candidate {cand_name}: {candidate_reason} tại {cand_eval_json}")
            continue
        if resume and os.path.isfile(cand_eval_json):
            print(f"[RESUME] Không tái sử dụng Candidate {cand_name}: {candidate_reason}. Sẽ đo lại.")

        execute_pipeline_step(
            [
                sys.executable, "tools/eval_communityforensics.py",
                "--data_dir", data_dir,
                "--clip_path", clip_path,
                "--checkpoint", c_path,
                "--model_type", "candidate",
                "--model_name", cand_name,
                "--condition", "all",
                "--save_logits_npz",
                "--calibration", base_calib_json,
                "--batch_size", batch_size,
                "--num_workers", num_workers,
                "--device", device,
                "--seed", str(seed),
                "--output_dir", out_logs_dir,
                "--export_candidate_eval", c_path.replace(".pth", "_eval.json"),
            ],
            f"Đo đạc ứng viên WiSE-FT: {cand_name} (Clean & Q30)"
        )
        cand_clean_npz = os.path.join(out_logs_dir, f"logits_{cand_name}_clean.npz")
        cand_q30_npz = os.path.join(out_logs_dir, f"logits_{cand_name}_q30.npz")
        _, cand_clean_eval = split_calibration_and_eval_logits(cand_clean_npz, calib_ratio=calib_ratio, seed=seed)
        _, cand_q30_eval = split_calibration_and_eval_logits(cand_q30_npz, calib_ratio=calib_ratio, seed=seed)
        write_candidate_eval_sidecar_from_splits(
            cand_clean_eval,
            cand_q30_eval,
            base_calib_json,
            c_path.replace(".pth", "_eval.json"),
            c_path,
            base_clean_eval_npz,
        )

    # Chạy tuyển chọn Pareto
    sweep_csv = cfg["wise_ft_gen2"]["sweep_csv"]
    best_ckpt_path = cfg["wise_ft_gen2"]["best_checkpoint_path"]

    execute_pipeline_step(
        [
            sys.executable, "tools/sweep_wise_ft_candidates.py",
            "--candidates_dir", cands_dir,
            "--calibration", base_calib_json,
            "--alpha_grid", alpha_str,
            "--beta_grid", beta_str,
            "--output_csv", sweep_csv,
            "--best_output", best_ckpt_path,
        ],
        "Tuyển chọn ứng viên WiSE-FT Gen 2 tối ưu Pareto (selected=True duy nhất)"
    )

    # -------------------------------------------------------------------------
    # BƯỚC 5: Đánh giá mô hình chiến thắng WiSE-FT Gen 2 Best (Task 7.3)
    # -------------------------------------------------------------------------
    for cond in ["clean", "q30"]:
        npz_check = os.path.join(out_logs_dir, f"logits_wise_ft_v2_{cond}.npz")
        json_check = os.path.join(out_logs_dir, f"eval_v3_wise_ft_v2_{cond}.json")
        if resume and os.path.isfile(npz_check) and os.path.isfile(json_check):
            print(f"[RESUME] Bỏ qua Eval WiSE-FT Gen 2 Best ({cond.upper()}): Đã có kết quả tại {npz_check}")
            continue

        execute_pipeline_step(
            [
                sys.executable, "tools/eval_communityforensics.py",
                "--data_dir", data_dir,
                "--clip_path", clip_path,
                "--checkpoint", best_ckpt_path,
                "--model_type", "wise_ft_v2",
                "--model_name", "wise_ft_v2",
                "--condition", cond,
                "--save_logits_npz",
                "--calibration", base_calib_json,
                "--batch_size", batch_size,
                "--num_workers", num_workers,
                "--device", device,
                "--seed", str(seed),
                "--output_dir", out_logs_dir,
            ],
            f"Đánh giá WiSE-FT Gen 2 Best trên điều kiện {cond.upper()}"
        )

    # Tách phân vùng calibration / evaluation cho WiSE-FT Gen 2 Best
    wise_clean_npz = os.path.join(out_logs_dir, "logits_wise_ft_v2_clean.npz")
    wise_q30_npz = os.path.join(out_logs_dir, "logits_wise_ft_v2_q30.npz")
    split_calibration_and_eval_logits(wise_clean_npz, calib_ratio=calib_ratio, seed=seed)
    split_calibration_and_eval_logits(wise_q30_npz, calib_ratio=calib_ratio, seed=seed)

    # -------------------------------------------------------------------------
    # BƯỚC 6: Re-baseline Ma Trận 4 Cấu Hình từ Số Đo Thật (Task 7.6)
    # Khóa chặt chỉ đo đạc trên phân vùng eval_split (1.600 mẫu) để Zero Leakage
    # -------------------------------------------------------------------------
    rebase_json = cfg["rebaseline"]["output_json"]
    rebase_md = cfg["rebaseline"]["output_md"]

    execute_pipeline_step(
        [
            sys.executable, "tools/rebaseline_cfe_matrix.py",
            "--calibration_baseline", os.path.join(calib_out_dir, "baseline.json"),
            "--calibration_anchored", os.path.join(calib_out_dir, "anchored.json"),
            "--measured_dir", out_logs_dir,
            "--use_eval_split",
            "--must_fake_at_real90_q30", str(cfg["gate_g7_1"]["must"]["fake_at_real90_q30"]),
            "--must_auc_clean", str(cfg["gate_g7_1"]["must"]["auc_clean"]),
            "--must_auc_q30", str(cfg["gate_g7_1"]["must"]["auc_q30"]),
            "--stretch_fake_at_real90_q30", str(cfg["gate_g7_1"]["stretch"]["fake_at_real90_q30"]),
            "--seed", str(seed),
            "--output_json", rebase_json,
            "--output_md", rebase_md,
        ],
        "Tái lập ma trận sàn 4 cấu hình x 2 điều kiện trên tập eval_split (Zero Data Leakage)"
    )

    # -------------------------------------------------------------------------
    # BƯỚC 7: Validator Kiểm Tra Nghiệm Thu Cổng G7.1 (Fail-Loud)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("FATFORMER-XLA: TIẾN TRÌNH THẨM ĐỊNH CHẤT LƯỢNG CỔNG G7.1")
    print("=" * 88)

    with open(rebase_json, "r", encoding="utf-8") as f:
        matrix_data = json.load(f)

    if matrix_data.get("evidence_level") != "measured":
        raise ValueError("[G7.1 FAIL] Ma trận kết quả không có evidence_level=measured!")

    res = matrix_data["results"]
    best_res = res["wise_ft_v2"]

    far90_q30 = best_res["q30"]["fake_at_real90"]
    auc_clean = best_res["clean"]["auc"]
    auc_q30 = best_res["q30"]["auc"]

    must_req = cfg["gate_g7_1"]["must"]
    stretch_req = cfg["gate_g7_1"]["stretch"]

    pass_far90 = far90_q30 >= must_req["fake_at_real90_q30"]
    pass_auc_clean = auc_clean >= must_req["auc_clean"]
    pass_auc_q30 = auc_q30 >= must_req["auc_q30"]
    pass_stretch = far90_q30 >= stretch_req["fake_at_real90_q30"]

    all_must_pass = pass_far90 and pass_auc_clean and pass_auc_q30
    verdict = "PASS" if all_must_pass else "FAIL"

    print(f"• Fake@Real90 (JPEG Q=30) : {far90_q30:.2f}% (Yêu cầu Must: >={must_req['fake_at_real90_q30']}%, Stretch: >={stretch_req['fake_at_real90_q30']}%) -> {'PASS' if pass_far90 else 'FAIL'}")
    print(f"• ROC-AUC (Clean CFE)    : {auc_clean:.2f}% (Yêu cầu Must: >={must_req['auc_clean']}%) -> {'PASS' if pass_auc_clean else 'FAIL'}")
    print(f"• ROC-AUC (JPEG Q=30)    : {auc_q30:.2f}% (Yêu cầu Must: >={must_req['auc_q30']}%) -> {'PASS' if pass_auc_q30 else 'FAIL'}")
    print(f"• Đạt mức Stretch        : {'CÓ' if pass_stretch else 'CHƯA'}")
    print("-" * 88)
    print(f"🏆 PHÁN QUYẾT CHUNG CUỘC CỔNG G7.1: [{verdict}]")
    print("=" * 88 + "\n")

    if verdict != "PASS":
        print("[!] DỪNG KHẨN CẤP: Cổng G7.1 KHÔNG ĐẠT tiêu chuẩn Must!")
        raise RuntimeError(
            f"[G7.1 FAIL] Nghiệm thu thất bại! Cổng G7.1 KHÔNG ĐẠT tiêu chuẩn Must:\n"
            f"  • Fake@Real90 (JPEG Q=30): {far90_q30:.2f}% (Yêu cầu Must: >= {must_req['fake_at_real90_q30']}%)\n"
            f"  • ROC-AUC (Clean CFE)   : {auc_clean:.2f}% (Yêu cầu Must: >= {must_req['auc_clean']}%)\n"
            f"  • ROC-AUC (JPEG Q=30)   : {auc_q30:.2f}% (Yêu cầu Must: >= {must_req['auc_q30']}%)\n"
            f"  Pipeline dừng ngay lập tức với exit code lỗi; không cấp chứng nhận nhầm."
        )
    else:
        print("[✓] CHÚC MỪNG: Cổng G7.1 chính thức ĐẠT CHUẨN bằng số đo thực tế trên GPU!")


def main():
    parser = argparse.ArgumentParser(description="FatFormer-XLA: Phase 7 Full GPU Acceptance Runner")
    parser.add_argument("--config", type=str, default="configs/phase7_gpu_acceptance.yaml",
                        help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--allow_cpu", action="store_true",
                        help="Cho phép chạy debug trên CPU (chỉ dùng test luồng logic)")
    parser.add_argument("--allow_dirty", action="store_true",
                        help="Cho phép chạy khi Git working tree dirty")
    parser.add_argument("--no_resume", dest="resume", action="store_false",
                        help="Bắt buộc chạy lại toàn bộ từ đầu, không tái sử dụng kết quả đã đo")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    cfg_file = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    if not os.path.isfile(cfg_file):
        print(f"[!] Không tìm thấy file cấu hình tại: {cfg_file}")
        sys.exit(1)

    try:
        run_phase7_acceptance(cfg_file, allow_cpu=args.allow_cpu, allow_dirty=args.allow_dirty, resume=args.resume)
    except Exception as e:
        print(f"\n[!] DỪNG KHẨN CẤP DO LỖI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
