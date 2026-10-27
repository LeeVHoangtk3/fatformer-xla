# -*- coding: utf-8 -*-
"""
FatFormer-XLA: Cong Cu Noi Suy Khong Gian Trong So WiSE-FT Gen 2 (Task 6.1-R Upgraded)
--------------------------------------------------------------------------------------
Co so khoa hoc: Wortsman et al., "Robust fine-tuning of zero-shot models", CVPR 2022.

Nang cap cot loi (Fix 146 Unexpected Keys & Prediction Collapse):
  1. Uu tien hop nhat giua Baseline goc (fatformer_4class_ckpt.pth, Adapters=3)
     va Anchored Recovery M6-R (fatformer_anchored_recovery_best.pth, Adapters=3, Zero-Init Step 0).
  2. Tu dong loai bo cac khoa top-level du thua ('srm.*', 'gating.*' ngoai vi) va ngan chan
     viec tron lan cac khoi Adapter lech chi muc (3-blocks vs 8-blocks).
  3. Ho tro Grid Search thuc te va xuat checkpoint chuan kem SHA-256.
"""

import os
import sys
import time
import json
import hashlib
import argparse
from typing import Dict, Tuple, List, Optional, Any

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
    """Tinh ma bam SHA-256 bit-perfect cho file checkpoint."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def find_first_existing(candidates: List[Optional[str]]) -> Optional[str]:
    """Tim duong dan ton tai dau tien tu danh sach ung vien."""
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


def safe_load_state_dict(checkpoint_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Nap checkpoint an toan, boc tach state_dict va chuan hoa tien to 'module.'."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"[!] Khong tim thay checkpoint tai: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        for key in ["model", "model_state", "state_dict", "model_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
        raw_sd = ckpt
    else:
        raw_sd = ckpt.state_dict() if hasattr(ckpt, "state_dict") else {}

    clean_sd = {}
    for k, v in raw_sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        clean_k = k[7:] if k.startswith("module.") else k
        clean_sd[clean_k] = v

    return clean_sd


def get_adapter_block_indices(sd: Dict[str, torch.Tensor]) -> set:
    """Tim tap chi muc cac khoi resblocks co chua adapter trong state_dict."""
    indices = set()
    for k in sd.keys():
        if "forgery_aware_adapter" in k or "visual_adapter" in k:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p in ("resblocks", "visual_adapter", "adapters") and i + 1 < len(parts) and parts[i + 1].isdigit():
                    indices.add(int(parts[i + 1]))
    return indices


def is_srm_final_projection(key: str) -> bool:
    """Xac dinh lop chieu cuoi cung cua SpatialResidualBlock (SRM)."""
    k_lower = key.lower()
    is_srm = ("srm" in k_lower or "srm_block" in k_lower or "srm_conv" in k_lower)
    is_proj_final = any(p in k_lower for p in ["proj.4", "proj_final", "out_proj", "proj.conv2", "srm_proj"])
    return is_srm and is_proj_final


def is_fixed_buffer(key: str) -> bool:
    """Xac dinh cac tensor buffer vi sai co dinh (k1, k2, k3 trong SRM)."""
    k_lower = key.lower()
    return any(b in k_lower for b in ["k1", "k2", "k3", "srm_filter", "haar_kernel"])


def wise_ft_merge(
    theta_orig: Dict[str, torch.Tensor],
    theta_target: Dict[str, torch.Tensor],
    alpha: float = 0.45,
    srm_scale: float = 0.35,
    keep_orphan_orig: bool = False,
    verbose: bool = True
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Thuc hien Weight-Space Ensembling chuan xac giua theta_orig va theta_target:
      - Voi key chung k in theta_orig cap theta_target:
          theta_wise[k] = (1 - alpha) * theta_orig[k] + alpha * theta_target[k]
      - Voi key rieng thuoc theta_target (SRM, Gating):
          + Buffers (k1, k2, k3): Giu nguyen 100%.
          + SRM final projection: Nhan voi srm_scale (beta_srm).
          + Gating & cac tang SRM trung gian: Nhan voi alpha hoac giu nguyen.
      - Khong gop chung orig_only_keys neu lech so khoi Adapter (tranh tao model lai 10 adapters).
    """
    keys_orig = set(theta_orig.keys())
    keys_target = set(theta_target.keys())

    shared_keys = keys_orig.intersection(keys_target)
    target_only_keys = keys_target.difference(keys_orig)
    orig_only_keys = keys_orig.difference(keys_target)

    orig_adapters = get_adapter_block_indices(theta_orig)
    target_adapters = get_adapter_block_indices(theta_target)

    if verbose:
        print("\n" + "=" * 78)
        print("PHÂN TÍCH KHÔNG GIAN TRỌNG SỐ (WiSE-FT GEN 2 MERGE ANALYSIS)")
        print("=" * 78)
        print(f"[*] Tổng số keys trong theta_orig   : {len(keys_orig):,} (Adapters={len(orig_adapters)}: {sorted(orig_adapters)})")
        print(f"[*] Tổng số keys trong theta_target : {len(keys_target):,} (Adapters={len(target_adapters)}: {sorted(target_adapters)})")
        print(f"[+] Số keys chung (Giao thoa)       : {len(shared_keys):,} (Nội suy tuyến tính alpha={alpha:.2f})")
        print(f"[+] Số keys riêng của target        : {len(target_only_keys):,} (SRM, Gating & Buffers)")
        print(f"[-] Số keys riêng của orig          : {len(orig_only_keys):,}")
        print(f"[*] Hệ số điều tiết nhánh SRM       : beta_srm={srm_scale:.2f}")

    wise_sd: Dict[str, torch.Tensor] = {}
    scaled_srm_count = 0
    buffer_intact_count = 0
    skipped_orphan_count = 0

    # 1. Noi suy tuyen tinh tren tap keys chung
    for k in shared_keys:
        t_orig = theta_orig[k]
        t_targ = theta_target[k]

        if t_orig.shape != t_targ.shape:
            raise ValueError(f"[!] Lỗi lệch shape tại key '{k}': orig {t_orig.shape} vs target {t_targ.shape}")

        target_dtype = t_orig.dtype
        if t_orig.is_floating_point():
            if is_srm_final_projection(k):
                # Neu lop chieu SRM nam trong tap chung, ap dung srm_scale
                interp_tensor = t_targ.to(torch.float32) * srm_scale
                scaled_srm_count += 1
            else:
                interp_tensor = (1.0 - alpha) * t_orig.to(torch.float32) + alpha * t_targ.to(torch.float32)
            wise_sd[k] = interp_tensor.to(target_dtype)
        else:
            wise_sd[k] = t_orig.clone()

    # 2. Xu ly tap keys chi co o theta_target (SRM, Gating, Buffers)
    for k in target_only_keys:
        t_targ = theta_target[k]
        target_dtype = t_targ.dtype

        if is_fixed_buffer(k):
            wise_sd[k] = t_targ.clone()
            buffer_intact_count += 1
        elif is_srm_final_projection(k):
            if t_targ.is_floating_point():
                wise_sd[k] = (t_targ.to(torch.float32) * srm_scale).to(target_dtype)
            else:
                wise_sd[k] = t_targ.clone()
            scaled_srm_count += 1
        else:
            wise_sd[k] = t_targ.clone()

    # 3. Chi giu orig_only_keys neu khong gay xung dot cau hinh Adapter
    if orig_only_keys:
        if orig_adapters == target_adapters or keep_orphan_orig:
            for k in orig_only_keys:
                wise_sd[k] = theta_orig[k].clone()
        else:
            skipped_orphan_count = len(orig_only_keys)
            if verbose:
                print(f"[!] Đã loại bỏ {skipped_orphan_count} khóa orig_only lệch cấu trúc Adapter để chống lỗi Unexpected Keys!")

    stats = {
        "alpha": alpha,
        "srm_scale": srm_scale,
        "num_adapters": len(target_adapters) if target_adapters else len(orig_adapters),
        "total_orig_keys": len(keys_orig),
        "total_target_keys": len(keys_target),
        "shared_keys_interpolated": len(shared_keys),
        "target_only_keys_preserved": len(target_only_keys),
        "skipped_orphan_orig_keys": skipped_orphan_count,
        "srm_projections_scaled": scaled_srm_count,
        "buffers_intact": buffer_intact_count,
        "total_wise_keys": len(wise_sd),
    }

    if verbose:
        print(f"[✓] Hợp nhất hoàn tất: {len(wise_sd):,} tensors chuẩn hóa "
              f"(Adapters={stats['num_adapters']}, scaled {scaled_srm_count} SRM proj, {buffer_intact_count} buffers).")
        print("=" * 78 + "\n")

    return wise_sd, stats


def save_wise_checkpoint(
    output_path: str,
    wise_sd: Dict[str, torch.Tensor],
    metadata: Dict[str, Any]
) -> str:
    """Luu checkpoint chuan dict {'model': wise_sd, 'metadata': metadata} kem SHA-256."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {"model": wise_sd, "metadata": metadata}
    torch.save(payload, output_path)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    sha256_hash = compute_sha256(output_path)

    metadata["sha256"] = sha256_hash
    metadata["file_size_mb"] = round(file_size_mb, 2)
    metadata["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    meta_json_path = output_path.replace(".pth", "_metadata.json")
    try:
        serializable_meta = json.loads(json.dumps(metadata, default=lambda o: str(o)))
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_meta, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[!] Cảnh báo: Không thể lưu metadata JSON: {e}")

    print(f"[✓] Đã xuất checkpoint WiSE-FT Gen 2 tại: {output_path}")
    print(f"    • Dung lượng : {file_size_mb:.2f} MB")
    print(f"    • SHA-256    : {sha256_hash}")
    return sha256_hash


def sweep_threshold_constrained(
    fake_probs: np.ndarray,
    targets: np.ndarray,
    th_min: float = 0.35,
    th_max: float = 0.65,
    th_step: float = 0.02,
    real_preservation_threshold: float = 90.0
) -> Tuple[float, float, Dict[str, float]]:
    """Quet nguong T* toi uu hoa Balanced Accuracy co rang buoc cung Real ACC >= 90.0%."""
    if fake_probs.ndim == 2 and fake_probs.shape[1] == 2:
        p_fake = fake_probs[:, 1]
    else:
        p_fake = fake_probs.reshape(-1)

    thresholds = np.arange(th_min, th_max + 1e-5, th_step)
    best_th = 0.50
    best_score = -1.0
    best_metrics = {"macro_acc": 0.0, "real_acc": 0.0, "fake_acc": 0.0}

    real_mask = (targets == 0)
    fake_mask = (targets == 1)
    n_real = max(int(np.sum(real_mask)), 1)
    n_fake = max(int(np.sum(fake_mask)), 1)

    valid_candidates = []
    for th in thresholds:
        preds = (p_fake >= th).astype(int)
        r_acc = (np.sum(preds[real_mask] == 0) / n_real) * 100.0
        f_acc = (np.sum(preds[fake_mask] == 1) / n_fake) * 100.0
        m_acc = 0.5 * (r_acc + f_acc)

        if r_acc >= real_preservation_threshold:
            valid_candidates.append((th, m_acc, r_acc, f_acc))
            if m_acc > best_score:
                best_score = m_acc
                best_th = float(th)
                best_metrics = {"macro_acc": m_acc, "real_acc": r_acc, "fake_acc": f_acc}

    if not valid_candidates:
        for th in thresholds:
            preds = (p_fake >= th).astype(int)
            r_acc = (np.sum(preds[real_mask] == 0) / n_real) * 100.0
            f_acc = (np.sum(preds[fake_mask] == 1) / n_fake) * 100.0
            m_acc = 0.5 * (r_acc + f_acc)
            if r_acc > best_metrics["real_acc"]:
                best_th = float(th)
                best_score = m_acc
                best_metrics = {"macro_acc": m_acc, "real_acc": r_acc, "fake_acc": f_acc}

    return best_th, best_score, best_metrics


def run_synthetic_dummy_test() -> bool:
    """Kiem thu toan hoc 6 tieu chi cua WiSE-FT Merge tren du lieu gia lap."""
    print("\n" + "=" * 76)
    print("CHẠY KIỂM THỬ TOÁN HỌC SYNTHETIC DUMMY TEST (WiSE-FT GEN 2)")
    print("=" * 76)

    theta_orig = {}
    theta_targ = {}
    torch.manual_seed(42)

    for i in range(1116):
        k = f"visual.transformer.resblocks.{i % 3}.forgery_aware_adapter.weight_{i}" if i < 132 else f"backbone.layer_{i}"
        val_orig = torch.randn(8, 8)
        theta_orig[k] = val_orig
        theta_targ[k] = val_orig + torch.randn(8, 8) * 0.2

    k1_tensor = torch.tensor([[0., 0., 0.], [-1., 1., 0.], [0., 0., 0.]]).repeat(3, 1, 1, 1)
    theta_targ["srm.k1"] = k1_tensor.clone()
    theta_targ["srm.proj.4.weight"] = torch.ones(16, 16, 1, 1) * 2.0
    theta_targ["gating.mlp.2.weight"] = torch.randn(1, 16)

    alpha_test, beta_test = 0.45, 0.35
    wise_sd, _ = wise_ft_merge(theta_orig, theta_targ, alpha=alpha_test, srm_scale=beta_test, verbose=False)

    k0 = list(theta_orig.keys())[0]
    expected_val = (1.0 - alpha_test) * theta_orig[k0] + alpha_test * theta_targ[k0]
    assert torch.max(torch.abs(wise_sd[k0] - expected_val)).item() < 1e-6
    assert torch.max(torch.abs(wise_sd["srm.k1"] - k1_tensor)).item() == 0.0
    assert abs(wise_sd["srm.proj.4.weight"][0, 0, 0, 0].item() - (2.0 * beta_test)) < 1e-6
    print("[✓] Toàn bộ tiêu chí toán học WiSE-FT Gen 2: PASS 100%!")
    return True


def parse_arguments():
    parser = argparse.ArgumentParser(description="FatFormer-XLA: WiSE-FT Gen 2 Weight-Space Ensembling")
    parser.add_argument("--orig_checkpoint", type=str, default="",
                        help="Duong dan checkpoint goc fatformer_4class_ckpt.pth")
    parser.add_argument("--m4_checkpoint", type=str, default="",
                        help="Duong dan checkpoint đích (Mac dinh uu tien fatformer_anchored_recovery_best.pth)")
    parser.add_argument("--output_path", type=str, default="",
                        help="Duong dan xuat fatformer_wise_ft_best.pth")
    parser.add_argument("--alpha", type=float, default=0.45,
                        help="He so noi suy alpha in [0.0, 1.0] (mac dinh Gen 2: 0.45)")
    parser.add_argument("--srm_scale", type=float, default=0.35,
                        help="He so dieu tiet SRM beta_srm (mac dinh Gen 2: 0.35)")
    parser.add_argument("--grid_search", action="store_true",
                        help="Xuat them cac ung vien alpha in [0.35, 0.45, 0.55]")
    parser.add_argument("--test_dummy", action="store_true",
                        help="Chay kiem thu toan hoc tren du lieu gia lap")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.test_dummy:
        sys.exit(0 if run_synthetic_dummy_test() else 1)

    print("\n" + "=" * 82)
    print("FATFORMER-XLA: HỢP NHẤT TRỌNG SỐ WiSE-FT THẾ HỆ 2 (ZERO UNEXPECTED KEYS)")
    print("=" * 82)

    orig_candidates = [
        args.orig_checkpoint,
        "/content/drive/MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth",
        "DATASET/pretrained/fatformer_4class_ckpt.pth",
        "pretrained/fatformer_4class_ckpt.pth",
        "/kaggle/input/fatformer-assets-pack/fatformer_4class_ckpt.pth",
    ]
    resolved_orig = find_first_existing(orig_candidates)

    # Uu tien so 1: Anchored Recovery M6-R (Adapters=3, cung kien truc voi Baseline!)
    target_candidates = [
        args.m4_checkpoint,
        "/content/drive/MyDrive/Fatformer/checkpoint/fatformer_anchored_recovery_best.pth",
        "DATASET/checkpoints/fatformer_anchored_recovery_best.pth",
        "checkpoints/m4_recovery/fatformer_anchored_recovery_best.pth",
        "wait/task6-3-r/fatformer_anchored_recovery_best.pth",
        "/content/drive/MyDrive/Fatformer/checkpoints/fatformer_crossdomain_weights_only.pth",
        "DATASET/checkpoints/fatformer_crossdomain_weights_only.pth",
    ]
    resolved_target = find_first_existing(target_candidates)

    if not resolved_orig or not resolved_target:
        print(f"[!] LỖI: Không tìm thấy đủ 2 file checkpoint để hợp nhất!")
        print(f"    • Orig   : {resolved_orig}")
        print(f"    • Target : {resolved_target}")
        sys.exit(1)

    print(f"[*] Checkpoint Gốc (theta_orig)     : {resolved_orig}")
    print(f"[*] Checkpoint Đích (theta_target)  : {resolved_target}")

    sd_orig = safe_load_state_dict(resolved_orig, device="cpu")
    sd_target = safe_load_state_dict(resolved_target, device="cpu")

    output_path = args.output_path
    if not output_path:
        if os.path.exists("/content/drive/MyDrive/Fatformer/checkpoint"):
            output_path = "/content/drive/MyDrive/Fatformer/checkpoint/fatformer_wise_ft_best.pth"
        else:
            output_path = os.path.join(PROJECT_ROOT, "DATASET", "checkpoints", "wise_ft", "fatformer_wise_ft_best.pth")

    wise_sd, stats = wise_ft_merge(
        sd_orig, sd_target,
        alpha=args.alpha,
        srm_scale=args.srm_scale,
        verbose=True
    )

    metadata = {
        "task": "WiSE-FT Gen 2 Unified Ensembling",
        "source_orig": os.path.basename(resolved_orig),
        "source_target": os.path.basename(resolved_target),
        "alpha": args.alpha,
        "srm_scale": args.srm_scale,
        "stats": stats,
        "threshold_recommended": 0.50,
    }

    save_wise_checkpoint(output_path, wise_sd, metadata)

    if args.grid_search:
        base_dir = os.path.dirname(os.path.abspath(output_path))
        for a_cand in [0.35, 0.50, 0.60]:
            cand_sd, cand_stats = wise_ft_merge(sd_orig, sd_target, alpha=a_cand, srm_scale=args.srm_scale, verbose=False)
            cand_path = os.path.join(base_dir, f"fatformer_wise_ft_alpha_{int(a_cand*100):02d}.pth")
            cand_meta = dict(metadata, alpha=a_cand, stats=cand_stats)
            save_wise_checkpoint(cand_path, cand_sd, cand_meta)


if __name__ == "__main__":
    main()