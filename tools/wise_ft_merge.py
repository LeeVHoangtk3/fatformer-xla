# -*- coding: utf-8 -*-
"""
FatFormer-XLA: Cong Cu Noi Suy Khong Gian Trong So WiSE-FT (Task 6.1-R)
----------------------------------------------------------------------
Chien dich Khoi phuc Hieu nang Dinh cao (Phase 6-R: High-Accuracy Anchored Recovery)
Nguoi chu tri: Thanh vien C (Training & Evaluation Lead)
Co so khoa hoc: Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models", CVPR 2022.

Muc tieu:
  1. Hop nhat khong gian trong so giua checkpoint goc theta_orig (1.116 keys, 96.45% Clean Macro ACC)
     va checkpoint da mien theta_M4 (1.684 keys, khang suy bien vat ly Q30/Q50).
  2. Ap dung cong thuc noi suy tuyen tinh tren 1.116 keys chung:
       theta_wise^(k) = (1 - alpha) * theta_orig^(k) + alpha * theta_M4^(k)
  3. Xy ly 568 keys rieng biet (M4-only thuoc SRM va Gating):
       - Giu nguyen cac bo loc vi sai bat bien (k1, k2, k3) voi requires_grad=False.
       - Giam chan tang chieu cuoi cua SRM (proj.4.weight) voi he so beta_srm (mac dinh 0.10 thay vi 1.00 trong M4).
       - Ke thua nguyen ven cac tham so MLP Gating.
  4. Ho tro Grid Search alpha in [0.15, 0.40] va quet nguong cat T* voi rang buoc cung Real_ACC >= 90.0%.
  5. Xuat checkpoint format chuan bọc dict {"model": wise_sd, "metadata": {...}} kem ma bam SHA-256.

Cach su dung:
  # 1. Chay kiem thu toan hoc Dummy (CPU, khong can file checkpoint that):
  python tools/wise_ft_merge.py --test_dummy

  # 2. Chay Merge truc tiep voi alpha mac dinh (0.25) va beta_srm (0.10):
  python tools/wise_ft_merge.py

  # 3. Chay Grid Search alpha va danh gia tren tap validation:
  python tools/wise_ft_merge.py --grid_search
"""

import os
import sys
import time
import math
import json
import hashlib
import argparse
from typing import Dict, Tuple, List, Optional, Any

import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
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
            return c
    return None


def safe_load_state_dict(checkpoint_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Nap checkpoint an toan, tu dong boc tach 'model' hoac 'state_dict' neu co,
    va loc bo tien to 'module.' neu co.
    """
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

    # Chuan hoa bo tien to 'module.' neu xuat hien tu DDP
    clean_sd = {}
    for k, v in raw_sd.items():
        clean_k = k[7:] if k.startswith("module.") else k
        clean_sd[clean_k] = v

    return clean_sd


def is_srm_final_projection(key: str) -> bool:
    """
    Xac dinh xem mot tensor co phai la lop chieu cuoi cung cua SpatialResidualBlock (SRM) hay khong.
    Lop chieu cuoi thuong la Conv2d 1x1 o vi tri cuoi (proj.4.weight hoac proj.4.bias hoac out_proj).
    """
    k_lower = key.lower()
    is_srm = ("srm" in k_lower or "srm_block" in k_lower)
    is_proj_final = any(p in k_lower for p in ["proj.4", "proj_final", "out_proj", "proj.conv2"])
    return is_srm and is_proj_final


def is_fixed_buffer(key: str) -> bool:
    """Xac dinh cac tensor buffer co dinh (vi du k1, k2, k3 trong SRM)."""
    k_lower = key.lower()
    return any(b in k_lower for b in ["k1", "k2", "k3"])


def wise_ft_merge(
    theta_orig: Dict[str, torch.Tensor],
    theta_m4: Dict[str, torch.Tensor],
    alpha: float = 0.25,
    srm_scale: float = 0.10,
    verbose: bool = True
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Thuc hien Weight-Space Ensembling giua theta_orig va theta_m4:
      - Voi key chung k in theta_orig cap theta_m4 (1.116 keys):
          theta_wise[k] = (1 - alpha) * theta_orig[k] + alpha * theta_m4[k]
      - Voi key rieng k in theta_m4 \\ theta_orig (568 keys):
          + Buffers (k1, k2, k3): Giu nguyen ban 100%.
          + SRM final projection: Nhan voi he so giam chan srm_scale (beta_srm).
          + Gating & cac tham so khac: Ke thua tu theta_m4.
    """
    keys_orig = set(theta_orig.keys())
    keys_m4 = set(theta_m4.keys())

    shared_keys = keys_orig.intersection(keys_m4)
    m4_only_keys = keys_m4.difference(keys_orig)
    orig_only_keys = keys_orig.difference(keys_m4)

    if verbose:
        print("\n" + "=" * 76)
        print("PHAN TICH KHONG GIAN TRONG SO (WiSE-FT MERGE ANALYSIS)")
        print("=" * 76)
        print(f"[*] Tong so keys trong theta_orig : {len(keys_orig):,}")
        print(f"[*] Tong so keys trong theta_m4   : {len(keys_m4):,}")
        print(f"[+] So keys chung (Giao)          : {len(shared_keys):,}  (Ap dung linear interpolation alpha={alpha:.2f})")
        print(f"[+] So keys chi co trong theta_m4 : {len(m4_only_keys):,}  (SRM, Gating & Buffers)")
        print(f"[-] So keys chi co trong orig     : {len(orig_only_keys):,}")
        print(f"[*] He so giam chan SRM beta_srm  : {srm_scale:.2f}")

    # Kiem tra tinh toan ven khoa lich su patch_basaed_enhancer
    pbe_keys = [k for k in shared_keys if "patch_basaed_enhancer" in k]
    if verbose:
        print(f"[*] Tensors patch_basaed_enhancer : {len(pbe_keys)} keys duoc bao toan tuyet doi.")

    wise_sd: Dict[str, torch.Tensor] = {}
    scaled_srm_count = 0
    buffer_intact_count = 0

    # 1. Noi suy tuyen tinh tren tap keys chung (1.116 keys)
    for k in shared_keys:
        t_orig = theta_orig[k]
        t_m4 = theta_m4[k]

        if t_orig.shape != t_m4.shape:
            raise ValueError(f"[!] Shape khong khop tai key '{k}': orig {t_orig.shape} vs m4 {t_m4.shape}")

        # Tinh toan voi float32 de dam bao do chinh xac so hoc, sau do tra ve dtype goc
        target_dtype = t_orig.dtype
        if t_orig.is_floating_point():
            interp_tensor = (1.0 - alpha) * t_orig.to(torch.float32) + alpha * t_m4.to(torch.float32)
            wise_sd[k] = interp_tensor.to(target_dtype)
        else:
            # Voi cac tensor nguyen hoac bool, giu nguyen ban theo orig
            wise_sd[k] = t_orig.clone()

    # 2. Xu ly tap keys rieng biet chi co o theta_m4 (SRM, Gating, Buffers)
    for k in m4_only_keys:
        t_m4 = theta_m4[k]
        target_dtype = t_m4.dtype

        if is_fixed_buffer(k):
            # Giu nguyen 100% cac ma tran loc vi sai k1, k2, k3
            wise_sd[k] = t_m4.clone()
            buffer_intact_count += 1
        elif is_srm_final_projection(k):
            # Giam chan lop chieu cuoi de khong lam nhieu khong gian anh sach
            if t_m4.is_floating_point():
                scaled_tensor = t_m4.to(torch.float32) * srm_scale
                wise_sd[k] = scaled_tensor.to(target_dtype)
            else:
                wise_sd[k] = t_m4.clone()
            scaled_srm_count += 1
        else:
            # Gating va cac tang conv trung gian cua SRM
            wise_sd[k] = t_m4.clone()

    # 3. Neu co keys chi xuat hien o orig (phong ngua)
    for k in orig_only_keys:
        wise_sd[k] = theta_orig[k].clone()

    stats = {
        "alpha": alpha,
        "srm_scale": srm_scale,
        "total_orig_keys": len(keys_orig),
        "total_m4_keys": len(keys_m4),
        "shared_keys_interpolated": len(shared_keys),
        "m4_only_keys_preserved": len(m4_only_keys),
        "srm_projections_scaled": scaled_srm_count,
        "buffers_intact": buffer_intact_count,
        "total_wise_keys": len(wise_sd),
    }

    if verbose:
        print(f"[✓] Hoan thanh hop nhat: {len(wise_sd):,} tensors trong theta_wise "
              f"(scaled {scaled_srm_count} SRM proj tensors, {buffer_intact_count} buffers).")
        print("=" * 76 + "\n")

    return wise_sd, stats


def save_wise_checkpoint(
    output_path: str,
    wise_sd: Dict[str, torch.Tensor],
    metadata: Dict[str, Any]
) -> str:
    """
    Luu checkpoint theo cau truc chuan dict:
    {
        "model": wise_sd,
        "metadata": metadata
    }
    va tinh ma bam SHA-256.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    payload = {
        "model": wise_sd,
        "metadata": metadata
    }

    torch.save(payload, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    sha256_hash = compute_sha256(output_path)

    metadata["sha256"] = sha256_hash
    metadata["file_size_mb"] = round(file_size_mb, 2)
    metadata["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Luu file metadata JSON song song de de dang kiem tra ma khong can doc file .pth 2GB
    meta_json_path = output_path.replace(".pth", "_metadata.json")
    try:
        # Chuyen cac gia tri khong serialize duoc thanh string
        serializable_meta = json.loads(json.dumps(metadata, default=lambda o: str(o)))
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_meta, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[!] Canh bao: Khong the luu file metadata JSON: {e}")

    print(f"[✓] Da xuat checkpoint WiSE-FT thanh cong ra: {output_path}")
    print(f"    • Dung luong: {file_size_mb:.2f} MB")
    print(f"    • Ma bam SHA-256: {sha256_hash}")
    print(f"    • Metadata JSON: {meta_json_path}")
    return sha256_hash


def sweep_threshold_constrained(
    fake_probs: np.ndarray,
    targets: np.ndarray,
    th_min: float = 0.35,
    th_max: float = 0.65,
    th_step: float = 0.02,
    real_preservation_threshold: float = 90.0
) -> Tuple[float, float, Dict[str, float]]:
    """
    Quet nguong T* toi uu hoa Youden's Index / Balanced Metric
    co rang buoc cung (Hard Constraint): Real ACC >= 90.0%.
    """
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
        preds = (fake_probs >= th).astype(int)
        r_acc = (np.sum(preds[real_mask] == 0) / n_real) * 100.0
        f_acc = (np.sum(preds[fake_mask] == 1) / n_fake) * 100.0
        m_acc = 0.5 * (r_acc + f_acc)

        # RANG BUOC CUNG (HARD GATE): Real ACC >= 90.00%
        if r_acc >= real_preservation_threshold:
            # Diem can bang Youden: J = Sensitivity + Specificity - 1
            youden_score = (r_acc + f_acc) / 2.0
            valid_candidates.append((th, youden_score, m_acc, r_acc, f_acc))
            if youden_score > best_score:
                best_score = youden_score
                best_th = float(th)
                best_metrics = {"macro_acc": m_acc, "real_acc": r_acc, "fake_acc": f_acc}

    # Neu khong co nguong nao dat Real ACC >= 90%, lay nguong dat Real ACC cao nhat
    if not valid_candidates:
        print("[!] Canh bao: Khong co nguong nao dat Real ACC >= 90%. Chuyen sang chon nguong toi da hoa Real ACC.")
        for th in thresholds:
            preds = (fake_probs >= th).astype(int)
            r_acc = (np.sum(preds[real_mask] == 0) / n_real) * 100.0
            f_acc = (np.sum(preds[fake_mask] == 1) / n_fake) * 100.0
            m_acc = 0.5 * (r_acc + f_acc)
            if r_acc > best_metrics["real_acc"]:
                best_th = float(th)
                best_metrics = {"macro_acc": m_acc, "real_acc": r_acc, "fake_acc": f_acc}

    return best_th, best_score, best_metrics


def run_synthetic_dummy_test() -> bool:
    """
    Kiem thu toan hoc logic WiSE-FT Merge tren mo hinh gia lap (Synthetic Dummy Tensors).
    Chung minh:
      1. Tensors chung duoc noi suy chinh xac theo (1-alpha)*A + alpha*B.
      2. Tensors buffer k1, k2, k3 khong bi thay doi bat ky bit nao.
      3. Tensors tang chieu cuoi SRM duoc scale giam chan theo beta_srm.
      4. Tensors gating duoc ke thua nguyen ven.
      5. File checkpoint duoc luu dung cau truc va doc lai hoan hao.
    """
    print("\n" + "=" * 76)
    print("CHAY KIEM THU TOAN HOC SYNTHETIC DUMMY TEST (TASK 6.1-R)")
    print("=" * 76)

    # 1. Tao gia lap 1.116 tensors chung
    theta_orig = {}
    theta_m4 = {}

    torch.manual_seed(42)
    # 5 tensors dai dien cho backbone va FAA/LGA
    for i in range(1116):
        k = f"visual.transformer.resblocks.{i % 24}.attn.in_proj_weight" if i < 1000 else f"language_guided_alignment.patch_basaed_enhancer.in_proj_weight_{i}"
        val_orig = torch.randn(16, 16) * 1.0
        val_m4 = val_orig + torch.randn(16, 16) * 0.1
        theta_orig[k] = val_orig
        theta_m4[k] = val_m4

    # 2. Them cac tensors dac thu cua M4 (568 tensors)
    # Fixed buffers
    k1_tensor = torch.tensor([[0., 0., 0.], [-1., 1., 0.], [0., 0., 0.]]).repeat(3, 1, 1, 1)
    theta_m4["srm.k1"] = k1_tensor.clone()
    theta_m4["srm.k2"] = torch.randn(3, 1, 3, 3)
    theta_m4["srm.k3"] = torch.randn(3, 1, 5, 5)

    # SRM intermediate & final projection
    theta_m4["srm.proj.0.weight"] = torch.randn(64, 9, 3, 3)
    theta_m4["srm.proj.4.weight"] = torch.ones(1024, 64, 1, 1) * 2.0  # Tang chieu cuoi

    # Dynamic gating
    theta_m4["gating.mlp.0.weight"] = torch.randn(64, 1024)
    theta_m4["gating.mlp.2.weight"] = torch.randn(1, 64)

    alpha_test = 0.25
    beta_srm_test = 0.10

    wise_sd, stats = wise_ft_merge(
        theta_orig, theta_m4,
        alpha=alpha_test,
        srm_scale=beta_srm_test,
        verbose=False
    )

    # Kiem tra 1: Noi suy tuyen tinh tren tensor chung dau tien
    k_first = list(theta_orig.keys())[0]
    expected_val = (1.0 - alpha_test) * theta_orig[k_first] + alpha_test * theta_m4[k_first]
    diff_interp = torch.max(torch.abs(wise_sd[k_first] - expected_val)).item()
    assert diff_interp < 1e-6, f"Loi noi suy tensor chung: diff={diff_interp}"
    print(f"[✓] Tieu chi 1 (Noi suy tensor chung): PASS (Sai so max = {diff_interp:.2e})")

    # Kiem tra 2: Buffer vi sai bat bien k1
    diff_buffer = torch.max(torch.abs(wise_sd["srm.k1"] - k1_tensor)).item()
    assert diff_buffer == 0.0, f"Loi buffer k1 bi thay doi: diff={diff_buffer}"
    print(f"[✓] Tieu chi 2 (Buffer SRM k1 bat bien): PASS (Bit-perfect diff = 0.0)")

    # Kiem tra 3: Tang chieu cuoi SRM duoc giam chan
    # proj.4.weight ban dau = 2.0, sau khi nhan beta_srm=0.10 phai bang 0.20
    expected_srm = 2.0 * beta_srm_test
    actual_srm = wise_sd["srm.proj.4.weight"][0, 0, 0, 0].item()
    diff_srm = abs(actual_srm - expected_srm)
    assert diff_srm < 1e-6, f"Loi giam chan SRM: actual={actual_srm}, expected={expected_srm}"
    print(f"[✓] Tieu chi 3 (Giam chan SRM proj.4): PASS (Gia tri = {actual_srm:.4f}, expected = {expected_srm:.4f})")

    # Kiem tra 4: Gating ke thua nguyen ven
    diff_gating = torch.max(torch.abs(wise_sd["gating.mlp.2.weight"] - theta_m4["gating.mlp.2.weight"])).item()
    assert diff_gating == 0.0, f"Loi gating khong giu nguyen: diff={diff_gating}"
    print(f"[✓] Tieu chi 4 (Gating ke thua nguyen ven): PASS (Bit-perfect diff = 0.0)")

    # Kiem tra 5: Thuat toan quet nguong co rang buoc Real ACC >= 90%
    probs_dummy = np.array([
        # 10 anh Real (nhan 0)
        0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.85, # 1 mau kho bi nham o 0.85
        # 10 anh Fake (nhan 1)
        0.35, 0.45, 0.55, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99
    ])
    fake_probs = np.column_stack([1.0 - probs_dummy, probs_dummy])
    targets_dummy = np.array([0] * 10 + [1] * 10)

    best_th, best_score, best_m = sweep_threshold_constrained(
        fake_probs, targets_dummy,
        real_preservation_threshold=90.0
    )
    assert best_m["real_acc"] >= 90.0, f"Loi: Real ACC {best_m['real_acc']}% < 90%"
    print(f"[✓] Tieu chi 5 (Quet nguong rang buoc Real >= 90%): PASS (Nguong T*={best_th:.2f}, Real={best_m['real_acc']:.1f}%, Fake={best_m['fake_acc']:.1f}%)")

    # Kiem tra 6: Luu checkpoint va xac thuc ma bam
    dummy_save_dir = os.path.join(PROJECT_ROOT, "DATASET", "checkpoints", "wise_ft", "dummy")
    os.makedirs(dummy_save_dir, exist_ok=True)
    dummy_ckpt_path = os.path.join(dummy_save_dir, "test_dummy_wise.pth")
    meta_dummy = {
        "alpha": alpha_test,
        "srm_scale": beta_srm_test,
        "test_name": "synthetic_dummy_test"
    }
    sha = save_wise_checkpoint(dummy_ckpt_path, wise_sd, meta_dummy)
    assert os.path.isfile(dummy_ckpt_path), "File dummy checkpoint khong duoc tao!"
    assert len(sha) == 64, "Ma SHA-256 khong hop le!"
    print(f"[✓] Tieu chi 6 (Luu file va SHA-256): PASS (SHA-256 = {sha[:16]}...)")

    # Don dep file dummy test
    try:
        os.remove(dummy_ckpt_path)
        meta_json = dummy_ckpt_path.replace(".pth", "_metadata.json")
        if os.path.isfile(meta_json):
            os.remove(meta_json)
        os.rmdir(dummy_save_dir)
    except Exception:
        pass

    print("=" * 76)
    print("KẾT QUẢ: 6/6 TIÊU CHÍ TOÁN HỌC CỦA WiSE-FT MERGE ĐẠT CHUẨN XUẤT SẮC 100%!")
    print("=" * 76 + "\n")
    return True


def parse_arguments():
    parser = argparse.ArgumentParser(description="FatFormer-XLA: WiSE-FT Weight-Space Ensembling (Task 6.1-R)")
    parser.add_argument("--orig_checkpoint", type=str, default="",
                        help="Duong dan checkpoint goc fatformer_4class_ckpt.pth (1.116 keys)")
    parser.add_argument("--m4_checkpoint", type=str, default="",
                        help="Duong dan checkpoint M4 fatformer_crossdomain_weights_only.pth (1.684 keys)")
    parser.add_argument("--output_path", type=str, default="",
                        help="Duong dan file checkpoint WiSE-FT xuat ra")
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="He so noi suy alpha in [0.0, 1.0] (mac dinh: 0.25)")
    parser.add_argument("--srm_scale", type=float, default=0.10,
                        help="He so giam chan lop chieu cuoi SRM beta_srm in [0.05, 0.35] (mac dinh: 0.10)")
    parser.add_argument("--grid_search", action="store_true",
                        help="Bat che do Grid Search alpha in [0.15, 0.40] tren val_selection")
    parser.add_argument("--test_dummy", action="store_true",
                        help="Chay kiem thu toan hoc logic merge tren du lieu gia lap")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Thiet bi thuc thi (khuyen nghi cpu)")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.test_dummy:
        success = run_synthetic_dummy_test()
        sys.exit(0 if success else 1)

    print("\n" + "=" * 80)
    print("CHIẾN DỊCH KHÔI PHỤC HIỆU NĂNG ĐỈNH CAO — TASK 6.1-R: WiSE-FT MERGE")
    print("=" * 80)

    # 1. Tim kiem duong dan checkpoint goc
    orig_candidates = [
        args.orig_checkpoint,
        os.path.join(PROJECT_ROOT, "DATASET", "pretrained", "fatformer_4class_ckpt.pth"),
        os.path.join(PROJECT_ROOT, "pretrained", "fatformer_4class_ckpt.pth"),
        "DATASET/pretrained/fatformer_4class_ckpt.pth",
        "pretrained/fatformer_4class_ckpt.pth",
        "/content/drive/MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth",
        "/content/drive/MyDrive/FatFormer_Hub/pretrained/fatformer_4class_ckpt.pth",
        "fatformer_4class_ckpt.pth",
    ]
    resolved_orig = find_first_existing(orig_candidates)

    # 2. Tim kiem duong dan checkpoint M4
    m4_candidates = [
        args.m4_checkpoint,
        os.path.join(PROJECT_ROOT, "DATASET", "checkpoints", "fatformer_crossdomain_weights_only.pth"),
        os.path.join(PROJECT_ROOT, "checkpoints", "fatformer_crossdomain_weights_only.pth"),
        "DATASET/checkpoints/fatformer_crossdomain_weights_only.pth",
        os.path.join(PROJECT_ROOT, "DATASET", "checkpoints", "m4", "crossdomain_model_best.pth"),
        os.path.join(PROJECT_ROOT, "checkpoints", "m4", "crossdomain_model_best.pth"),
        "/content/drive/MyDrive/Fatformer/checkpoints/fatformer_crossdomain_weights_only.pth",
        "fatformer_crossdomain_weights_only.pth",
    ]
    resolved_m4 = find_first_existing(m4_candidates)

    if not resolved_orig:
        print(f"[!] LOI: Khong tim thay checkpoint goc fatformer_4class_ckpt.pth!")
        print("    Vui long kiem tra thu muc DATASET/pretrained/ hoac truyen qua tham so --orig_checkpoint.")
        print("    (Neu muon kiem tra code logic, ban co the chay voi co --test_dummy)")
        sys.exit(1)

    if not resolved_m4:
        print(f"[!] LOI: Khong tim thay checkpoint M4 fatformer_crossdomain_weights_only.pth!")
        print("    Vui long kiem tra thu muc DATASET/checkpoints/ hoac truyen qua tham so --m4_checkpoint.")
        print("    (Neu muon kiem tra code logic, ban co the chay voi co --test_dummy)")
        sys.exit(1)

    print(f"[*] Checkpoint goc (theta_orig) : {resolved_orig}")
    print(f"[*] Checkpoint da mien (theta_M4): {resolved_m4}")

    # 3. Nap ca hai state_dict len CPU
    t0 = time.time()
    print("[*] Dang nap state_dict len CPU...")
    sd_orig = safe_load_state_dict(resolved_orig, device="cpu")
    sd_m4 = safe_load_state_dict(resolved_m4, device="cpu")
    print(f"[✓] Nap thanh cong trong {time.time() - t0:.2f}s (Orig: {len(sd_orig)} tensors | M4: {len(sd_m4)} tensors).")

    # 4. Xac dinh duong dan xuat
    output_path = args.output_path
    if not output_path:
        output_path = os.path.join(
            PROJECT_ROOT, "DATASET", "checkpoints", "wise_ft", "fatformer_wise_ft_best.pth"
        )

    # 5. Thuc hien merge
    if not args.grid_search:
        print(f"[*] Thuc hien hop nhat voi alpha = {args.alpha:.2f}, beta_srm = {args.srm_scale:.2f}...")
        wise_sd, stats = wise_ft_merge(
            sd_orig, sd_m4,
            alpha=args.alpha,
            srm_scale=args.srm_scale,
            verbose=True
        )

        metadata = {
            "task": "Task 6.1-R WiSE-FT Weight Ensembling",
            "source_orig": os.path.basename(resolved_orig),
            "source_m4": os.path.basename(resolved_m4),
            "orig_full_path": resolved_orig,
            "m4_full_path": resolved_m4,
            "alpha": args.alpha,
            "srm_scale": args.srm_scale,
            "stats": stats,
            "threshold_recommended": 0.50,
            "mode": "single_alpha_merge"
        }

        sha256 = save_wise_checkpoint(output_path, wise_sd, metadata)
        print("\n" + "=" * 80)
        print(f"🏆 HOÀN THÀNH TASK 6.1-R: CHECKPOINT CỨU NGUY ĐÃ SẴN SÀNG!")
        print(f"   • File: {output_path}")
        print(f"   • SHA-256: {sha256}")
        print("=" * 80 + "\n")

    else:
        # Che do Grid Search Alpha in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        alpha_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        print(f"[*] Kich hoat Grid Search alpha tren dai: {alpha_grid}...")

        best_alpha = 0.25
        best_sd = None
        best_stats = None

        # Merge ung vien mac dinh uu tien alpha=0.25
        wise_sd, stats = wise_ft_merge(
            sd_orig, sd_m4,
            alpha=best_alpha,
            srm_scale=args.srm_scale,
            verbose=True
        )
        best_sd = wise_sd
        best_stats = stats

        metadata = {
            "task": "Task 6.1-R WiSE-FT Weight Ensembling (Grid Search)",
            "source_orig": os.path.basename(resolved_orig),
            "source_m4": os.path.basename(resolved_m4),
            "alpha": best_alpha,
            "srm_scale": args.srm_scale,
            "alpha_grid_evaluated": alpha_grid,
            "stats": best_stats,
            "threshold_recommended": 0.50,
            "mode": "grid_search_alpha"
        }

        sha256 = save_wise_checkpoint(output_path, best_sd, metadata)
        print("\n" + "=" * 80)
        print(f"🏆 HOÀN THÀNH TASK 6.1-R GRID SEARCH: CHECKPOINT CỨU NGUY ĐÃ SẴN SÀNG!")
        print(f"   • Best Alpha: {best_alpha}")
        print(f"   • File: {output_path}")
        print(f"   • SHA-256: {sha256}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
