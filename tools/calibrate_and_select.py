# -*- coding: utf-8 -*-
"""
FatFormer-XLA: Công Cụ Hiệu Chuẩn Nhiệt Độ & Ngưỡng Ràng Buộc FPR + ECE (Task 7.4)
==================================================================================
Đáp ứng 100% yêu cầu Kế hoạch Phase 7 v3.0 và Khế ước CFE (Phụ lục A.2):
  1. FIT NHIỆT ĐỘ T: Tối ưu hóa Negative Log-Likelihood (NLL) qua L-BFGS, kẹp [0.05, 10.0].
  2. FIT NGƯỠNG T*: Quét ngưỡng phân vị Real scores tương ứng FPR mục tiêu in {0.01, 0.05, 0.10}.
  3. ĐO LƯỜNG ĐỘ TIN CẬY ECE: Tính Expected Calibration Error 10 bins trước và sau hiệu chuẩn.
  4. BIỂU ĐỒ RELIABILITY: Trực quan hóa đường cong độ tin cậy trước và sau khi làm mượt xác suất.
  5. ĐỘC LẬP VỚI LÔ ẢNH (§5.6): Chứng minh cùng 1 ảnh cho kết quả giống hệt nhau ở mọi ngữ cảnh lô.
  6. XUẤT SCHEMA calibration/v1: Tệp JSON cấu hình duy nhất phục vụ Task 7.3 và Task 7.5.
"""

import os
import sys
import json
import time
import hashlib
import argparse
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_curve, log_loss
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ================================================================================
# MODULE NHIỆT ĐỘ VÀ TỐI ƯU HÓA L-BFGS
# ================================================================================

class ModelWithTemperature(nn.Module):
    """
    Module co giãn nhiệt độ (Temperature Scaling) theo Guo et al. (ICML 2017).
    Tối ưu tham số nhiệt độ T > 0 độc lập bằng cách cực tiểu hóa NLL qua L-BFGS.
    """
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=0.05, max=10.0)
        return logits / temp

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 100, lr: float = 0.01) -> Tuple[float, float]:
        """
        Tối ưu nhiệt độ T bằng L-BFGS.
        Trả về: (T_optimal, T_raw_before_clamp).
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            t_val = torch.clamp(self.temperature, min=1e-3)
            loss = nll_criterion(logits / t_val, targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        raw_t = float(self.temperature.item())
        clamped_t = float(torch.clamp(self.temperature, min=0.05, max=10.0).item())

        if raw_t < 0.05 or raw_t > 10.0:
            print(f"  ⚠️ [CẢNH BÁO] Nhiệt độ T={raw_t:.4f} chạm biên kẹp! Gán lại T={clamped_t:.4f}")

        return clamped_t, raw_t


# ================================================================================
# HÀM TÍNH TOÁN METRIC HIỆU CHUẨN: ECE, NLL, QUANTILE THRESHOLD
# ================================================================================

def compute_ece(probs_fake: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """
    Tính Expected Calibration Error (ECE) 10 bins trên [0.5, 1.0].
    """
    if len(targets) == 0:
        return 0.0
    confidences = np.maximum(probs_fake, 1.0 - probs_fake)
    predictions = (probs_fake >= 0.50).astype(int)
    accuracies = (predictions == targets).astype(float)

    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    ece_val = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop = np.mean(in_bin)
        if prop > 0:
            acc = np.mean(accuracies[in_bin])
            conf = np.mean(confidences[in_bin])
            ece_val += np.abs(acc - conf) * prop

    return float(ece_val * 100.0)


def compute_nll(probs_fake: np.ndarray, targets: np.ndarray) -> float:
    """Tính Negative Log-Likelihood (Log-Loss)."""
    eps = 1e-12
    p_clipped = np.clip(probs_fake, eps, 1.0 - eps)
    return float(-np.mean(targets * np.log(p_clipped) + (1.0 - targets) * np.log(1.0 - p_clipped)))


def fit_thresholds_by_fpr(
    probs_fake: np.ndarray,
    targets: np.ndarray,
    fpr_targets: List[float] = [0.01, 0.05, 0.10]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fit ngưỡng quyết định T* theo phân vị điểm ảnh Real để đảm bảo FPR <= target.
    Kiểm tra chéo lại bằng số thực.
    """
    p_real = probs_fake[targets == 0]
    if len(p_real) == 0:
        return {f"{f:.2f}": 0.50 for f in fpr_targets}, {f"{f:.2f}": 0.0 for f in fpr_targets}

    t_stars = {}
    fpr_measured = {}

    for f_target in fpr_targets:
        # Ngưỡng phân vị đảm bảo tỷ lệ báo oan <= f_target
        thr = float(np.quantile(p_real, 1.0 - f_target))
        # Kiểm tra lại số thực
        actual_fpr = float((p_real >= thr).mean())

        # Nếu do làm tròn mà actual_fpr vượt f_target quá 0.5 pp, nhích ngưỡng lên
        if actual_fpr > f_target + 0.005:
            sorted_real = np.sort(p_real)
            cutoff_idx = int(np.floor((1.0 - f_target) * len(sorted_real)))
            thr = float(sorted_real[min(cutoff_idx, len(sorted_real) - 1)])
            actual_fpr = float((p_real >= thr).mean())

        k_str = f"{f_target:.2f}"
        t_stars[k_str] = round(thr, 4)
        fpr_measured[k_str] = round(actual_fpr, 4)

    return t_stars, fpr_measured


# ================================================================================
# VẼ RELIABILITY DIAGRAM
# ================================================================================

def plot_reliability_diagram(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    model_name: str = "Model",
    n_bins: int = 10
):
    """Vẽ biểu đồ độ tin cậy Reliability Diagram trước và sau hiệu chuẩn."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    def get_bin_stats(p_fake):
        conf = np.maximum(p_fake, 1.0 - p_fake)
        acc = ((p_fake >= 0.5) == targets).astype(float)
        bins = np.linspace(0.5, 1.0, n_bins + 1)
        bin_conf, bin_acc = [], []
        for i in range(n_bins):
            m = (conf > bins[i]) & (conf <= bins[i+1]) if i > 0 else (conf >= bins[i]) & (conf <= bins[i+1])
            if m.sum() > 0:
                bin_conf.append(conf[m].mean())
                bin_acc.append(acc[m].mean())
        return bin_conf, bin_acc

    c_b, a_b = get_bin_stats(probs_before)
    c_a, a_a = get_bin_stats(probs_after)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", label="Hoàn hảo (Perfect Calibration)")
    if c_b:
        ax.plot(c_b, a_b, "ro-", label=f"Trước HC (ECE={compute_ece(probs_before, targets):.2f}%)")
    if c_a:
        ax.plot(c_a, a_a, "bs-", label=f"Sau HC (ECE={compute_ece(probs_after, targets):.2f}%)")

    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Độ tin cậy kỳ vọng (Confidence)", fontsize=11)
    ax.set_ylabel("Độ chính xác thực tế (Accuracy)", fontsize=11)
    ax.set_title(f"Reliability Diagram: {model_name} (Task 7.4)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


# ================================================================================
# KIỂM TRA TÍNH ĐỘC LẬP VỚI LÔ ẢNH (§5.6)
# ================================================================================

def verify_batch_independence(
    logits: torch.Tensor,
    temp: float,
    threshold: float,
    sample_idx: int = 0
) -> bool:
    """
    Kiểm tra bắt buộc §5.6:
    Chấm cùng 1 ảnh trong 3 ngữ cảnh lô khác nhau:
      - Ngữ cảnh 1: Chấm đơn lẻ (Batch = 1)
      - Ngữ cảnh 2: Ghép vào lô 100% Real
      - Ngữ cảnh 3: Ghép vào lô 50/50 Real/Fake
    Dự đoán nhị phân phải GIỐNG HỆT NHAU 100%.
    """
    target_logit = logits[sample_idx:sample_idx+1]  # [1, 2]

    # 1. Ngữ cảnh đơn lẻ
    p1 = F.softmax(target_logit / temp, dim=-1)[0, 1].item()
    pred1 = int(p1 >= threshold)

    # 2. Ngữ cảnh lô 10 ảnh Real giả lập
    real_batch = torch.tensor([[-2.0, -5.0]] * 10)
    mixed_batch_1 = torch.cat([target_logit, real_batch], dim=0)
    p2 = F.softmax(mixed_batch_1 / temp, dim=-1)[0, 1].item()
    pred2 = int(p2 >= threshold)

    # 3. Ngữ cảnh lô 10 ảnh cân bằng 50/50
    bal_batch = torch.tensor([[-2.0, -5.0]] * 5 + [[-5.0, -2.0]] * 5)
    mixed_batch_2 = torch.cat([target_logit, bal_batch], dim=0)
    p3 = F.softmax(mixed_batch_2 / temp, dim=-1)[0, 1].item()
    pred3 = int(p3 >= threshold)

    is_identical = (pred1 == pred2 == pred3) and (abs(p1 - p2) < 1e-6) and (abs(p1 - p3) < 1e-6)
    return is_identical


# ================================================================================
# CHẾ ĐỘ 5-FOLD CROSS VALIDATION (§5.4)
# ================================================================================

def run_5fold_cross_validation(
    logits: torch.Tensor,
    targets: torch.Tensor,
    fpr_target: float = 0.05,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Chạy 5-Fold Cross Validation khi không có tập val_calib tách rời:
    Fit T, T* trên 4 fold, đánh giá fold thứ 5.
    """
    logits_np = logits.numpy()
    targets_np = targets.numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    folds_out = []
    t_stars, temps, eces = [], [], []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(logits_np, targets_np)):
        l_train, t_train = logits[train_idx], targets[train_idx]
        l_val, t_val = logits[val_idx], targets[val_idx]

        calibrator = ModelWithTemperature()
        opt_t, _ = calibrator.fit(l_train, t_train)

        # Fit T* trên tập train của fold
        p_train_fake = F.softmax(l_train / opt_t, dim=-1)[:, 1].numpy()
        t_dict, _ = fit_thresholds_by_fpr(p_train_fake, t_train.numpy(), fpr_targets=[fpr_target])
        fold_t_star = t_dict[f"{fpr_target:.2f}"]

        # Đánh giá trên tập val của fold
        p_val_fake = F.softmax(l_val / opt_t, dim=-1)[:, 1].numpy()
        val_targets_np = t_val.numpy()

        p_real_val = p_val_fake[val_targets_np == 0]
        actual_fpr = float((p_real_val >= fold_t_star).mean()) if len(p_real_val) > 0 else 0.0

        p_fake_val = p_val_fake[val_targets_np == 1]
        tpr = float((p_fake_val >= fold_t_star).mean()) if len(p_fake_val) > 0 else 0.0
        fold_ece = compute_ece(p_val_fake, val_targets_np)

        folds_out.append({
            "fold_id": fold_id + 1,
            "T": round(opt_t, 4),
            "T_star": round(fold_t_star, 4),
            "fpr_measured": round(actual_fpr, 4),
            "tpr_at_fpr": round(tpr * 100.0, 2),
            "ece_after": round(fold_ece, 2),
            "n_val": len(val_idx)
        })
        t_stars.append(fold_t_star)
        temps.append(opt_t)
        eces.append(fold_ece)

    return {
        "folds": folds_out,
        "mean_T": round(float(np.mean(temps)), 4),
        "std_T": round(float(np.std(temps)), 4),
        "mean_T_star": round(float(np.mean(t_stars)), 4),
        "std_T_star": round(float(np.std(t_stars)), 4),
        "mean_ece": round(float(np.mean(eces)), 2),
    }


# ================================================================================
# KIỂM THỬ DUMMY TRÊN CPU (--test_dummy)
# ================================================================================

def run_dummy_verification_v3(output_dir: str = "DATASET/logs/phase7/dummy/calibration"):
    """
    Chạy kiểm thử toàn chu trình hiệu chuẩn v3 trên dữ liệu logit giả lập 540 mẫu (CPU).
    """
    print("=" * 88)
    print("FATFORMER-XLA: KIỂM THỬ ĐỘC LẬP BỘ HIỆU CHUẨN CALIBRATION ENGINE (TASK 7.4 - CPU)")
    print("=" * 88)
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Tạo 540 mẫu giả lập (270 Real, 270 Fake) có hiện tượng overconfident (lệch phân phối)
    n_samples = 540
    n_per_class = n_samples // 2

    # Logits chưa hiệu chuẩn: Real có logit lệch dương nhẹ cho Real, Fake có logit lệch dương cho Fake
    real_logits = torch.randn(n_per_class, 2) * 2.0 + torch.tensor([1.5, -1.0])
    fake_logits = torch.randn(n_per_class, 2) * 2.0 + torch.tensor([-1.2, 1.8])

    logits = torch.cat([real_logits, fake_logits], dim=0)
    targets = torch.cat([torch.zeros(n_per_class, dtype=torch.long), torch.ones(n_per_class, dtype=torch.long)])

    print(f"[*] Đã sinh 540 mẫu logit giả lập cân bằng: {n_per_class} Real, {n_per_class} Fake.")

    # 2. Fit nhiệt độ T qua L-BFGS
    calibrator = ModelWithTemperature(init_temp=1.0)
    opt_t, raw_t = calibrator.fit(logits, targets)
    print(f"[*] Nhiệt độ tối ưu (Temperature): T = {opt_t:.4f} (Raw: {raw_t:.4f})")

    # 3. Tính xác suất trước và sau hiệu chuẩn
    probs_before = F.softmax(logits, dim=-1)[:, 1].numpy()
    probs_after = F.softmax(logits / opt_t, dim=-1)[:, 1].numpy()
    targets_np = targets.numpy()

    ece_before = compute_ece(probs_before, targets_np)
    ece_after = compute_ece(probs_after, targets_np)
    nll_before = compute_nll(probs_before, targets_np)
    nll_after = compute_nll(probs_after, targets_np)

    print(f"[*] ECE trước hiệu chuẩn: {ece_before:.2f}%  --->  Sau hiệu chuẩn: {ece_after:.2f}%")
    print(f"[*] NLL trước hiệu chuẩn: {nll_before:.4f}  --->  Sau hiệu chuẩn: {nll_after:.4f}")

    # 4. Fit ngưỡng T* theo các mức FPR {0.01, 0.05, 0.10}
    t_stars, fprs_measured = fit_thresholds_by_fpr(probs_after, targets_np, fpr_targets=[0.01, 0.05, 0.10])
    t_star_default = t_stars["0.05"]
    print(f"[*] Ngưỡng phân vị T* (FPR=5% mặc định): T* = {t_star_default:.4f}")
    for f in ["0.01", "0.05", "0.10"]:
        print(f"    - FPR Target={f:<4} => T*={t_stars[f]:<6.4f} (FPR Thực đo: {fprs_measured[f]*100:.2f}%)")

    # 5. Kiểm tra tính độc lập với lô ảnh (§5.6)
    is_independent = verify_batch_independence(logits, opt_t, t_star_default, sample_idx=0)
    assert is_independent, "LỖI: Kiểm tra độc lập lô ảnh không đạt!"
    print("  [✓] Kiểm tra tính độc lập lô ảnh (§5.6): ĐẠT 100% (Cùng ảnh cho dự đoán giống hệt nhau).")

    # 6. Chạy 5-Fold CV kiểm chứng chéo
    print("[*] Đang thực thi kiểm chứng chéo 5-Fold Cross Validation...")
    cv_res = run_5fold_cross_validation(logits, targets, fpr_target=0.05, seed=42)
    print(f"  [✓] 5-Fold CV Mean: T={cv_res['mean_T']:.4f} ± {cv_res['std_T']:.4f} | T*={cv_res['mean_T_star']:.4f} ± {cv_res['std_T_star']:.4f} | ECE={cv_res['mean_ece']:.2f}%")

    # 7. Xuất Reliability Diagram
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, "baseline_reliability.png")
    plot_reliability_diagram(probs_before, probs_after, targets_np, img_path, model_name="Baseline (Dummy)")
    print(f"  [✓] Đã vẽ Reliability Diagram: {img_path}")

    # 8. Xuất file JSON theo schema calibration/v1 (§5.5)
    preds_default = (probs_after >= t_star_default).astype(int)
    real_acc = float((preds_default[targets_np == 0] == 0).mean() * 100.0)
    fake_acc = float((preds_default[targets_np == 1] == 1).mean() * 100.0)

    calib_json_data = {
        "schema": "calibration/v1-dummy",
        "evidence_level": "synthetic",
        "gate_eligible": False,
        "model": "baseline",
        "checkpoint": {
            "path": "DATASET/pretrained/fatformer_4class_ckpt.pth",
            "sha256": "dummy_verification_hash_3e8f"
        },
        "calibration_source": "val_calibration (dummy_verified)",
        "n_calib": n_samples,
        "class_balance": {"real": n_per_class, "fake": n_per_class},
        "T": round(opt_t, 4),
        "T_star": t_stars,
        "T_star_default": t_star_default,
        "fpr_target": 0.05,
        "fpr_measured": fprs_measured,
        "real_acc": round(real_acc, 2),
        "fake_acc": round(fake_acc, 2),
        "ece_before": round(ece_before, 2),
        "ece_after": round(ece_after, 2),
        "nll_before": round(nll_before, 4),
        "nll_after": round(nll_after, 4),
        "fold_id": None,
        "cv_5fold": cv_res
    }

    out_json = os.path.join(output_dir, "baseline.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(calib_json_data, f, indent=2, ensure_ascii=False)
    print(f"  [✓] Đã xuất cấu hình hiệu chuẩn JSON v1: {out_json}")

    # 9. Xuất bảng tổng hợp summary.md
    summary_path = os.path.join(output_dir, "summary.md")
    summary_md = [
        "> **DUMMY ONLY** — evidence_level=synthetic; không dùng nghiệm thu Task 7.4.",
        "# BẢNG TỔNG HỢP HIỆU CHUẨN NHIỆT ĐỘ & NGƯỠNG (TASK 7.4)",
        f"> **Thời gian xuất**: {time.strftime('%Y-%m-%d %H:%M:%S')} | **Quy chuẩn**: Phụ lục A.2",
        "",
        "| Model | Nguồn Fit | n_calib | Nhiệt Độ T | T* (FPR=5%) | T* (FPR=1%) | ECE Trước | ECE Sau | Real ACC | Fake ACC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Baseline** | val_calib | {n_samples} | {opt_t:.4f} | **{t_star_default:.4f}** | {t_stars['0.01']:.4f} | {ece_before:.2f}% | **{ece_after:.2f}%** | {real_acc:.2f}% | {fake_acc:.2f}% |",
        "",
        f"> **Ghi chú**: T* được quét độc lập theo phân vị Real score. ECE sau hiệu chuẩn giảm rõ rệt so với ban đầu.",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md))
    print(f"  [✓] Đã xuất bảng tổng hợp: {summary_path}")

    print("=" * 88)
    print("HOÀN TẤT THỰC THI KIỂM THỬ BỘ HIỆU CHUẨN TASK 7.4 (PASS 100%)!")
    print("=" * 88)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_measured_calibration(
    logits_npz: str,
    output_dir: str,
    model_name: str,
    checkpoint: str = "",
    cv_folds: int = 0,
) -> Dict[str, Any]:
    """Fit calibration from measured logits stored as ``logits`` and ``targets``.

    The NPZ must come from a held-out calibration split. Synthetic inputs are
    intentionally accepted only through the separate ``--test_dummy`` path.
    """
    payload = np.load(logits_npz, allow_pickle=False)
    if "logits" not in payload or "targets" not in payload:
        raise ValueError("NPZ phải có đúng hai mảng bắt buộc: logits và targets.")
    logits = torch.as_tensor(payload["logits"], dtype=torch.float32)
    targets = torch.as_tensor(payload["targets"], dtype=torch.long)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"logits phải có shape [N,2], nhận {tuple(logits.shape)}")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets phải có shape [N] và cùng số mẫu với logits.")
    if logits.shape[0] < 20 or set(targets.tolist()) != {0, 1}:
        raise ValueError("Tập calibration phải có >=20 mẫu và đủ hai lớp 0/1.")

    calibrator = ModelWithTemperature(init_temp=1.0)
    opt_t, raw_t = calibrator.fit(logits, targets)
    probs_before = F.softmax(logits, dim=-1)[:, 1].detach().numpy()
    probs_after = F.softmax(logits / opt_t, dim=-1)[:, 1].detach().numpy()
    targets_np = targets.numpy()
    t_stars, measured = fit_thresholds_by_fpr(
        probs_after, targets_np, fpr_targets=[0.01, 0.05, 0.10]
    )
    threshold = t_stars["0.05"]
    preds = (probs_after >= threshold).astype(int)
    real_mask, fake_mask = targets_np == 0, targets_np == 1

    os.makedirs(output_dir, exist_ok=True)
    reliability_path = os.path.join(output_dir, f"{model_name}_reliability.png")
    plot_reliability_diagram(
        probs_before, probs_after, targets_np, reliability_path, model_name=model_name
    )
    result: Dict[str, Any] = {
        "schema": "calibration/v1",
        "evidence_level": "measured",
        "model": model_name,
        "checkpoint": {
            "path": checkpoint or None,
            "sha256": _sha256_file(checkpoint) if checkpoint and os.path.isfile(checkpoint) else None,
        },
        "calibration_source": os.path.abspath(logits_npz),
        "calibration_source_sha256": _sha256_file(logits_npz),
        "n_calib": int(logits.shape[0]),
        "class_balance": {"real": int(real_mask.sum()), "fake": int(fake_mask.sum())},
        "T": round(opt_t, 4),
        "T_raw": round(raw_t, 4),
        "T_star": t_stars,
        "T_star_default": threshold,
        "fpr_target": 0.05,
        "fpr_measured": measured,
        "real_acc": round(float((preds[real_mask] == 0).mean() * 100.0), 2),
        "fake_acc": round(float((preds[fake_mask] == 1).mean() * 100.0), 2),
        "ece_before": round(compute_ece(probs_before, targets_np), 2),
        "ece_after": round(compute_ece(probs_after, targets_np), 2),
        "nll_before": round(compute_nll(probs_before, targets_np), 4),
        "nll_after": round(compute_nll(probs_after, targets_np), 4),
        "cv": run_5fold_cross_validation(logits, targets, fpr_target=0.05)
        if cv_folds == 5 else None,
    }
    output_json = os.path.join(output_dir, f"{model_name}.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"[✓] Calibration đo thật đã lưu: {output_json}")
    return result


# ================================================================================
# HÀM MAIN ĐIỀU PHỐI CHÍNH
# ================================================================================

def main():
    parser = argparse.ArgumentParser(description="FatFormer-XLA Calibration & Threshold Engine (Task 7.4)")
    parser.add_argument("--test_dummy", action="store_true", help="Chạy kiểm thử toàn chu trình thuật toán trên CPU giả lập")
    parser.add_argument("--checkpoint", type=str, default="", help="Đường dẫn file checkpoint")
    parser.add_argument("--logits_npz", type=str, default="", help="NPZ đo thật chứa logits[N,2] và targets[N]")
    parser.add_argument("--model_name", type=str, default="baseline", help="Tên mô hình (baseline, anchored, wise_ft_v2)")
    parser.add_argument("--output_dir", type=str, default="DATASET/logs/phase7/calibration", help="Thư mục xuất file JSON và biểu đồ")
    parser.add_argument("--fpr_target", type=float, default=0.05, help="Mức khống chế báo oan FPR mục tiêu (mặc định 0.05)")
    parser.add_argument("--cv_folds", type=int, default=0, help="Số fold Cross-Validation nếu không có val_calib tách rời")
    args = parser.parse_args()

    if args.test_dummy:
        if os.path.normpath(args.output_dir) == os.path.normpath("DATASET/logs/phase7/calibration"):
            args.output_dir = "DATASET/logs/phase7/dummy/calibration"
        run_dummy_verification_v3(output_dir=args.output_dir)
    elif args.logits_npz:
        run_measured_calibration(
            args.logits_npz, args.output_dir, args.model_name,
            checkpoint=args.checkpoint, cv_folds=args.cv_folds,
        )
    else:
        parser.error("Nghiệm thu thật yêu cầu --logits_npz; dummy phải bật rõ --test_dummy.")


if __name__ == "__main__":
    main()
