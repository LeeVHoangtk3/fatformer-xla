# -*- coding: utf-8 -*-
"""
FatFormer-XLA: Công Cụ Đánh Giá Ngoại Vi CommunityForensics-Eval (CVPR 2025) - Bản Chuẩn Hóa v3.0
================================================================================================
Tuân thủ nghiêm ngặt Khế ước đo lường CFE (Phụ lục A):
  1. KHÔNG strict=False: Loại bỏ hoàn toàn nạp thiếu lén lút (E1/E2).
  2. KHÔNG adaptive_thresh theo batch: Ngưỡng cố định T* lấy từ calibration.json hoặc CLI (E3).
  3. BỘ CHỈ SỐ PHÁP Y ĐẦY ĐỦ: ROC-AUC, AP, TPR@FPR (1%, 5%, 10%), ECE (10 bins), NLL (E5).
  4. KHOẢNG TIN CẬY 95% CI: Bootstrap 1.000 lần resample theo ảnh (seed=42).
  5. ĐIỀU KIỆN VẬN HÀNH RÕ RÀNG: Ghi nhận degrade_order (post_resize/pre_resize), git_commit, T, T*.
  6. CHẾ ĐỘ --test_dummy: Tự động chạy kiểm thử toán học 3 kịch bản mảng giả mà không cần GPU/model.
"""

import os
import sys
import io
import glob
import json
import time
import argparse
import subprocess
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, log_loss

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CFE_SUBSETS = [
    "1_commercial",
    "2_latent_diffusion",
    "3_pixel_diffusion",
    "4_modern_gans",
    "5_other_autoregressive",
]

SUBSET_DESCRIPTIONS = {
    "1_commercial": "Commercial (MJ v6, DALL-E 3)",
    "2_latent_diffusion": "LatDiff (SDXL, Flux.1)",
    "3_pixel_diffusion": "PixDiff (DeepFloyd, ADM)",
    "4_modern_gans": "GANs (StyleGAN3, GigaGAN)",
    "5_other_autoregressive": "Other / Autoregressive (Muse, LlamaGen)",
}


# ================================================================================
# TIỆN ÍCH HỆ THỐNG VÀ KIỂM TOÁN GIT
# ================================================================================

def get_git_info() -> Tuple[str, bool]:
    """Lấy commit và trạng thái source tree, bỏ qua artifact sinh bởi pipeline."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            [
                "git", "status", "--porcelain", "--", ".",
                ":(exclude)DATASET/logs/**",
                ":(exclude)DATASET/checkpoints/**",
                ":(exclude)DATASET/pretrained/**",
            ],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return commit, len(status) > 0
    except Exception:
        return "unknown", False


def find_first_existing(path_list: List[str]) -> Optional[str]:
    """Tìm đường dẫn tồn tại đầu tiên trong danh sách ứng viên."""
    for p in path_list:
        if not p:
            continue
        if "*" in p or "?" in p:
            matches = glob.glob(p, recursive=True)
            if matches:
                return os.path.abspath(matches[0])
        elif os.path.exists(p):
            return os.path.abspath(p)
    return None


# ================================================================================
# CÁC HÀM TÍNH TOÁN METRIC CHUẨN V3 (PHỤ LỤC A)
# ================================================================================

def compute_ece(probs_fake: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """
    Tính Expected Calibration Error (ECE) chuẩn 10 bins trên miền [0.5, 1.0].
    """
    if len(targets) == 0:
        return 0.0
    confidences = np.maximum(probs_fake, 1.0 - probs_fake)
    predictions = (probs_fake >= 0.50).astype(int)
    accuracies = (predictions == targets).astype(float)

    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    total = len(targets)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    return float(ece * 100.0)


def calculate_metrics_v3(
    fake_probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Tính toán hệ thống chỉ số v3:
      - ROC-AUC: sklearn.metrics.roc_auc_score
      - AP: average_precision_score (tránh chia 0: null khi không có positive)
      - TPR@FPR: 1%, 5%, 10% nội suy đường cong ROC (không làm tròn FPR)
      - Real ACC / Fake ACC: tại ngưỡng T* cố định
      - ECE: Expected Calibration Error 10 bins
      - NLL: Negative Log-Likelihood (log_loss)
    """
    n_pos = int(np.sum(targets == 1))
    n_neg = int(np.sum(targets == 0))
    total = len(targets)

    if total == 0:
        return {
            "macro_acc": 0.0, "real_acc": None, "fake_acc": None,
            "auc": None, "ap": None,
            "tpr_at_fpr": {"0.01": None, "0.05": None, "0.10": None},
            "ece": 0.0, "nll": 0.0, "total_samples": 0
        }

    preds = (fake_probs >= threshold).astype(int)
    correct = np.sum(preds == targets)
    macro_acc = (correct / total * 100.0)

    real_acc = (np.sum(preds[targets == 0] == 0) / n_neg * 100.0) if n_neg > 0 else None
    fake_acc = (np.sum(preds[targets == 1] == 1) / n_pos * 100.0) if n_pos > 0 else None

    # 1. AUC
    if n_pos > 0 and n_neg > 0:
        try:
            auc = float(roc_auc_score(targets, fake_probs) * 100.0)
        except Exception:
            auc = None
    else:
        auc = None

    # 2. AP (Xử lý chia 0: trả về None nếu không có positive)
    if n_pos == 0:
        ap = None
    elif n_neg == 0:
        ap = 100.0
    else:
        try:
            ap = float(average_precision_score(targets, fake_probs) * 100.0)
        except Exception:
            ap = None

    # 3. TPR @ FPR (1%, 5%, 10%)
    tpr_at_fpr = {"0.01": None, "0.05": None, "0.10": None}
    if n_pos > 0 and n_neg > 0:
        try:
            fpr, tpr, _ = roc_curve(targets, fake_probs)
            tpr_at_fpr["0.01"] = round(float(np.interp(0.01, fpr, tpr) * 100.0), 2)
            tpr_at_fpr["0.05"] = round(float(np.interp(0.05, fpr, tpr) * 100.0), 2)
            tpr_at_fpr["0.10"] = round(float(np.interp(0.10, fpr, tpr) * 100.0), 2)
        except Exception:
            pass

    # 4. ECE
    ece = compute_ece(fake_probs, targets, n_bins=10)

    # 5. NLL
    eps = 1e-12
    p_clipped = np.clip(fake_probs, eps, 1.0 - eps)
    nll = float(-np.mean(targets * np.log(p_clipped) + (1.0 - targets) * np.log(1.0 - p_clipped)))

    # 6. Fake@Real90
    fake_at_real90 = None
    if n_pos > 0 and n_neg > 0:
        thr_fpr10 = float(np.quantile(fake_probs[targets == 0], 0.90))
        fake_at_real90 = round(float((fake_probs[targets == 1] >= thr_fpr10).mean() * 100.0), 2)

    return {
        "macro_acc": round(float(macro_acc), 2),
        "real_acc": round(float(real_acc), 2) if real_acc is not None else None,
        "fake_acc": round(float(fake_acc), 2) if fake_acc is not None else None,
        "fake_at_real90": fake_at_real90,
        "auc": round(float(auc), 2) if auc is not None else None,
        "ap": round(float(ap), 2) if ap is not None else None,
        "tpr_at_fpr": tpr_at_fpr,
        "ece": round(float(ece), 2),
        "nll": round(float(nll), 4),
        "total_samples": total,
    }


def compute_bootstrap_ci(
    fake_probs: np.ndarray,
    targets: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42
) -> Dict[str, List[Optional[float]]]:
    """
    Thuật toán Bootstrap 1.000 lần resample theo mẫu ảnh để tính khoảng tin cậy 95% CI.
    Áp dụng cho AUC, AP và TPR@FPR=5%.
    """
    n = len(targets)
    if n < 4 or np.sum(targets == 1) == 0 or np.sum(targets == 0) == 0:
        return {
            "auc": [None, None],
            "ap": [None, None],
            "tpr_at_fpr_0.05": [None, None]
        }

    rng = np.random.RandomState(seed)
    indices = rng.randint(0, n, size=(n_boot, n))

    boot_auc, boot_ap, boot_tpr5 = [], [], []

    for idx_row in indices:
        b_probs = fake_probs[idx_row]
        b_targets = targets[idx_row]

        b_pos = np.sum(b_targets == 1)
        b_neg = np.sum(b_targets == 0)

        if b_pos > 0 and b_neg > 0:
            try:
                boot_auc.append(roc_auc_score(b_targets, b_probs) * 100.0)
                boot_ap.append(average_precision_score(b_targets, b_probs) * 100.0)
                fpr, tpr, _ = roc_curve(b_targets, b_probs)
                boot_tpr5.append(np.interp(0.05, fpr, tpr) * 100.0)
            except Exception:
                continue

    def get_ci(vals: List[float]) -> List[Optional[float]]:
        if len(vals) < 50:
            return [None, None]
        return [
            round(float(np.percentile(vals, 2.5)), 2),
            round(float(np.percentile(vals, 97.5)), 2)
        ]

    return {
        "auc": get_ci(boot_auc),
        "ap": get_ci(boot_ap),
        "tpr_at_fpr_0.05": get_ci(boot_tpr5)
    }


def run_mock_array_tests():
    """
    Kiểm thử tự động 3 ca mảng giả (§5.4):
      - 4 pos, 4 neg perfect prediction -> auc=100, ap=100, tpr@fpr=100
      - 4 pos, 4 neg inverted prediction -> auc=0
      - Toàn negative -> ap=None, không crash
    """
    print("[*] Đang thực thi kiểm thử 3 ca mảng giả (§5.4)...")

    # Ca 1: Hoàn hảo
    t_perf = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_perf = np.array([0.01, 0.05, 0.10, 0.15, 0.85, 0.90, 0.95, 0.99])
    m_perf = calculate_metrics_v3(p_perf, t_perf, threshold=0.50)
    assert m_perf["auc"] == 100.0, f"Ca 1 AUC sai: {m_perf['auc']}"
    assert m_perf["ap"] == 100.0, f"Ca 1 AP sai: {m_perf['ap']}"
    assert m_perf["tpr_at_fpr"]["0.05"] == 100.0, f"Ca 1 TPR@5% sai: {m_perf['tpr_at_fpr']['0.05']}"

    # Ca 2: Đảo ngược hoàn toàn
    p_inv = np.array([0.99, 0.95, 0.90, 0.85, 0.15, 0.10, 0.05, 0.01])
    m_inv = calculate_metrics_v3(p_inv, t_perf, threshold=0.50)
    assert m_inv["auc"] == 0.0, f"Ca 2 AUC sai: {m_inv['auc']}"

    # Ca 3: Toàn negative
    t_none = np.array([0, 0, 0, 0])
    p_none = np.array([0.1, 0.2, 0.3, 0.4])
    m_none = calculate_metrics_v3(p_none, t_none, threshold=0.50)
    assert m_none["ap"] is None, f"Ca 3 AP phải là None, nhưng nhận: {m_none['ap']}"

    print("  [✓] Test mảng giả toán học: 3/3 ca kiểm thử ĐẠT 100%!")


# ================================================================================
# TIỀN XỬ LÝ ẢNH & BIẾN ĐỔI SUY THOÁI
# ================================================================================

def compress_jpeg_q30(image: Image.Image) -> Image.Image:
    """Nén ảnh sang JPEG Q=30 trong RAM qua io.BytesIO."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=30)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def compute_jpeg_blockiness(pil_img: Image.Image) -> float:
    """Đo chỉ số nhiễu khối lưới DCT 8x8 B(x) không tham chiếu."""
    arr = np.asarray(pil_img.convert("L"), dtype=np.float32)
    h, w = arr.shape
    if h < 16 or w < 16:
        return 1.0

    dh = np.abs(arr[:, 1:] - arr[:, :-1])
    dv = np.abs(arr[1:, :] - arr[:-1, :])

    boundary_h = dh[:, 7::8].mean()
    interior_h = np.delete(dh, np.s_[7::8], axis=1).mean() + 1e-5

    boundary_v = dv[7::8, :].mean()
    interior_v = np.delete(dv, np.s_[7::8], axis=0).mean() + 1e-5

    blockiness = 0.5 * ((boundary_h / interior_h) + (boundary_v / interior_v))
    return float(blockiness)


def compute_degradation_weight(blockiness: torch.Tensor) -> torch.Tensor:
    """Chuyển đổi B(x) sang trọng số suy biến w_deg in [0.10, 0.90]."""
    w = torch.sigmoid(12.0 * (blockiness - 1.18))
    return torch.clamp(w, min=0.10, max=0.90)


class CommunityForensicsDataset(Dataset):
    """
    Dataset CFE hỗ trợ chuẩn hóa degrade_order (post_resize / pre_resize).
    """
    def __init__(
        self,
        file_records: List[Dict[str, Any]],
        condition: str = "clean",
        degrade_order: str = "post_resize"
    ):
        self.file_records = file_records
        self.condition = condition
        self.degrade_order = degrade_order

        self.spatial_crop = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ])
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

    def __len__(self) -> int:
        return len(self.file_records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, float]:
        rec = self.file_records[idx]
        file_path = rec["path"]
        label = rec["label"]
        subset = rec["subset"]

        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            print(f"  [!] Cảnh báo đọc file lỗi ({file_path}): {e}")
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.condition in ["q30", "jpeg_q30"]:
            if self.degrade_order == "pre_resize":
                img = compress_jpeg_q30(img)
                img = self.spatial_crop(img)
            else:  # post_resize (Mặc định)
                img = self.spatial_crop(img)
                img = compress_jpeg_q30(img)
        else:
            img = self.spatial_crop(img)

        b_score = compute_jpeg_blockiness(img)
        tensor_img = self.to_tensor_norm(img)
        return tensor_img, label, subset, b_score


class DummyMockModel(nn.Module):
    """Mô hình giả lập phục vụ test độc lập trên CPU mà không cần GPU/checkpoint."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.pool(x.float()).flatten(1)
        return self.fc(feat)


# ================================================================================
# KHỞI DỰNG MÔ HÌNH VÀ CỔNG CHỜ CANONICAL LOADER (TASK 7.2)
# ================================================================================

def build_cfe_model_v3(
    model_type: str,
    ckpt_path: Optional[str],
    clip_path: str,
    device: torch.device,
    is_dummy: bool = False
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Khởi tạo mô hình tuân thủ Khế ước CFE:
      - KHÔNG strict=False.
      - Sử dụng Canonical Loader từ Task 7.2 nếu có.
      - Từ chối nạp nếu vi phạm cấu trúc (invalid_layout).
    """
    if is_dummy:
        print(f"[*] [DUMMY] Khởi tạo DummyMockModel cho [{model_type}]...")
        return DummyMockModel(num_classes=2).to(device), {
            "path": "dummy", "status": "dummy_ok", "missing": 0, "unexpected": 0
        }

    # Cổng chờ Canonical Loader từ Task 7.2
    try:
        from src.models.canonical_loader import load_canonical
        HAS_CANONICAL = True
    except ImportError:
        HAS_CANONICAL = False

    if not HAS_CANONICAL:
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"\n[!] CẢNH BÁO TỪ CHỐI NẠP: Chưa tìm thấy src/models/canonical_loader.py (thuộc Task 7.2 của Thành viên B).")
            print(f"    Theo Khế ước đo lường CFE v3 (Phụ lục A.5), nghiêm cấm nạp bằng strict=False.")
            print(f"    Vui lòng đợi Thành viên B hoàn thành Task 7.2 hoặc chạy cờ --test_dummy để kiểm thử metric.")
            sys.exit(2)
        else:
            print(f"[!] Không tìm thấy file checkpoint tại: {ckpt_path}")
            sys.exit(1)

    from src.models import build_model
    use_modules = (model_type != "baseline")
    model_args = SimpleNamespace(
        backbone="CLIP:ViT-L/14",
        clip_path=clip_path,
        num_classes=2,
        use_srm=use_modules,
        use_gating=use_modules,
        srm_out_dim=1024,
        num_context_embedding=8,
    )

    model, load_report = load_canonical(ckpt_path, model_args)
    assert load_report.get("missing", 0) == 0 and load_report.get("unexpected", 0) == 0, (
        f"Vi phạm Khế ước nạp: Missing={load_report.get('missing')}, Unexpected={load_report.get('unexpected')}"
    )
    model = model.to(device)
    model.eval()
    return model, load_report


# ================================================================================
# QUY TRÌNH SUY LUẬN & ĐO ĐẠC
# ================================================================================

@torch.no_grad()
def infer_records_v3(
    model: nn.Module,
    records: List[Dict[str, Any]],
    condition: str,
    degrade_order: str,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 2,
    temperature: float = 1.0,
    threshold: float = 0.50
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Suy luận trên danh sách ảnh và tính toán toàn bộ chỉ số v3, kèm logits thô."""
    if not records:
        empty_m = calculate_metrics_v3(np.array([]), np.array([]), threshold=threshold)
        return empty_m, torch.empty((0, 2)), torch.empty((0,)), torch.empty((0,)), torch.empty((0, 2))

    ds = CommunityForensicsDataset(records, condition=condition, degrade_order=degrade_order)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    all_logits = []
    all_targets = []
    all_blockiness = []

    for imgs, labels, _, b_scores in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        all_logits.append(outputs.cpu().float())
        all_targets.append(labels)
        all_blockiness.append(b_scores.float())

    logits_cat = torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, 2))
    targets_cat = torch.cat(all_targets, dim=0) if all_targets else torch.empty((0,), dtype=torch.long)
    blockiness_cat = torch.cat(all_blockiness, dim=0) if all_blockiness else torch.empty((0,))

    scaled_logits = logits_cat / max(temperature, 1e-4)
    probs = F.softmax(scaled_logits, dim=-1)
    fake_probs_np = probs[:, 1].numpy()
    targets_np = targets_cat.numpy()

    metrics = calculate_metrics_v3(fake_probs_np, targets_np, threshold=threshold)
    return metrics, probs, targets_cat, blockiness_cat, logits_cat


def compute_unified_router_predictions_v3(
    probs_wise: torch.Tensor,
    probs_anchored: torch.Tensor,
    blockiness: torch.Tensor,
    targets: torch.Tensor,
    fixed_threshold: float = 0.50,
    router_params_path: Optional[str] = None
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """
    Bộ định tuyến suy biến tự động (Unified Auto-Router):
      - ĐÃ BỎ adaptive_thresh theo batch: Sử dụng fixed_threshold cố định theo khế ước.
      - Tính trọng số w_deg(x) trên từng ảnh riêng biệt.
    """
    w_deg = compute_degradation_weight(blockiness).unsqueeze(1)
    unified_probs = (1.0 - w_deg) * probs_wise + w_deg * probs_anchored

    fake_probs_np = unified_probs[:, 1].numpy()
    targets_np = targets.numpy()

    metrics = calculate_metrics_v3(fake_probs_np, targets_np, threshold=fixed_threshold)
    return metrics, unified_probs


def scan_cfe_dataset(data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Quét dữ liệu CFE từ 5 tập con."""
    records_by_subset = {}
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}

    for sub in CFE_SUBSETS:
        records_by_subset[sub] = []
        sub_dir = os.path.join(data_dir, sub)
        if not os.path.exists(sub_dir):
            continue

        for lbl, folder in [(0, "0_real"), (1, "1_fake")]:
            target_dir = os.path.join(sub_dir, folder)
            if os.path.exists(target_dir):
                for fname in sorted(os.listdir(target_dir)):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in valid_exts:
                        records_by_subset[sub].append({
                            "path": os.path.join(target_dir, fname),
                            "label": lbl,
                            "subset": sub,
                            "fname": fname
                        })

    return records_by_subset


def create_dummy_cfe_data(temp_root: str):
    """Tạo dữ liệu giả lập cho 5 tập con để kiểm thử dummy."""
    for sub in CFE_SUBSETS:
        r_dir = os.path.join(temp_root, sub, "0_real")
        f_dir = os.path.join(temp_root, sub, "1_fake")
        os.makedirs(r_dir, exist_ok=True)
        os.makedirs(f_dir, exist_ok=True)
        for i in range(4):
            Image.new("RGB", (224, 224), color=(30 + i * 10, 40, 50)).save(os.path.join(r_dir, f"dummy_r_{i:04d}.png"))
            Image.new("RGB", (224, 224), color=(150, 100 + i * 10, 200)).save(os.path.join(f_dir, f"dummy_f_{i:04d}.png"))


# ================================================================================
# XUẤT BÁO CÁO MARKDOWN & JSON SCHEMA V3
# ================================================================================

def generate_markdown_report_v3(
    all_results: Dict[str, Any],
    git_commit: str,
    is_dirty: bool,
    degrade_order: str,
    threshold: float
) -> str:
    """Tạo báo cáo Markdown chuẩn hóa v3 có cột TPR@FPR=10% và dòng Trạng thái nạp."""
    md = []
    md.append("# BÁO CÁO KẾT QUẢ KIỂM ĐỊNH NGOẠI VI: COMMUNITYFORENSICS-EVAL (CVPR 2025) - v3.0")
    md.append(f"> **Dự án**: FatFormer-XLA | **Thời gian**: {datetime.now(timezone.utc).isoformat()}")
    md.append(f"> **Git Commit**: `{git_commit}` {'*(Dirty)*' if is_dirty else '*(Clean)*'} | **Thứ tự nén**: `{degrade_order}` | **Ngưỡng T***: `{threshold}`")
    md.append("> **Quy chuẩn**: Bắt buộc kèm 95% CI và đầy đủ TPR@FPR 1/5/10% theo Phụ lục A.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("### Bảng A: Kết Quả Kiểm Chuẩn Trên Miền Ảnh Sạch (Clean CFE)")
    md.append("")
    md.append("| Họ Kiến Trúc / Tập Con | Số Mẫu | Macro ACC | Real ACC | Fake ACC | ROC-AUC | AP | TPR@1% | TPR@5% | **TPR@10%** |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    clean_subsets = all_results.get("clean", {}).get("baseline", {}).get("by_subset", {})
    for sub in CFE_SUBSETS:
        sub_desc = SUBSET_DESCRIPTIONS[sub]
        m = clean_subsets.get(sub, {})
        n_samp = m.get("total_samples", 0)
        tpr1 = m.get("tpr_at_fpr", {}).get("0.01", "—")
        tpr5 = m.get("tpr_at_fpr", {}).get("0.05", "—")
        tpr10 = m.get("tpr_at_fpr", {}).get("0.10", "—")
        md.append(
            f"| **{sub_desc}** | {n_samp} | "
            f"{m.get('macro_acc', 0.0):.2f}% | "
            f"{m.get('real_acc', '—')}% | "
            f"{m.get('fake_acc', '—')}% | "
            f"{m.get('auc', '—')}% | "
            f"{m.get('ap', '—')}% | "
            f"{tpr1}% | {tpr5}% | **{tpr10}%** |"
        )

    ov = all_results.get("clean", {}).get("baseline", {}).get("overall", {})
    tpr1_ov = ov.get("tpr_at_fpr", {}).get("0.01", "—")
    tpr5_ov = ov.get("tpr_at_fpr", {}).get("0.05", "—")
    tpr10_ov = ov.get("tpr_at_fpr", {}).get("0.10", "—")

    md.append(
        f"| 🏆 **OVERALL POOLED** | {ov.get('total_samples', 0)} | "
        f"**{ov.get('macro_acc', 0.0):.2f}%** | "
        f"**{ov.get('real_acc', '—')}%** | "
        f"**{ov.get('fake_acc', '—')}%** | "
        f"**{ov.get('auc', '—')}%** | "
        f"**{ov.get('ap', '—')}%** | "
        f"{tpr1_ov}% | {tpr5_ov}% | **{tpr10_ov}%** |"
    )

    ci = ov.get("ci95", {})
    md.append("")
    md.append(f"> **Khoảng tin cậy 95% CI (Bootstrap 1.000)**: AUC: `{ci.get('auc', ['—','—'])}` | AP: `{ci.get('ap', ['—','—'])}` | TPR@5%: `{ci.get('tpr_at_fpr_0.05', ['—','—'])}`")
    md.append(f"> **Trạng thái nạp Checkpoint**: `ok` (0 missing, 0 unexpected keys)")
    md.append("")
    md.append("---")
    return "\n".join(md)


# ================================================================================
# HÀM ĐIỀU PHỐI CHÍNH (MAIN ENTRYPOINT)
# ================================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="FatFormer-XLA CommunityForensics-Eval Benchmark Pipeline v3.0")
    parser.add_argument("--data_dir", type=str, default="/content/dataset_local/test_communityforensics")
    parser.add_argument("--clip_path", type=str, default="")
    parser.add_argument("--checkpoint", "-c", type=str, default="")
    parser.add_argument("--baseline_ckpt", type=str, default="")
    parser.add_argument("--wise_ckpt", type=str, default="")
    parser.add_argument("--anchored_ckpt", type=str, default="")
    parser.add_argument("--model_type", type=str, default="baseline",
                        choices=["baseline", "anchored", "wise_ft_v2", "candidate", "custom"],
                        help="Loại kiến trúc mô hình (baseline, anchored, wise_ft_v2, candidate)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Tên định danh mô hình cho tệp đầu ra (mặc định lấy theo model_type)")
    parser.add_argument("--condition", type=str, default="clean",
                        choices=["clean", "q30", "jpeg_q30", "all"],
                        help="Điều kiện ảnh: clean, q30, hoặc all (cả clean và q30)")
    parser.add_argument("--save_logits_npz", action="store_true", default=True,
                        help="Tự động xuất tệp NPZ chứa logits [N,2] và targets [N]")
    parser.add_argument("--no_save_logits_npz", dest="save_logits_npz", action="store_false")
    parser.add_argument("--export_candidate_eval", type=str, default=None,
                        help="Đường dẫn xuất tệp sidecar _eval.json cho candidate sweep 7.5")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default="DATASET/logs/phase7")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Ngưỡng cố định T* (mặc định đọc từ calibration hoặc 0.50)")
    parser.add_argument("--calibration", type=str, default=None, help="Đường dẫn file calibration.json (Task 7.4)")
    parser.add_argument("--degrade_order", type=str, default="post_resize", choices=["post_resize", "pre_resize"],
                        help="Thứ tự nén ảnh (mặc định post_resize theo chuẩn CFE)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên cho bootstrap CI")
    parser.add_argument("--test_dummy", action="store_true", help="Chạy chế độ kiểm thử mảng giả và dummy pipeline trên CPU")
    return parser.parse_args()


def main():
    args = parse_args()
    device_str = "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    device = torch.device(device_str)

    git_commit, is_dirty = get_git_info()
    model_name = args.model_name or args.model_type or "baseline"

    print("=" * 88)
    print("FATFORMER-XLA: PIPELINE KIỂM CHUẨN NGOẠI VI COMMUNITYFORENSICS-EVAL (CHUẨN HÓA v3.0)")
    print("=" * 88)
    print(f"• Mô hình đánh giá   : {model_name} (Kiến trúc: {args.model_type})")
    print(f"• Thiết bị tính toán : {device_str.upper()} ({torch.cuda.get_device_name(0) if device_str == 'cuda' else 'CPU'})")
    print(f"• Git Commit         : {git_commit} {'(DIRTY)' if is_dirty else '(CLEAN)'}")
    print(f"• Thứ tự nén         : {args.degrade_order}")
    print(f"• Random Seed        : {args.seed}")

    # Chạy kiểm thử mảng giả toán học (§5.4)
    run_mock_array_tests()

    # Nạp calibration nếu có
    calib_data = {}
    fixed_thresh = 0.50
    temp_val = 1.0
    thresh_source = "default (0.50)"

    if args.calibration and os.path.exists(args.calibration):
        try:
            with open(args.calibration, "r", encoding="utf-8") as f:
                calib_data = json.load(f)
            temp_val = float(calib_data.get("T", 1.0))
            raw_t_star = calib_data.get("T_star_default", calib_data.get("T_star", 0.50))
            if isinstance(raw_t_star, dict):
                fixed_thresh = float(raw_t_star.get("0.05", 0.50))
            else:
                fixed_thresh = float(raw_t_star)
            thresh_source = f"calibration.json ({args.calibration})"
            print(f"[*] Đã nạp cấu hình hiệu chuẩn: T={temp_val}, T*={fixed_thresh}")
        except Exception as e:
            print(f"[!] Cảnh báo nạp calibration thất bại: {e}")

    if args.threshold is not None:
        fixed_thresh = args.threshold
        thresh_source = f"CLI (--threshold {args.threshold})"

    print(f"• Ngưỡng phán quyết  : T* = {fixed_thresh} (Nguồn: {thresh_source})")

    # Dữ liệu
    if args.test_dummy:
        if os.path.normpath(args.output_dir) == os.path.normpath("DATASET/logs/phase7"):
            args.output_dir = "DATASET/logs/phase7/dummy"
        import tempfile
        dummy_dir = tempfile.mkdtemp(prefix="cfe_dummy_")
        create_dummy_cfe_data(dummy_dir)
        records_by_subset = scan_cfe_dataset(dummy_dir)
    else:
        records_by_subset = scan_cfe_dataset(args.data_dir)

    total_scanned = sum(len(recs) for recs in records_by_subset.values())
    if total_scanned == 0 and not args.test_dummy:
        print(f"[!] Lỗi: Không tìm thấy ảnh nào trong {args.data_dir}!")
        return 1

    # Khởi tạo mô hình
    ckpt_target = (
        args.checkpoint
        or (args.baseline_ckpt if args.model_type == "baseline" else "")
        or (args.anchored_ckpt if args.model_type == "anchored" else "")
        or (args.wise_ckpt if args.model_type == "wise_ft_v2" else "")
    )
    model, load_report = build_cfe_model_v3(
        model_type=args.model_type,
        ckpt_path=ckpt_target,
        clip_path=args.clip_path,
        device=device,
        is_dummy=args.test_dummy
    )

    # Xác định danh sách điều kiện cần đánh giá
    if args.condition == "all":
        conditions_to_run = ["clean", "q30"]
    elif args.condition in ["q30", "jpeg_q30"]:
        conditions_to_run = ["q30"]
    else:
        conditions_to_run = ["clean"]

    os.makedirs(args.output_dir, exist_ok=True)
    cond_overall_metrics: Dict[str, Dict[str, Any]] = {}
    all_results_for_md: Dict[str, Dict[str, Any]] = {}

    for cond in conditions_to_run:
        cond_label = "ẢNH SẠCH (CLEAN CFE)" if cond == "clean" else f"NÉN JPEG Q=30 ({cond.upper()})"
        print("\n" + "=" * 88)
        print(f"BẮT ĐẦU ĐÁNH GIÁ ĐIỀU KIỆN: {cond_label} — {total_scanned} ẢNH")
        print("=" * 88)

        subset_metrics = {}
        pooled_probs_list = []
        pooled_targets_list = []
        pooled_logits_list = []
        pooled_sample_ids: List[str] = []

        for sub in CFE_SUBSETS:
            recs = records_by_subset.get(sub, [])
            m_sub, sub_probs, sub_targets, _, sub_logits = infer_records_v3(
                model=model,
                records=recs,
                condition=cond,
                degrade_order=args.degrade_order,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                temperature=temp_val,
                threshold=fixed_thresh
            )
            subset_metrics[sub] = m_sub
            if len(recs) > 0:
                pooled_probs_list.append(sub_probs)
                pooled_targets_list.append(sub_targets)
                pooled_logits_list.append(sub_logits)
                pooled_sample_ids.extend(
                    f"{rec['subset']}/{rec['label']}/{rec['fname']}" for rec in recs
                )

            tpr1 = m_sub.get("tpr_at_fpr", {}).get("0.01", "—")
            tpr5 = m_sub.get("tpr_at_fpr", {}).get("0.05", "—")
            tpr10 = m_sub.get("tpr_at_fpr", {}).get("0.10", "—")
            print(f"  • {sub:<23}: ACC={m_sub['macro_acc']:>5.2f}% | Real={str(m_sub['real_acc']):>5}% | Fake={str(m_sub['fake_acc']):>5}% | AUC={str(m_sub['auc']):>5}% | TPR@5%={str(tpr5):>5}% | TPR@10%={str(tpr10):>5}%")

        if pooled_probs_list:
            all_probs_cat = torch.cat(pooled_probs_list, dim=0)
            all_targets_cat = torch.cat(pooled_targets_list, dim=0)
            all_logits_cat = torch.cat(pooled_logits_list, dim=0)
            all_sample_ids = np.asarray(pooled_sample_ids, dtype=str)

            overall_m = calculate_metrics_v3(all_probs_cat[:, 1].numpy(), all_targets_cat.numpy(), threshold=fixed_thresh)
            ci95 = compute_bootstrap_ci(all_probs_cat[:, 1].numpy(), all_targets_cat.numpy(), n_boot=1000, seed=args.seed)
            overall_m["ci95"] = ci95
        else:
            all_logits_cat = torch.empty((0, 2))
            all_targets_cat = torch.empty((0,), dtype=torch.long)
            all_sample_ids = np.asarray([], dtype=str)
            overall_m = calculate_metrics_v3(np.array([]), np.array([]), threshold=fixed_thresh)

        print(f"\n  🏆 OVERALL POOLED ({cond}): ACC={overall_m['macro_acc']:.2f}% | Real={str(overall_m['real_acc'])}% | Fake={str(overall_m['fake_acc'])}% | Fake@Real90={str(overall_m.get('fake_at_real90'))}% | AUC={str(overall_m['auc'])}% | ECE={overall_m['ece']:.2f}%")
        print(f"     Khoảng tin cậy 95% CI: AUC={overall_m.get('ci95', {}).get('auc')} | AP={overall_m.get('ci95', {}).get('ap')} | TPR@5%={overall_m.get('ci95', {}).get('tpr_at_fpr_0.05')}")

        cond_overall_metrics[cond] = overall_m
        if cond not in all_results_for_md:
            all_results_for_md[cond] = {}
        all_results_for_md[cond][model_name] = {
            "by_subset": subset_metrics,
            "overall": overall_m,
        }

        # Lưu logits và targets NPZ nếu được yêu cầu
        if args.save_logits_npz and len(all_logits_cat) > 0:
            npz_name = f"logits_{model_name}_{cond}.npz"
            npz_path = os.path.join(args.output_dir, npz_name)
            np.savez_compressed(
                npz_path,
                logits=all_logits_cat.numpy(),
                targets=all_targets_cat.numpy(),
                sample_ids=all_sample_ids,
                model=model_name,
                condition=cond,
                temperature=temp_val,
                threshold=fixed_thresh,
            )
            print(f"  [✓] Logits NPZ: {os.path.abspath(npz_path)}")

        # Xuất file schema v3 JSON & MD cho condition này
        schema_v3_output = {
            "report_schema": "v3-dummy" if args.test_dummy else "v3",
            "evidence_level": "synthetic" if args.test_dummy else "measured",
            "gate_eligible": not args.test_dummy and not is_dirty,
            "git_commit": git_commit,
            "git_dirty": is_dirty,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "degrade_order": args.degrade_order,
            "dataset": {"name": "communityforensics_eval", "n": total_scanned, "condition": cond},
            "checkpoints": [{
                "path": ckpt_target or "dummy",
                "load_report": load_report,
                "status": "ok" if args.test_dummy else load_report.get("status", "ok")
            }],
            "calibration": {
                "source": thresh_source,
                "T": temp_val,
                "T_star": fixed_thresh,
                "fpr_target": calib_data.get("fpr_target", 0.05)
            },
            "metrics": overall_m,
            "per_subset": subset_metrics
        }

        json_path = os.path.join(args.output_dir, f"eval_v3_{model_name}_{cond}.json")
        md_path = os.path.join(args.output_dir, f"eval_v3_{model_name}_{cond}.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(schema_v3_output, f, indent=2, ensure_ascii=False)

        report_md = generate_markdown_report_v3(
            all_results={cond: {model_name: all_results_for_md[cond][model_name]}},
            git_commit=git_commit,
            is_dirty=is_dirty,
            degrade_order=args.degrade_order,
            threshold=fixed_thresh
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        print(f"  [✓] JSON v3   : {os.path.abspath(json_path)}")
        print(f"  [✓] MD v3     : {os.path.abspath(md_path)}")

    # Xuất sidecar cho WiSE-FT Candidate Sweep nếu có đủ clean và q30
    if args.export_candidate_eval or (args.model_type == "candidate" and "clean" in cond_overall_metrics and "q30" in cond_overall_metrics):
        sidecar_path = args.export_candidate_eval
        if not sidecar_path and ckpt_target:
            sidecar_path = ckpt_target.replace(".pth", "_eval.json")
        if sidecar_path:
            clean_m = cond_overall_metrics.get("clean", {})
            q30_m = cond_overall_metrics.get("q30", {})
            sidecar_data = {
                "schema": "candidate_eval/v1",
                "evidence_level": "synthetic" if args.test_dummy else "measured",
                "candidate": os.path.basename(ckpt_target or model_name),
                "checkpoint": ckpt_target,
                "metrics": {
                    "real_acc_clean": float(clean_m.get("real_acc", 0.0) or 0.0),
                    "fake_at_real90_clean": float(clean_m.get("fake_at_real90", 0.0) or clean_m.get("tpr_at_fpr", {}).get("0.10", 0.0) or 0.0),
                    "auc_clean": float(clean_m.get("auc", 0.0) or 0.0),
                    "real_acc_q30": float(q30_m.get("real_acc", 0.0) or 0.0),
                    "fake_at_real90_q30": float(q30_m.get("fake_at_real90", 0.0) or q30_m.get("tpr_at_fpr", {}).get("0.10", 0.0) or 0.0),
                    "auc_q30": float(q30_m.get("auc", 0.0) or 0.0),
                }
            }
            os.makedirs(os.path.dirname(os.path.abspath(sidecar_path)), exist_ok=True)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar_data, f, indent=2, ensure_ascii=False)
            print(f"\n[✓] Đã xuất sidecar ứng viên WiSE-FT: {sidecar_path}")

    print("\n" + "=" * 88)
    print("HOÀN TẤT THỰC THI EVALUATION v3 THEO ĐÚNG KHẾ ƯỚC PHỤ LỤC A!")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
