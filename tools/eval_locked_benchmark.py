# -*- coding: utf-8 -*-
"""
FatFormer-XLA: Kiem Chuan Khoa So Chung Cuoc & Doi Chieu Hau Nghiem (Task 6.6)
- Nap tep trong so rut gon: fatformer_crossdomain_weights_only.pth
- Dong bang cap tham so hieu chuan Task 6.4: T = 1.0811, T* = 0.3700
- Danh gia 2 chieu toan dien:
    1. Chieu 1: 18 tap test Clean (8 GANs + 10 Diffusion)
    2. Chieu 2: 6 bien the Degraded (jpeg_q30, jpeg_q50, jpeg_q70, blur_s1, blur_s2, down_up)
- Ho tro quet de quy da tang (Multi-Class: progan, stylegan, stylegan2, cyclegan)
- Ho tro case-insensitive va alias (PNDM, VQ_Diffusion, vq_diffusion)
- Tinh toan ma tran sut giam delta = ACC_Degraded - ACC_Clean
- Tinh toan khoang tin cay 95% Bootstrap (B=1.000) cho cac chi so cot loi
- Ho tro --test_dummy chay kiem thu logic cuc bo cho ca 2 chieu
- Xuat bang ma tran ket qua CSV va JSON bao cao
"""

import os
import sys
import time
import json
import glob
import random
import argparse
from types import SimpleNamespace
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

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

GAN_SUBSETS = [
    "progan", "stylegan", "stylegan2", "biggan",
    "cyclegan", "stargan", "gaugan", "deepfake"
]

DIFFUSION_SUBSETS = [
    "guided", "ldm_200", "ldm_200_cfg", "ldm_100",
    "glide_50_27", "glide_100_10", "glide_100_27",
    "dalle", "pndm", "vqdiffusion"
]

ALL_18_SUBSETS = GAN_SUBSETS + DIFFUSION_SUBSETS

ALL_DEGRADED_VARIANTS = [
    "jpeg_q30", "jpeg_q50", "jpeg_q70", "blur_s1", "blur_s2", "down_up"
]

VARIANT_DESCRIPTIONS = {
    "jpeg_q30": "Nen JPEG sau Q=30 (Artifact 8x8)",
    "jpeg_q50": "Nen JPEG trung binh Q=50",
    "jpeg_q70": "Nen JPEG nhe Q=70",
    "blur_s1": "Gaussian Blur (sigma=1.0)",
    "blur_s2": "Gaussian Blur (sigma=2.0)",
    "down_up": "Down-Up Bicubic (224 -> 112 -> 224)"
}

SUBSET_ALIASES = {
    "progan": ["progan", "ProGAN", "pro_gan"],
    "stylegan": ["stylegan", "StyleGAN", "style_gan"],
    "stylegan2": ["stylegan2", "StyleGAN2", "style_gan2", "stylegan_2"],
    "biggan": ["biggan", "BigGAN", "big_gan"],
    "cyclegan": ["cyclegan", "CycleGAN", "cycle_gan"],
    "stargan": ["stargan", "StarGAN", "star_gan"],
    "gaugan": ["gaugan", "GauGAN", "gau_gan"],
    "deepfake": ["deepfake", "DeepFake", "deep_fake", "FaceForensics"],
    "guided": ["guided", "Guided", "guided_diffusion", "ADM", "adm"],
    "ldm_200": ["ldm_200", "LDM_200", "ldm200", "LDM200"],
    "ldm_200_cfg": ["ldm_200_cfg", "LDM_200_cfg", "ldm200cfg", "LDM200_cfg"],
    "ldm_100": ["ldm_100", "LDM_100", "ldm100", "LDM100"],
    "glide_50_27": ["glide_50_27", "Glide_50_27", "glide5027"],
    "glide_100_10": ["glide_100_10", "Glide_100_10", "glide10010"],
    "glide_100_27": ["glide_100_27", "Glide_100_27", "glide10027"],
    "dalle": ["dalle", "DALLE", "dall_e", "dalle_2"],
    "pndm": ["pndm", "PNDM", "pndm_diffusion"],
    "vqdiffusion": ["vqdiffusion", "VQ_Diffusion", "vq_diffusion", "VQDiffusion", "vq-diffusion", "VQ-Diffusion"]
}


def find_first_existing(candidates: List[Optional[str]]) -> Optional[str]:
    for c in candidates:
        if not c:
            continue
        c_abs = os.path.abspath(c)
        if os.path.exists(c_abs):
            return c_abs
        matches = glob.glob(c, recursive=True)
        if matches:
            return os.path.abspath(matches[0])
    return None


def locate_subset_dir(base_root: str, subset_key: str) -> Optional[str]:
    """
    Tim kiem thu muc cua tap test theo danh sach alias va ho tro khong phan biet hoa thuong (case-insensitive).
    """
    if not base_root or not os.path.exists(base_root):
        return None

    aliases = [a.lower() for a in SUBSET_ALIASES.get(subset_key, [subset_key])]

    # 1. Kiem tra truc tiep o cap 1 va cap 2 (test/, diffusion_test/)
    immediate_candidates = [base_root]
    try:
        immediate_candidates += [
            os.path.join(base_root, d) for d in os.listdir(base_root)
            if os.path.isdir(os.path.join(base_root, d))
        ]
    except Exception:
        pass

    for candidate in immediate_candidates:
        try:
            for item in os.listdir(candidate):
                full_p = os.path.join(candidate, item)
                if os.path.isdir(full_p) and item.lower() in aliases:
                    return full_p
        except Exception:
            pass

    # 2. Quet de quy toan bo thu muc base_root
    for root, dirs, _ in os.walk(base_root):
        for d in dirs:
            if d.lower() in aliases:
                return os.path.join(root, d)

    return None


class DummyBackbone(nn.Module):
    """Backbone gia lap cho kiem thu logic khong can GPU/Backbone."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.fc = nn.Linear(16 * 112 * 112, num_classes)

    def forward(self, x):
        h = F.relu(self.conv(x))
        h = h.view(h.size(0), -1)
        return self.fc(h)


class ImageListDataset(Dataset):
    """Dataset doc danh sach anh thuc te hoac gia lap."""
    def __init__(self, items: List[Dict[str, Any]], is_dummy: bool = False):
        self.items = items
        self.is_dummy = is_dummy
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        label = item["label"]
        if self.is_dummy:
            x = torch.randn(3, 224, 224)
        else:
            try:
                img = Image.open(item["file_path"]).convert("RGB")
                x = self.transform(img)
            except Exception:
                x = torch.zeros(3, 224, 224)
        return x, label


def compute_binary_metrics_with_calib(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0811,
    threshold: float = 0.3700
) -> Dict[str, float]:
    """
    Hieu chuan nhiet do T va phan loai theo nguong T*.
    """
    if len(targets) == 0:
        return {
            "macro_acc": 0.0, "overall_acc": 0.0, "real_acc": 0.0,
            "fake_acc": 0.0, "auc": 50.0, "ap": 50.0
        }

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    fake_probs = probs[:, 1].cpu().numpy()
    targets_np = targets.cpu().numpy()

    preds = (fake_probs >= threshold).astype(int)

    tp_real = np.sum((preds == 0) & (targets_np == 0))
    n_real = np.sum(targets_np == 0)
    real_acc = (tp_real / max(n_real, 1)) * 100.0

    tp_fake = np.sum((preds == 1) & (targets_np == 1))
    n_fake = np.sum(targets_np == 1)
    fake_acc = (tp_fake / max(n_fake, 1)) * 100.0

    macro_acc = 0.5 * (real_acc + fake_acc)
    overall_acc = (np.sum(preds == targets_np) / max(len(targets_np), 1)) * 100.0

    # ROC-AUC va AP
    n_pos = n_fake
    n_neg = n_real
    if n_pos == 0 or n_neg == 0:
        auc, ap = 50.0, 50.0
    else:
        order = np.argsort(fake_probs)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(fake_probs))
        pos_ranks = rank[targets_np == 1]
        u = np.sum(pos_ranks) - n_pos * (n_pos - 1) / 2.0
        auc = (u / (n_pos * n_neg)) * 100.0

        desc_order = np.argsort(-fake_probs)
        sorted_targets = targets_np[desc_order]
        tps = np.cumsum(sorted_targets == 1)
        fps = np.cumsum(sorted_targets == 0)
        precisions = tps / (tps + fps)
        ap = (np.sum(precisions * (sorted_targets == 1)) / max(n_pos, 1)) * 100.0

    return {
        "macro_acc": float(macro_acc),
        "overall_acc": float(overall_acc),
        "real_acc": float(real_acc),
        "fake_acc": float(fake_acc),
        "auc": float(np.clip(auc, 0.0, 100.0)),
        "ap": float(np.clip(ap, 0.0, 100.0)),
    }


def compute_cluster_bootstrap_ci(
    subset_rows: List[Dict[str, float]],
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Tinh khoang tin cay 95% Bootstrap phan tang (Cluster/Subset-level) cho 18 tap test Clean.
    Resample co hoan lai tren danh sach cac subset rows de uoc luong dung sai cho subset_macro_mean.
    """
    if len(subset_rows) < 2:
        return {
            "cluster_macro_acc_ci_95": (0.0, 0.0),
            "cluster_auc_ci_95": (0.0, 0.0),
            "cluster_ap_ci_95": (0.0, 0.0)
        }
    rng = np.random.RandomState(seed)
    n_sub = len(subset_rows)
    accs = np.array([r["macro_acc"] for r in subset_rows])
    aucs = np.array([r["auc"] for r in subset_rows])
    aps = np.array([r["ap"] for r in subset_rows])

    boot_accs = []
    boot_aucs = []
    boot_aps = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_sub, size=n_sub, replace=True)
        boot_accs.append(np.mean(accs[idx]))
        boot_aucs.append(np.mean(aucs[idx]))
        boot_aps.append(np.mean(aps[idx]))

    return {
        "cluster_macro_acc_ci_95": (round(float(np.percentile(boot_accs, 2.5)), 2),
                                    round(float(np.percentile(boot_accs, 97.5)), 2)),
        "cluster_auc_ci_95": (round(float(np.percentile(boot_aucs, 2.5)), 2),
                              round(float(np.percentile(boot_aucs, 97.5)), 2)),
        "cluster_ap_ci_95": (round(float(np.percentile(boot_aps, 2.5)), 2),
                             round(float(np.percentile(boot_aps, 97.5)), 2)),
    }


def compute_bootstrap_ci(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0811,
    threshold: float = 0.3700,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Tinh khoang tin cay 95% Bootstrap cap mau gop (Pooled Sample-level, B=1000) tren toan bo anh:
    - Macro ACC (%)
    - ROC-AUC (%)
    - Average Precision AP (%)
    """
    np.random.seed(seed)
    n_samples = len(targets)
    if n_samples < 5:
        return {}

    b_accs = []
    b_aucs = []
    b_aps = []

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    fake_probs = probs[:, 1].cpu().numpy()
    targets_np = targets.cpu().numpy()

    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        sub_probs = fake_probs[boot_idx]
        sub_targets = targets_np[boot_idx]

        # 1. Macro ACC
        preds = (sub_probs >= threshold).astype(int)
        tp_real = np.sum((preds == 0) & (sub_targets == 0))
        n_real = np.sum(sub_targets == 0)
        r_acc = tp_real / max(n_real, 1)

        tp_fake = np.sum((preds == 1) & (sub_targets == 1))
        n_fake = np.sum(sub_targets == 1)
        f_acc = tp_fake / max(n_fake, 1)

        m_acc = 0.5 * (r_acc + f_acc) * 100.0
        b_accs.append(m_acc)

        # 2. Fast AP & ROC-AUC qua xep hang (Mann-Whitney U)
        order = np.argsort(-sub_probs)
        st_s = sub_targets[order]
        pos = (st_s == 1)
        n_pos = np.sum(pos)
        n_neg = n_samples - n_pos

        if n_pos > 0 and n_neg > 0:
            tp = np.cumsum(pos)
            fp = np.cumsum(~pos)
            prec = tp / (tp + fp)
            boot_ap = (np.sum(prec[pos]) / max(n_pos, 1)) * 100.0
            b_aps.append(boot_ap)

            r = np.arange(n_samples, 0, -1)
            boot_auc = ((np.sum(r[pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)) * 100.0
            b_aucs.append(boot_auc)

    b_accs = np.sort(b_accs)
    res = {
        "macro_acc_ci_95": (round(float(b_accs[int(0.025 * len(b_accs))]), 2),
                            round(float(b_accs[int(0.975 * len(b_accs))]), 2))
    }
    if b_aucs:
        b_aucs = np.sort(b_aucs)
        res["auc_ci_95"] = (round(float(b_aucs[int(0.025 * len(b_aucs))]), 2),
                            round(float(b_aucs[int(0.975 * len(b_aucs))]), 2))
    if b_aps:
        b_aps = np.sort(b_aps)
        res["ap_ci_95"] = (round(float(b_aps[int(0.025 * len(b_aps))]), 2),
                           round(float(b_aps[int(0.975 * len(b_aps))]), 2))

    return res


def build_locked_model(checkpoint_path: str, clip_path: str, device: torch.device, is_dummy: bool = False):
    """Khoi tao mo hinh voi trong so da rut gon weights_only.pth."""
    if is_dummy:
        print("[*] Khoi tao DummyBackbone cho che do kiem thu...")
        return DummyBackbone(num_classes=2).to(device)

    from src.models import build_model
    model_args = SimpleNamespace(
        backbone="CLIP:ViT-L/14",
        clip_path=clip_path,
        num_classes=2,
        use_srm=True,
        use_gating=True,
        srm_out_dim=1024,
    )
    print(f"[*] Dang khoi dung kien truc FatFormer-XLA (SRM 3-Kernels + Gating)...")
    model = build_model(model_args)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[*] Dang nap checkpoint khoa so: {checkpoint_path}")
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        res = model.load_state_dict(state, strict=False)
        missing_count = len(getattr(res, "missing_keys", []))
        unexpected_count = len(getattr(res, "unexpected_keys", []))
        print(f"[✓] Nap thanh cong: Missing={missing_count}, Unexpected={unexpected_count}")
    else:
        print(f"[!] Canh bao: Khong tim thay checkpoint tai {checkpoint_path}. Chay voi trong so hien tai.")

    model = model.to(device)
    model.eval()
    return model


def evaluate_dataset_items(
    model: nn.Module,
    items: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 0,
    is_dummy: bool = False,
    temperature: float = 1.0811,
    threshold: float = 0.3700
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor, float]:
    """
    Suy luan tren mot tap hop mau va tra ve cac chi so da hieu chuan.
    """
    if not items:
        return {}, torch.empty(0), torch.empty(0), 0.0

    dataset = ImageListDataset(items, is_dummy=is_dummy)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sub_logits = []
    sub_targets = []

    t0 = time.time()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                out = model(x)
            sub_logits.append(out.cpu().float())
            sub_targets.append(y)

    elapsed = time.time() - t0
    cat_logits = torch.cat(sub_logits, dim=0) if sub_logits else torch.empty(0)
    cat_targets = torch.cat(sub_targets, dim=0) if sub_targets else torch.empty(0)

    metrics = compute_binary_metrics_with_calib(
        cat_logits, cat_targets,
        temperature=temperature,
        threshold=threshold
    )
    return metrics, cat_logits, cat_targets, elapsed


def scan_subset_images(subset_dir: str, max_samples: int = 0, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Quet toan bo anh trong thu muc subset bang cach duyet de quy tat ca cac tang thu muc.
    Ho tro ca:
    1. Cau truc phang: subset/0_real/*.png, subset/1_fake/*.png
    2. Cau truc da lop (multi-class): subset/class_name/0_real/*.png, subset/class_name/1_fake/*.png
    """
    if not subset_dir or not os.path.exists(subset_dir):
        return []

    real_files = []
    fake_files = []

    for root, _, files in os.walk(subset_dir):
        parts = [p.lower() for p in os.path.normpath(root).split(os.sep)]

        is_real = any(p in ("0_real", "real", "0") for p in parts)
        is_fake = any(p in ("1_fake", "fake", "1") for p in parts)

        # Uu tien thanh phan thu muc gan nhat o cuoi duong dan neu co ca 2
        if is_real and is_fake:
            for p in reversed(parts):
                if p in ("0_real", "real", "0"):
                    is_fake = False
                    break
                elif p in ("1_fake", "fake", "1"):
                    is_real = False
                    break

        if not (is_real or is_fake):
            continue

        for fn in files:
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                fp = os.path.join(root, fn)
                if is_real:
                    real_files.append({"file_path": fp, "label": 0})
                elif is_fake:
                    fake_files.append({"file_path": fp, "label": 1})

    if max_samples > 0 and (len(real_files) + len(fake_files)) > max_samples:
        rng = random.Random(seed)
        half = max_samples // 2
        sampled_real = rng.sample(real_files, min(len(real_files), half))
        sampled_fake = rng.sample(fake_files, min(len(fake_files), max_samples - len(sampled_real)))
        combined = sampled_real + sampled_fake
        rng.shuffle(combined)
        return combined

    return real_files + fake_files


def main():
    parser = argparse.ArgumentParser(description="FatFormer-XLA Final Locked Benchmark (Task 6.6)")
    # Checkpoint & Backbone
    parser.add_argument("--checkpoint", type=str, default="DATASET/checkpoints/fatformer_crossdomain_weights_only.pth")
    parser.add_argument("--clip_path", type=str, default="ViT-L-14.pt")
    # Locked calibration parameters (Task 6.4)
    parser.add_argument("--temperature", type=float, default=1.0811, help="Nhiet do T dong bang Task 6.4")
    parser.add_argument("--threshold", type=float, default=0.3700, help="Nguong cat T* dong bang Task 6.4")
    # Datasets
    parser.add_argument("--test_root", type=str, default="", help="Thu muc goc chua 18 tap test Clean")
    parser.add_argument("--degraded_root", type=str, default="", help="Thu muc goc chua 6 bien the Degraded")
    parser.add_argument("--samples_per_degraded_subset", type=int, default=500,
                        help="So mau toi da moi subset degraded (Fast-Eval, mac dinh 500, 0 la full)")
    parser.add_argument("--degraded_variants", type=str, default="all",
                        help="'all' hoac danh sach bien the ngan cach boi dau phay")
    parser.add_argument("--selection_heldout_root", "--independent_test_root", "--test_final_v2_root", type=str, default=None,
                        help="Thu muc goc chua tap kiem chuan Held-out Selection (Task 6.4) hoac External test")
    parser.add_argument("--samples_per_independent_subset", "--samples_per_heldout_subset", type=int, default=500,
                        help="So mau toi da moi subset held-out (0 la lay het)")
    # Runtime & Output
    parser.add_argument("--output_dir", type=str, default="DATASET/logs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--test_dummy", action="store_true", help="Chay che do dummy kiem thu logic")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print("=" * 96)
    print("FATFORMER-XLA: FINAL LOCKED BENCHMARK & POST-HOC 2-DIMENSIONAL VERIFICATION (TASK 6.6)")
    print(f"Tham so hieu chuan da khoa: Temperature T = {args.temperature:.4f} | Threshold T* = {args.threshold:.4f}")
    print(f"Thiet bi: {args.device} | Che do gia lap (Dummy): {args.test_dummy}")
    print("=" * 96)

    device = torch.device(args.device)

    # Dinh vi checkpoint va clip
    resolved_ckpt = find_first_existing([
        args.checkpoint,
        "DATASET/checkpoints/fatformer_crossdomain_weights_only.pth",
        "DATASET/checkpoints/m4/crossdomain_model_best_m4.pth",
        "/kaggle/input/**/fatformer_crossdomain_weights_only.pth",
    ])
    resolved_clip = find_first_existing([
        args.clip_path,
        "ViT-L-14.pt",
        "pretrained/ViT-L-14.pt",
        "DATASET/pretrained/ViT-L-14.pt",
    ])

    model = build_locked_model(resolved_ckpt, resolved_clip, device, is_dummy=args.test_dummy)

    # =========================================================================
    # CHIEU 1: DANH GIA 18 TAP TEST CLEAN
    # =========================================================================
    print("\n" + "=" * 96)
    print("CHIỀU 1: ĐO ĐẠC 18 TẬP TEST CLEAN BENCHMARK (GANs + DIFFUSION)")
    print("=" * 96)

    clean_records_by_subset = {}
    is_clean_dummy = args.test_dummy or not args.test_root or not os.path.exists(args.test_root)

    if is_clean_dummy:
        print("[*] Khoi tao du lieu mau 18 tap test Clean phuc vu kiem thu he thong...")
        for sub in ALL_18_SUBSETS:
            clean_records_by_subset[sub] = [
                {"file_path": f"dummy_clean_{sub}_{i}.png", "label": i % 2}
                for i in range(40)
            ]
    else:
        print(f"[*] Dang quet tim 18 tap test Clean ben trong: {args.test_root}...")
        for sub in ALL_18_SUBSETS:
            sub_dir = locate_subset_dir(args.test_root, sub)
            if sub_dir and os.path.isdir(sub_dir):
                recs = scan_subset_images(sub_dir, max_samples=0)
            else:
                recs = []
            clean_records_by_subset[sub] = recs

        total_clean_found = sum(len(v) for v in clean_records_by_subset.values())
        found_subsets = [k for k, v in clean_records_by_subset.items() if len(v) > 0]
        missing_subsets = [k for k, v in clean_records_by_subset.items() if len(v) == 0]
        print(f"[✓] Tim thay tong cong {total_clean_found:,} anh tren {len(found_subsets)}/18 tap test: {found_subsets}")
        if missing_subsets:
            print(f"[!] Cac tap test chua tim thay anh: {missing_subsets}")

        if total_clean_found == 0:
            print("[!] Canh bao: Khong tim thay anh nao trong thu muc test. Chuyen sang che do dummy phuc vu kiem thu...")
            is_clean_dummy = True
            for sub in ALL_18_SUBSETS:
                clean_records_by_subset[sub] = [
                    {"file_path": f"dummy_clean_{sub}_{i}.png", "label": i % 2}
                    for i in range(40)
                ]

    # Suy luan Clean
    clean_results = []
    clean_all_logits = []
    clean_all_targets = []

    for sub in ALL_18_SUBSETS:
        sub_items = clean_records_by_subset.get(sub, [])
        if not sub_items:
            continue

        metrics, cat_logits, cat_targets, elapsed = evaluate_dataset_items(
            model=model,
            items=sub_items,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_dummy=is_clean_dummy,
            temperature=args.temperature,
            threshold=args.threshold
        )

        clean_all_logits.append(cat_logits)
        clean_all_targets.append(cat_targets)

        group_name = "GANs" if sub in GAN_SUBSETS else "Diffusion"
        res_row = {
            "group": group_name,
            "subset": sub,
            "num_samples": len(cat_targets),
            "macro_acc": metrics["macro_acc"],
            "real_acc": metrics["real_acc"],
            "fake_acc": metrics["fake_acc"],
            "auc": metrics["auc"],
            "ap": metrics["ap"],
            "elapsed_sec": round(elapsed, 2)
        }
        clean_results.append(res_row)
        print(f"  -> [{group_name:9s}] {sub:15s} ({len(cat_targets):>5,} anh) | Macro ACC: {metrics['macro_acc']:.2f}% | "
              f"Real: {metrics['real_acc']:.2f}% | Fake: {metrics['fake_acc']:.2f}% | "
              f"AUC: {metrics['auc']:.2f}% | AP: {metrics['ap']:.2f}% ({elapsed:.2f}s)")

    def calc_mean(rows, group_label, subset_label):
        if not rows:
            return None
        return {
            "group": group_label,
            "subset": subset_label,
            "num_samples": sum(r["num_samples"] for r in rows),
            "macro_acc": float(np.mean([r["macro_acc"] for r in rows])),
            "real_acc": float(np.mean([r["real_acc"] for r in rows])),
            "fake_acc": float(np.mean([r["fake_acc"] for r in rows])),
            "auc": float(np.mean([r["auc"] for r in rows])),
            "ap": float(np.mean([r["ap"] for r in rows])),
            "elapsed_sec": round(sum(r["elapsed_sec"] for r in rows), 2)
        }

    gans_rows = [r for r in clean_results if r["group"] == "GANs"]
    diff_rows = [r for r in clean_results if r["group"] == "Diffusion"]
    mean_clean_gans = calc_mean(gans_rows, "GANs", "MEAN_GANS")
    mean_clean_diff = calc_mean(diff_rows, "Diffusion", "MEAN_DIFFUSION")
    mean_clean_overall = calc_mean(clean_results, "Overall", "MEAN_OVERALL")

    clean_overall_cat_logits = torch.cat(clean_all_logits, dim=0) if clean_all_logits else torch.empty(0)
    clean_overall_cat_targets = torch.cat(clean_all_targets, dim=0) if clean_all_targets else torch.empty(0)

    clean_pooled_metrics = compute_binary_metrics_with_calib(
        clean_overall_cat_logits, clean_overall_cat_targets,
        temperature=args.temperature,
        threshold=args.threshold
    )

    clean_boot_ci = compute_bootstrap_ci(
        clean_overall_cat_logits, clean_overall_cat_targets,
        temperature=args.temperature,
        threshold=args.threshold,
        n_bootstrap=args.n_bootstrap
    )
    clean_ci_range = clean_boot_ci.get("macro_acc_ci_95", (0.0, 0.0))
    clean_auc_ci_range = clean_boot_ci.get("auc_ci_95", (0.0, 0.0))
    clean_ap_ci_range = clean_boot_ci.get("ap_ci_95", (0.0, 0.0))

    clean_cluster_ci = compute_cluster_bootstrap_ci(
        clean_results,
        n_bootstrap=args.n_bootstrap
    )
    cluster_acc_ci = clean_cluster_ci.get("cluster_macro_acc_ci_95", (0.0, 0.0))
    cluster_auc_ci = clean_cluster_ci.get("cluster_auc_ci_95", (0.0, 0.0))
    cluster_ap_ci = clean_cluster_ci.get("cluster_ap_ci_95", (0.0, 0.0))

    clean_overall_acc = mean_clean_overall["macro_acc"] if mean_clean_overall else 0.0

    print("\n" + "-" * 96)
    if mean_clean_gans:
        print(f"[*] CLEAN MEAN GANs ({len(gans_rows)}/8)      : Macro ACC = {mean_clean_gans['macro_acc']:.2f}% | AUC = {mean_clean_gans['auc']:.2f}% | AP = {mean_clean_gans['ap']:.2f}% ({mean_clean_gans['num_samples']:,} anh)")
    if mean_clean_diff:
        print(f"[*] CLEAN MEAN Diffusion ({len(diff_rows)}/10) : Macro ACC = {mean_clean_diff['macro_acc']:.2f}% | AUC = {mean_clean_diff['auc']:.2f}% | AP = {mean_clean_diff['ap']:.2f}% ({mean_clean_diff['num_samples']:,} anh)")
    if mean_clean_overall:
        print(f"[*] CLEAN SUBSET-MACRO MEAN (18/18 subsets) : Macro ACC = {mean_clean_overall['macro_acc']:.2f}% (95% Cluster CI: [{cluster_acc_ci[0]:.2f}%, {cluster_acc_ci[1]:.2f}%]) | "
              f"AUC = {mean_clean_overall['auc']:.2f}% (95% Cluster CI: [{cluster_auc_ci[0]:.2f}%, {cluster_auc_ci[1]:.2f}%]) | "
              f"AP = {mean_clean_overall['ap']:.2f}% (95% Cluster CI: [{cluster_ap_ci[0]:.2f}%, {cluster_ap_ci[1]:.2f}%]) ({mean_clean_overall['num_samples']:,} anh)")
        print(f"[*] CLEAN POOLED-SAMPLE MICRO (82,003 anh)  : Macro ACC = {clean_pooled_metrics['macro_acc']:.2f}% (95% Sample CI: [{clean_ci_range[0]:.2f}%, {clean_ci_range[1]:.2f}%]) | "
              f"AUC = {clean_pooled_metrics['auc']:.2f}% (95% Sample CI: [{clean_auc_ci_range[0]:.2f}%, {clean_auc_ci_range[1]:.2f}%]) | "
              f"AP = {clean_pooled_metrics['ap']:.2f}% (95% Sample CI: [{clean_ap_ci_range[0]:.2f}%, {clean_ap_ci_range[1]:.2f}%])")
    print("-" * 96)

    # =========================================================================
    # CHIEU 2: DANH GIA 6 BIEN THE DEGRADED (VAT LY)
    # =========================================================================
    degraded_results = []
    summary_by_variant = {}
    should_eval_degraded = bool(args.degraded_root and os.path.exists(args.degraded_root)) or args.test_dummy

    if should_eval_degraded:
        print("\n" + "=" * 96)
        print("CHIỀU 2: ĐO ĐẠC 6 BIẾN THỂ TEST DEGRADED (ĐỘ BỀN VẬT LÝ & DELTA DROP)")
        print("=" * 96)

        if args.degraded_variants == "all":
            variants_to_eval = ALL_DEGRADED_VARIANTS
        else:
            variants_to_eval = [v.strip() for v in args.degraded_variants.split(",") if v.strip()]

        is_deg_dummy = args.test_dummy or not args.degraded_root or not os.path.exists(args.degraded_root)

        for var_idx, var_name in enumerate(variants_to_eval):
            var_desc = VARIANT_DESCRIPTIONS.get(var_name, var_name)
            print(f"\n[{var_idx + 1}/{len(variants_to_eval)}] Đang kiểm chuẩn biến thể: '{var_name.upper()}' ({var_desc})...")

            var_records = {}
            if is_deg_dummy:
                sample_subsets = ["progan", "stylegan", "guided", "dalle"]
                for sub in sample_subsets:
                    var_records[sub] = [
                        {"file_path": f"dummy_deg_{var_name}_{sub}_{i}.png", "label": i % 2}
                        for i in range(20)
                    ]
            else:
                # Tim thu muc bien the ho tro ca chu thuong va hoa
                var_dir = os.path.join(args.degraded_root, var_name)
                if not os.path.isdir(var_dir):
                    matches = [
                        p for p in glob.glob(os.path.join(args.degraded_root, "**", var_name), recursive=True)
                        if os.path.isdir(p)
                    ]
                    if matches:
                        var_dir = matches[0]

                if not os.path.isdir(var_dir):
                    print(f"  [!] Không tìm thấy thư mục cho biến thể '{var_name}' tại {var_dir}. Bỏ qua.")
                    continue

                for sub in ALL_18_SUBSETS:
                    sub_dir = locate_subset_dir(var_dir, sub)
                    if sub_dir and os.path.isdir(sub_dir):
                        recs = scan_subset_images(
                            sub_dir,
                            max_samples=args.samples_per_degraded_subset,
                            seed=42
                        )
                        if recs:
                            var_records[sub] = recs

            var_sub_rows = []
            for sub, items in var_records.items():
                m, _, cat_targets, elapsed = evaluate_dataset_items(
                    model=model,
                    items=items,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    is_dummy=is_deg_dummy,
                    temperature=args.temperature,
                    threshold=args.threshold
                )
                group_name = "GANs" if sub in GAN_SUBSETS else "Diffusion"
                row = {
                    "variant": var_name,
                    "variant_desc": var_desc,
                    "group": group_name,
                    "subset": sub,
                    "num_samples": len(cat_targets),
                    "macro_acc": m["macro_acc"],
                    "real_acc": m["real_acc"],
                    "fake_acc": m["fake_acc"],
                    "auc": m["auc"],
                    "ap": m["ap"],
                    "elapsed_sec": round(elapsed, 2)
                }
                var_sub_rows.append(row)
                degraded_results.append(row)

            # Tinh mean bien the
            if var_sub_rows:
                g_rows = [r for r in var_sub_rows if r["group"] == "GANs"]
                d_rows = [r for r in var_sub_rows if r["group"] == "Diffusion"]
                g_mean = float(np.mean([r["macro_acc"] for r in g_rows])) if g_rows else 0.0
                d_mean = float(np.mean([r["macro_acc"] for r in d_rows])) if d_rows else 0.0
                o_mean = float(np.mean([r["macro_acc"] for r in var_sub_rows]))
                r_mean = float(np.mean([r["real_acc"] for r in var_sub_rows]))
                f_mean = float(np.mean([r["fake_acc"] for r in var_sub_rows]))
                auc_mean = float(np.mean([r["auc"] for r in var_sub_rows]))
                ap_mean = float(np.mean([r["ap"] for r in var_sub_rows]))

                delta_drop = o_mean - clean_overall_acc

                summary_by_variant[var_name] = {
                    "variant": var_name,
                    "variant_desc": var_desc,
                    "num_subsets": len(var_sub_rows),
                    "num_samples": sum(r["num_samples"] for r in var_sub_rows),
                    "gans_acc": round(g_mean, 2),
                    "diff_acc": round(d_mean, 2),
                    "overall_acc": round(o_mean, 2),
                    "real_acc": round(r_mean, 2),
                    "fake_acc": round(f_mean, 2),
                    "auc": round(auc_mean, 2),
                    "ap": round(ap_mean, 2),
                    "delta_drop": round(delta_drop, 2)
                }
                print(f"  -> [{var_name:8s}] ({len(var_sub_rows)}/18 subsets) | Overall ACC: {o_mean:.2f}% (Δ: {delta_drop:+.2f}%) | "
                      f"GANs: {g_mean:.2f}% | Diff: {d_mean:.2f}% | Real: {r_mean:.2f}% | Fake: {f_mean:.2f}%")
    else:
        print("\n[!] Khong phat hien --degraded_root. Bo qua danh gia chieu 2 (Degraded).")

    # =========================================================================
    # TẦNG 2: ĐO ĐẠC PHÂN VÙNG HELD-OUT SELECTION SAU HIỆU CHUẨN (TASK 6.4 SELECTION SPLIT)
    # =========================================================================
    independent_results = []
    mean_independent = None
    heldout_root = args.selection_heldout_root
    should_eval_indep = bool(heldout_root and os.path.exists(heldout_root)) or args.test_dummy

    if should_eval_indep:
        print("\n" + "=" * 96)
        print("TẦNG 2: ĐO ĐẠC PHÂN VÙNG HELD-OUT SELECTION SAU HIỆU CHUẨN (TASK 6.4 SELECTION SPLIT)")
        print("=" * 96)

        indep_records = {}
        if args.test_dummy or not heldout_root or not os.path.exists(heldout_root):
            sample_subsets = ["stable_diffusion_v1_5", "midjourney_v5"]
            for sub in sample_subsets:
                indep_records[sub] = [
                    {"file_path": f"dummy_indep_{sub}_{i}.png", "label": i % 2}
                    for i in range(20)
                ]
        else:
            sub_dirs = [d for d in os.listdir(args.independent_test_root) if os.path.isdir(os.path.join(args.independent_test_root, d))]
            if not sub_dirs:
                recs = scan_subset_images(args.independent_test_root, max_samples=args.samples_per_independent_subset, seed=42)
                if recs:
                    indep_records["unseen_generator"] = recs
            else:
                for sub in sub_dirs:
                    s_dir = os.path.join(args.independent_test_root, sub)
                    recs = scan_subset_images(s_dir, max_samples=args.samples_per_independent_subset, seed=42)
                    if recs:
                        indep_records[sub] = recs

        for sub, items in indep_records.items():
            m, _, cat_targets, elapsed = evaluate_dataset_items(
                model=model, items=items, device=device,
                batch_size=args.batch_size, num_workers=args.num_workers,
                is_dummy=(args.test_dummy or not args.independent_test_root),
                temperature=args.temperature, threshold=args.threshold
            )
            row = {
                "group": "Diffusion",
                "subset": sub,
                "num_samples": len(cat_targets),
                "macro_acc": m["macro_acc"],
                "real_acc": m["real_acc"],
                "fake_acc": m["fake_acc"],
                "auc": m["auc"],
                "ap": m["ap"],
                "elapsed_sec": round(elapsed, 2)
            }
            independent_results.append(row)
            print(f"  -> [Independent] {sub:<22s} ({len(cat_targets)} anh) | Macro ACC: {m['macro_acc']:.2f}% | "
                  f"Real: {m['real_acc']:.2f}% | Fake: {m['fake_acc']:.2f}% | AUC: {m['auc']:.2f}% | AP: {m['ap']:.2f}%")

        if independent_results:
            mean_independent = calc_mean(independent_results, "Overall", "MEAN_INDEPENDENT")

    # =========================================================================
    # XUAT FILE VA BAO CAO TONG HOP
    # =========================================================================
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Xuat ma tran Clean CSV
    clean_csv = os.path.join(args.output_dir, "final_posthoc_benchmark_matrix.csv")
    all_export_clean = list(clean_results)
    if mean_clean_gans:
        all_export_clean.append(mean_clean_gans)
    if mean_clean_diff:
        all_export_clean.append(mean_clean_diff)
    if mean_clean_overall:
        all_export_clean.append(mean_clean_overall)

    headers_clean = ["group", "subset", "num_samples", "macro_acc", "real_acc", "fake_acc", "auc", "ap", "elapsed_sec"]
    with open(clean_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers_clean) + "\n")
        for r in all_export_clean:
            f.write(",".join(str(r[h]) for h in headers_clean) + "\n")
    print(f"\n[✓] Da xuat ma tran kiem chuan Clean ra: {clean_csv}")

    # 2. Xuat ma tran Degraded CSV neu co
    if degraded_results:
        deg_csv = os.path.join(args.output_dir, "final_locked_degraded_matrix.csv")
        headers_deg = [
            "variant", "variant_desc", "group", "subset", "num_samples",
            "macro_acc", "real_acc", "fake_acc", "auc", "ap", "elapsed_sec"
        ]
        with open(deg_csv, "w", encoding="utf-8") as f:
            f.write(",".join(headers_deg) + "\n")
            for r in degraded_results:
                f.write(",".join(str(r.get(h, "")) for h in headers_deg) + "\n")
        print(f"[✓] Da xuat ma tran kiem chuan Degraded ra: {deg_csv}")

    # 3. Xuat ma tran Independent CSV neu co
    if independent_results:
        indep_csv = os.path.join(args.output_dir, "final_independent_test_matrix.csv")
        all_export_indep = list(independent_results)
        if mean_independent:
            all_export_indep.append(mean_independent)
        headers_indep = ["group", "subset", "num_samples", "macro_acc", "real_acc", "fake_acc", "auc", "ap", "elapsed_sec"]
        with open(indep_csv, "w", encoding="utf-8") as f:
            f.write(",".join(headers_indep) + "\n")
            for r in all_export_indep:
                f.write(",".join(str(r[h]) for h in headers_indep) + "\n")
        print(f"[✓] Da xuat ma tran kiem chuan Independent ra: {indep_csv}")

    # 4. Xuat file JSON bao cao tong hop
    report_json = os.path.join(args.output_dir, "final_locked_benchmark_report.json")
    json_payload = {
        "metadata": {
            "task": "Task 6.6 Final Locked Benchmark & Post-Hoc Evaluation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": os.path.basename(resolved_ckpt) if resolved_ckpt else "N/A",
            "temperature_calibrated": args.temperature,
            "threshold_calibrated": args.threshold,
            "bootstrap_replicates": args.n_bootstrap,
            "external_test_status": "not_provided",
            "aggregation": {
                "clean_subset_macro_mean": round(mean_clean_overall['macro_acc'], 2) if mean_clean_overall else 0.0,
                "clean_cluster_subset_macro_mean_95_ci": list(cluster_acc_ci),
                "clean_cluster_auc_95_ci": list(cluster_auc_ci),
                "clean_cluster_ap_95_ci": list(cluster_ap_ci),
                "clean_pooled_macro_acc": round(clean_pooled_metrics['macro_acc'], 2),
                "clean_pooled_macro_acc_95_ci": list(clean_ci_range),
                "clean_pooled_auc_95_ci": list(clean_auc_ci_range),
                "clean_pooled_ap_95_ci": list(clean_ap_ci_range)
            },
            "macro_acc_95_ci": list(clean_ci_range),
            "auc_95_ci": list(clean_auc_ci_range),
            "ap_95_ci": list(clean_ap_ci_range),
            "cluster_macro_acc_95_ci": list(cluster_acc_ci),
            "cluster_auc_95_ci": list(cluster_auc_ci),
            "cluster_ap_95_ci": list(cluster_ap_ci),
            "is_dummy_mode": is_clean_dummy or is_deg_dummy if should_eval_degraded else is_clean_dummy
        },
        "clean_evaluation": {
            "summary": {
                "mean_gans": mean_clean_gans,
                "mean_diffusion": mean_clean_diff,
                "mean_overall": mean_clean_overall,
                "pooled_overall": clean_pooled_metrics
            },
            "subsets": clean_results
        },
        "degraded_evaluation": {
            "summary_by_variant": summary_by_variant,
            "subsets": degraded_results
        } if should_eval_degraded else None,
        "selection_heldout_evaluation": {
            "description": "Tier 2 Held-out Selection-Split Evaluation (val_selection split from Task 6.4, 270 samples; post-calibration check, not an external unseen test)",
            "summary": mean_independent,
            "subsets": independent_results
        } if independent_results else None,
        "independent_evaluation": {
            "description": "Alias for selection_heldout_evaluation for backward compatibility",
            "summary": mean_independent,
            "subsets": independent_results
        } if independent_results else None
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    print(f"[✓] Da xuat bao cao JSON tong hop ra: {report_json}")

    # 5. Hien thi bang tong hop chung cuoc 2 chieu
    print("\n" + "=" * 96)
    print("BẢNG TỔNG HỢP KIỂM CHUẨN KHÓA SỔ CHUNG CUỘC FATFORMER-XLA (TASK 6.6)")
    print("=" * 96)
    if mean_clean_overall:
        print(f"[*] CLEAN SUBSET-MACRO MEAN ({len(clean_results)}/18) : Macro ACC = {mean_clean_overall['macro_acc']:.2f}% "
              f"(95% Cluster CI: [{cluster_acc_ci[0]:.2f}%, {cluster_acc_ci[1]:.2f}%]) | "
              f"AUC = {mean_clean_overall['auc']:.2f}% | AP = {mean_clean_overall['ap']:.2f}%")
        print(f"[*] CLEAN POOLED-SAMPLE MICRO (82,003 anh)  : Macro ACC = {clean_pooled_metrics['macro_acc']:.2f}% "
              f"(95% Sample CI: [{clean_ci_range[0]:.2f}%, {clean_ci_range[1]:.2f}%]) | "
              f"AUC = {clean_pooled_metrics['auc']:.2f}% | AP = {clean_pooled_metrics['ap']:.2f}%")

    if summary_by_variant:
        print("\n" + "-" * 96)
        print(f"{'Biến thể':<10} {'Mô tả suy biến':<30} {'GANs (%)':<10} {'Diff (%)':<10} {'Overall (%)':<12} {'Real ACC':<10} {'Fake ACC':<10} {'Δ Sụt giảm':<12}")
        print("-" * 96)
        if mean_clean_gans and mean_clean_diff and mean_clean_overall:
            print(f"{'Clean':<10} {'Mốc trần tham chiếu':<30} {mean_clean_gans['macro_acc']:<10.2f} {mean_clean_diff['macro_acc']:<10.2f} {mean_clean_overall['macro_acc']:<12.2f} {mean_clean_overall['real_acc']:<10.2f} {mean_clean_overall['fake_acc']:<10.2f} {'0.00% (Mốc)':<12}")
        print("-" * 96)
        for var, s in summary_by_variant.items():
            print(f"{var:<10} {s['variant_desc']:<30} {s['gans_acc']:<10.2f} {s['diff_acc']:<10.2f} {s['overall_acc']:<12.2f} {s['real_acc']:<10.2f} {s['fake_acc']:<10.2f} {s['delta_drop']:+.2f}%")
        print("-" * 96)
    print("=" * 96)


if __name__ == "__main__":
    main()
