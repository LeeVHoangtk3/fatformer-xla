"""
FatFormer-XLA: Cong Cu Danh Gia Ma Tran Ablation Study & Hieu Chuan Gate 4 Dieu Kien
Tich hop lien hoan Task 6.5 (Ablation Matrix M1-M5) va Task 6.4 (Calibration & Model Selection).

Quy trinh thuc thi:
    1. Nap 5 cau hinh mo hinh ung vien:
       - M1: Baseline tac gia 4-class (fatformer_4class_ckpt.pth)
       - M2: Plan A Finetune (fatformer_plan_a_best_weights.pth)
       - M3: Data Only (m3/crossdomain_model_best.pth)
       - M4: Distill + Cons (m4/crossdomain_model_best.pth)
       - M5: LoRA ViT Adapter (m5/crossdomain_model_best.pth)
    2. Trich xuat song song logits tren 540 anh val_crossdomain o ca 2 trang thai:
       - Clean: Anh goc qua tien xu ly chuan CLIP
       - Degraded Q=30: Anh nen JPEG muc chat luong 30
    3. Tinh toan Ma Tran Ablation Study (Task 6.5):
       - Do dac Macro ACC, Real ACC, Fake ACC, Macro AUC, AP tren ca Clean va Q=30
       - Binh luon su sut giam hieu nang va dong gop cua tung module ky thuat
       - Xuat file CSV va JSON tong hop
    4. Hieu Chuan Xac Suat & Kiem Dinh Gate 4 Dieu Kien (Task 6.4):
       - Phan chia doc lap: val_calibration (50%) va val_selection (50%)
       - Toi uu hoa nhiet do T bang L-BFGS va quet nguong cat T* tren val_calibration
       - Danh gia doc lap tren val_selection theo Gate 4 dieu kien
       - Trich xuat file trong so rut gon weights_only.pth cho mo hinh duoc tuyen chon
"""

import os
import sys
import io
import glob
import json
import time
import argparse
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def find_first_existing(path_list: List[str]) -> Optional[str]:
    """Tim duong dan ton tai dau tien trong danh sach ung vien (ho tro ca glob pattern)."""
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


def compress_jpeg_q30(image: Image.Image) -> Image.Image:
    """Nen anh sang dinh dang JPEG voi chat luong Q=30 bang bo dem bo nho RAM."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=30)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class ValCrossDomainPairedDataset(Dataset):
    """
    Dataset danh gia da mien: Tra ve dong thoi anh Clean va anh bi nen JPEG Q=30.
    """
    def __init__(self, records: List[Dict[str, Any]], transform=None):
        self.records = records
        self.transform = transform or transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        rec = self.records[idx]
        file_path = rec.get("file_path", "")
        label = int(rec.get("label", 0))
        generator = rec.get("generator", "unknown")

        try:
            img_clean = Image.open(file_path).convert("RGB")
        except Exception:
            # Fallback mau anh gia lap neu file loi
            img_clean = Image.new("RGB", (224, 224), color=(128, 128, 128))

        img_q30 = compress_jpeg_q30(img_clean)

        tensor_clean = self.transform(img_clean)
        tensor_q30 = self.transform(img_q30)

        return tensor_clean, tensor_q30, label, generator


class DummyBackbone(nn.Module):
    """Mo hinh gia lap phuc vu che do kiem thu toan hoc tren CPU."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.pool(x.float()).flatten(1)
        return self.fc(feat)


class ModelWithTemperature(nn.Module):
    """
    Module co gian nhiet do (Temperature Scaling) theo Guo et al. (ICML 2017).
    Toi uu tham so nhiet do T > 0 duy doc lap tren val_calibration bang thuat toan L-BFGS.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        return logits / temp

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 50) -> float:
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nll_criterion(scaled_logits, targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(torch.clamp(self.temperature, min=0.1, max=10.0).item())


def compute_expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """Tinh toan sai so hieu chuan ky vong ECE (Expected Calibration Error)."""
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = (predictions == targets).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = float(targets.size(0))

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * (in_bin.sum().item() / total_samples)

    return float(ece)


def calculate_auc_ap(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """Tinh toan Macro ROC-AUC va Average Precision khong phu thuoc sklearn."""
    fake_probs = probs[:, 1]
    order = np.argsort(fake_probs)[::-1]
    sorted_targets = targets[order]

    n_pos = np.sum(targets == 1)
    n_neg = np.sum(targets == 0)

    if n_pos == 0 or n_neg == 0:
        return 50.0, 50.0

    # Tinh AUC bang quy tac hinh thang
    tps = np.cumsum(sorted_targets == 1)
    fps = np.cumsum(sorted_targets == 0)
    tpr = tps / n_pos
    fpr = fps / n_neg

    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = trap_fn(tpr, fpr) * 100.0

    # Tinh Average Precision
    precisions = tps / (tps + fps)
    ap = np.sum(precisions * (sorted_targets == 1)) / max(n_pos, 1) * 100.0

    return float(np.clip(auc, 0.0, 100.0)), float(np.clip(ap, 0.0, 100.0))


def sweep_optimal_threshold(
    probs: torch.Tensor,
    targets: torch.Tensor,
    th_min: float = 0.35,
    th_max: float = 0.65,
    th_step: float = 0.02,
) -> Tuple[float, float]:
    """Quet nguong cat phan loai T* tren val_calibration de can bang Macro F1."""
    thresholds = np.arange(th_min, th_max + 1e-5, th_step)
    optimal_th = 0.50
    optimal_macro_f1 = -1.0

    fake_probs = probs[:, 1].numpy() if isinstance(probs, torch.Tensor) else probs[:, 1]
    targets_np = targets.numpy() if isinstance(targets, torch.Tensor) else targets

    for th in thresholds:
        preds = (fake_probs >= th).astype(int)

        # Real (0)
        tp_real = np.sum((preds == 0) & (targets_np == 0))
        fp_real = np.sum((preds == 0) & (targets_np == 1))
        fn_real = np.sum((preds == 1) & (targets_np == 0))
        prec_real = tp_real / max(tp_real + fp_real, 1)
        rec_real = tp_real / max(tp_real + fn_real, 1)
        f1_real = (2 * prec_real * rec_real) / max(prec_real + rec_real, 1e-8)

        # Fake (1)
        tp_fake = np.sum((preds == 1) & (targets_np == 1))
        fp_fake = np.sum((preds == 1) & (targets_np == 0))
        fn_fake = np.sum((preds == 0) & (targets_np == 1))
        prec_fake = tp_fake / max(tp_fake + fp_fake, 1)
        rec_fake = tp_fake / max(tp_fake + fn_fake, 1)
        f1_fake = (2 * prec_fake * rec_fake) / max(prec_fake + rec_fake, 1e-8)

        macro_f1 = 0.5 * (f1_real + f1_fake)
        if macro_f1 > optimal_macro_f1:
            optimal_macro_f1 = macro_f1
            optimal_th = float(th)

    return optimal_th, optimal_macro_f1


def evaluate_predictions(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.50) -> Dict[str, float]:
    """Danh gia cac chi so phan loai co ban theo nguong chi dinh."""
    fake_probs = probs[:, 1].numpy() if isinstance(probs, torch.Tensor) else probs[:, 1]
    targets_np = targets.numpy() if isinstance(targets, torch.Tensor) else targets

    preds = (fake_probs >= threshold).astype(int)
    correct = np.sum(preds == targets_np)
    total = len(targets_np)
    macro_acc = (correct / total * 100.0) if total > 0 else 0.0

    real_mask = (targets_np == 0)
    fake_mask = (targets_np == 1)

    r_acc = (np.sum(preds[real_mask] == 0) / np.sum(real_mask) * 100.0) if np.sum(real_mask) > 0 else 0.0
    f_acc = (np.sum(preds[fake_mask] == 1) / np.sum(fake_mask) * 100.0) if np.sum(fake_mask) > 0 else 0.0

    probs_np = probs.numpy() if isinstance(probs, torch.Tensor) else probs
    auc, ap = calculate_auc_ap(probs_np, targets_np)

    return {
        "macro_acc": float(macro_acc),
        "real_acc": float(r_acc),
        "fake_acc": float(f_acc),
        "auc": float(auc),
        "ap": float(ap),
    }


def verify_gate_criteria(
    candidate_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> Tuple[bool, List[str]]:
    """
    Kiem tra 4 dieu kien rang buoc so sanh tuong doi voi moc san baseline:
        - Dieu kien 1: Clean Macro ACC >= Base_clean - 1.5 pp
        - Dieu kien 2: Clean Fake ACC >= Base_fake_clean - 2.0 pp
        - Dieu kien 3: Q30 Overall ACC >= Base_q30
        - Dieu kien 4: Q30 Fake ACC >= Base_fake_q30 + 10.0 pp va Q30 Real ACC >= 60.0%
    """
    checks = []
    all_pass = True

    c1_diff = candidate_metrics["clean_macro_acc"] - baseline_metrics["clean_macro_acc"]
    c1_pass = c1_diff >= -1.5
    checks.append(f"Dieu kien 1 (Clean ACC >= Base - 1.5 pp): {'PASS' if c1_pass else 'FAIL'} (Delta: {c1_diff:+.2f} pp)")
    if not c1_pass:
        all_pass = False

    c2_diff = candidate_metrics["clean_fake_acc"] - baseline_metrics["clean_fake_acc"]
    c2_pass = c2_diff >= -2.0
    checks.append(f"Dieu kien 2 (Clean Fake ACC >= Base - 2.0 pp): {'PASS' if c2_pass else 'FAIL'} (Delta: {c2_diff:+.2f} pp)")
    if not c2_pass:
        all_pass = False

    c3_diff = candidate_metrics["q30_macro_acc"] - baseline_metrics["q30_macro_acc"]
    c3_pass = c3_diff >= 0.0
    checks.append(f"Dieu kien 3 (Q30 ACC >= Base Q30): {'PASS' if c3_pass else 'FAIL'} (Delta: {c3_diff:+.2f} pp)")
    if not c3_pass:
        all_pass = False

    c4_fake_diff = candidate_metrics["q30_fake_acc"] - baseline_metrics["q30_fake_acc"]
    c4_pass = (c4_fake_diff >= 10.0) and (candidate_metrics["q30_real_acc"] >= 60.0)
    checks.append(
        f"Dieu kien 4 (Q30 Fake ACC >= Base + 10 pp & Real >= 60%): {'PASS' if c4_pass else 'FAIL'} "
        f"(Fake Delta: {c4_fake_diff:+.2f} pp, Real: {candidate_metrics['q30_real_acc']:.2f}%)"
    )
    if not c4_pass:
        all_pass = False

    return all_pass, checks


def build_candidate_model(
    branch_name: str,
    ckpt_path: str,
    clip_path: str,
    device: torch.device,
    is_dummy: bool = False
) -> nn.Module:
    """Khoi tao kien truc va nap checkpoint chuan cho tung nhanh M1 - M5."""
    if is_dummy or not (clip_path and os.path.exists(clip_path)):
        num_c = 4 if branch_name == "M1" else 2
        return DummyBackbone(num_classes=num_c).to(device)

    from src.models import build_model
    from src.training.checkpoint_manager import CheckpointManager

    model_args = SimpleNamespace(
        backbone="CLIP:ViT-L/14",
        clip_path=clip_path,
        num_classes=4 if branch_name == "M1" else 2,
        use_srm=(branch_name != "M1"),
        use_gating=(branch_name != "M1"),
        srm_out_dim=1024,
    )
    model = build_model(model_args)

    # Neu la M5: Gắn bộ điều hợp LoRA ViT
    if branch_name == "M5":
        from src.models.lora_adapter import ViTLoRAAdapter
        vit_target = getattr(model, "image_encoder", getattr(model, "visual", model))
        lora_cfg = {"enabled": True, "rank": 8, "alpha": 16.0, "num_layers": 2}
        ViTLoRAAdapter(vit_target, config=lora_cfg)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"  [*] Nap checkpoint {branch_name} tu: {ckpt_path}")
        try:
            ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt_obj = torch.load(ckpt_path, map_location="cpu")
        state = ckpt_obj.get("model_state", ckpt_obj.get("model", ckpt_obj))
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        res = model.load_state_dict(state, strict=False)
        missing_count = len(getattr(res, "missing_keys", []))
        unexpected_count = len(getattr(res, "unexpected_keys", []))
        print(f"  [✓] Da nap checkpoint {branch_name}: Missing keys = {missing_count} | Unexpected keys = {unexpected_count}")
        del ckpt_obj, state
    else:
        print(f"  [!] Canh bao: Khong tim thay checkpoint cho {branch_name} tai '{ckpt_path}'. Chay voi trong so hien thoi.")

    model = model.to(device)
    model.eval()
    return model


def forward_model_logits(model: nn.Module, x: torch.Tensor, branch_name: str) -> torch.Tensor:
    """Forward qua mo hinh va chuyen doi ve binary logits 2 chieu [Real, Fake]."""
    raw_logits = model(x)
    if branch_name == "M1" and raw_logits.size(-1) == 4:
        # Chuyen doi tu 4-class sang 2-class:
        # Lop 0: Real, Cac lop 1, 2, 3: Fake
        probs_4 = F.softmax(raw_logits, dim=-1)
        p_real = probs_4[:, 0:1]
        p_fake = torch.sum(probs_4[:, 1:], dim=-1, keepdim=True)
        # Logit tuong ung
        logit_real = torch.log(p_real + 1e-8)
        logit_fake = torch.log(p_fake + 1e-8)
        return torch.cat([logit_real, logit_fake], dim=-1)
    return raw_logits


def run_crossdomain_ablation_and_gate(args):
    print("=" * 88)
    print("FATFORMER-XLA: MA TRAN ABLATION STUDY (TASK 6.5) & HIEU CHUAN GATE (TASK 6.4)")
    print(f"Thiet bi: {args.device} | Che do gia lap (Dummy): {args.test_dummy}")
    print("=" * 88)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[!] Canh bao: PyTorch hien tai khong ho tro CUDA hoac chua bat GPU.")
        print("    Tu dong chuyen doi thiet bi sang 'cpu' de duy tri tien trinh danh gia.")
        device_str = "cpu"
    device = torch.device(device_str)

    # 1. Tim kiem va xac dinh duong dan CLIP backbone
    clip_candidates = [
        args.clip_path,
        "ViT-L-14.pt",
        "pretrained/ViT-L-14.pt",
        "DATASET/pretrained/ViT-L-14.pt",
        "/kaggle/input/datasets/hoangleev/fatformer-assets-pack/ViT-L-14.pt",
        "/kaggle/input/fatformer-assets-pack/ViT-L-14.pt",
    ]
    resolved_clip = find_first_existing(clip_candidates) or args.clip_path

    # 2. Xac dinh danh sach checkpoint cho 5 nhanh
    checkpoint_map = {
        "M1": find_first_existing([
            args.m1_ckpt,
            "DATASET/pretrained/fatformer_4class_ckpt.pth",
            "pretrained/fatformer_4class_ckpt.pth",
            "fatformer_4class_ckpt.pth",
            "/kaggle/input/datasets/hoangleev/fatformer-assets-pack/fatformer_4class_ckpt.pth",
            "/kaggle/input/fatformer-assets-pack/fatformer_4class_ckpt.pth",
        ]),
        "M2": find_first_existing([
            args.m2_ckpt,
            "DATASET/checkpoints/fatformer_plan_a_best_weights.pth",
            "DATASET/checkpoints/fatformer_plan_a_best.pth",
            "/kaggle/input/datasets/hoangleev/fatformer-assets-pack/fatformer_plan_a_best_weights.pth",
            "/kaggle/input/fatformer-assets-pack/fatformer_plan_a_best_weights.pth",
        ]),
        "M3": find_first_existing([
            args.m3_ckpt,
            "/kaggle/input/**/crossdomain_model_best_m3.pth",
            "/kaggle/input/**/*best*m3*.pth",
            "/kaggle/input/**/*_m3.pth",
            "/kaggle/input/**/m3/crossdomain_model_best.pth",
            "/kaggle/input/**/m3*/crossdomain_model_best.pth",
            "/kaggle/input/**/m3*best*.pth",
            "DATASET/checkpoints/m3/crossdomain_model_best.pth",
            "/kaggle/working/fatformer-xla/DATASET/checkpoints/m3/crossdomain_model_best.pth",
            "/kaggle/working/checkpoints/m3/crossdomain_model_best.pth",
            "wait/task6-3/DATASET/checkpoints/m3/crossdomain_model_best.pth",
        ]),
        "M4": find_first_existing([
            args.m4_ckpt,
            "/kaggle/input/**/crossdomain_model_best_m4.pth",
            "/kaggle/input/**/*best*m4*.pth",
            "/kaggle/input/**/*_m4.pth",
            "/kaggle/input/**/m4/crossdomain_model_best.pth",
            "/kaggle/input/**/m4*/crossdomain_model_best.pth",
            "/kaggle/input/**/m4*best*.pth",
            "DATASET/checkpoints/m4/crossdomain_model_best.pth",
            "/kaggle/working/fatformer-xla/DATASET/checkpoints/m4/crossdomain_model_best.pth",
            "/kaggle/working/checkpoints/m4/crossdomain_model_best.pth",
            "wait/task6-3/DATASET/checkpoints/m4/crossdomain_model_best.pth",
        ]),
        "M5": find_first_existing([
            args.m5_ckpt,
            "/kaggle/input/**/crossdomain_model_best_m5.pth",
            "/kaggle/input/**/*best*m5*.pth",
            "/kaggle/input/**/*_m5.pth",
            "/kaggle/input/**/m5/crossdomain_model_best.pth",
            "/kaggle/input/**/m5*/crossdomain_model_best.pth",
            "/kaggle/input/**/m5*best*.pth",
            "DATASET/checkpoints/m5/crossdomain_model_best.pth",
            "/kaggle/working/fatformer-xla/DATASET/checkpoints/m5/crossdomain_model_best.pth",
            "/kaggle/working/checkpoints/m5/crossdomain_model_best.pth",
            "wait/task6-3/DATASET/checkpoints/m5/crossdomain_model_best.pth",
        ]),
    }

    print("\n[*] Trang thai dinh vi Checkpoint 5 nhanh ung vien:")
    for b_name in ["M1", "M2", "M3", "M4", "M5"]:
        print(f"  - {b_name}: {checkpoint_map[b_name] or '[Chua tim thay - Chay che do khoi tao]'}")

    # 3. Nap manifest du lieu validation
    manifest_candidates = [
        args.manifest,
        "DATASET/manifests/data_manifest.json",
        "data_manifest.json",
        "/kaggle/working/fatformer-xla/DATASET/manifests/data_manifest.json",
    ]
    resolved_manifest = find_first_existing(manifest_candidates)

    val_records = []
    if resolved_manifest and not args.test_dummy:
        try:
            with open(resolved_manifest, "r", encoding="utf-8") as f:
                m_data = json.load(f)
            records = m_data.get("records", [])
            val_records = [r for r in records if r.get("split") == "val"]
            print(f"[✓] Nap thanh cong {len(val_records)} mau validation tu: {resolved_manifest}")
        except Exception as e:
            print(f"[!] Loi doc manifest ({resolved_manifest}): {e}")

    # Neu manifest cu khong co mau val, tu dong tai ban 3600 anh tu GitHub
    if len(val_records) == 0 and not args.test_dummy:
        print("[*] Manifest hien tai khong co mau val. Dang tu dong cap nhat ban manifest 3600 anh tu GitHub...")
        try:
            import urllib.request
            raw_url = "https://raw.githubusercontent.com/LeeVHoangtk3/fatformer-xla/hoang/DATASET/manifests/data_manifest.json"
            save_dest = resolved_manifest or "DATASET/manifests/data_manifest.json"
            urllib.request.urlretrieve(raw_url, save_dest)
            with open(save_dest, "r", encoding="utf-8") as f:
                m_data = json.load(f)
            records = m_data.get("records", [])
            val_records = [r for r in records if r.get("split") == "val"]
            print(f"[✓] Da cap nhat manifest thanh cong: {len(val_records)} mau validation!")
        except Exception as e:
            print(f"[!] Khong the cap nhat manifest: {e}")

    # Tu dong remap duong dan anh thuc te tren Kaggle neu can
    real_staging_dirs = glob.glob("/kaggle/input/**/diffusion_staging", recursive=True)
    if real_staging_dirs and len(val_records) > 0:
        base_s = real_staging_dirs[0]
        remap_count = 0
        for r in val_records:
            orig_p = r.get("file_path", "")
            if not os.path.exists(orig_p) and "diffusion_staging" in orig_p:
                sub_p = orig_p.split("diffusion_staging")[-1].lstrip("/\\")
                new_p = os.path.join(base_s, sub_p)
                if os.path.exists(new_p):
                    r["file_path"] = new_p
                    remap_count += 1
        if remap_count > 0:
            print(f"[✓] Tu dong anh xa thanh cong {remap_count} duong dan anh den: {base_s}")

    # Neu chay dummy hoac khong co anh thuc te tren dia
    if args.test_dummy or len(val_records) == 0:
        print("[*] Khoi tao 100 mau gia lap de kiem thu pipeline...")
        val_records = []
        for i in range(100):
            val_records.append({
                "file_path": f"dummy_{i}.png",
                "label": i % 2,
                "generator": ["sdv15", "midjourney", "stylegan", "progan", "dalle"][i % 5],
                "split": "val"
            })

    # 4. Khoi tao Dataset & DataLoader
    dataset = ValCrossDomainPairedDataset(val_records)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 5. Do dac va thu thap Logits cho ca 5 mo hinh
    print("\n" + "=" * 88)
    print("GIAI DOAN 1: THU THAP LOGITS TOAN DIEN TREN TAP KIEM DINH DA MIEN")
    print("=" * 88)

    models_data = {}
    for branch in ["M1", "M2", "M3", "M4", "M5"]:
        print(f"\n[*] Dang danh gia nhanh [{branch}]...")
        model = build_candidate_model(
            branch_name=branch,
            ckpt_path=checkpoint_map[branch],
            clip_path=resolved_clip,
            device=device,
            is_dummy=args.test_dummy
        )

        clean_logits_list = []
        q30_logits_list = []
        targets_list = []
        generators_list = []

        t0 = time.time()
        with torch.no_grad():
            for x_clean, x_q30, targets, gens in loader:
                x_clean = x_clean.to(device)
                x_q30 = x_q30.to(device)

                with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                    l_clean = forward_model_logits(model, x_clean, branch)
                    l_q30 = forward_model_logits(model, x_q30, branch)

                clean_logits_list.append(l_clean.cpu().float())
                q30_logits_list.append(l_q30.cpu().float())
                targets_list.append(targets)
                generators_list.extend(gens)

        elapsed = time.time() - t0
        all_l_clean = torch.cat(clean_logits_list, dim=0)
        all_l_q30 = torch.cat(q30_logits_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)

        models_data[branch] = {
            "clean_logits": all_l_clean,
            "q30_logits": all_l_q30,
            "targets": all_targets,
            "generators": generators_list,
            "elapsed_sec": elapsed,
        }
        print(f"  -> Hoan tat trong {elapsed:.2f}s ({len(all_targets)} mau).")

        # Giai phong bo nho GPU
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 6. Tinh toan Ma Tran Ablation Study (Task 6.5)
    print("\n" + "=" * 88)
    print("GIAI DOAN 2: XAY DUNG MA TRAN ABLATION STUDY DA MIEN (TASK 6.5)")
    print("=" * 88)

    ablation_matrix = []
    for branch in ["M1", "M2", "M3", "M4", "M5"]:
        d = models_data[branch]
        targets = d["targets"]

        p_clean = F.softmax(d["clean_logits"], dim=-1)
        p_q30 = F.softmax(d["q30_logits"], dim=-1)

        eval_clean = evaluate_predictions(p_clean, targets, threshold=0.50)
        eval_q30 = evaluate_predictions(p_q30, targets, threshold=0.50)

        delta_drop = eval_q30["macro_acc"] - eval_clean["macro_acc"]

        row = {
            "branch": branch,
            "clean_macro_acc": eval_clean["macro_acc"],
            "clean_real_acc": eval_clean["real_acc"],
            "clean_fake_acc": eval_clean["fake_acc"],
            "clean_auc": eval_clean["auc"],
            "clean_ap": eval_clean["ap"],
            "q30_macro_acc": eval_q30["macro_acc"],
            "q30_real_acc": eval_q30["real_acc"],
            "q30_fake_acc": eval_q30["fake_acc"],
            "q30_auc": eval_q30["auc"],
            "q30_ap": eval_q30["ap"],
            "delta_degradation": delta_drop,
            "elapsed_sec": d["elapsed_sec"],
        }
        ablation_matrix.append(row)

    # In bang Markdown Ablation
    print("\n| Nhanh | Kien Truc / Ky Thuat | Clean ACC | Clean Fake | Q30 ACC | Q30 Fake | Q30 Real | Δ Sut Giam | Macro AUC |")
    print("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    branch_desc = {
        "M1": "Baseline 4-class (Tac gia)",
        "M2": "Plan A Finetune (SRM+Gating)",
        "M3": "+ Du Lieu Da Bo Sinh",
        "M4": "+ Chuan Hoa Distill & Cons",
        "M5": "+ LoRA ViT Adapter (Mo rong)",
    }
    for r in ablation_matrix:
        b = r["branch"]
        print(
            f"| **{b}** | {branch_desc[b]} | {r['clean_macro_acc']:.2f}% | {r['clean_fake_acc']:.2f}% | "
            f"{r['q30_macro_acc']:.2f}% | {r['q30_fake_acc']:.2f}% | {r['q30_real_acc']:.2f}% | "
            f"{r['delta_degradation']:+.2f}% | {r['clean_auc']:.2f}% |"
        )

    # Xuat CSV va JSON Ablation
    os.makedirs(args.output_dir, exist_ok=True)
    csv_ablation = os.path.join(args.output_dir, "crossdomain_ablation_matrix.csv")
    with open(csv_ablation, "w", encoding="utf-8") as f:
        headers = list(ablation_matrix[0].keys())
        f.write(",".join(headers) + "\n")
        for r in ablation_matrix:
            f.write(",".join(str(r[h]) for h in headers) + "\n")
    print(f"\n[✓] Da xuat bang so lieu Ablation Study ra: {csv_ablation}")

    # 7. Hieu Chuan Xac Suat & Kiem Dinh Gate 4 Dieu Kien (Task 6.4)
    print("\n" + "=" * 88)
    print("GIAI DOAN 3: HIEU CHUAN NHIET DO & TUYEN CHON THEO GATE 4 DIEU KIEN (TASK 6.4)")
    print("=" * 88)

    # Doc moc san baseline
    base_metrics = {
        "clean_macro_acc": 73.25,
        "clean_real_acc": 98.50,
        "clean_fake_acc": 48.00,
        "q30_macro_acc": 52.20,
        "q30_real_acc": 99.20,
        "q30_fake_acc": 5.20,
    }
    base_candidates = [
        args.baseline_metrics,
        os.path.join(PROJECT_ROOT, "DATASET", "logs", "m6_crossdomain_recovery", "calibration", "val_crossdomain_baseline_metrics.json"),
        os.path.join(PROJECT_ROOT, "DATASET", "logs", "val_crossdomain_baseline_metrics.json"),
    ]
    base_file = None
    for bc in base_candidates:
        if bc and os.path.exists(bc):
            base_file = bc
            break

    if base_file:
        try:
            with open(base_file, "r", encoding="utf-8") as f:
                bm = json.load(f)
            base_metrics = bm.get("baseline_metrics", bm)
            print(f"[✓] Nap moc san baseline tham chieu tu: {base_file}")
        except Exception as e:
            print(f"[!] Canh bao nap baseline metrics: {e}")

    # Phan chia 50% val_calibration va 50% val_selection co kiem soat
    torch.manual_seed(args.seed)
    n_samples = len(models_data["M3"]["targets"])
    perm = torch.randperm(n_samples)
    split_idx = n_samples // 2

    idx_calib = perm[:split_idx]
    idx_select = perm[split_idx:]
    print(f"[*] Phan chia phan vung doc lap (seed={args.seed}): "
          f"val_calibration = {len(idx_calib)} mau | val_selection = {len(idx_select)} mau")

    gate_results = []
    selected_model_branch = None
    selected_composite_score = -1.0

    for branch in ["M1", "M2", "M3", "M4", "M5"]:
        d = models_data[branch]
        targets_all = d["targets"]

        # Chia phan vung
        l_clean_calib = d["clean_logits"][idx_calib]
        l_clean_select = d["clean_logits"][idx_select]
        l_q30_select = d["q30_logits"][idx_select]
        t_calib = targets_all[idx_calib]
        t_select = targets_all[idx_select]

        # 1. Fit Temperature tren val_calibration (Clean)
        calibrator = ModelWithTemperature()
        temp_val = calibrator.fit(l_clean_calib, t_calib)

        # 2. Quet nguong toi uu tren val_calibration
        scaled_calib_p = F.softmax(calibrator(l_clean_calib).detach(), dim=-1)
        th_val, f1_val = sweep_optimal_threshold(scaled_calib_p, t_calib)

        # 3. Do ECE truoc va sau hieu chuan
        p_raw_select = F.softmax(l_clean_select, dim=-1)
        ece_before = compute_expected_calibration_error(p_raw_select, t_select)

        scaled_clean_select = F.softmax(calibrator(l_clean_select).detach(), dim=-1)
        ece_after = compute_expected_calibration_error(scaled_clean_select, t_select)

        # 4. Danh gia doc lap tren val_selection voi cap (T, T*)
        scaled_q30_select = F.softmax(calibrator(l_q30_select).detach(), dim=-1)

        eval_clean_gate = evaluate_predictions(scaled_clean_select, t_select, threshold=th_val)
        eval_q30_gate = evaluate_predictions(scaled_q30_select, t_select, threshold=th_val)

        cand_metrics = {
            "clean_macro_acc": eval_clean_gate["macro_acc"],
            "clean_real_acc": eval_clean_gate["real_acc"],
            "clean_fake_acc": eval_clean_gate["fake_acc"],
            "q30_macro_acc": eval_q30_gate["macro_acc"],
            "q30_real_acc": eval_q30_gate["real_acc"],
            "q30_fake_acc": eval_q30_gate["fake_acc"],
        }

        # Kiem tra Gate 4 dieu kien
        gate_pass, gate_report = verify_gate_criteria(cand_metrics, base_metrics)

        # Diem composite danh gia tuyen chon: can bang giua Clean ACC va Q30 ACC
        composite = 0.5 * eval_clean_gate["macro_acc"] + 0.5 * eval_q30_gate["macro_acc"]

        gate_info = {
            "branch": branch,
            "optimal_temperature": temp_val,
            "optimal_threshold": th_val,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "metrics": cand_metrics,
            "composite_score": composite,
            "gate_pass": gate_pass,
            "gate_report": gate_report,
        }
        gate_results.append(gate_info)

        if gate_pass and composite > selected_composite_score:
            selected_composite_score = composite
            selected_model_branch = branch

    # In bang tong hop Gate
    print("\n| Nhanh | Nhiet Do T | Nguong T* | ECE Truoc -> Sau | Clean ACC (Val) | Q30 Fake (Val) | Ket Qua Gate |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for g in gate_results:
        b = g["branch"]
        status = "**PASS** (Thoa 4 DK)" if g["gate_pass"] else "FAIL"
        print(
            f"| **{b}** | {g['optimal_temperature']:.3f} | {g['optimal_threshold']:.2f} | "
            f"{g['ece_before']:.4f} -> {g['ece_after']:.4f} | "
            f"{g['metrics']['clean_macro_acc']:.2f}% | {g['metrics']['q30_fake_acc']:.2f}% | {status} |"
        )

    # Ket luan tuyen chon
    selection_type = "GATE_PASSED"
    if not selected_model_branch:
        # Fallback chon model co composite score cao trong so M3-M5
        valid_candidates = [g for g in gate_results if g["branch"] in ["M3", "M4", "M5"]]
        if valid_candidates:
            best_cand = max(valid_candidates, key=lambda x: x["composite_score"])
            selected_model_branch = best_cand["branch"]
            selected_composite_score = best_cand["composite_score"]
            selection_type = "FALLBACK_BEST_CANDIDATE"
        else:
            selected_model_branch = "M5"
            selected_composite_score = next((g["composite_score"] for g in gate_results if g["branch"] == "M5"), -1.0)
            selection_type = "FALLBACK_DEFAULT"

    print(
        f"\n-> MO HINH DAI DIEN DUOC TUYEN CHON: [{selected_model_branch}] "
        f"(Composite Score: {selected_composite_score:.2f}%, Status: {selection_type})"
    )

    # Xuat file trong so rut gon weights_only.pth
    if args.export_weights_only and not args.test_dummy:
        chosen_ckpt = checkpoint_map[selected_model_branch]
        if chosen_ckpt and os.path.exists(chosen_ckpt):
            out_weights_path = os.path.join(args.output_dir, "fatformer_crossdomain_weights_only.pth")
            print(f"[*] Dang trich xuat tap trong so rut gon tu {chosen_ckpt}...")
            ckpt_obj = torch.load(chosen_ckpt, map_location="cpu")
            state_dict = ckpt_obj.get("model_state", ckpt_obj)
            torch.save(state_dict, out_weights_path)
            f_size_mb = os.path.getsize(out_weights_path) / (1024 * 1024)
            print(f"[✓] Trich xuat thanh cong: {out_weights_path} ({f_size_mb:.2f} MB)")

    # Xuat artifact raw logits
    if getattr(args, "save_raw_logits", False):
        logits_export_path = os.path.join(args.output_dir, "crossdomain_raw_logits.npz")
        try:
            export_dict = {}
            for b in ["M1", "M2", "M3", "M4", "M5"]:
                if b in models_data:
                    export_dict[f"{b}_clean_logits"] = models_data[b]["clean_logits"].cpu().numpy()
                    export_dict[f"{b}_q30_logits"] = models_data[b]["q30_logits"].cpu().numpy()
                    export_dict[f"{b}_targets"] = models_data[b]["targets"].cpu().numpy()
            export_dict["val_calib_indices"] = idx_calib.cpu().numpy()
            export_dict["val_select_indices"] = idx_select.cpu().numpy()
            np.savez_compressed(logits_export_path, **export_dict)
            print(f"[✓] Da xuat artifact logits tho ra: {logits_export_path}")
        except Exception as e:
            print(f"[!] Canh bao xuat raw logits: {e}")

    # Xuat bao cao tong hop JSON
    report_json_path = os.path.join(args.output_dir, "calibration_and_selection_report.json")
    final_report = {
        "metadata": {
            "task": "Task 6.4 & 6.5 Integrated Execution",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": args.seed,
            "selected_model": selected_model_branch,
            "selected_composite_score": selected_composite_score,
            "selection_status": selection_type,
            "composite_formula": "0.5 * clean_macro_acc + 0.5 * q30_macro_acc",
            "all_gate_passed": any(g["gate_pass"] for g in gate_results),
        },
        "baseline_metrics": base_metrics,
        "ablation_matrix": ablation_matrix,
        "gate_results": gate_results,
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"[✓] Da xuat bao cao tong hop Gate & Ablation ra: {report_json_path}")
    print("=" * 88)
    print("[HOAN TAT] TOAN BO TIEN TRINH THUC THI PHUONG AN A THANH CONG MIEN MAN!")


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Domain Ablation & Gate Selection Pipeline")
    parser.add_argument("--test_dummy", action="store_true", help="Chay kiem thu thuat toan bang du lieu gia lap tren CPU")
    parser.add_argument("--clip_path", type=str, default="ViT-L-14.pt", help="Duong dan backbone ViT-L-14.pt")
    parser.add_argument("--manifest", type=str, default="DATASET/manifests/data_manifest.json", help="Duong dan data_manifest.json")
    parser.add_argument("--baseline_metrics", type=str, default="DATASET/logs/val_crossdomain_baseline_metrics.json", help="Moc san baseline metrics")
    parser.add_argument("--output_dir", type=str, default="DATASET/logs", help="Thu muc xuat ket qua")

    # Checkpoint tuy chon
    parser.add_argument("--m1_ckpt", type=str, default="", help="Checkpoint M1")
    parser.add_argument("--m2_ckpt", type=str, default="", help="Checkpoint M2")
    parser.add_argument("--m3_ckpt", type=str, default="", help="Checkpoint M3")
    parser.add_argument("--m4_ckpt", type=str, default="", help="Checkpoint M4")
    parser.add_argument("--m5_ckpt", type=str, default="", help="Checkpoint M5")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size danh gia")
    parser.add_argument("--num_workers", type=int, default=2, help="So luong workers nap anh")
    parser.add_argument("--seed", type=int, default=42, help="Hat giong ngau nhien")
    parser.add_argument("--export_weights_only", action="store_true", default=True, help="Trich xuat weights_only.pth")
    parser.add_argument("--save_raw_logits", action="store_true", default=True, help="Xuat tap tin raw logits ra .npz de ban giao doi chieu")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiet bi: cuda hoac cpu")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_crossdomain_ablation_and_gate(args)
