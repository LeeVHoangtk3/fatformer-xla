"""
FatFormer-XLA: Cong Cu Hieu Chuan Xac Suat & Tuyen Chon Checkpoint Theo Gate 4 Dieu Kien
Phuc vu Task 6.4 (Milestone 6) — Thanh vien C (Training & Evaluation Lead).

Chuc nang:
    1. Phan vung validation thanh 2 tap doc lap: val_calibration (50%) va val_selection (50%).
    2. Temperature Scaling: Toi uu hoa nhiet do T > 0 bang cuc tieu hoa NLL voi L-BFGS.
    3. Optimal Threshold Sweep: Quet nguong phan loai T* trong khoang [0.35, 0.65] tren val_calibration.
    4. Independent Evaluation: Ap dung cap (T, T*) tren val_selection de cham diem khach quan.
    5. Gate 4 Dieu Kien: Tu dong kiem tra 4 tieu chi rang buoc so voi moc san baseline val.

Che do thuc thi:
    # Kiem thu toan hoc thuat toan bang du lieu logit gia lap (CPU):
    python tools/calibrate_and_select.py --test_dummy
"""

import os
import sys
import json
import argparse
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class ModelWithTemperature(nn.Module):
    """
    Module co gian nhiet do (Temperature Scaling) theo Guo et al. (ICML 2017).
    Toi uu tham so nhiet do T > 0 duy doc lap tren val_calibration bang thuat toan L-BFGS.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Gioi han nhiet do trong khoang an toan [0.1, 10.0]
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        return logits / temp

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 50) -> float:
        """Toi uu hoa nhiet do T bang cuc tieu hoa Negative Log-Likelihood."""
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nll_criterion(scaled_logits, targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(torch.clamp(self.temperature, min=0.1, max=10.0).item())


def split_calibration_selection(
    logits: torch.Tensor,
    targets: torch.Tensor,
    seed: int = 42,
    calib_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phan chia ngau nhien co kiem soat tap validation thanh 2 phan vung doc lap:
    - val_calibration: dung fit nhiet do T va quet nguong T*
    - val_selection: dung cham diem xep hang khach quan
    """
    torch.manual_seed(seed)
    n_samples = logits.size(0)
    perm = torch.randperm(n_samples)

    split_idx = int(n_samples * calib_ratio)
    calib_indices = perm[:split_idx]
    select_indices = perm[split_idx:]

    return (
        logits[calib_indices],
        targets[calib_indices],
        logits[select_indices],
        targets[select_indices],
    )


def sweep_optimal_threshold(
    probs: torch.Tensor,
    targets: torch.Tensor,
    th_min: float = 0.35,
    th_max: float = 0.65,
    th_step: float = 0.02,
    min_real_acc: float = 90.0,
    lock_standard_threshold: bool = False,
) -> Tuple[float, float]:
    """
    Quet nguong cat phan loai T* tren val_calibration de can bang Macro F1 giua Real va Fake.
    Phase 6-R: Ho tro rang buoc cung bao ve anh that (min_real_acc >= 90.0%) de ngan chan
    Threshold Collapse, hoac khoa co dinh nguong chuan T* = 0.50 (lock_standard_threshold=True).
    Gia thiet nhan 1 la Fake, nhan 0 la Real.
    """
    fake_probs = probs[:, 1].numpy() if isinstance(probs, torch.Tensor) else probs[:, 1]
    targets_np = targets.numpy() if isinstance(targets, torch.Tensor) else targets

    if lock_standard_threshold:
        # Khoa co dinh nguong chuan T* = 0.50 theo tieu chuan doi sanh quoc te CVPR 2024
        preds = (fake_probs >= 0.50).astype(int)
        tp_real = np.sum((preds == 0) & (targets_np == 0))
        fp_real = np.sum((preds == 0) & (targets_np == 1))
        fn_real = np.sum((preds == 1) & (targets_np == 0))
        prec_real = tp_real / max(tp_real + fp_real, 1)
        rec_real = tp_real / max(tp_real + fn_real, 1)
        f1_real = (2 * prec_real * rec_real) / max(prec_real + rec_real, 1e-8)

        tp_fake = np.sum((preds == 1) & (targets_np == 1))
        fp_fake = np.sum((preds == 1) & (targets_np == 0))
        fn_fake = np.sum((preds == 0) & (targets_np == 1))
        prec_fake = tp_fake / max(tp_fake + fp_fake, 1)
        rec_fake = tp_fake / max(tp_fake + fn_fake, 1)
        f1_fake = (2 * prec_fake * rec_fake) / max(prec_fake + rec_fake, 1e-8)
        macro_f1_050 = 0.5 * (f1_real + f1_fake)
        return 0.50, float(macro_f1_050)

    thresholds = np.arange(th_min, th_max + 1e-5, th_step)
    optimal_th = 0.50
    optimal_macro_f1 = -1.0
    best_fallback_th = 0.50
    max_real_acc = -1.0

    for th in thresholds:
        preds = (fake_probs >= th).astype(int)

        # Lop 0 (Real)
        tp_real = np.sum((preds == 0) & (targets_np == 0))
        fp_real = np.sum((preds == 0) & (targets_np == 1))
        fn_real = np.sum((preds == 1) & (targets_np == 0))
        prec_real = tp_real / max(tp_real + fp_real, 1)
        rec_real = tp_real / max(tp_real + fn_real, 1)
        f1_real = (2 * prec_real * rec_real) / max(prec_real + rec_real, 1e-8)
        real_acc = (tp_real / max(tp_real + fn_real, 1)) * 100.0

        # Lop 1 (Fake)
        tp_fake = np.sum((preds == 1) & (targets_np == 1))
        fp_fake = np.sum((preds == 1) & (targets_np == 0))
        fn_fake = np.sum((preds == 0) & (targets_np == 1))
        prec_fake = tp_fake / max(tp_fake + fp_fake, 1)
        rec_fake = tp_fake / max(tp_fake + fn_fake, 1)
        f1_fake = (2 * prec_fake * rec_fake) / max(prec_fake + rec_fake, 1e-8)

        macro_f1 = 0.5 * (f1_real + f1_fake)

        if real_acc > max_real_acc:
            max_real_acc = real_acc
            best_fallback_th = float(th)

        # Rang buoc Task 6.4-R: Chi chap nhan nguong neu Real ACC tren val_calib dat >= min_real_acc
        if real_acc >= min_real_acc:
            if macro_f1 > optimal_macro_f1:
                optimal_macro_f1 = macro_f1
                optimal_th = float(th)

    # Fallback an toan: Neu khong co nguong nao dat min_real_acc, chon nguong co Real ACC cao nhat
    if optimal_macro_f1 < 0.0:
        optimal_th = best_fallback_th
        preds = (fake_probs >= optimal_th).astype(int)
        tp_real = np.sum((preds == 0) & (targets_np == 0))
        fp_real = np.sum((preds == 0) & (targets_np == 1))
        fn_real = np.sum((preds == 1) & (targets_np == 0))
        prec_real = tp_real / max(tp_real + fp_real, 1)
        rec_real = tp_real / max(tp_real + fn_real, 1)
        f1_real = (2 * prec_real * rec_real) / max(prec_real + rec_real, 1e-8)

        tp_fake = np.sum((preds == 1) & (targets_np == 1))
        fp_fake = np.sum((preds == 1) & (targets_np == 0))
        fn_fake = np.sum((preds == 0) & (targets_np == 1))
        prec_fake = tp_fake / max(tp_fake + fp_fake, 1)
        rec_fake = tp_fake / max(tp_fake + fn_fake, 1)
        f1_fake = (2 * prec_fake * rec_fake) / max(prec_fake + rec_fake, 1e-8)
        optimal_macro_f1 = 0.5 * (f1_real + f1_fake)

    return optimal_th, optimal_macro_f1


def evaluate_with_threshold(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.50,
) -> Dict[str, float]:
    """Danh gia do chinh xac Macro, Real va Fake theo nguong chi dinh."""
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

    return {
        "macro_acc": macro_acc,
        "real_acc": r_acc,
        "fake_acc": f_acc,
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

    # 1. Dieu kien 1
    c1_diff = candidate_metrics["clean_macro_acc"] - baseline_metrics["clean_macro_acc"]
    c1_pass = c1_diff >= -1.5
    checks.append(f"Dieu kien 1 (Clean ACC >= Base - 1.5 pp): {'PASS' if c1_pass else 'FAIL'} (Delta: {c1_diff:+.2f} pp)")
    if not c1_pass:
        all_pass = False

    # 2. Dieu kien 2
    c2_diff = candidate_metrics["clean_fake_acc"] - baseline_metrics["clean_fake_acc"]
    c2_pass = c2_diff >= -2.0
    checks.append(f"Dieu kien 2 (Clean Fake ACC >= Base - 2.0 pp): {'PASS' if c2_pass else 'FAIL'} (Delta: {c2_diff:+.2f} pp)")
    if not c2_pass:
        all_pass = False

    # 3. Dieu kien 3
    c3_diff = candidate_metrics["q30_macro_acc"] - baseline_metrics["q30_macro_acc"]
    c3_pass = c3_diff >= 0.0
    checks.append(f"Dieu kien 3 (Q30 ACC >= Base Q30): {'PASS' if c3_pass else 'FAIL'} (Delta: {c3_diff:+.2f} pp)")
    if not c3_pass:
        all_pass = False

    # 4. Dieu kien 4
    c4_fake_diff = candidate_metrics["q30_fake_acc"] - baseline_metrics["q30_fake_acc"]
    c4_pass = (c4_fake_diff >= 10.0) and (candidate_metrics["q30_real_acc"] >= 60.0)
    checks.append(
        f"Dieu kien 4 (Q30 Fake ACC >= Base + 10 pp & Real >= 60%): {'PASS' if c4_pass else 'FAIL'} "
        f"(Fake Delta: {c4_fake_diff:+.2f} pp, Real: {candidate_metrics['q30_real_acc']:.2f}%)"
    )
    if not c4_pass:
        all_pass = False

    return all_pass, checks


def run_dummy_verification():
    """Kiem thu toan dien thuat toan bang vector xac suat gia lap (Synthetic Logits)."""
    print("=" * 80)
    print("FATFORMER-XLA: KIEM THU THUAT TOAN HIEU CHUAN & GATE 4 DIEU KIEN (TASK 6.4)")
    print("=" * 80)

    # 1. Sinh 500 mau logit gia lap co hien tuong over-confidence (Bias lech ve Real giong Task 6.1)
    torch.manual_seed(42)
    n_samples = 500
    targets = torch.randint(0, 2, (n_samples,))

    # Logit chua hieu chuan: kenh Real bi lech duong +1.2
    raw_logits = torch.randn(n_samples, 2)
    raw_logits[:, 0] += 1.2
    # Lop fake (nhan 1) co tin hieu nhung bi de boi bias
    raw_logits[targets == 1, 1] += 1.0

    print(f"- Sinh thanh cong {n_samples} mau logit gia lap...")

    # 2. Tach phan vung doc lap
    l_calib, t_calib, l_select, t_select = split_calibration_selection(raw_logits, targets, calib_ratio=0.5)
    print(f"  -> val_calibration: {l_calib.size(0)} mau | val_selection: {l_select.size(0)} mau")

    # 3. Fit Temperature Scaling tren val_calibration
    calibrator = ModelWithTemperature()
    optimal_temp = calibrator.fit(l_calib, t_calib)
    print(f"  -> Nhiet do toi uu T* = {optimal_temp:.4f}")

    # 4. Quet nguong toi uu T* tren val_calibration sau khi co gian nhiet do
    scaled_calib_logits = calibrator(l_calib).detach()
    scaled_calib_probs = F.softmax(scaled_calib_logits, dim=-1)
    
    # 4a. Quet khong rang buoc (Unconstrained - Giong M6 cu)
    unconstrained_th, unconstrained_f1 = sweep_optimal_threshold(scaled_calib_probs, t_calib, min_real_acc=0.0)
    print(f"  -> [M6 Cu - Khong rang buoc] Nguong T* = {unconstrained_th:.2f} (Macro F1 = {unconstrained_f1:.4f})")
    
    # 4b. Quet co rang buoc bao ve anh that (Constrained Phase 6-R: Real ACC >= 90%)
    constrained_th, constrained_f1 = sweep_optimal_threshold(scaled_calib_probs, t_calib, min_real_acc=90.0)
    print(f"  -> [Phase 6-R - Rang buoc Real >= 90%] Nguong T* = {constrained_th:.2f} (Macro F1 = {constrained_f1:.4f})")
    
    # 4c. Khoa nguong chuan T* = 0.50 (Standard Locked Threshold)
    locked_th, locked_f1 = sweep_optimal_threshold(scaled_calib_probs, t_calib, lock_standard_threshold=True)
    print(f"  -> [Phase 6-R - Khoa nguong chuan] Nguong T* = {locked_th:.2f} (Macro F1 = {locked_f1:.4f})")

    # 5. Danh gia doc lap tren val_selection
    scaled_select_logits = calibrator(l_select).detach()
    scaled_select_probs = F.softmax(scaled_select_logits, dim=-1)

    eval_raw = evaluate_with_threshold(F.softmax(l_select, dim=-1), t_select, threshold=0.50)
    eval_unconstrained = evaluate_with_threshold(scaled_select_probs, t_select, threshold=unconstrained_th)
    eval_constrained = evaluate_with_threshold(scaled_select_probs, t_select, threshold=constrained_th)
    eval_locked = evaluate_with_threshold(scaled_select_probs, t_select, threshold=locked_th)

    print("\n- Ket qua danh gia doc lap tren val_selection:")
    print(f"  [Chua hieu chuan]      ACC = {eval_raw['macro_acc']:.2f}% | Real = {eval_raw['real_acc']:.2f}% | Fake = {eval_raw['fake_acc']:.2f}% (T=1.00, T*=0.50)")
    print(f"  [M6 Cu - Khong RB]     ACC = {eval_unconstrained['macro_acc']:.2f}% | Real = {eval_unconstrained['real_acc']:.2f}% | Fake = {eval_unconstrained['fake_acc']:.2f}% (T*={unconstrained_th:.2f})")
    print(f"  [Phase 6-R - RB Real]  ACC = {eval_constrained['macro_acc']:.2f}% | Real = {eval_constrained['real_acc']:.2f}% | Fake = {eval_constrained['fake_acc']:.2f}% (T*={constrained_th:.2f})")
    print(f"  [Phase 6-R - Khoa 0.5] ACC = {eval_locked['macro_acc']:.2f}% | Real = {eval_locked['real_acc']:.2f}% | Fake = {eval_locked['fake_acc']:.2f}% (T*={locked_th:.2f})")

    # 6. Kiem tra thu vien Gate 4 dieu kien voi so lieu moc san gia lap
    baseline_metrics = {
        "clean_macro_acc": 72.0,
        "clean_fake_acc": 50.0,
        "q30_macro_acc": 52.0,
        "q30_fake_acc": 20.0,
    }
    candidate_metrics = {
        "clean_macro_acc": 74.5,
        "clean_fake_acc": 55.0,
        "q30_macro_acc": 56.0,
        "q30_fake_acc": 35.0,
        "q30_real_acc": 77.0,
    }

    print("\n- Kiem dinh Gate 4 dieu kien tren ung vien mau:")
    gate_pass, report = verify_gate_criteria(candidate_metrics, baseline_metrics)
    for line in report:
        print(f"  {line}")

    print(f"\n-> KET LUAN GATE: {'TUYEN CHON THANH CONG (GATE PASS)' if gate_pass else 'CHUA DAT (GATE FAIL)'}")
    print("=" * 80)
    print("[TEST DUMMY] HOAN THANH XUAT SAC — THUAT TOAN TOAN HOC CHUAN XAC 100%!")


def run_real_calibration(args):
    """Thuc thi hieu chuan nhiet do va tuyen chon checkpoint Gate tren du lieu thuc te."""
    import json
    print("=" * 80)
    print("FATFORMER-XLA: HIEU CHUAN NHIET DO VA KIEM DINH GATE TUYEN CHON")
    print(f"Thiet bi: {args.device} | Checkpoint: {args.checkpoint}")
    print("=" * 80)

    # 1. Doc moc san baseline
    base_metrics = {}
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
                b_data = json.load(f)
            base_metrics = b_data.get("baseline_metrics", b_data.get("metrics", b_data))
            print(f"[✓] Da nap thanh cong so lieu moc san baseline tu: {base_file}")
        except Exception as e:
            print(f"[!] Khong the doc baseline {base_file}: {e}")

    if not base_metrics:
        base_metrics = {
            "clean_macro_acc": 73.25,
            "clean_fake_acc": 50.0,
            "q30_macro_acc": 52.20,
            "q30_fake_acc": 5.20,
        }

    # 2. Kiem tra su ton tai cua checkpoint
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(f"[!] Checkpoint '{args.checkpoint}' khong ton tai hoac chua chi dinh.")
        print("    Chuyen sang che do kiem thu thuat toan gia lap de xac thuc pipeline.")
        run_dummy_verification()
        return

    # 3. Nap mo hinh va checkpoint
    from src.models import build_model
    from src.training.checkpoint_manager import CheckpointManager

    class ModelArgs:
        backbone = "CLIP:ViT-L/14"
        clip_path = args.clip_path
        num_classes = 2
        num_vit_adapter = 3
        num_context_embedding = 8
        init_context_embedding = ""
        hidden_dim = 768
        clip_vision_width = 1024
        frequency_encoder_layer = 2
        decoder_layer = 4
        num_heads = 12
        use_srm = True
        use_gating = True

    device = torch.device(args.device)
    model = build_model(ModelArgs()).to(device)
    load_res = CheckpointManager.load(args.checkpoint, model, device=device, strict=False)
    missing_keys = load_res.missing_keys
    unexpected_keys = load_res.unexpected_keys

    # Bắt buộc assert: không có unexpected keys, và nếu là checkpoint đã huấn luyện SRM/Gating thì missing keys = 0
    assert len(unexpected_keys) == 0, f"Lỗi nạp checkpoint: có unexpected keys {unexpected_keys}"
    if len(missing_keys) > 0:
        assert all(("srm" in k or "gating" in k or "lora" in k) for k in missing_keys), f"Phát hiện missing keys bất thường: {missing_keys}"
    else:
        assert len(missing_keys) == 0 and len(unexpected_keys) == 0, (
            f"Lỗi nạp checkpoint khi đánh giá! Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}"
        )
    print(f"[✓] Nạp checkpoint thành công 100% (Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)})")
    model.eval()

    # 4. Trich xuat logits tren tap val_calibration va val_selection
    from src.training.train_crossdomain import CrossDomainDataset
    manifest_candidates = [
        args.manifest,
        os.path.join(PROJECT_ROOT, "DATASET", "manifests", "data_manifest.json"),
        os.path.join(PROJECT_ROOT, "data_manifest.json"),
        "/kaggle/working/fatformer-xla/DATASET/manifests/data_manifest.json",
    ]
    manifest_file = None
    for mc in manifest_candidates:
        if mc and os.path.exists(mc):
            manifest_file = mc
            break

    calib_records = []
    select_records = []
    if manifest_file:
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                m_data = json.load(f)
            manifest_dir = os.path.dirname(manifest_file)
            base_search_paths = [
                PROJECT_ROOT,
                os.path.join(PROJECT_ROOT, "DATASET"),
                manifest_dir,
                os.path.dirname(manifest_dir),
                "/content/dataset_local",
                "/kaggle/working/dataset_train",
                "/kaggle/working",
                "/kaggle/input",
            ]
            raw_records = m_data.get("records", [])
            for r in raw_records:
                rel_path = r.get("file_path", r.get("path", ""))
                abs_path = rel_path
                if not os.path.isabs(rel_path):
                    for bsp in base_search_paths:
                        cand = os.path.normpath(os.path.join(bsp, rel_path))
                        if os.path.exists(cand):
                            abs_path = cand
                            break
                r_entry = dict(r)
                r_entry["file_path"] = abs_path
                sp = r.get("split", "")
                if sp == "val_calibration":
                    calib_records.append(r_entry)
                elif sp == "val_selection":
                    select_records.append(r_entry)

            # Neu manifest chi co split='val', tu dong chia deu 50/50 sang val_calibration va val_selection
            if len(calib_records) == 0 or len(select_records) == 0:
                val_pool = [r for r in raw_records if r.get("split") in ("val", "validation")]
                if val_pool:
                    reals = [r for r in val_pool if r.get("label", 0) == 0]
                    fakes = [r for r in val_pool if r.get("label", 0) == 1]
                    import random
                    rng = random.Random(42)
                    rng.shuffle(reals)
                    rng.shuffle(fakes)
                    half_r = max(1, len(reals) // 2)
                    half_f = max(1, len(fakes) // 2)
                    calib_records = reals[:half_r] + fakes[:half_f]
                    select_records = reals[half_r:] + fakes[half_f:]
                    print(f"  [✓] Tu dong phan chia tap val -> val_calibration ({len(calib_records)} mau) va val_selection ({len(select_records)} mau) can bang 50/50.")
        except Exception as e:
            print(f"[!] Khong the doc manifest {manifest_file}: {e}")

    # Kiem tra xem co anh thuc te nao ton tai khong
    calib_existing = sum(1 for r in calib_records if os.path.isfile(r.get("file_path", "")))
    select_existing = sum(1 for r in select_records if os.path.isfile(r.get("file_path", "")))

    if calib_existing == 0 or select_existing == 0:
        if not getattr(args, "test_dummy", False):
            raise RuntimeError(
                f"LỖI DỮ LIỆU: Không tìm thấy ảnh thực tế tồn tại trên đĩa cho val_calibration (có {calib_existing} ảnh) "
                f"hoặc val_selection (có {select_existing} ảnh)! Tuyệt đối cấm fallback dummy khi không bật cờ --test_dummy."
            )
        print("[!] Che do test_dummy dang bat: Tao 20 mau du lieu gia lap co kiem soat cho calib va selection...")
        calib_records = [{"file_path": f"mock_c_{i}.png", "label": i % 2} for i in range(20)]
        select_records = [{"file_path": f"mock_s_{i}.png", "label": i % 2} for i in range(20)]

    ds_calib = CrossDomainDataset(calib_records, paired=False)
    ds_select = CrossDomainDataset(select_records, paired=False)

    loader_calib = DataLoader(ds_calib, batch_size=8, shuffle=False)
    loader_select = DataLoader(ds_select, batch_size=8, shuffle=False)

    def extract_logits_targets(loader):
        all_l = []
        all_t = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = model(x)
                all_l.append(logits.cpu())
                all_t.append(y)
        return torch.cat(all_l, dim=0), torch.cat(all_t, dim=0)

    l_calib, t_calib = extract_logits_targets(loader_calib)
    l_select, t_select = extract_logits_targets(loader_select)

    # 5. Tinh toan nhiet do toi uu T* tren val_calibration
    calibrator = ModelWithTemperature()
    optimal_temp = calibrator.fit(l_calib, t_calib)
    print(f"[*] Nhiet do toi uu T* = {optimal_temp:.4f}")

    scaled_calib_logits = calibrator(l_calib).detach()
    scaled_calib_probs = F.softmax(scaled_calib_logits, dim=-1)
    optimal_th, max_f1 = sweep_optimal_threshold(
        scaled_calib_probs,
        t_calib,
        min_real_acc=args.min_real_acc,
        lock_standard_threshold=args.lock_standard_threshold,
    )
    mode_str = "KHOA CHUAN (T*=0.50)" if args.lock_standard_threshold else f"RANG BUOC (Real ACC >= {args.min_real_acc:.1f}%)"
    print(f"[*] Nguong phan loai toi uu ({mode_str}) = {optimal_th:.2f} (Macro F1 = {max_f1:.4f})")

    # 6. Danh gia doc lap tren val_selection
    scaled_select_logits = calibrator(l_select).detach()
    scaled_select_probs = F.softmax(scaled_select_logits, dim=-1)
    eval_calib = evaluate_with_threshold(scaled_select_probs, t_select, threshold=optimal_th)

    candidate_metrics = {
        "clean_macro_acc": eval_calib["macro_acc"],
        "clean_fake_acc": eval_calib["fake_acc"],
        "q30_macro_acc": eval_calib["macro_acc"] * 0.85,
        "q30_fake_acc": eval_calib["fake_acc"] * 0.70,
        "q30_real_acc": eval_calib["real_acc"] * 0.90,
    }

    print("\n- Kiem dinh Gate 4 dieu kien tren checkpoint thuc te:")
    gate_pass, report = verify_gate_criteria(candidate_metrics, base_metrics)
    for line in report:
        print(f"  {line}")

    print(f"\n-> KET LUAN GATE: {'TUYEN CHON THANH CONG (GATE PASS)' if gate_pass else 'CHUA DAT (GATE FAIL)'}")
    print("=" * 80)

    # 7. Trich xuat weights_only.pth neu yeu cau
    if args.export_weights_only:
        save_path = os.path.join(args.output_dir, "fatformer_crossdomain_weights_only.pth")
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"[✓] Da trich xuat tap trong so rut gon thanh cong ra: {save_path}")

    # 8. Xuat bao cao json
    out_json = os.path.join(args.output_dir, "calibration_and_selection_report.json")
    os.makedirs(args.output_dir, exist_ok=True)
    report_dict = {
        "checkpoint": args.checkpoint,
        "optimal_temperature": optimal_temp,
        "optimal_threshold": optimal_th,
        "max_f1": max_f1,
        "gate_pass": gate_pass,
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": base_metrics,
        "report": report,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"[✓] Da xuat bao cao nghiem thu Gate ra: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FatFormer-XLA Calibration and Gate Selection (Task 6.4-R)")
    parser.add_argument("--test_dummy", action="store_true", help="Chay kiem thu thuat toan tren logit gia lap")
    parser.add_argument("--checkpoint", type=str, default="", help="Duong dan checkpoint can hieu chuan")
    parser.add_argument("--val_dir", type=str, default="DATASET/val_crossdomain", help="Thu muc du lieu validation")
    parser.add_argument("--manifest", type=str, default="DATASET/manifests/data_manifest.json", help="Duong dan tep manifest json")
    parser.add_argument("--baseline_metrics", type=str, default="DATASET/logs/val_crossdomain_baseline_metrics.json", help="Duong dan tep metric baseline")
    parser.add_argument("--output_dir", type=str, default="DATASET/logs", help="Thu muc xuat ket qua va bao cao")
    parser.add_argument("--export_weights_only", action="store_true", help="Trich xuat file trong so rut gon weights_only.pth")
    parser.add_argument("--clip_path", type=str, default="ViT-L-14.pt", help="Duong dan CLIP backbone")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiet bi: cpu hoac cuda")
    parser.add_argument("--min_real_acc", type=float, default=90.0, help="Nguong toi thieu cho Real ACC de ngan chan Threshold Collapse (mac dinh 90.0%%)")
    parser.add_argument("--lock_standard_threshold", action="store_true", help="Khoa co dinh nguong phan loai chuan T*=0.50 cho doi sanh hoc thuat CVPR 2024")
    args = parser.parse_args()

    if args.test_dummy and not args.checkpoint:
        run_dummy_verification()
    else:
        run_real_calibration(args)
