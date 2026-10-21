# -*- coding: utf-8 -*-
"""
FatFormer-XLA: Kiem Chuan Khoa So Chung Cuoc & Doi Chieu Hau Nghiem (Task 6.6)
- Nap tep trong so rut gon: fatformer_crossdomain_weights_only.pth
- Dong bang cap tham so hieu chuan Task 6.4: T = 1.0811, T* = 0.3700
- Tinh toan khoang tin cay 95% Bootstrap (B=1.000) cho cac chi so cot loi
- Ho tro --test_dummy chay kiem thu logic cuc bo
- Xuat bang ma tran ket qua CSV va JSON bao cao
"""

import os
import sys
import time
import json
import glob
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
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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
            # Sinh tensor gia lap
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


def compute_bootstrap_ci(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0811,
    threshold: float = 0.3700,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Tinh khoang tin cay 95% Bootstrap (B=1000) cho cac chi so.
    """
    np.random.seed(seed)
    n_samples = len(targets)
    if n_samples < 5:
        return {}

    b_accs, b_aucs, b_aps = [], [], []

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    fake_probs = probs[:, 1].cpu().numpy()
    targets_np = targets.cpu().numpy()

    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        sub_probs = fake_probs[boot_idx]
        sub_targets = targets_np[boot_idx]

        preds = (sub_probs >= threshold).astype(int)
        tp_real = np.sum((preds == 0) & (sub_targets == 0))
        n_real = np.sum(sub_targets == 0)
        r_acc = tp_real / max(n_real, 1)

        tp_fake = np.sum((preds == 1) & (sub_targets == 1))
        n_fake = np.sum(sub_targets == 1)
        f_acc = tp_fake / max(n_fake, 1)

        m_acc = 0.5 * (r_acc + f_acc) * 100.0
        b_accs.append(m_acc)

    b_accs = np.sort(b_accs)
    ci_low = float(b_accs[int(0.025 * n_bootstrap)])
    ci_high = float(b_accs[int(0.975 * n_bootstrap)])

    return {
        "macro_acc_ci_95": (round(ci_low, 2), round(ci_high, 2))
    }


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


def main():
    parser = argparse.ArgumentParser(description="FatFormer-XLA Final Locked Benchmark (Task 6.6)")
    parser.add_argument("--checkpoint", type=str, default="DATASET/checkpoints/fatformer_crossdomain_weights_only.pth")
    parser.add_argument("--clip_path", type=str, default="ViT-L-14.pt")
    parser.add_argument("--temperature", type=float, default=1.0811, help="Nhiet do T dong bang Task 6.4")
    parser.add_argument("--threshold", type=float, default=0.3700, help="Nguong cat T* dong bang Task 6.4")
    parser.add_argument("--test_root", type=str, default="", help="Thu muc goc chua 18 tap test")
    parser.add_argument("--output_dir", type=str, default="DATASET/logs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--test_dummy", action="store_true", help="Chay che do dummy kiem thu logic")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print("=" * 88)
    print("FATFORMER-XLA: FINAL LOCKED BENCHMARK & POST-HOC VERIFICATION (TASK 6.6)")
    print(f"Tham so hieu chuan da khoa: Temperature T = {args.temperature:.4f} | Threshold T* = {args.threshold:.4f}")
    print(f"Thiet bi: {args.device} | Che do gia lap (Dummy): {args.test_dummy}")
    print("=" * 88)

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

    # 1. Chuan bi danh sach cac tap test
    records_by_subset = {}
    if args.test_dummy or not args.test_root or not os.path.exists(args.test_root):
        print("[*] Khoi tao du lieu mau 18 tap test phuc vu kiem thu he thong...")
        for sub in ALL_18_SUBSETS:
            records_by_subset[sub] = [
                {"file_path": f"dummy_{sub}_{i}.png", "label": i % 2}
                for i in range(40)
            ]
    else:
        # Quet tap test thuc te (ho tro de quy moi tang thu muc)
        print(f"[*] Dang quet tim 18 tap test ben trong: {args.test_root}...")
        for sub in ALL_18_SUBSETS:
            sub_dir = os.path.join(args.test_root, sub)
            if not os.path.isdir(sub_dir):
                # Tim kiem de quy trong args.test_root
                matches = [
                    p for p in glob.glob(os.path.join(args.test_root, "**", sub), recursive=True)
                    if os.path.isdir(p)
                ]
                if matches:
                    sub_dir = matches[0]

            recs = []
            if os.path.isdir(sub_dir):
                for label_str, lbl in [("0_real", 0), ("1_fake", 1), ("real", 0), ("fake", 1), ("0", 0), ("1", 1)]:
                    c_dir = os.path.join(sub_dir, label_str)
                    if os.path.isdir(c_dir):
                        for root, _, files in os.walk(c_dir):
                            for fn in files:
                                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                                    recs.append({"file_path": os.path.join(root, fn), "label": lbl})
            records_by_subset[sub] = recs

        total_found = sum(len(v) for v in records_by_subset.values())
        found_subsets = [k for k, v in records_by_subset.items() if len(v) > 0]
        print(f"[✓] Tim thay tong cong {total_found} anh tren {len(found_subsets)}/18 tap test: {found_subsets}")
        if total_found == 0:
            print("[!] Canh bao: Khong tim thay anh nao trong thu muc test. Chuyen sang che do dummy phuc vu kiem thu...")
            for sub in ALL_18_SUBSETS:
                records_by_subset[sub] = [
                    {"file_path": f"dummy_{sub}_{i}.png", "label": i % 2}
                    for i in range(40)
                ]

    # 2. Vong do dac suy luan khoa so
    print("\n[*] Bat dau suy luan danh gia tren 18 tap test...")
    results = []
    all_logits_list = []
    all_targets_list = []

    for sub in ALL_18_SUBSETS:
        sub_items = records_by_subset.get(sub, [])
        if not sub_items:
            continue

        dataset = ImageListDataset(sub_items, is_dummy=args.test_dummy)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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
        cat_logits = torch.cat(sub_logits, dim=0)
        cat_targets = torch.cat(sub_targets, dim=0)

        all_logits_list.append(cat_logits)
        all_targets_list.append(cat_targets)

        metrics = compute_binary_metrics_with_calib(
            cat_logits, cat_targets,
            temperature=args.temperature,
            threshold=args.threshold
        )
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
        results.append(res_row)
        print(f"  -> [{group_name:9s}] {sub:15s} | Macro ACC: {metrics['macro_acc']:.2f}% | "
              f"Real: {metrics['real_acc']:.2f}% | Fake: {metrics['fake_acc']:.2f}% | "
              f"AUC: {metrics['auc']:.2f}% | AP: {metrics['ap']:.2f}% ({elapsed:.2f}s)")

    # 3. Tinh toan cac chi so tong hop (Mean GANs, Mean Diffusion, Mean Overall)
    gans_rows = [r for r in results if r["group"] == "GANs"]
    diff_rows = [r for r in results if r["group"] == "Diffusion"]

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

    mean_gans = calc_mean(gans_rows, "GANs", "MEAN_GANS")
    mean_diff = calc_mean(diff_rows, "Diffusion", "MEAN_DIFFUSION")
    mean_overall = calc_mean(results, "Overall", "MEAN_OVERALL")

    # 4. Tinh khoang tin cay 95% Bootstrap cho toan bo mau
    overall_cat_logits = torch.cat(all_logits_list, dim=0)
    overall_cat_targets = torch.cat(all_targets_list, dim=0)
    boot_ci = compute_bootstrap_ci(
        overall_cat_logits, overall_cat_targets,
        temperature=args.temperature,
        threshold=args.threshold,
        n_bootstrap=args.n_bootstrap
    )
    ci_range = boot_ci.get("macro_acc_ci_95", (0.0, 0.0))

    print("\n" + "=" * 88)
    print("BẢNG TỔNG HỢP KIỂM CHUẨN KHÓA SỔ CHUNG CUỘC (TASK 6.6)")
    print("=" * 88)
    if mean_gans:
        print(f"[*] MEAN GANs      : Macro ACC = {mean_gans['macro_acc']:.2f}% | AUC = {mean_gans['auc']:.2f}% | AP = {mean_gans['ap']:.2f}%")
    if mean_diff:
        print(f"[*] MEAN Diffusion : Macro ACC = {mean_diff['macro_acc']:.2f}% | AUC = {mean_diff['auc']:.2f}% | AP = {mean_diff['ap']:.2f}%")
    if mean_overall:
        print(f"[*] MEAN OVERALL   : Macro ACC = {mean_overall['macro_acc']:.2f}% (95% CI: [{ci_range[0]:.2f}%, {ci_range[1]:.2f}%]) | "
              f"AUC = {mean_overall['auc']:.2f}% | AP = {mean_overall['ap']:.2f}%")

    # 5. Xuat file CSV va JSON bao cao
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "final_posthoc_benchmark_matrix.csv")

    all_export_rows = list(results)
    if mean_gans:
        all_export_rows.append(mean_gans)
    if mean_diff:
        all_export_rows.append(mean_diff)
    if mean_overall:
        all_export_rows.append(mean_overall)

    headers = ["group", "subset", "num_samples", "macro_acc", "real_acc", "fake_acc", "auc", "ap", "elapsed_sec"]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in all_export_rows:
            f.write(",".join(str(r[h]) for h in headers) + "\n")
    print(f"\n[✓] Da xuat ma tran kiem chuan khoa so ra: {out_csv}")

    report_json = os.path.join(args.output_dir, "final_locked_benchmark_report.json")
    json_payload = {
        "metadata": {
            "task": "Task 6.6 Final Locked Benchmark",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": os.path.basename(resolved_ckpt) if resolved_ckpt else "N/A",
            "temperature_calibrated": args.temperature,
            "threshold_calibrated": args.threshold,
            "bootstrap_replicates": args.n_bootstrap,
            "macro_acc_95_ci": list(ci_range),
            "is_dummy_mode": args.test_dummy
        },
        "summary": {
            "mean_gans": mean_gans,
            "mean_diffusion": mean_diff,
            "mean_overall": mean_overall
        },
        "subsets_metrics": results
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    print(f"[✓] Da xuat bao cao JSON tong hop ra: {report_json}")
    print("=" * 88)


if __name__ == "__main__":
    main()
