"""
tools/eval_gen1_reinterpretation.py
Task 7.1 (Phase 7 — D1-D2)
Đo lại Gen 1 (fatformer_wise_ft_best.pth) dưới 2 layout n=3 vs n=8 trên 2.000 ảnh clean của CommunityForensics.
[DIAGNOSTIC-ONLY] partial load — Mục đích phân xử: artifact_loader vs different_basin.
"""

import os
import sys
import glob
import json
import argparse
import datetime
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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


class CFECleanDataset(Dataset):
    """Dataset nạp ảnh clean từ 5 tập con CFE, chuẩn hóa 224x224 theo CLIP."""
    def __init__(self, file_records: List[Dict[str, Any]]):
        self.file_records = file_records
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

    def __len__(self) -> int:
        return len(self.file_records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.file_records[idx]
        file_path = rec["path"]
        label = rec["label"]
        subset = rec["subset"]

        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            print(f"  [!] Cảnh báo đọc file lỗi ({file_path}): {e}")
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        tensor_img = self.transform(img)
        return tensor_img, label, subset


def scan_cfe_clean_samples(data_dir: str, max_samples: int = 2000) -> List[Dict[str, Any]]:
    """Quét dữ liệu clean CFE và lấy đúng max_samples (cân bằng 0_real và 1_fake)."""
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    real_records = []
    fake_records = []

    for sub in CFE_SUBSETS:
        sub_dir = os.path.join(data_dir, sub)
        if not os.path.exists(sub_dir):
            continue

        r_dir = os.path.join(sub_dir, "0_real")
        if os.path.exists(r_dir):
            for fn in sorted(os.listdir(r_dir)):
                if os.path.splitext(fn)[1].lower() in valid_exts:
                    real_records.append({"path": os.path.join(r_dir, fn), "label": 0, "subset": sub})

        f_dir = os.path.join(sub_dir, "1_fake")
        if os.path.exists(f_dir):
            for fn in sorted(os.listdir(f_dir)):
                if os.path.splitext(fn)[1].lower() in valid_exts:
                    fake_records.append({"path": os.path.join(f_dir, fn), "label": 1, "subset": sub})

    # Cân bằng 50/50 nếu có thể
    samples_per_class = max_samples // 2
    selected_real = real_records[:samples_per_class] if len(real_records) >= samples_per_class else real_records
    selected_fake = fake_records[:samples_per_class] if len(fake_records) >= samples_per_class else fake_records

    all_records = selected_real + selected_fake
    print(f"[*] Quét được {len(all_records)} ảnh ({len(selected_real)} Real, {len(selected_fake)} Fake) từ {data_dir}")
    return all_records


def calculate_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    """Tính ACC, Real ACC, Fake ACC, AUC và AP."""
    fake_probs = probs[:, 1]
    preds = (fake_probs >= threshold).astype(int)

    total = len(targets)
    acc = float(np.mean(preds == targets) * 100.0) if total > 0 else 0.0

    real_mask = (targets == 0)
    fake_mask = (targets == 1)

    real_acc = float(np.mean(preds[real_mask] == 0) * 100.0) if np.sum(real_mask) > 0 else 0.0
    fake_acc = float(np.mean(preds[fake_mask] == 1) * 100.0) if np.sum(fake_mask) > 0 else 0.0

    # ROC-AUC & AP
    order = np.argsort(fake_probs)[::-1]
    sorted_targets = targets[order]
    n_pos = np.sum(targets == 1)
    n_neg = np.sum(targets == 0)

    if n_pos == 0 or n_neg == 0:
        auc, ap = 50.0, 50.0
    else:
        tps = np.cumsum(sorted_targets == 1)
        fps = np.cumsum(sorted_targets == 0)
        tpr = np.concatenate([[0.0], tps / n_pos])
        fpr = np.concatenate([[0.0], fps / n_neg])
        trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc = float(trap_fn(tpr, fpr) * 100.0)

        precisions = tps / np.maximum(tps + fps, 1e-8)
        tpr_no_zero = tps / n_pos
        ap = float(np.sum((tpr_no_zero[1:] - tpr_no_zero[:-1]) * precisions[1:]) * 100.0)
        if len(tpr_no_zero) > 0 and tpr_no_zero[0] > 0:
            ap += float(tpr_no_zero[0] * precisions[0] * 100.0)

    return {
        "acc": round(acc, 2),
        "real_acc": round(real_acc, 2),
        "fake_acc": round(fake_acc, 2),
        "auc": round(auc, 2),
        "ap": round(ap, 2),
    }


def build_and_load_model(
    num_adapters: int,
    ckpt_path: str,
    clip_path: str,
    device: torch.device
) -> Tuple[nn.Module, List[int], List[int]]:
    """Dựng mô hình FatFormer với số adapter chỉ định và nạp partial weights."""
    from src.models import build_model

    model_args = SimpleNamespace(
        backbone="CLIP:ViT-L/14",
        clip_path=clip_path,
        num_classes=2,
        use_srm=True,
        use_gating=True,
        srm_out_dim=1024,
        num_vit_adapter=num_adapters,
        num_context_embedding=8,
    )
    print(f"\n[*] [DIAGNOSTIC-ONLY] partial load: Dựng mô hình với num_vit_adapter={num_adapters}...")
    model = build_model(model_args)

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw_ckpt
    if isinstance(raw_ckpt, dict):
        for k in ["model", "model_state_dict", "state_dict", "model_state"]:
            if k in raw_ckpt and isinstance(raw_ckpt[k], dict):
                state = raw_ckpt[k]
                break

    clean_state = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            clean_k = k[7:] if k.startswith("module.") else k
            clean_state[clean_k] = v

    model_sd = model.state_dict()
    filtered_sd = {}
    dropped_keys = []

    for k, v in clean_state.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered_sd[k] = v
        else:
            dropped_keys.append(k)

    # Nạp partial
    missing_keys = [k for k in model_sd.keys() if k not in filtered_sd]
    model.load_state_dict(filtered_sd, strict=False)

    # Xác định các block được nạp và bị bỏ
    split = 24 // num_adapters
    loaded_blocks = [(i + 1) * split - 1 for i in range(num_adapters)]

    # 10 blocks gốc của Gen 1
    gen1_all_blocks = [2, 5, 7, 8, 11, 14, 15, 17, 20, 23]
    dropped_blocks = [b for b in gen1_all_blocks if b not in loaded_blocks]

    print(f"    • Loaded blocks ({len(loaded_blocks)}): {loaded_blocks}")
    print(f"    • Dropped blocks ({len(dropped_blocks)}): {dropped_blocks}")
    print(f"    • Nạp thành công: {len(filtered_sd)}/{len(model_sd)} tensors (Missing: {len(missing_keys)})")

    model = model.to(device)
    model.eval()
    return model, loaded_blocks, dropped_blocks


@torch.no_grad()
def run_eval(
    model: nn.Module,
    records: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 32,
    temperature: float = 1.0,
    threshold: float = 0.50
) -> Dict[str, float]:
    """Chạy suy luận batch trên danh sách records và tính toán metrics."""
    dataset = CFECleanDataset(records)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    all_probs = []
    all_targets = []

    for imgs, targets, _ in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        scaled_logits = logits / max(temperature, 1e-4)
        probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()

        all_probs.append(probs)
        all_targets.append(targets.numpy())

    probs_arr = np.concatenate(all_probs, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    metrics = calculate_metrics(probs_arr, targets_arr, threshold=threshold)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Eval Gen 1 Reinterpretation n=3 vs n=8 (Task 7.1)")
    parser.add_argument("--checkpoint", type=str, default="DATASET/checkpoints/wise_ft/fatformer_wise_ft_best.pth")
    parser.add_argument("--clip_path", type=str, default="DATASET/ViT-L-14.pt")
    parser.add_argument("--cfe_dir", type=str, default="DATASET/test_communityforensics")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="DATASET/logs/phase7/wise_ft_legacy_reinterpretation.json")
    parser.add_argument("--dummy", action="store_true", help="Chạy chế độ giả lập để kiểm tra cú pháp khi không có GPU/data")
    args = parser.parse_args()

    print("=" * 80)
    print("FATFORMER-XLA: ĐO LẠI GEN 1 (n=3 vs n=8) TRÊN CFE CLEAN (TASK 7.1)")
    print(f"Thời gian: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"Thiết bị:  {args.device.upper()}")
    print("=" * 80)

    # Tìm đường dẫn dự phòng (fallback search kế thừa hạ tầng MyDrive/Fatformer)
    ckpt_candidates = [
        args.checkpoint,
        "/content/drive/MyDrive/Fatformer/checkpoint/fatformer_wise_ft_best.pth",
        "/content/drive/MyDrive/Fatformer/checkpoint/wise_ft/fatformer_wise_ft_best.pth",
        "/content/drive/MyDrive/FatFormer_Hub/checkpoints/wise_ft/fatformer_wise_ft_best.pth",
        "/content/fatformer_wise_ft_best.pth",
    ]
    clip_candidates = [
        args.clip_path,
        "/content/drive/MyDrive/Fatformer/pretrained/ViT-L-14.pt",
        "/content/drive/MyDrive/Fatformer/checkpoint/ViT-L-14.pt",
        "/content/drive/MyDrive/FatFormer_Hub/checkpoints/ViT-L-14.pt",
        "/content/ViT-L-14.pt",
        "checkpoints/ViT-L-14.pt",
    ]
    cfe_candidates = [
        args.cfe_dir,
        "/content/dataset_local/test_communityforensics",
        "/content/test_communityforensics",
        "/content/drive/MyDrive/Fatformer/datasets/test_communityforensics",
        "/content/drive/MyDrive/FatFormer_Hub/datasets/test_communityforensics",
    ]

    def find_path(cands):
        for c in cands:
            if c and os.path.exists(c):
                return c
        return cands[0]

    real_ckpt = find_path(ckpt_candidates)
    real_clip = find_path(clip_candidates)
    real_cfe = find_path(cfe_candidates)

    print(f"• Checkpoint: {real_ckpt} (Exists: {os.path.exists(real_ckpt)})")
    print(f"• CLIP Path:  {real_clip} (Exists: {os.path.exists(real_clip)})")
    print(f"• CFE Dir:    {real_cfe} (Exists: {os.path.exists(real_cfe)})")

    device = torch.device(args.device)

    if args.dummy or not (os.path.exists(real_clip) and os.path.exists(real_cfe)):
        print("\n[!] CẢNH BÁO: Thiếu dữ liệu ảnh CFE hoặc ViT-L-14.pt trên máy hiện tại.")
        print("    -> Vui lòng chạy script này trên Google Colab có GPU và mount Google Drive!")
        if not args.dummy:
            sys.exit(1)
        print("    -> Đang chạy chế độ --dummy để mô phỏng...")
        # Mock results
        runs = [
            {"layout": "n3", "loaded_blocks": [7, 15, 23], "dropped": [2, 5, 8, 11, 14, 17, 20],
             "acc": 52.05, "real_acc": 100.0, "fake_acc": 4.10, "auc": 54.20, "ap": 51.30},
            {"layout": "n8", "loaded_blocks": [2, 5, 8, 11, 14, 17, 20, 23], "dropped": [7, 15],
             "acc": 54.10, "real_acc": 98.50, "fake_acc": 9.70, "auc": 56.40, "ap": 53.10}
        ]
    else:
        # Quét tập 2.000 ảnh CFE clean
        records = scan_cfe_clean_samples(real_cfe, max_samples=args.max_samples)
        if len(records) == 0:
            raise RuntimeError(f"Không tìm thấy ảnh nào trong thư mục: {real_cfe}")

        runs = []

        # --- RUN 1: n = 3 ---
        print("\n" + "-" * 80)
        print("[1/2] ĐANG ĐO CẤU HÌNH n=3 (Layout Loader cũ: 3 blocks)")
        print("-" * 80)
        model_n3, loaded_n3, dropped_n3 = build_and_load_model(3, real_ckpt, real_clip, device)
        m_n3 = run_eval(model_n3, records, device, batch_size=args.batch_size, temperature=args.temperature, threshold=args.threshold)
        print(f"  [✓] Kết quả n=3: ACC={m_n3['acc']}% | Real={m_n3['real_acc']}% | Fake={m_n3['fake_acc']}% | AUC={m_n3['auc']}% | AP={m_n3['ap']}%")
        runs.append({
            "layout": "n3",
            "loaded_blocks": loaded_n3,
            "dropped": dropped_n3,
            **m_n3
        })
        del model_n3
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- RUN 2: n = 8 ---
        print("\n" + "-" * 80)
        print("[2/2] ĐANG ĐO CẤU HÌNH n=8 (Layout Loader mới: 8 blocks)")
        print("-" * 80)
        model_n8, loaded_n8, dropped_n8 = build_and_load_model(8, real_ckpt, real_clip, device)
        m_n8 = run_eval(model_n8, records, device, batch_size=args.batch_size, temperature=args.temperature, threshold=args.threshold)
        print(f"  [✓] Kết quả n=8: ACC={m_n8['acc']}% | Real={m_n8['real_acc']}% | Fake={m_n8['fake_acc']}% | AUC={m_n8['auc']}% | AP={m_n8['ap']}%")
        runs.append({
            "layout": "n8",
            "loaded_blocks": loaded_n8,
            "dropped": dropped_n8,
            **m_n8
        })
        del model_n8
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Xác định phán quyết (Verdict) theo luật §5.4
    n3_fake = runs[0]["fake_acc"]
    n8_fake = runs[1]["fake_acc"]
    delta_fake = n8_fake - n3_fake

    if delta_fake >= 10.0:
        verdict = "artifact_loader"
        explanation = "n=8 hơn n=3 >= 10 pp Fake ACC: Nguyên nhân sụp đổ chính là do loader cũ ép n=3 và vứt 7 khối adapter."
    elif delta_fake <= -10.0:
        verdict = "inconclusive"
        explanation = (
            "n=8 thấp hơn n=3 >= 10 pp. Luật chỉ cho phép kết luận "
            "artifact_loader khi n=8 tốt hơn; kết quả đảo chiều này không đủ "
            "để gán different_basin nếu chưa có CI, nên giữ invalid_layout."
        )
    elif abs(delta_fake) < 3.0:
        verdict = "different_basin"
        explanation = "n=8 và n=3 chênh < 3 pp và đều thấp: Bản thân trọng số WiSE-FT đã trượt vào basin nghiệm kém, không phải do loader."
    else:
        verdict = "inconclusive"
        explanation = "Độ lớn chênh lệch từ 3 đến 10 pp: Không đủ bằng chứng kết luận loader là nguyên nhân duy nhất; giữ nhãn invalid_layout."

    output_doc = {
        "schema": "gen1_reinterpretation/v1",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint": real_ckpt,
        "split": "cfe_clean_2000",
        "temperature": args.temperature,
        "threshold": args.threshold,
        "runs": runs,
        "delta_fake_acc": round(delta_fake, 2),
        "verdict": verdict,
        "verdict_explanation": explanation
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"PHÁN QUYẾT CUỐI CÙNG: {verdict.upper()}")
    print(f"• Fake ACC n=3: {n3_fake}%")
    print(f"• Fake ACC n=8: {n8_fake}%")
    print(f"• Chênh lệch (Delta): {delta_fake:+.2f} pp")
    print(f"• Giải thích: {explanation}")
    print(f"• Đã lưu kết quả tại: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()

