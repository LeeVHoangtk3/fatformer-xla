#!/usr/bin/env python3
"""Unified CLI Inference Tool cho FatFormer-XLA (Task 7.15).

Một lệnh duy nhất để suy luận phát hiện ảnh giả mạo:
- Nhận ảnh đơn, thư mục, hoặc mẫu wildcard.
- Tự động nạp kiến trúc và siêu dữ liệu (metadata) nhúng từ checkpoint.
- Tiền xử lý chuẩn CLIP (Resize 224, CenterCrop 224, ToTensor, Normalize).
- In bảng kết quả gọn gàng và xuất JSON chi tiết nếu có --output.
- Hỗ trợ cờ --benchmark_latency đo đạc overhead phân tầng (Batch 1, 8, 32).
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import build_model
from src.models.canonical_loader import load_canonical
from src.models.unified_fatformer import UnifiedFatFormerXLA


def get_default_transform():
    """Tạo pipeline tiền xử lý chuẩn CLIP (không augmentation)."""
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])


def collect_image_paths(inputs: List[str]) -> List[str]:
    """Thu thập danh sách đường dẫn ảnh từ các tệp, thư mục hoặc glob."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    image_paths = []

    for item in inputs:
        p_obj = Path(item)
        if p_obj.is_file():
            if p_obj.suffix.lower() in valid_exts:
                image_paths.append(str(p_obj.resolve()))
            continue
        elif p_obj.is_dir():
            for root, _, files in os.walk(str(p_obj)):
                for f in files:
                    if Path(f).suffix.lower() in valid_exts:
                        image_paths.append(os.path.abspath(os.path.join(root, f)))
            continue

        # Nếu là pattern wildcard
        matched = glob.glob(item)
        if matched:
            for p in matched:
                if os.path.isfile(p) and Path(p).suffix.lower() in valid_exts:
                    image_paths.append(os.path.abspath(p))
                elif os.path.isdir(p):
                    for root, _, files in os.walk(p):
                        for f in files:
                            if Path(f).suffix.lower() in valid_exts:
                                image_paths.append(os.path.abspath(os.path.join(root, f)))
        else:
            print(f"[!] Bỏ qua: Không tìm thấy đường dẫn '{item}'")

    return sorted(list(set(image_paths)))


def load_unified_model(
    checkpoint_path: str,
    calibration_path: Optional[str] = None,
    clip_path: str = "DATASET/pretrained/ViT-L-14.pt",
    device: str = "cpu"
) -> Tuple[UnifiedFatFormerXLA, Dict[str, Any]]:
    """Nạp UnifiedFatFormerXLA từ checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    raw_payload = torch.load(checkpoint_path, map_location="cpu")
    metadata = {}
    if isinstance(raw_payload, dict) and "metadata" in raw_payload:
        metadata = raw_payload["metadata"]
        layout_n = metadata.get("layout", {}).get("n", 3)
        layout_indices = metadata.get("layout", {}).get("indices", [7, 15, 23])
        calib_dict = metadata.get("calibration", None)
        router_dict = metadata.get("router", None)
        router_ablation = metadata.get("router_ablation", None)
        ckpt_sha = metadata.get("source_checkpoint_sha256", None)
    else:
        # Checkpoint thô chưa đóng gói
        layout_n = 3
        layout_indices = [7, 15, 23]
        calib_dict = None
        router_dict = None
        router_ablation = None
        ckpt_sha = None

    model_args = SimpleNamespace(
        arch="ViT-L/14",
        backbone="CLIP:ViT-L/14",
        num_classes=2,
        clip_path=clip_path if os.path.exists(clip_path) else None,
        use_srm=True,
        use_gating=True,
        num_adapter_blocks=layout_n,
        adapter_block_indices=layout_indices
    )

    backbone, load_report = load_canonical(checkpoint_path, model_args)
    backbone = backbone.to(device).eval()

    sha256_to_use = ckpt_sha or load_report.get("sha256")

    # Nếu người dùng truyền --calibration từ bên ngoài
    if calibration_path and os.path.exists(calibration_path):
        with open(calibration_path, "r", encoding="utf-8") as f:
            calib_dict = json.load(f)

    # Kiểm tra calibration: BẮT BUỘC có calibration hợp lệ, KHÔNG tự tạo dummy
    if not calib_dict:
        raise ValueError(
            f"[!] Lỗi nghiêm trọng: Checkpoint '{checkpoint_path}' không chứa calibration trong metadata, "
            f"và không có tệp --calibration được truyền vào. Theo đặc tả Task 7.15, mô hình yêu cầu calibration đo thật từ val_calib."
        )

    calib_ckpt_sha = calib_dict.get("checkpoint", {}).get("sha256")
    if calib_ckpt_sha and sha256_to_use and calib_ckpt_sha != sha256_to_use:
        raise ValueError(
            f"[!] Lỗi nghiêm trọng: Tệp calibration thuộc checkpoint khác!\n"
            f"  - Checkpoint SHA-256   : {sha256_to_use}\n"
            f"  - Calibration CKPT SHA : {calib_ckpt_sha}\n"
            f"Không được dùng calibration của baseline cho checkpoint v2!"
        )

    unified = UnifiedFatFormerXLA(
        backbone=backbone,
        calibration=calib_dict,
        router_params=router_dict,
        router_ablation=router_ablation,
        checkpoint_sha256=sha256_to_use
    ).to(device).eval()

    return unified, metadata


def benchmark_latency(
    model: UnifiedFatFormerXLA,
    device: torch.device,
    batch_sizes: List[int] = (1, 8, 32),
    n_warmup: int = 15,
    n_steps: int = 40,
    output_json: Optional[str] = None
) -> Dict[str, Any]:
    """Đo đạc độ trễ phân tầng: Backbone vs Unified Package, tính overhead Delta T."""
    results = {
        "device": str(device),
        "precision": "FP16" if device.type == "cuda" else "FP32",
        "batches": {},
        "overhead_meets_target": None
    }

    print("\n" + "=" * 80)
    print("TIẾN HÀNH ĐO ĐẠC ĐỘ TRỄ SUY LUẬN (LATENCY OVERHEAD BENCHMARK - G7.3)")
    print(f"Thiết bị: {device} | Precision: {results['precision']} | Warmup: {n_warmup} | Steps: {n_steps}")
    print("=" * 80)

    for b in batch_sizes:
        x = torch.randn(b, 3, 224, 224, device=device)
        times_backbone = []
        times_package = []

        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                # 1. Warmup
                for _ in range(n_warmup):
                    _ = model.backbone(x)
                    _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

                # 2. Đo Backbone thô
                for _ in range(n_steps):
                    t0 = time.perf_counter()
                    _ = model.backbone(x)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times_backbone.append((time.perf_counter() - t0) * 1000.0)

                # 3. Đo Unified Wrapper (Package)
                for _ in range(n_steps):
                    t0 = time.perf_counter()
                    _ = model(x)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times_package.append((time.perf_counter() - t0) * 1000.0)

        bb_median_batch = float(np.median(times_backbone))
        pkg_median_batch = float(np.median(times_package))
        pkg_p95_batch = float(np.percentile(times_package, 95))

        bb_per_img = bb_median_batch / b
        pkg_per_img = pkg_median_batch / b
        overhead_per_img = max(0.0, pkg_per_img - bb_per_img)
        throughput = (1000.0 / pkg_median_batch) * b

        results["batches"][str(b)] = {
            "batch_size": b,
            "backbone_batch_ms": round(bb_median_batch, 2),
            "package_batch_median_ms": round(pkg_median_batch, 2),
            "package_batch_p95_ms": round(pkg_p95_batch, 2),
            "backbone_per_image_ms": round(bb_per_img, 2),
            "package_per_image_ms": round(pkg_per_img, 2),
            "overhead_delta_ms": round(overhead_per_img, 4),
            "throughput_fps": round(throughput, 1)
        }

        print(f"• Batch {b:>2}: Package = {pkg_median_batch:>6.2f} ms/batch | {pkg_per_img:>5.2f} ms/ảnh | Overhead = {overhead_per_img:>6.4f} ms | {throughput:>5.1f} FPS")

    # Kiểm tra tiêu chuẩn G7.3: Overhead do wrapper thêm vào < 5.0 ms/ảnh (chỉ nghiệm thu trên GPU CUDA)
    if device.type == "cuda":
        b1_overhead = results["batches"].get("1", {}).get("overhead_delta_ms", 999.0)
        results["overhead_meets_target"] = bool(b1_overhead < 5.0)
    else:
        results["overhead_meets_target"] = None
        results["notes"] = "Kiểm chuẩn chạy trên CPU; mục tiêu G7.3 (<5.0 ms overhead) chỉ nghiệm thu chính thức trên GPU T4."

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[✓] Đã xuất báo cáo độ trễ ra: {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="FatFormer-XLA Unified Inference CLI (Task 7.15)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn file .pth (đóng gói hoặc thô)")
    parser.add_argument("--calibration", type=str, default=None, help="Đường dẫn file calibration JSON chính chủ (nếu checkpoint chưa đóng gói)")
    parser.add_argument("--images", nargs="+", default=[], help="Danh sách ảnh, thư mục hoặc pattern")
    parser.add_argument("--clip_path", type=str, default="DATASET/pretrained/ViT-L-14.pt")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file JSON lưu kết quả suy luận")
    parser.add_argument("--batch_size", type=int, default=1, help="Kích thước batch suy luận")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--benchmark_latency", action="store_true", help="Chạy kiểm chuẩn độ trễ phân tầng (B=1,8,32)")
    parser.add_argument("--latency_out", type=str, default=None, help="Đường dẫn lưu file JSON độ trễ")
    args = parser.parse_args()

    # Xác định device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[*] Khởi tạo môi trường suy luận trên thiết bị: {device}")
    unified_model, metadata = load_unified_model(
        args.checkpoint,
        calibration_path=args.calibration,
        clip_path=args.clip_path,
        device=str(device)
    )
    print("[✓] Nạp UnifiedFatFormerXLA thành công!")

    # Nếu có cờ đo latency
    if args.benchmark_latency:
        benchmark_latency(unified_model, device=device, output_json=args.latency_out)

    if not args.images:
        if not args.benchmark_latency:
            print("[!] Không có ảnh đầu vào (--images). Hoàn tất khởi động.")
        return

    # Thu thập ảnh
    image_paths = collect_image_paths(args.images)
    if not image_paths:
        print("[!] Không tìm thấy ảnh hợp lệ nào từ danh sách đầu vào.")
        return

    print(f"[*] Tìm thấy {len(image_paths)} ảnh hợp lệ. Đang tiến hành suy luận...")
    transform = get_default_transform()
    results = []

    # Suy luận phân tầng theo batch
    for idx in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[idx : idx + args.batch_size]
        tensors = []
        valid_paths = []

        for p in batch_paths:
            try:
                with Image.open(p) as img:
                    img_rgb = img.convert("RGB")
                    tensor = transform(img_rgb)
                    tensors.append(tensor)
                    valid_paths.append(p)
            except Exception as e:
                print(f"[SKIP] {p}: Không thể mở hoặc lỗi định dạng ({e})")

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors, dim=0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = unified_model.predict(batch_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        ms_per_img = t_elapsed_ms / len(valid_paths)

        probs = outputs["prob_fake"].cpu().numpy()
        labels = outputs["label"].cpu().numpy()
        w_degs = outputs["w_deg"].cpu().numpy()
        thrs = outputs["threshold"].cpu().numpy()

        for p, prob, lbl, w, thr in zip(valid_paths, probs, labels, w_degs, thrs):
            lbl_str = "fake" if lbl == 1 else "real"
            results.append({
                "path": p,
                "prob_fake": round(float(prob), 4),
                "label": lbl_str,
                "w_deg": round(float(w), 4),
                "threshold": round(float(thr), 4),
                "latency_ms": round(float(ms_per_img), 2)
            })

    # In bảng kết quả
    print("\n" + "=" * 90)
    print(f"{'Ảnh (Path)':<45} | {'Prob Fake':<10} | {'Dự Đoán':<8} | {'Ngưỡng':<8} | {'Latency':<8}")
    print("-" * 90)
    for r in results[:50]:  # In tối đa 50 dòng nếu danh sách dài
        p_name = os.path.basename(r["path"])
        if len(p_name) > 42:
            p_name = p_name[:39] + "..."
        print(f"{p_name:<45} | {r['prob_fake']:<10.4f} | {r['label'].upper():<8} | {r['threshold']:<8.4f} | {r['latency_ms']:<6.1f} ms")
    if len(results) > 50:
        print(f"... và {len(results) - 50} ảnh khác.")
    print("=" * 90)

    # Xuất file kết quả JSON nếu có
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        final_doc = {
            "checkpoint": args.checkpoint,
            "layout": metadata.get("layout", {}),
            "total_images": len(results),
            "results": results
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_doc, f, indent=2, ensure_ascii=False)
        print(f"[✓] Đã lưu kết quả suy luận ra: {args.output}")


if __name__ == "__main__":
    main()
