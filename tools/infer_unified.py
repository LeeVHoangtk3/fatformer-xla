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
import gc
from datetime import datetime, timezone
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
from tools.package_unified_fatformer import validate_calibration_against_checkpoint, compute_file_sha256


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

    raw_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
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

    sha256_to_use = ckpt_sha or compute_file_sha256(checkpoint_path)

    # Nếu người dùng truyền --calibration từ bên ngoài
    if calibration_path and not os.path.isfile(calibration_path):
        raise FileNotFoundError(calibration_path)
    if calibration_path:
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
    validate_calibration_against_checkpoint(calib_dict, sha256_to_use)
    # The already mapped payload avoids a second multi-GB checkpoint read.
    backbone, load_report = load_canonical(raw_payload, model_args)
    backbone = backbone.to(device).eval()

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
    n_warmup: int = 20,
    n_steps: int = 200,
    output_json: Optional[str] = None,
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """Đo đạc độ trễ phân tầng: Backbone vs Unified Package, tính overhead Delta T."""
    results = {
        "device": str(device),
        "precision": "FP16" if device.type == "cuda" else "FP32",
        "batches": {},
        "overhead_meets_target": None
    }
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    results['allow_tf32'] = False
    if n_steps < 2 or n_warmup < 1 or any(b <= 0 for b in batch_sizes):
        raise ValueError("Require positive batches/warmup and at least two timing steps")
    results["schema"] = "unified_latency/v3"
    results["gpu_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    results["protocol"] = dict(warmup=n_warmup, steps=n_steps,
                               order="paired, alternating backbone/package first",
                               input="synthetic normalized tensor, excludes I/O/preprocessing")
    results["target_gpu"] = "allocated CUDA GPU"
    results["hardware_protocol"] = "any_compatible_cuda_gpu_with_sufficient_memory"
    results["original_t4_reference_meets_target"] = None
    results["requested_batch_sizes"] = list(batch_sizes)
    if device.type == "cuda":
        free,total = torch.cuda.mem_get_info(device)
        results["total_vram_gib"] = total/(1024**3)
        results["free_vram_gib_at_benchmark_start"] = free/(1024**3)
    results["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    if checkpoint_path:
        results["checkpoint_sha256"] = compute_file_sha256(checkpoint_path)

    print("\n" + "=" * 80)
    print("TIẾN HÀNH ĐO ĐẠC ĐỘ TRỄ SUY LUẬN (LATENCY OVERHEAD BENCHMARK - G7.3)")
    print(f"Thiết bị: {device} | Precision: {results['precision']} | Warmup: {n_warmup} | Steps: {n_steps}")
    print("=" * 80)

    def measure_batch(b):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(b, 3, 224, 224, device=device)
        times_backbone = []
        times_package = []

        with torch.no_grad():
            with torch.amp.autocast(device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                # 1. Warmup
                for _ in range(n_warmup):
                    _ = model.backbone(x)
                    _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

                for step in range(n_steps):
                    calls = [(model.backbone, times_backbone), (model, times_package)]
                    if step % 2:
                        calls.reverse()
                    for call, timings in calls:
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        t0 = time.perf_counter()
                        _ = call(x)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        timings.append((time.perf_counter() - t0) * 1000.0)

        bb_median_batch = float(np.median(times_backbone))
        pkg_median_batch = float(np.median(times_package))
        pkg_p95_batch = float(np.percentile(times_package, 95))

        bb_per_img = bb_median_batch / b
        pkg_per_img = pkg_median_batch / b
        deltas = (np.array(times_package) - np.array(times_backbone)) / b
        overhead_per_img = float(np.median(deltas))
        throughput = (1000.0 / pkg_median_batch) * b

        results["batches"][str(b)] = {
            "batch_size": b,
            "backbone_batch_ms": round(bb_median_batch, 2),
            "package_batch_median_ms": round(pkg_median_batch, 2),
            "package_batch_p95_ms": round(pkg_p95_batch, 2),
            "backbone_per_image_ms": round(bb_per_img, 2),
            "package_per_image_ms": round(pkg_per_img, 2),
            "overhead_delta_ms": overhead_per_img,
            "throughput_fps": round(throughput, 1)
        }
        results["batches"][str(b)].update(
            overhead_delta_p95_ms=float(np.percentile(deltas, 95)),
            backbone_samples_ms=times_backbone, package_samples_ms=times_package,
            paired_delta_per_image_ms=deltas.tolist())

        print(f"• Batch {b:>2}: Package = {pkg_median_batch:>6.2f} ms/batch | {pkg_per_img:>5.2f} ms/ảnh | Overhead = {overhead_per_img:>6.4f} ms | {throughput:>5.1f} FPS")

        results["batches"][str(b)]["status"] = "measured"
        if device.type == 'cuda':
            results["batches"][str(b)]["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device)/(1024**3)

    for b in batch_sizes:
        oom = False
        try:
            measure_batch(b)
        except torch.OutOfMemoryError:
            if device.type != 'cuda':
                raise
            oom = True
            results["batches"][str(b)] = dict(batch_size=b, status="out_of_memory",
                                             reason="Batch does not fit available VRAM; no timing reported")
            print(f"Batch {b}: out of memory; other batches remain valid", flush=True)
        if oom:
            gc.collect()
            torch.cuda.empty_cache()

    # Hardware requirement revised by the user: evaluate the GPU actually used.
    results["observed_device_meets_5ms"] = None
    b1_overhead = results["batches"].get("1", {}).get("overhead_delta_ms")
    if device.type == "cuda" and b1_overhead is not None and np.isfinite(b1_overhead):
        results["overhead_meets_target"] = bool(b1_overhead < 5.0)
        results["observed_device_meets_5ms"] = results["overhead_meets_target"]
        if "T4" in results["gpu_name"]:
            results["original_t4_reference_meets_target"] = results["overhead_meets_target"]
        results["notes"] = "Acceptance applies to the reported GPU only; T4 reference is informational."
    else:
        results["notes"] = "No valid CUDA batch-1 timing; GPU overhead acceptance remains unavailable."

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[✓] Đã xuất báo cáo độ trễ ra: {output_json}")

    return results


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="FatFormer-XLA Unified Inference CLI (Task 7.15)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn file .pth (đóng gói hoặc thô)")
    parser.add_argument("--calibration", type=str, default=None, help="Đường dẫn file calibration JSON chính chủ (nếu checkpoint chưa đóng gói)")
    parser.add_argument("--images", nargs="+", default=[], help="Danh sách ảnh, thư mục hoặc pattern")
    parser.add_argument("--clip_path", type=str, default="DATASET/pretrained/ViT-L-14.pt")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file JSON lưu kết quả suy luận")
    parser.add_argument("--batch_size", type=int, default=1, help="Kích thước batch suy luận")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--benchmark_latency", action="store_true", help="Chạy kiểm chuẩn độ trễ phân tầng (B=1,8,32)")
    parser.add_argument("--latency_batches", nargs="+", type=int, default=[1,8,32], help="Requested latency batches; CUDA OOM is recorded per batch")
    parser.add_argument("--latency_out", type=str, default=None, help="Đường dẫn lưu file JSON độ trễ")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")

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
        benchmark_latency(unified_model, device=device, batch_sizes=args.latency_batches, output_json=args.latency_out,
                          checkpoint_path=args.checkpoint)

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
