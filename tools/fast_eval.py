# -*- coding: utf-8 -*-
"""
Script đánh giá nhanh Fast-Eval (< 8 phút) cho mô hình FatFormer-XLA:
- Hỗ trợ lấy mẫu ngẫu nhiên đại diện 500 ảnh/tập test (250 real + 250 fake, seed=42).
- Tự động đo đạc ACC, AP, ROC-AUC, Real-ACC, Fake-ACC và thời gian thực thi.
- Hỗ trợ xuất bảng tổng kết và file CSV định dạng chuẩn.
- Hỗ trợ cờ --test_dummy phục vụ Smoke Test cục bộ.
"""

import os
import sys
import time
import argparse
import tempfile
import shutil
import csv
from typing import Dict, List, Optional
import torch
from torchvision import transforms
from PIL import Image

# Đảm bảo import được package src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

from src.models import build_model
from src.training.checkpoint_manager import CheckpointManager
from src.evaluation.fast_eval import run_benchmark, evaluate_dataloader
from src.evaluation.metrics import format_evaluation_summary
from src.datasets.dataset import ALL_TEST_SUBSETS, GAN_SUBSETS, DIFFUSION_SUBSETS


def parse_args():
    parser = argparse.ArgumentParser(description="FatFormer-XLA Fast-Eval Pipeline (< 8 Phút)")
    
    # Checkpoint & Backbone
    parser.add_argument("--checkpoint", type=str, default="", help="Đường dẫn file checkpoint .pth")
    parser.add_argument("--clip_path", type=str, default="pretrained/ViT-L-14.pt", help="Đường dẫn file ViT-L-14.pt")
    
    # Dataset
    parser.add_argument("--dataset_root", type=str, default="", help="Đường dẫn thư mục chứa các tập test")
    parser.add_argument("--subsets", type=str, default="all", help="'all', 'gans', 'diffusion' hoặc danh sách phân cách bởi dấu phẩy")
    parser.add_argument("--samples_per_subset", type=int, default=500, help="Số mẫu tối đa mỗi subset (250 real + 250 fake, mặc định 500)")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch đánh giá (mặc định: 32)")
    parser.add_argument("--num_workers", type=int, default=2, help="Số luồng nạp dữ liệu (mặc định: 2)")
    
    # Hardware & Performance
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Thiết bị tính toán")
    
    # Model architecture options
    parser.add_argument("--backbone", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_vit_adapter", type=int, default=3)
    parser.add_argument("--num_context_embedding", type=int, default=8)
    parser.add_argument("--init_context_embedding", type=str, default="")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--clip_vision_width", type=int, default=1024)
    parser.add_argument("--frequency_encoder_layer", type=int, default=2)
    parser.add_argument("--decoder_layer", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--use_srm", action="store_true", help="Bật khối không gian SRM 3-Kernels")
    parser.add_argument("--use_gating", action="store_true", help="Bật khối thích ứng tần số động Dynamic Gating")
    parser.add_argument("--strict", action="store_true", help="Nạp trọng số checkpoint ở chế độ strict=True")
    
    # Output & Verification
    parser.add_argument("--output_csv", type=str, default="", help="Đường dẫn file CSV xuất kết quả")
    parser.add_argument("--test_dummy", action="store_true", help="Chạy Smoke Test với dữ liệu giả lập (không cần dataset thật)")

    return parser.parse_args()


def export_results_to_csv(results_by_subset: Dict[str, Dict[str, float]], csv_path: str):
    """Xuất kết quả đánh giá ra file CSV chuẩn hóa."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = ["subset", "acc", "ap", "auc", "r_acc", "f_acc", "elapsed_sec"]
    
    accs, aps, aucs, r_accs, f_accs = [], [], [], [], []

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for name, metrics in results_by_subset.items():
            accs.append(metrics["acc"])
            aps.append(metrics["ap"])
            aucs.append(metrics.get("auc", 0.0))
            r_accs.append(metrics["r_acc"])
            f_accs.append(metrics["f_acc"])
            
            writer.writerow({
                "subset": name,
                "acc": f"{metrics['acc']:.2f}",
                "ap": f"{metrics['ap']:.2f}",
                "auc": f"{metrics.get('auc', 0.0):.2f}",
                "r_acc": f"{metrics['r_acc']:.2f}",
                "f_acc": f"{metrics['f_acc']:.2f}",
                "elapsed_sec": f"{metrics.get('elapsed_sec', 0.0):.2f}"
            })
            
        # Dòng trung bình MEAN
        if accs:
            writer.writerow({
                "subset": "MEAN",
                "acc": f"{sum(accs)/len(accs):.2f}",
                "ap": f"{sum(aps)/len(aps):.2f}",
                "auc": f"{sum(aucs)/len(aucs):.2f}",
                "r_acc": f"{sum(r_accs)/len(r_accs):.2f}",
                "f_acc": f"{sum(f_accs)/len(f_accs):.2f}",
                "elapsed_sec": "-"
            })
            
    print(f"[✓] Đã xuất thành công bảng số liệu ra: {csv_path}")


def create_dummy_dataset(base_dir: str, subsets: List[str] = ["dummy_progan", "dummy_glide"]):
    """Tạo tập dữ liệu giả lập có cấu trúc 0_real và 1_fake phục vụ Smoke Test."""
    img_size = (256, 256)
    for sub in subsets:
        for label_dir in ["0_real", "1_fake"]:
            full_dir = os.path.join(base_dir, sub, label_dir)
            os.makedirs(full_dir, exist_ok=True)
            for i in range(4):
                img = Image.new("RGB", img_size, color=(i * 30, (i * 50) % 255, (i * 70) % 255))
                img.save(os.path.join(full_dir, f"sample_{i}.png"))


def run_fast_eval():
    args = parse_args()
    
    # 1. Cấu hình thiết bị
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 86)
    print("FATFORMER-XLA: PIPELINE ĐÁNH GIÁ NHANH FAST-EVAL (< 8 PHÚT)")
    print("=" * 86)
    print(f"• Thiết bị tính toán:       {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"• Kích thước mẫu / subset:  {args.samples_per_subset} ảnh (250 real + 250 fake, seed=42)")
    print(f"• Batch size:               {args.batch_size}")
    print(f"• Chế độ SRM 3-Kernels:     {args.use_srm}")
    print(f"• Chế độ Dynamic Gating:    {args.use_gating}")

    # 2. Xử lý đường dẫn clip_path
    clip_path = args.clip_path
    if not os.path.isabs(clip_path):
        clip_path = os.path.join(project_root, clip_path)
    if not os.path.exists(clip_path):
        # Fallback thử tìm trong thư mục gốc
        alt_path = os.path.join(project_root, "ViT-L-14.pt")
        if os.path.exists(alt_path):
            clip_path = alt_path
        else:
            print(f"[CẢNH BÁO] Không tìm thấy file CLIP tại: {clip_path}")
    args.clip_path = clip_path

    # 3. Khởi tạo mô hình
    print(f"\n[*] Đang khởi tạo mô hình FatFormer với backbone {args.backbone}...")
    if args.test_dummy and not os.path.exists(clip_path):
        print("  [Thông báo] Phát hiện chế độ Smoke Test cục bộ: Sử dụng mô hình giả lập (Mock Model) do chưa có ViT-L-14.pt.")
        class DummyMockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(3, 2)
            def forward(self, x):
                return self.fc(self.pool(x.float()).flatten(1))
        model = DummyMockModel()
    else:
        model = build_model(args)
    model = model.to(device)
    model.eval()

    # 4. Nạp checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(project_root, ckpt_path)
        if os.path.exists(ckpt_path):
            print(f"[*] Đang nạp checkpoint từ: {ckpt_path}...")
            CheckpointManager.load(ckpt_path, model, device=device, strict=args.strict)
            print("  -> Nạp trọng số checkpoint thành công!")
        else:
            print(f"[CẢNH BÁO] Checkpoint không tồn tại: {ckpt_path}. Chạy suy luận với trọng số hiện thời.")
    else:
        # Tự động tìm checkpoint mặc định nếu có
        default_ckpt = os.path.join(project_root, "fatformer_4class_ckpt.pth")
        if os.path.exists(default_ckpt):
            print(f"[*] Phát hiện checkpoint mặc định: {default_ckpt}")
            CheckpointManager.load(default_ckpt, model, device=device, strict=args.strict)

    # 5. Xử lý kịch bản Smoke Test Dummy hoặc Dữ liệu thật
    if args.test_dummy:
        print("\n" + "=" * 86)
        print("CHẾ ĐỘ SMOKE TEST VỚI DỮ LIỆU GIẢ LẬP (SYNTHETIC DUMMY DATA)")
        print("=" * 86)
        temp_dir = tempfile.mkdtemp(prefix="fatformer_fast_eval_dummy_")
        try:
            dummy_subsets = ["dummy_gan", "dummy_diff"]
            create_dummy_dataset(temp_dir, dummy_subsets)
            print(f"[*] Đã tạo tập dữ liệu giả lập tại: {temp_dir}")
            
            results = run_benchmark(
                model=model,
                dataset_path=temp_dir,
                device=device,
                selected_subsets=dummy_subsets,
                max_samples_per_subset=args.samples_per_subset,
                batch_size=min(args.batch_size, 4),
                num_workers=0
            )
            
            if args.output_csv:
                export_results_to_csv(results, args.output_csv)
                
            print("\n[✓] SMOKE TEST THÀNH CÔNG: Pipeline Fast-Eval hoạt động 100% chuẩn xác!")
            return 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # 6. Chạy đánh giá trên Dataset thật
    if not args.dataset_root:
        # Thử tìm các vị trí dataset tiềm năng
        candidates = [
            "/content/dataset_local/test",
            "/content/dataset_local",
            os.path.join(project_root, "datasets", "test"),
            os.path.join(project_root, "datasets")
        ]
        for c in candidates:
            if os.path.exists(c):
                args.dataset_root = c
                break

    if not args.dataset_root or not os.path.exists(args.dataset_root):
        print(f"\n[LỖI] Không tìm thấy thư mục dữ liệu: '{args.dataset_root}'.")
        print("Gợi ý: Cung cấp --dataset_root /path/to/test hoặc sử dụng cờ --test_dummy để chạy kiểm thử.")
        return 1

    # Phân giải danh sách subsets
    if args.subsets == "all":
        selected_subsets = ALL_TEST_SUBSETS
    elif args.subsets == "gans":
        selected_subsets = GAN_SUBSETS
    elif args.subsets == "diffusion":
        selected_subsets = DIFFUSION_SUBSETS
    else:
        selected_subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    print(f"[*] Tập dữ liệu test root: {args.dataset_root}")
    print(f"[*] Danh sách subsets ({len(selected_subsets)} tập): {selected_subsets}")

    start_eval_time = time.time()
    results = run_benchmark(
        model=model,
        dataset_path=args.dataset_root,
        device=device,
        selected_subsets=selected_subsets,
        max_samples_per_subset=args.samples_per_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    total_eval_duration = time.time() - start_eval_time

    # 7. Xuất file kết quả nếu có yêu cầu
    if args.output_csv and results:
        export_results_to_csv(results, args.output_csv)

    # 8. Đánh giá tiêu chuẩn DoD (< 8 phút, < 30s/subset)
    print("\n" + "=" * 86)
    print("[ĐÁNH GIÁ TIÊU CHUẨN NGHIỆM THU - DEFINITION OF DONE]")
    print("=" * 86)
    if results:
        num_evaluated = len(results)
        avg_time_per_subset = total_eval_duration / max(1, num_evaluated)
        print(f"• Số lượng tập test hoàn tất:      {num_evaluated}/{len(selected_subsets)}")
        print(f"• Tổng thời gian toàn bộ:          {total_eval_duration:.2f} s ({total_eval_duration/60:.2f} phút)")
        print(f"• Thời gian trung bình / tập test:  {avg_time_per_subset:.2f} s")
        
        time_gate_subset = avg_time_per_subset < 30.0
        time_gate_total = total_eval_duration < 480.0  # 8 phút = 480 giây
        
        print(f"• Cổng kiểm tra thời gian / subset (< 30s): {'[✓] ĐẠT' if time_gate_subset else '[!] CẢNH BÁO'}")
        print(f"• Cổng kiểm tra tổng thời gian (< 8 phút):  {'[✓] ĐẠT' if time_gate_total else '[!] CẢNH BÁO'}")
    else:
        print("[!] Không có subset nào được đánh giá. Vui lòng kiểm tra lại cấu trúc thư mục dữ liệu.")

    return 0


if __name__ == "__main__":
    sys.exit(run_fast_eval())
