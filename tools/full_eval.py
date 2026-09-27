# -*- coding: utf-8 -*-
"""
Script đánh giá toàn diện Full Benchmark (Task 2.2 / Task 5.1) cho FatFormer-XLA:
- Đánh giá 100% mẫu ảnh trên 18 tập test chuẩn (8 GANs + 10 Diffusion).
- Tự động phân nhóm và tính toán 3 dòng tổng kết: MEAN GANs, MEAN Diffusion, MEAN Overall.
- Tự động đối chiếu mốc sàn học thuật paper CVPR 2024 (Dung sai DoD <= +-0.5%).
- Xuất dữ liệu định dạng CSV và Markdown phục vụ báo cáo khoa học.
- Hỗ trợ cờ --test_dummy phục vụ Smoke Test cục bộ.
"""

import os
import sys
import time
import argparse
import tempfile
import shutil
import csv
from typing import Dict, List, Tuple, Optional
import numpy as np
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
from src.evaluation.fast_eval import run_benchmark
from src.datasets.dataset import ALL_TEST_SUBSETS, GAN_SUBSETS, DIFFUSION_SUBSETS


PAPER_BASELINE_TARGETS = {
    "gans_mean_acc": 98.4,
    "diffusion_mean_acc": 95.0,
    "guided_acc": 76.1
}


def parse_args():
    parser = argparse.ArgumentParser(description="FatFormer-XLA Full-Benchmark Clean Pipeline (Task 2.2)")
    
    # Checkpoint & Backbone
    parser.add_argument("--checkpoint", type=str, default="", help="Đường dẫn file checkpoint .pth")
    parser.add_argument("--clip_path", type=str, default="pretrained/ViT-L-14.pt", help="Đường dẫn file ViT-L-14.pt")
    
    # Dataset
    parser.add_argument("--test_root", "--dataset_root", dest="dataset_root", type=str, default="", 
                        help="Đường dẫn thư mục chứa 18 tập test Clean đã giải nén")
    parser.add_argument("--subsets", type=str, default="all", 
                        help="'all', 'gans', 'diffusion' hoặc danh sách phân cách bởi dấu phẩy")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch đánh giá (mặc định: 32)")
    parser.add_argument("--num_workers", type=int, default=4, help="Số luồng nạp dữ liệu (mặc định: 4)")
    
    # Hardware & Model Architecture
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Thiết bị tính toán")
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
    parser.add_argument("--use_srm", action="store_true", help="Bật nhánh SRM (mặc định False cho Baseline gốc)")
    parser.add_argument("--use_gating", action="store_true", help="Bật nhánh Gating (mặc định False cho Baseline gốc)")
    parser.add_argument("--strict", action="store_true", default=True, help="Nạp checkpoint ở chế độ strict=True")
    
    # Output & Verification
    parser.add_argument("--output_csv", type=str, default="", help="Đường dẫn file CSV xuất kết quả")
    parser.add_argument("--output_markdown", type=str, default="", help="Đường dẫn file Markdown xuất kết quả")
    parser.add_argument("--test_dummy", action="store_true", help="Chạy Smoke Test với dữ liệu giả lập")

    return parser.parse_args()


def calculate_group_mean(metrics_dict: Dict[str, Dict[str, float]], group_subsets: List[str]) -> Optional[Dict[str, float]]:
    """Tính trung bình các chỉ số cho một danh sách subset cụ thể."""
    accs, aps, aucs, r_accs, f_accs = [], [], [], [], []
    for sub in group_subsets:
        if sub in metrics_dict:
            m = metrics_dict[sub]
            accs.append(m["acc"])
            aps.append(m["ap"])
            aucs.append(m.get("auc", 0.0))
            r_accs.append(m["r_acc"])
            f_accs.append(m["f_acc"])
            
    if not accs:
        return None
        
    return {
        "acc": float(np.mean(accs)),
        "ap": float(np.mean(aps)),
        "auc": float(np.mean(aucs)),
        "r_acc": float(np.mean(r_accs)),
        "f_acc": float(np.mean(f_accs)),
    }


def format_grouped_summary(results_by_subset: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, Optional[Dict[str, float]]]]:
    """
    Tạo bảng tổng hợp kết quả phân nhóm GANs (8 tập) và Diffusion (10 tập) kèm 3 dòng MEAN.
    """
    lines = []
    lines.append("=" * 86)
    lines.append("BẢNG KẾT QUẢ BENCHMARK BASELINE TOÀN DIỆN (CVPR 2024 PAPER REPRODUCTION)")
    lines.append("=" * 86)
    lines.append(f"{'Index':<6} {'Subset':<16} {'ACC (%)':<10} {'AP (%)':<10} {'AUC (%)':<10} {'Real ACC (%)':<14} {'Fake ACC (%)':<14}")
    lines.append("-" * 86)

    # 1. Nhóm GANs
    lines.append("--- [NHÓM 1: 8 KIẾN TRÚC GANS] ---")
    gan_evaluated = []
    for idx, name in enumerate(GAN_SUBSETS):
        if name in results_by_subset:
            m = results_by_subset[name]
            gan_evaluated.append(name)
            lines.append(
                f"({idx:<3})  {name:<16} {m['acc']:<10.2f} {m['ap']:<10.2f} "
                f"{m.get('auc', 0.0):<10.2f} {m['r_acc']:<14.2f} {m['f_acc']:<14.2f}"
            )
            
    mean_gans = calculate_group_mean(results_by_subset, GAN_SUBSETS)
    if mean_gans:
        lines.append("-" * 86)
        lines.append(
            f"{'MEAN GANS (8)':<23} {mean_gans['acc']:<10.2f} {mean_gans['ap']:<10.2f} "
            f"{mean_gans['auc']:<10.2f} {mean_gans['r_acc']:<14.2f} {mean_gans['f_acc']:<14.2f}"
        )
    lines.append("-" * 86)

    # 2. Nhóm Diffusion
    lines.append("--- [NHÓM 2: 10 KIẾN TRÚC DIFFUSION] ---")
    diff_evaluated = []
    start_diff_idx = len(GAN_SUBSETS)
    for idx, name in enumerate(DIFFUSION_SUBSETS):
        if name in results_by_subset:
            m = results_by_subset[name]
            diff_evaluated.append(name)
            lines.append(
                f"({start_diff_idx + idx:<3})  {name:<16} {m['acc']:<10.2f} {m['ap']:<10.2f} "
                f"{m.get('auc', 0.0):<10.2f} {m['r_acc']:<14.2f} {m['f_acc']:<14.2f}"
            )
            
    mean_diff = calculate_group_mean(results_by_subset, DIFFUSION_SUBSETS)
    if mean_diff:
        lines.append("-" * 86)
        lines.append(
            f"{'MEAN DIFF (10)':<23} {mean_diff['acc']:<10.2f} {mean_diff['ap']:<10.2f} "
            f"{mean_diff['auc']:<10.2f} {mean_diff['r_acc']:<14.2f} {mean_diff['f_acc']:<14.2f}"
        )
    lines.append("=" * 86)

    # 3. Các subset khác (nếu có)
    other_subsets = [s for s in results_by_subset if s not in GAN_SUBSETS and s not in DIFFUSION_SUBSETS]
    if other_subsets:
        lines.append("--- [NHÓM KHÁC / DUMMY SUBSETS] ---")
        for idx, name in enumerate(other_subsets):
            m = results_by_subset[name]
            lines.append(
                f"(*{idx:<2})  {name:<16} {m['acc']:<10.2f} {m['ap']:<10.2f} "
                f"{m.get('auc', 0.0):<10.2f} {m['r_acc']:<14.2f} {m['f_acc']:<14.2f}"
            )
        lines.append("-" * 86)

    # 4. Toàn bộ Overall
    all_evaluated = list(results_by_subset.keys())
    mean_overall = calculate_group_mean(results_by_subset, all_evaluated)
    if mean_overall:
        lines.append(
            f"{'MEAN OVERALL':<23} {mean_overall['acc']:<10.2f} {mean_overall['ap']:<10.2f} "
            f"{mean_overall['auc']:<10.2f} {mean_overall['r_acc']:<14.2f} {mean_overall['f_acc']:<14.2f}"
        )
    lines.append("=" * 86)

    summary_dict = {
        "mean_gans": mean_gans,
        "mean_diff": mean_diff,
        "mean_overall": mean_overall
    }
    return "\n".join(lines), summary_dict


def export_full_results(results_by_subset: Dict[str, Dict[str, float]], 
                        summary_dict: Dict[str, Optional[Dict[str, float]]],
                        csv_path: str = "", markdown_path: str = ""):
    """Xuất kết quả chi tiết ra file CSV và Markdown."""
    fieldnames = ["group", "subset", "acc", "ap", "auc", "r_acc", "f_acc", "elapsed_sec"]
    
    # 1. Xuất CSV
    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Ghi GANs
            for name in GAN_SUBSETS:
                if name in results_by_subset:
                    m = results_by_subset[name]
                    writer.writerow({
                        "group": "GANs", "subset": name,
                        "acc": f"{m['acc']:.2f}", "ap": f"{m['ap']:.2f}", "auc": f"{m.get('auc', 0.0):.2f}",
                        "r_acc": f"{m['r_acc']:.2f}", "f_acc": f"{m['f_acc']:.2f}",
                        "elapsed_sec": f"{m.get('elapsed_sec', 0.0):.2f}"
                    })
            if summary_dict.get("mean_gans"):
                mg = summary_dict["mean_gans"]
                writer.writerow({
                    "group": "GANs", "subset": "MEAN_GANS",
                    "acc": f"{mg['acc']:.2f}", "ap": f"{mg['ap']:.2f}", "auc": f"{mg['auc']:.2f}",
                    "r_acc": f"{mg['r_acc']:.2f}", "f_acc": f"{mg['f_acc']:.2f}", "elapsed_sec": "-"
                })

            # Ghi Diffusion
            for name in DIFFUSION_SUBSETS:
                if name in results_by_subset:
                    m = results_by_subset[name]
                    writer.writerow({
                        "group": "Diffusion", "subset": name,
                        "acc": f"{m['acc']:.2f}", "ap": f"{m['ap']:.2f}", "auc": f"{m.get('auc', 0.0):.2f}",
                        "r_acc": f"{m['r_acc']:.2f}", "f_acc": f"{m['f_acc']:.2f}",
                        "elapsed_sec": f"{m.get('elapsed_sec', 0.0):.2f}"
                    })
            if summary_dict.get("mean_diff"):
                md = summary_dict["mean_diff"]
                writer.writerow({
                    "group": "Diffusion", "subset": "MEAN_DIFFUSION",
                    "acc": f"{md['acc']:.2f}", "ap": f"{md['ap']:.2f}", "auc": f"{md['auc']:.2f}",
                    "r_acc": f"{md['r_acc']:.2f}", "f_acc": f"{md['f_acc']:.2f}", "elapsed_sec": "-"
                })

            # Ghi Other
            for name, m in results_by_subset.items():
                if name not in GAN_SUBSETS and name not in DIFFUSION_SUBSETS:
                    writer.writerow({
                        "group": "Other", "subset": name,
                        "acc": f"{m['acc']:.2f}", "ap": f"{m['ap']:.2f}", "auc": f"{m.get('auc', 0.0):.2f}",
                        "r_acc": f"{m['r_acc']:.2f}", "f_acc": f"{m['f_acc']:.2f}",
                        "elapsed_sec": f"{m.get('elapsed_sec', 0.0):.2f}"
                    })

            # Ghi Overall
            if summary_dict.get("mean_overall"):
                mo = summary_dict["mean_overall"]
                writer.writerow({
                    "group": "Overall", "subset": "MEAN_OVERALL",
                    "acc": f"{mo['acc']:.2f}", "ap": f"{mo['ap']:.2f}", "auc": f"{mo['auc']:.2f}",
                    "r_acc": f"{mo['r_acc']:.2f}", "f_acc": f"{mo['f_acc']:.2f}", "elapsed_sec": "-"
                })
        print(f"[✓] Đã lưu kết quả CSV ra: {csv_path}")

    # 2. Xuất Markdown
    if markdown_path:
        os.makedirs(os.path.dirname(os.path.abspath(markdown_path)), exist_ok=True)
        table_text, _ = format_grouped_summary(results_by_subset)
        with open(markdown_path, mode="w", encoding="utf-8") as f:
            f.write("# BÁO CÁO KẾT QUẢ BENCHMARK BASELINE CLEAN\n\n```\n")
            f.write(table_text)
            f.write("\n```\n")
        print(f"[✓] Đã lưu báo cáo Markdown ra: {markdown_path}")


def verify_paper_reproduction(summary_dict: Dict[str, Optional[Dict[str, float]]], results_by_subset: Dict[str, Dict[str, float]]):
    """Kiểm tra điều kiện Definition of Done: sai số <= +-0.5% so với paper gốc CVPR 2024."""
    print("\n" + "=" * 86)
    print("CỔNG KIỂM TRA TÁI LẬP BÀI BÁO GỐC CVPR 2024 (DoD VERIFICATION GATE)")
    print("=" * 86)

    mg = summary_dict.get("mean_gans")
    md = summary_dict.get("mean_diff")
    guided_acc = results_by_subset.get("guided", {}).get("acc", None)

    all_pass = True

    # 1. Kiểm tra GANs Mean
    if mg:
        target = PAPER_BASELINE_TARGETS["gans_mean_acc"]
        delta = abs(mg["acc"] - target)
        status = "[✓] ĐẠT" if delta <= 0.5 else "[!] CẢNH BÁO"
        if delta > 0.5: all_pass = False
        print(f"• Trung bình GANs (8 tập):      Thực tế={mg['acc']:.2f}% | Kỳ vọng paper={target:.1f}% (Δ={delta:+.2f}%) -> {status}")
    else:
        print("• Trung bình GANs:              [Chưa đánh giá đủ tập GANs]")

    # 2. Kiểm tra Diffusion Mean
    if md:
        target = PAPER_BASELINE_TARGETS["diffusion_mean_acc"]
        delta = abs(md["acc"] - target)
        status = "[✓] ĐẠT" if delta <= 0.5 else "[!] CẢNH BÁO"
        if delta > 0.5: all_pass = False
        print(f"• Trung bình Diffusion (10 tập): Thực tế={md['acc']:.2f}% | Kỳ vọng paper={target:.1f}% (Δ={delta:+.2f}%) -> {status}")
    else:
        print("• Trung bình Diffusion:         [Chưa đánh giá đủ tập Diffusion]")

    # 3. Kiểm tra Guided Diffusion
    if guided_acc is not None:
        target = PAPER_BASELINE_TARGETS["guided_acc"]
        delta = abs(guided_acc - target)
        status = "[✓] ĐẠT" if delta <= 0.5 else "[!] CẢNH BÁO"
        if delta > 0.5: all_pass = False
        print(f"• Tập khó nhất (Guided Diff):   Thực tế={guided_acc:.2f}% | Kỳ vọng paper={target:.1f}% (Δ={delta:+.2f}%) -> {status}")

    print("-" * 86)
    if all_pass:
        print("[✓] KẾT LUẬN: ĐẠT CHUẨN NGHIỆM THU TASK 2.2 (Tái lập thành công paper CVPR 2024 trong dung sai +-0.5%)!")
    else:
        print("[*] GHI CHÚ: Sai số cần được phân tích đối chứng chi tiết trong Báo cáo Milestone 1.")
    print("=" * 86)


def create_dummy_dataset(base_dir: str):
    """Tạo tập dữ liệu giả lập cho cả họ GAN và Diffusion phục vụ Smoke Test."""
    img_size = (256, 256)
    subsets = ["progan", "stylegan", "guided", "glide_50_27"]
    for sub in subsets:
        for label_dir in ["0_real", "1_fake"]:
            full_dir = os.path.join(base_dir, sub, label_dir)
            os.makedirs(full_dir, exist_ok=True)
            for i in range(4):
                img = Image.new("RGB", img_size, color=((i * 40) % 255, (i * 70) % 255, (i * 100) % 255))
                img.save(os.path.join(full_dir, f"sample_{i}.png"))
    return subsets


def run_full_eval():
    args = parse_args()

    # 1. Cấu hình thiết bị
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 86)
    print("FATFORMER-XLA: PIPELINE BENCHMARK BASELINE TOÀN BỘ 18 TẬP CLEAN (TASK 2.2)")
    print("=" * 86)
    print(f"• Thiết bị tính toán:       {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"• Chế độ đánh giá:          FULL-BENCHMARK (100% số ảnh trong từng tập test)")
    print(f"• Batch size:               {args.batch_size}")
    print(f"• Số luồng nạp (Workers):   {args.num_workers}")
    print(f"• Chế độ Baseline gốc:      SRM={args.use_srm}, Gating={args.use_gating}, Strict={args.strict}")

    # 2. Xử lý đường dẫn CLIP
    clip_path = args.clip_path
    if not os.path.isabs(clip_path):
        clip_path = os.path.join(project_root, clip_path)
    if not os.path.exists(clip_path):
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

    # 4. Nạp checkpoint tác giả (strict=True)
    if args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(project_root, ckpt_path)
        if os.path.exists(ckpt_path):
            print(f"[*] Đang nạp checkpoint từ: {ckpt_path} (strict={args.strict})...")
            CheckpointManager.load(ckpt_path, model, device=device, strict=args.strict)
            print("  -> Nạp trọng số checkpoint thành công!")
        else:
            print(f"[CẢNH BÁO] Không tìm thấy checkpoint tại: {ckpt_path}")
    else:
        default_ckpt = os.path.join(project_root, "fatformer_4class_ckpt.pth")
        if os.path.exists(default_ckpt):
            print(f"[*] Phát hiện checkpoint mặc định: {default_ckpt}")
            CheckpointManager.load(default_ckpt, model, device=device, strict=args.strict)

    # 5. Kịch bản Smoke Test Dummy hoặc Dữ liệu thật
    if args.test_dummy:
        print("\n" + "=" * 86)
        print("CHẾ ĐỘ SMOKE TEST VỚI DỮ LIỆU GIẢ LẬP (SYNTHETIC DUMMY DATA)")
        print("=" * 86)
        temp_dir = tempfile.mkdtemp(prefix="fatformer_full_eval_dummy_")
        try:
            dummy_subsets = create_dummy_dataset(temp_dir)
            print(f"[*] Đã tạo tập dữ liệu giả lập tại: {temp_dir} ({len(dummy_subsets)} subsets)")
            
            results = run_benchmark(
                model=model,
                dataset_path=temp_dir,
                device=device,
                selected_subsets=dummy_subsets,
                max_samples_per_subset=None, # Đánh giá 100% mẫu
                batch_size=min(args.batch_size, 4),
                num_workers=0
            )
            
            table_text, summary_dict = format_grouped_summary(results)
            print("\n" + table_text)
            
            if args.output_csv or args.output_markdown:
                export_full_results(results, summary_dict, args.output_csv, args.output_markdown)
                
            verify_paper_reproduction(summary_dict, results)
            print("\n[✓] SMOKE TEST FULL BENCHMARK HOÀN TẤT THÀNH CÔNG!")
            return 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # 6. Chạy đánh giá trên Dataset thật
    if not args.dataset_root:
        candidates = [
            "/content/dataset_local/test_clean",
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
        print(f"\n[LỖI] Không tìm thấy thư mục dữ liệu test clean: '{args.dataset_root}'.")
        print("Gợi ý: Cung cấp --test_root /content/dataset_local/test_clean hoặc sử dụng cờ --test_dummy.")
        return 1

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

    start_benchmark_time = time.time()
    results = run_benchmark(
        model=model,
        dataset_path=args.dataset_root,
        device=device,
        selected_subsets=selected_subsets,
        max_samples_per_subset=None, # Đánh giá toàn bộ tập
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    total_benchmark_time = time.time() - start_benchmark_time

    # 7. Định dạng phân nhóm kết quả và in bảng
    table_text, summary_dict = format_grouped_summary(results)
    print("\n" + table_text)
    print(f"Tổng thời gian Full Benchmark: {total_benchmark_time:.2f} s ({total_benchmark_time/60:.2f} phút)")

    # 8. Xuất file kết quả
    if args.output_csv or args.output_markdown:
        export_full_results(results, summary_dict, args.output_csv, args.output_markdown)

    # 9. Thẩm định điều kiện DoD
    verify_paper_reproduction(summary_dict, results)

    return 0


if __name__ == "__main__":
    sys.exit(run_full_eval())
