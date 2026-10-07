"""
================================================================================
FATFORMER FORENSIC INSPECTOR - PHÂN TÍCH VÀ ĐÁNH GIÁ PHÁP Y ẢNH ĐƠN LẺ
================================================================================
Công cụ kiểm thử pháp y cho ảnh với FatFormer Checkpoint (CVPR 2024 & Task 4.2):
1. Quản lý Logs chuẩn mực theo PHIÊN và theo PHẦN:
   - Theo Phiên: Mỗi lần chạy tạo một phiên duy nhất (Session ID, timestamp, check_XX),
     lưu trọn vẹn artifacts vào DATASET/inspection_results/<tên_ảnh>/check_XX/,
     đồng thời ghi nhận vào bảng lịch sử toàn cục DATASET/logs/inspection_history.csv.
   - Theo Phần: Cấu trúc log 5 Phần rõ ràng, minh bạch:
     + PHẦN 1: THIẾT LẬP PHIÊN & MÔI TRƯỜNG SUY LUẬN
     + PHẦN 2: THÔNG SỐ KIẾN TRÚC & CHECKPOINT
     + PHẦN 3: TIẾN TRÌNH QUÉT PHÁP Y THỊ GIÁC (GRID 4x4 / STANDARD)
     + PHẦN 4: HỒ SƠ ĐÁNH GIÁ PHÁP Y & KẾT LUẬN TOÀN CỤC
     + PHẦN 5: DANH MỤC THÀNH PHẨM & ARTIFACTS XUẤT RA
   - Tự động xuất file nhật ký phiên: session_report.txt ngay trong thư mục check_XX.
2. Chế độ Mặc định: Grid Multi-Patch 4x4 (16 mảnh 224x224 ở độ phân giải gốc).
   - Bảo toàn 100% vết vi sai tần số cao của AI, không bị làm mờ bởi Resize.
   - Hỗ trợ tùy chỉnh kích thước lưới qua --grid_size (3, 4, 5...).
3. Chế độ Standard: Resize 256x256 -> CenterCrop 224x224 (Quy chuẩn bài báo).
4. Tự động nhận diện cấu hình (SRM, Dynamic Gating, ViT Adapters).

VÍ DỤ SỬ DỤNG (USAGE EXAMPLES):
1. Chạy nhanh tương tác (Menu chọn ảnh & checkpoint):
   python tools/inspect_image.py

2. Chạy với Checkpoint Task 4.2 và ảnh chỉ định (Lưới mặc định 4x4):
   python tools/inspect_image.py -i DATASET/image/girl3.png -c DATASET/checkpoints/fatformer_srm_robust_final.pth

3. Chạy với Baseline CVPR 2024:
   python tools/inspect_image.py -i DATASET/image/girl3.png -c DATASET/pretrained/fatformer_4class_ckpt.pth

4. Tùy chỉnh kích thước lưới Grid (ví dụ 5x5 = 25 mảnh):
   python tools/inspect_image.py -i DATASET/image/girl3.png --grid_size 5

5. Chạy chế độ Standard (Resize 224x224):
   python tools/inspect_image.py -i DATASET/image/girl3.png --mode standard
================================================================================
"""

import os
import sys
import json
import csv
import argparse
import math
from datetime import datetime
import numpy as np
from PIL import Image

# Đảm bảo mã hóa UTF-8 cho Windows Terminal
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

# Thêm đường dẫn project_root vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models import build_model
from src.training.checkpoint_manager import CheckpointManager


# ================================================================================
# BỘ GHI NHẬT KÝ ĐỒNG BỘ CONSOLE & TỆP (DUAL LOGGER)
# ================================================================================

class SessionLogger:
    """Ghi nhật ký đồng thời ra Console và file session_report.txt."""
    def __init__(self, log_filepath):
        self.log_filepath = log_filepath
        os.makedirs(os.path.dirname(os.path.abspath(log_filepath)), exist_ok=True)
        self.file = open(log_filepath, "w", encoding="utf-8")

    def print(self, message=""):
        sys.stdout.write(message + "\n")
        sys.stdout.flush()
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        if self.file and not self.file.closed:
            self.file.close()


def record_global_history(csv_path, record):
    """Lưu tóm tắt phiên kiểm thử vào DATASET/logs/inspection_history.csv."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    fields = [
        "session_id", "timestamp", "image_name", "image_dims",
        "checkpoint_name", "model_type", "mode", "grid_size",
        "verdict", "max_fake_pct", "mean_fake_pct", "flagged_count",
        "total_patches", "run_dir"
    ]
    with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


# ================================================================================
# BỘ CÔNG CỤ TÌM KIẾM VÀ ĐỊNH VỊ TÀI NGUYÊN THÔNG MINH
# ================================================================================

def find_available_checkpoints():
    """Quét và trả về danh sách các file checkpoint (.pth) khả dụng trong project."""
    search_dirs = [
        os.path.join(project_root, "DATASET", "checkpoints"),
        os.path.join(project_root, "DATASET", "pretrained"),
        os.path.join(project_root, "checkpoints"),
        os.path.join(project_root, "checkpoint"),
        project_root
    ]
    seen = set()
    found = []
    for d in search_dirs:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".pth"):
                    full_p = os.path.abspath(os.path.join(d, f))
                    if full_p not in seen:
                        seen.add(full_p)
                        found.append(full_p)
    return found


def resolve_checkpoint(user_path):
    """Tìm file checkpoint từ đường dẫn do người dùng chỉ định."""
    avail = find_available_checkpoints()
    if not user_path:
        return None, avail
    
    if os.path.exists(user_path) and os.path.isfile(user_path):
        return os.path.abspath(user_path), avail
        
    basename = os.path.basename(user_path)
    candidates = [
        os.path.join(project_root, user_path),
        os.path.join(project_root, "DATASET", user_path),
        os.path.join(project_root, "DATASET", "checkpoints", basename),
        os.path.join(project_root, "DATASET", "pretrained", basename),
        os.path.join(project_root, "checkpoints", basename),
        os.path.join(project_root, "checkpoint", basename),
        os.path.join(project_root, basename)
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c), avail
            
    for p in avail:
        if os.path.basename(p).lower() == basename.lower():
            return p, avail
            
    return None, avail


def find_available_images():
    """Quét và trả về danh sách ảnh khả dụng trong DATASET/image."""
    img_dirs = [
        os.path.join(project_root, "DATASET", "image"),
        os.path.join(project_root, "DATASET", "images"),
        os.path.join(project_root, "images"),
        project_root
    ]
    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
    found = []
    seen = set()
    for d in img_dirs:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(valid_exts):
                    full_p = os.path.abspath(os.path.join(d, f))
                    if full_p not in seen:
                        seen.add(full_p)
                        found.append(full_p)
    return found


def resolve_image(user_path):
    """Tìm file ảnh từ tham số người dùng."""
    avail = find_available_images()
    if not user_path:
        return None, avail
        
    if os.path.exists(user_path) and os.path.isfile(user_path):
        return os.path.abspath(user_path), avail
        
    basename = os.path.basename(user_path)
    candidates = [
        os.path.join(project_root, user_path),
        os.path.join(project_root, "DATASET", user_path),
        os.path.join(project_root, "DATASET", "image", user_path),
        os.path.join(project_root, "DATASET", "image", basename),
        os.path.join(project_root, "images", basename),
        os.path.join(project_root, basename)
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c), avail
            
    for p in avail:
        if os.path.basename(p).lower() == basename.lower():
            return p, avail
            
    return None, avail


def select_checkpoint_interactive(user_path):
    """Xử lý chọn Checkpoint (CLI hoặc menu tương tác)."""
    found, avail = resolve_checkpoint(user_path)
    if user_path:
        if found:
            return found
        else:
            print(f"\n❌ [LỖI] Không tìm thấy file checkpoint do bạn chỉ định: {user_path}")
            print("   Danh sách các Checkpoint khả dụng tìm thấy trong máy:")
            for idx, p in enumerate(avail, 1):
                print(f"     [{idx}] {os.path.relpath(p, project_root)}")
            sys.exit(1)

    if not avail:
        print("\n❌ [LỖI] Không tìm thấy bất kỳ file checkpoint .pth nào trong hệ thống!")
        sys.exit(1)

    def rank_ckpt(p):
        n = os.path.basename(p).lower()
        if "robust" in n or "srm" in n:
            return 0
        if "4class" in n or "baseline" in n:
            return 1
        return 2

    avail = sorted(avail, key=rank_ckpt)

    if sys.stdin.isatty():
        print("\n" + "=" * 70)
        print("          CHỌN CHECKPOINT FATFORMER ĐỂ PHÂN TÍCH")
        print("=" * 70)
        for idx, p in enumerate(avail, 1):
            rel = os.path.relpath(p, project_root)
            size_mb = os.path.getsize(p) / (1024 * 1024)
            tag = " [KHUYẾN NGHỊ: Task 4.2 Final]" if "robust" in rel else ""
            if "4class" in rel:
                tag = " [CVPR 2024 Baseline]"
            print(f"  [{idx}] {rel} ({size_mb:.1f} MB){tag}")
        print(f"  [0] Nhập đường dẫn checkpoint tùy chỉnh khác...")
        print("-" * 70)
        try:
            choice = input(f"▶ Chọn Checkpoint [1-{len(avail)}, Mặc định: 1]: ").strip()
            if choice == "0":
                custom = input("  Nhập đường dẫn checkpoint: ").strip()
                c_found, _ = resolve_checkpoint(custom)
                if c_found:
                    return c_found
                print(f"❌ Không tìm thấy checkpoint tại: {custom}")
                sys.exit(1)
            elif choice == "":
                return avail[0]
            else:
                idx = int(choice)
                if 1 <= idx <= len(avail):
                    return avail[idx - 1]
                return avail[0]
        except (ValueError, EOFError, KeyboardInterrupt):
            return avail[0]
    else:
        return avail[0]


def select_image_interactive(user_path):
    """Xử lý chọn Ảnh (CLI hoặc menu tương tác)."""
    found, avail = resolve_image(user_path)
    if user_path:
        if found:
            return found
        else:
            print(f"\n❌ [LỖI] Không tìm thấy file ảnh do bạn chỉ định: {user_path}")
            print("   Danh sách các file ảnh mẫu có sẵn trong DATASET/image:")
            for idx, p in enumerate(avail, 1):
                print(f"     [{idx}] {os.path.relpath(p, project_root)}")
            sys.exit(1)

    if not avail:
        print("\n❌ [LỖI] Không tìm thấy file ảnh nào trong thư mục DATASET/image!")
        if sys.stdin.isatty():
            try:
                custom = input("Vui lòng nhập đường dẫn file ảnh cần kiểm tra: ").strip()
                c_found, _ = resolve_image(custom)
                if c_found:
                    return c_found
            except (EOFError, KeyboardInterrupt):
                pass
        sys.exit(1)

    if sys.stdin.isatty():
        print("\n" + "=" * 70)
        print("          CHỌN ẢNH CẦN PHÂN TÍCH PHÁP Y THỊ GIÁC")
        print("=" * 70)
        for idx, p in enumerate(avail, 1):
            rel = os.path.relpath(p, project_root)
            size_kb = os.path.getsize(p) / 1024.0
            print(f"  [{idx}] {rel} ({size_kb:.1f} KB)")
        print(f"  [0] Nhập đường dẫn ảnh khác...")
        print("-" * 70)
        try:
            choice = input(f"▶ Chọn ảnh cần kiểm tra [1-{len(avail)}, Mặc định: 1]: ").strip()
            if choice == "0":
                custom = input("  Nhập đường dẫn ảnh: ").strip()
                c_found, _ = resolve_image(custom)
                if c_found:
                    return c_found
                print(f"❌ Không tìm thấy ảnh tại: {custom}")
                sys.exit(1)
            elif choice == "":
                return avail[0]
            else:
                idx = int(choice)
                if 1 <= idx <= len(avail):
                    return avail[idx - 1]
                return avail[0]
        except (ValueError, EOFError, KeyboardInterrupt):
            return avail[0]
    else:
        return avail[0]


def auto_detect_checkpoint_config(ckpt_path, manual_srm=None, manual_gating=None, manual_adapters=None):
    """
    Tự động suy luận cấu hình mô hình (SRM, Dynamic Gating, ViT Adapters).
    """
    filename = os.path.basename(ckpt_path).lower()
    is_robust = ("robust" in filename) or ("srm" in filename) or ("final" in filename) or ("epoch" in filename)

    if manual_srm is not None:
        use_srm = manual_srm
    else:
        use_srm = is_robust

    if manual_gating is not None:
        use_gating = manual_gating
    else:
        use_gating = is_robust

    if manual_adapters is not None:
        num_vit_adapter = manual_adapters
    else:
        num_vit_adapter = 8

    model_type_str = "FatFormer-XLA Robust (Task 4.2 Final)" if is_robust else "FatFormer Baseline (CVPR 2024)"

    return use_srm, use_gating, num_vit_adapter, model_type_str


# ================================================================================
# CÁC HÀM TÍNH TOÁN VÀ SUY LUẬN MÔ HÌNH
# ================================================================================

def compute_haar_dwt_energy(img_tensor):
    """
    Tính toán tỷ lệ năng lượng tần số cao qua biến đổi sóng con 2D Haar DWT.
    img_tensor: [1, 3, 224, 224] (float32)
    """
    h0 = torch.tensor([1.0, 1.0], dtype=torch.float32, device=img_tensor.device).reshape(1, 1, 1, 2) / math.sqrt(2.0)
    h1 = torch.tensor([1.0, -1.0], dtype=torch.float32, device=img_tensor.device).reshape(1, 1, 1, 2) / math.sqrt(2.0)

    gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]

    l = F.conv2d(gray, h0, stride=(1, 2))
    h = F.conv2d(gray, h1, stride=(1, 2))

    v0 = h0.transpose(-1, -2)
    v1 = h1.transpose(-1, -2)

    ll = F.conv2d(l, v0, stride=(2, 1))
    lh = F.conv2d(l, v1, stride=(2, 1))
    hl = F.conv2d(h, v0, stride=(2, 1))
    hh = F.conv2d(h, v1, stride=(2, 1))

    e_ll = torch.sum(ll ** 2).item()
    e_high = torch.sum(lh ** 2 + hl ** 2 + hh ** 2).item()
    e_total = e_ll + e_high + 1e-12
    high_freq_ratio = (e_high / e_total) * 100.0

    return {
        "energy_low": e_ll,
        "energy_high": e_high,
        "high_freq_ratio_pct": high_freq_ratio
    }


def inspect_forward(model, image_tensor):
    """
    Chạy forward pass và trích xuất:
    - Tổng Logits S(i) + S'(i)
    - Thành phần Vanilla CLIP S(i)
    - Thành phần Text-Guided Adapter S'(i)
    - Activation map từ khối Forgery-Aware Adapter (FAA) cuối cùng
    """
    captured_adapter_acts = []

    def hook_fn(module, input_tensor, output_tensor):
        captured_adapter_acts.append(output_tensor.detach())

    hook_handle = None
    last_adapter = None
    for module in model.modules():
        if module.__class__.__name__ == "ForgeryAwareAdapterLayer":
            last_adapter = module

    if last_adapter is not None:
        hook_handle = last_adapter.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        tokenized_prompts = model.tokenized_prompts
        logit_scale = model.logit_scale.exp()

        image_features = model.image_encoder(image_tensor.type(model.dtype), return_full=True)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = model.language_guided_alignment(image_features)

        # 1. Eq. (1): Vanilla CLIP similarity S(i)
        logits_s = []
        text_feature_list = []
        for pts_i, imf_i in zip(prompts, image_features_norm):
            text_features = model.text_encoder(pts_i, tokenized_prompts)
            text_feature_list.append(text_features)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i[0] @ text_features_norm.t()
            logits_s.append(l_i)
        logits_s = torch.stack(logits_s)

        # 2. Eq. (9): Text-guided interactor similarity S'(i)
        text_features = torch.stack(text_feature_list, dim=1)
        tgt = image_features[:, 1:].transpose(0, 1)
        tgt2 = model.text_guided_interactor(tgt, text_features, text_features)[0]
        tgt = tgt + tgt2
        tgt = model.norm1(tgt)
        tgt = model.forward_FFN(tgt)

        aug_image_features = tgt.transpose(0, 1).mean(dim=1)
        aug_image_features = aug_image_features / aug_image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        aug_logits_s_prime = []
        for pts_i, imf_i in zip(aug_image_features, text_features_norm.transpose(0, 1)):
            aug_logits_s_prime.append(logit_scale * pts_i @ imf_i.t())
        aug_logits_s_prime = torch.stack(aug_logits_s_prime)

        total_logits = logits_s + aug_logits_s_prime

    if hook_handle is not None:
        hook_handle.remove()

    heatmap_16x16 = None
    if len(captured_adapter_acts) > 0:
        act = captured_adapter_acts[-1]
        spatial_tokens = act[1:, 0, :]
        spatial_norms = torch.norm(spatial_tokens, dim=-1)
        grid_size = int(math.sqrt(spatial_norms.shape[0]))
        heatmap_16x16 = spatial_norms.reshape(grid_size, grid_size).float().cpu().numpy()

    return total_logits, logits_s, aug_logits_s_prime, heatmap_16x16


# ================================================================================
# BỘ RENDER DASHBOARD TRỰC QUAN HÓA (CHUẨN ASCII AN TOÀN CHO FONT)
# ================================================================================

def render_standard_dashboard(orig_pil_img, heatmap_16x16, metrics, output_path, check_number=1, alpha=0.55):
    """Render Dashboard chuẩn 3 panel cho 1 ảnh đơn lẻ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h = orig_pil_img.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    cropped_img = orig_pil_img.crop((left, top, left + crop_size, top + crop_size)).resize((224, 224), Image.BICUBIC)
    cropped_np = np.array(cropped_img) / 255.0

    if heatmap_16x16 is not None:
        h_min, h_max = heatmap_16x16.min(), heatmap_16x16.max()
        norm_map = (heatmap_16x16 - h_min) / (h_max - h_min + 1e-8)
        norm_tensor = torch.from_numpy(norm_map).unsqueeze(0).unsqueeze(0)
        upscaled_tensor = F.interpolate(norm_tensor, size=(224, 224), mode="bicubic", align_corners=False)
        upscaled_map = upscaled_tensor.squeeze().numpy()
        upscaled_map = np.clip(upscaled_map, 0.0, 1.0)

        cmap = plt.get_cmap("jet")
        colored_heatmap = cmap(upscaled_map)[:, :, :3]
        blended_img = (1.0 - alpha) * cropped_np + alpha * colored_heatmap
        blended_img = np.clip(blended_img, 0.0, 1.0)
    else:
        blended_img = cropped_np

    fig = plt.figure(figsize=(16, 5.5), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35], wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cropped_np)
    ax1.set_title("Input Image (Crop 224x224)", fontsize=12, fontweight="bold", pad=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(blended_img)
    ax2.set_title("FAA Heatmap Overlay", fontsize=12, fontweight="bold", pad=10)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")

    is_fake = metrics["prediction"] == "FAKE"
    pred_color = "#D32F2F" if is_fake else "#2E7D32"

    p_real = metrics["prob_real_pct"]
    p_fake = metrics["prob_fake_pct"]
    
    bar_y = [0.85, 0.72]
    bar_height = 0.08
    ax3.barh(bar_y[0], p_real, height=bar_height, color="#4CAF50", edgecolor="#2E7D32", alpha=0.9)
    ax3.barh(bar_y[1], p_fake, height=bar_height, color="#F44336", edgecolor="#D32F2F", alpha=0.9)
    
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 1.0)

    ax3.text(5, bar_y[0], f"REAL: {p_real:.2f}%", va="center", ha="left", color="white", fontweight="bold", fontsize=10)
    ax3.text(5, bar_y[1], f"FAKE: {p_fake:.2f}%", va="center", ha="left", color="white", fontweight="bold", fontsize=10)

    ckpt_name = os.path.basename(metrics.get("checkpoint", "Unknown"))
    conclusion_text = (
        f"PREDICTION: {metrics['prediction']} ({metrics['confidence_label']})\n"
        f"----------------------------------------------------------\n"
        f"- Inspection Run     : Check #{check_number}\n"
        f"- Shannon Entropy    : {metrics['shannon_entropy']:.4f} / 1.0000\n"
        f"- Logit Margin (Dz)  : {metrics['logit_margin']:+.4f} (z_fake: {metrics['raw_logits']['fake']:.2f}, z_real: {metrics['raw_logits']['real']:.2f})\n"
        f"- CLIP S(i) Margin   : {metrics['architecture_decomposition']['clip_s_margin']:+.4f}\n"
        f"- Adapter S' Margin  : {metrics['architecture_decomposition']['adapter_s_prime_margin']:+.4f}\n"
        f"- DWT High-Freq Ratio: {metrics['haar_dwt_frequency']['high_freq_ratio_pct']:.2f}%\n"
        f"- Model Checkpoint   : {ckpt_name}"
    )

    bbox_props = dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5", edgecolor=pred_color, linewidth=1.8)
    ax3.text(0.0, 0.48, conclusion_text, transform=ax3.transAxes, fontsize=10.0,
             family="monospace", va="top", bbox=bbox_props)

    plt.suptitle(f"FATFORMER FORENSIC REPORT: {os.path.basename(metrics['image_path'])} [CHECK #{check_number}]",
                 fontsize=14, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def render_grid_dashboard(orig_pil_img, patch_results, summary_metrics, output_path, check_number=1, grid_size=4):
    """Render Dashboard phân tích lưới đa mảnh ở độ phân giải gốc."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    orig_w, orig_h = orig_pil_img.size
    orig_np = np.array(orig_pil_img) / 255.0

    fig = plt.figure(figsize=(18, 6.0), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.9, 1.25], wspace=0.18)

    # Panel 1: Ảnh gốc kèm bounding boxes của từng patch
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_np)
    ax1.set_title(f"Native Image ({orig_w}x{orig_h}) - Grid Patches ({grid_size}x{grid_size})",
                  fontsize=12, fontweight="bold", pad=10)

    for p in patch_results:
        x1, y1, x2, y2 = p["box"]
        p_fake = p["prob_fake_pct"]
        is_p_fake = p_fake >= 50.0
        box_color = "#F44336" if is_p_fake else "#4CAF50"

        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.8,
                                  edgecolor=box_color, facecolor="none", alpha=0.9)
        ax1.add_patch(rect)
        ax1.text(x1 + 4, y1 + 18, f"#{p['idx']} ({p_fake:.0f}%)", color="white", fontsize=7.5,
                 fontweight="bold", bbox=dict(boxstyle="square,pad=0.15", facecolor=box_color, alpha=0.85))

    ax1.axis("off")

    # Panel 2: Ma trận nhiệt Lưới G x G Fake Probabilities
    ax2 = fig.add_subplot(gs[0, 1])
    grid_matrix = np.zeros((grid_size, grid_size))
    for p in patch_results:
        r, c = p["row"], p["col"]
        grid_matrix[r, c] = p["prob_fake_pct"]

    im2 = ax2.imshow(grid_matrix, cmap="RdYlGn_r", vmin=0.0, vmax=100.0)
    ax2.set_title(f"Fake Probability Matrix ({grid_size}x{grid_size})", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(range(grid_size))
    ax2.set_yticks(range(grid_size))

    for r in range(grid_size):
        for c in range(grid_size):
            val = grid_matrix[r, c]
            text_color = "white" if val > 65 or val < 35 else "black"
            ax2.text(c, r, f"{val:.1f}%\n(#{r * grid_size + c + 1})", ha="center", va="center",
                     color=text_color, fontweight="bold", fontsize=9.0)

    cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Fake Prob (%)", fontsize=10)

    # Panel 3: Forensic Scorecard tổng hợp
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")

    worst_p = summary_metrics["max_fake_patch"]
    verdict = summary_metrics["verdict"]
    verdict_color = "#D32F2F" if "FAKE" in verdict or "SUSPECTED" in verdict or "CAN THIỆP" in verdict else "#2E7D32"

    bar_y = [0.85, 0.72]
    bar_height = 0.08
    ax3.barh(bar_y[0], 100.0 - summary_metrics["max_fake_prob"], height=bar_height,
             color="#4CAF50", edgecolor="#2E7D32", alpha=0.9)
    ax3.barh(bar_y[1], summary_metrics["max_fake_prob"], height=bar_height,
             color="#F44336", edgecolor="#D32F2F", alpha=0.9)

    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 1.0)

    ax3.text(5, bar_y[0], f"PEAK REAL: {100.0 - summary_metrics['max_fake_prob']:.2f}%",
             va="center", ha="left", color="white", fontweight="bold", fontsize=10)
    ax3.text(5, bar_y[1], f"PEAK FAKE: {summary_metrics['max_fake_prob']:.2f}% (Patch #{worst_p['idx']})",
             va="center", ha="left", color="white", fontweight="bold", fontsize=10)

    ckpt_name = os.path.basename(summary_metrics.get("checkpoint", "Unknown"))
    conclusion_text = (
        f"OVERALL VERDICT: {verdict}\n"
        f"----------------------------------------------------------\n"
        f"- Inspection Run Index    : Check #{check_number}\n"
        f"- Total Patches Scanned   : {len(patch_results)} (Grid {grid_size}x{grid_size})\n"
        f"- Flagged Fake Patches    : {summary_metrics['flagged_fake_count']} / {len(patch_results)}\n"
        f"- Peak Fake Probability   : {summary_metrics['max_fake_prob']:.2f}% (Patch #{worst_p['idx']})\n"
        f"- Mean Fake Probability   : {summary_metrics['mean_fake_prob']:.2f}%\n"
        f"- Model Checkpoint        : {ckpt_name}\n"
        f"----------------------------------------------------------\n"
        f"MOST SUSPICIOUS PATCH (#{worst_p['idx']}):\n"
        f"- Box Coordinates         : {worst_p['box']}\n"
        f"- Logit Margin (Delta-z)  : {worst_p['logit_margin']:+.4f}\n"
        f"- CLIP S(i) Margin        : {worst_p['clip_s_margin']:+.4f}\n"
        f"- Adapter S'(i) Margin    : {worst_p['adapter_s_prime_margin']:+.4f}\n"
        f"- DWT High-Freq Ratio     : {worst_p['high_freq_ratio']:.2f}%"
    )

    bbox_props = dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5", edgecolor=verdict_color, linewidth=1.8)
    ax3.text(0.0, 0.48, conclusion_text, transform=ax3.transAxes, fontsize=9.4,
             family="monospace", va="top", bbox=bbox_props)

    plt.suptitle(f"FATFORMER NATIVE GRID DASHBOARD: {os.path.basename(summary_metrics['image_path'])} [CHECK #{check_number}]",
                 fontsize=13.5, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def setup_run_directory(base_output_dir, image_path):
    """
    Quản lý phân cấp thư mục theo phiên:
    DATASET/inspection_results/<tên_ảnh>/check_XX/
    """
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.join(base_output_dir, image_stem)
    os.makedirs(image_dir, exist_ok=True)

    existing_runs = []
    for entry in os.listdir(image_dir):
        entry_path = os.path.join(image_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith("check_"):
            try:
                num = int(entry.split("_")[1])
                existing_runs.append(num)
            except (IndexError, ValueError):
                pass

    check_number = max(existing_runs, default=0) + 1
    run_dir = os.path.join(image_dir, f"check_{check_number:02d}")
    os.makedirs(run_dir, exist_ok=True)

    return run_dir, check_number, image_stem


# ================================================================================
# HÀM MAIN THỰC THI CHÍNH (PHÂN RÕ 5 PHẦN & TỪNG PHIÊN)
# ================================================================================

def main():
    default_dataset_output = os.path.join(project_root, "DATASET", "inspection_results")
    global_history_csv = os.path.join(project_root, "DATASET", "logs", "inspections", "inspection_history.csv")

    epilog_text = """
VÍ DỤ SỬ DỤNG:
  1. Chạy nhanh tương tác (Menu chọn ảnh & checkpoint):
     python tools/inspect_image.py

  2. Chạy với Checkpoint Task 4.2 và ảnh chỉ định (Lưới mặc định 4x4):
     python tools/inspect_image.py -i DATASET/image/girl3.png -c DATASET/checkpoints/fatformer_srm_robust_final.pth

  3. Chạy với Baseline CVPR 2024:
     python tools/inspect_image.py -i DATASET/image/girl3.png -c DATASET/pretrained/fatformer_4class_ckpt.pth

  4. Tùy chỉnh kích thước lưới Grid (ví dụ 5x5 = 25 mảnh):
     python tools/inspect_image.py -i DATASET/image/girl3.png --grid_size 5

  5. Chạy chế độ Standard (Resize 224x224):
     python tools/inspect_image.py -i DATASET/image/girl3.png --mode standard
"""
    parser = argparse.ArgumentParser(
        description="FatFormer Forensic Inspector (CVPR 2024 & Task 4.2)", 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_text,
        add_help=True
    )
    parser.add_argument("--image", "-i", type=str, default=None, 
                        help="Đường dẫn đến file ảnh cần kiểm tra (để trống để chọn tương tác).")
    parser.add_argument("--ckpt", "-c", "--checkpoint", type=str, default=None, 
                        help="Đường dẫn file checkpoint .pth (để trống để chọn tương tác).")
    parser.add_argument("--mode", "-m", type=str, default="grid", choices=["grid", "standard", "native_center"],
                        help="Chế độ kiểm tra: grid (Lưới đa mảnh độ phân giải gốc - Mặc định), standard (Resize 224), native_center (1 ô tâm).")
    parser.add_argument("--grid_size", "-g", type=int, default=4, 
                        help="Kích thước lưới khi chọn mode grid (Mặc định: 4, tức 4x4 = 16 mảnh).")
    parser.add_argument("--clip_path", type=str, default=None, help="Đường dẫn ViT-L-14.pt.")
    parser.add_argument("--output_dir", "-o", type=str, default=default_dataset_output,
                        help=f"Thư mục gốc lưu kết quả (Mặc định: DATASET/inspection_results).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Thiết bị tính toán: cuda hoặc cpu (Mặc định: tự động nhận diện).")
    parser.add_argument("--alpha", type=float, default=0.55, help="Độ trong suốt của Heatmap (0.0 đến 1.0).")
    parser.add_argument("--save_json", action="store_true", default=True, help="Lưu kết quả chi tiết ra JSON.")
    parser.add_argument("--use_srm", dest="use_srm", action="store_true", default=None, 
                        help="Kích hoạt khối SRM (Mặc định: tự nhận diện theo checkpoint)")
    parser.add_argument("--no_srm", dest="use_srm", action="store_false", help="Tắt khối SRM")
    parser.add_argument("--use_gating", dest="use_gating", action="store_true", default=None, 
                        help="Kích hoạt Dynamic Gating (Mặc định: tự nhận diện theo checkpoint)")
    parser.add_argument("--no_gating", dest="use_gating", action="store_false", help="Tắt khối Dynamic Gating")
    parser.add_argument("--num_vit_adapter", type=int, default=None, 
                        help="Số lượng adapter trong ViT (Mặc định: 8)")
    args = parser.parse_args()

    # 1. Chọn file Checkpoint & Ảnh
    ckpt_path = select_checkpoint_interactive(args.ckpt)
    image_path = select_image_interactive(args.image)

    # 2. Nhận diện cấu hình kiến trúc
    use_srm, use_gating, num_vit_adapter, model_type_str = auto_detect_checkpoint_config(
        ckpt_path,
        manual_srm=args.use_srm,
        manual_gating=args.use_gating,
        manual_adapters=args.num_vit_adapter
    )

    # 3. Nạp thông số ảnh & Thư mục phiên
    raw_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = raw_img.size
    image_filename = os.path.basename(image_path)
    run_dir, check_number, image_stem = setup_run_directory(args.output_dir, image_path)

    # 4. Khởi tạo mã phiên và Session Logger
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"SESSION_{timestamp_tag}_{image_stem.upper()}_CHECK{check_number:02d}"
    session_log_path = os.path.join(run_dir, "session_report.txt")
    logger = SessionLogger(session_log_path)

    logger.print("=" * 100)
    logger.print("              FATFORMER FORENSIC INSPECTOR - PHIÊN KIỂM ĐỊNH PHÁP Y THỊ GIÁC")
    logger.print("=" * 100)

    # -------------------------------------------------------------------------
    # PHẦN 1: THIẾT LẬP PHIÊN & MÔI TRƯỜNG SUY LUẬN
    # -------------------------------------------------------------------------
    logger.print("\n[PHẦN 1: THIẾT LẬP PHIÊN & MÔI TRƯỜNG SUY LUẬN (SESSION & ENVIRONMENT)]")
    logger.print("-" * 100)
    logger.print(f"  * Mã định danh phiên      : {session_id}")
    logger.print(f"  * Thời gian thực thi      : {now_str}")
    logger.print(f"  * Tệp ảnh kiểm thử        : {os.path.abspath(image_path)} ({orig_w}x{orig_h})")
    logger.print(f"  * Thiết bị tính toán      : {args.device.upper()}")
    logger.print(f"  * Số thứ tự phiên ảnh     : Lần kiểm tra #{check_number} đối với ảnh [{image_filename}]")
    logger.print(f"  * Thư mục lưu vết phiên   : {os.path.abspath(run_dir)}")

    # -------------------------------------------------------------------------
    # PHẦN 2: THÔNG SỐ KIẾN TRÚC & CHECKPOINT
    # -------------------------------------------------------------------------
    clip_candidates = [
        args.clip_path,
        os.path.join(project_root, "DATASET", "pretrained", "ViT-L-14.pt"),
        os.path.join(project_root, "ViT-L-14.pt"),
        os.path.join(project_root, "pretrained", "ViT-L-14.pt")
    ]
    clip_path = None
    for cand in clip_candidates:
        if cand and os.path.exists(cand):
            clip_path = os.path.abspath(cand)
            break
            
    if not clip_path:
        logger.print(f"❌ [LỖI] Không tìm thấy file backbone ViT-L-14.pt tại các đường dẫn mặc định!")
        logger.close()
        sys.exit(1)

    logger.print("\n[PHẦN 2: THÔNG SỐ KIẾN TRÚC & CHECKPOINT (ARCHITECTURE & MODEL SPEC)]")
    logger.print("-" * 100)
    logger.print(f"  * Checkpoint nạp vào      : {os.path.abspath(ckpt_path)}")
    logger.print(f"  * Phân loại mô hình       : {model_type_str}")
    logger.print(f"  * Cấu hình kiến trúc      : SRM 3-Kernels={'BẬT' if use_srm else 'TẮT'} | Dynamic Gating={'BẬT' if use_gating else 'TẮT'} | ViT Adapters={num_vit_adapter}")
    logger.print(f"  * Backbone thị giác       : CLIP:ViT-L/14 ({os.path.abspath(clip_path)})")

    device = torch.device(args.device)
    model_args = argparse.Namespace(
        backbone="CLIP:ViT-L/14",
        clip_path=clip_path,
        num_classes=2,
        num_vit_adapter=num_vit_adapter,
        num_context_embedding=8,
        init_context_embedding="",
        hidden_dim=768,
        clip_vision_width=1024,
        frequency_encoder_layer=2,
        decoder_layer=4,
        num_heads=12,
        use_srm=use_srm,
        use_gating=use_gating
    )

    logger.print("  * Khởi tạo mô hình        : Đang nạp trọng số mạng...")
    model = build_model(model_args)
    model = model.to(device)

    strict = not (use_srm or use_gating)
    CheckpointManager.load(ckpt_path, model, device=device, strict=strict)
    logger.print("  * Trạng thái mô hình      : [✓] Khớp trọng số thành công, sẵn sàng suy luận!")

    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # -------------------------------------------------------------------------
    # PHẦN 3: TIẾN TRÌNH QUÉT PHÁP Y THỊ GIÁC
    # -------------------------------------------------------------------------
    logger.print("\n[PHẦN 3: TIẾN TRÌNH QUÉT PHÁP Y THỊ GIÁC (FORENSIC SCAN EXECUTION)]")
    logger.print("-" * 100)

    if args.mode == "grid":
        if orig_w < 224 or orig_h < 224:
            logger.print(f"⚠️ [CẢNH BÁO] Ảnh ({orig_w}x{orig_h}) nhỏ hơn 224x224, tự động chuyển sang mode standard.")
            args.mode = "standard"
        else:
            g = max(2, args.grid_size)
            logger.print(f"  * Chế độ phân tích       : Lưới đa mảnh nguyên bản (Native Resolution Multi-Patch)")
            logger.print(f"  * Ma trận phân rã         : {g}x{g} = {g*g} mảnh 224x224 (Không qua Resize bicubic)")
            logger.print(f"  * Tiến trình quét         : Đang phân tích từng mảnh vi sai...")

            xs = np.linspace(0, orig_w - 224, g).astype(int)
            ys = np.linspace(0, orig_h - 224, g).astype(int)

            patch_results = []
            idx = 1
            for r_idx, y in enumerate(ys):
                for c_idx, x in enumerate(xs):
                    box = (int(x), int(y), int(x + 224), int(y + 224))
                    patch_img = raw_img.crop(box)
                    patch_tensor = norm_transform(patch_img).unsqueeze(0).to(device)

                    dwt_m = compute_haar_dwt_energy(patch_tensor)
                    total_logits, logits_s, aug_logits_s_prime, heatmap = inspect_forward(model, patch_tensor)

                    probs = F.softmax(total_logits, dim=-1).squeeze(0).tolist()
                    p_real, p_fake = probs[0] * 100.0, probs[1] * 100.0

                    z_real = total_logits[0, 0].item()
                    z_fake = total_logits[0, 1].item()
                    logit_margin = z_fake - z_real

                    s_real, s_fake = logits_s[0, 0].item(), logits_s[0, 1].item()
                    s_prime_real, s_prime_fake = aug_logits_s_prime[0, 0].item(), aug_logits_s_prime[0, 1].item()

                    patch_results.append({
                        "idx": idx,
                        "row": r_idx,
                        "col": c_idx,
                        "box": box,
                        "prob_real_pct": p_real,
                        "prob_fake_pct": p_fake,
                        "logit_margin": logit_margin,
                        "clip_s_margin": s_fake - s_real,
                        "adapter_s_prime_margin": s_prime_fake - s_prime_real,
                        "high_freq_ratio": dwt_m["high_freq_ratio_pct"],
                        "heatmap": heatmap
                    })
                    idx += 1

            fake_probs = [p["prob_fake_pct"] for p in patch_results]
            max_fake_prob = max(fake_probs)
            mean_fake_prob = float(np.mean(fake_probs))
            worst_patch = max(patch_results, key=lambda p: p["prob_fake_pct"])
            flagged_count = sum(1 for p in patch_results if p["prob_fake_pct"] >= 50.0)

            if max_fake_prob >= 75.0:
                verdict = "PHÁT HIỆN CAN THIỆP AI CỤC BỘ (Local Forgery Detected)"
            elif max_fake_prob >= 50.0:
                verdict = "NGHI NGỜ CÓ CAN THIỆP AI (Suspicious - Boundary Case)"
            else:
                verdict = "ẢNH THẬT (Tất cả các mảnh đều đạt chuẩn Real)"

            summary_metrics = {
                "session_id": session_id,
                "timestamp": now_str,
                "image_path": os.path.abspath(image_path),
                "image_filename": image_filename,
                "image_dims": f"{orig_w}x{orig_h}",
                "checkpoint": os.path.abspath(ckpt_path),
                "model_type": model_type_str,
                "inspection_run_index": check_number,
                "inspection_folder": os.path.abspath(run_dir),
                "mode": "grid",
                "grid_size": f"{g}x{g}",
                "total_patches": len(patch_results),
                "flagged_fake_count": flagged_count,
                "max_fake_prob": max_fake_prob,
                "mean_fake_prob": mean_fake_prob,
                "verdict": verdict,
                "max_fake_patch": {k: v for k, v in worst_patch.items() if k != "heatmap"}
            }

            logger.print("\n  CHI TIẾT MA TRẬN PHÂN RÃ CÁC MẢNH LƯỚI:")
            logger.print("  " + "-" * 96)
            logger.print(f"  {'Mảnh #':^8} | {'Tọa độ Box (x1, y1, x2, y2)':^26} | {'Real (%)':^10} | {'Fake (%)':^10} | {'DWT High-Freq':^14} | {'Đánh giá':^10}")
            logger.print("  " + "-" * 96)
            for p in patch_results:
                status = "[FAKE]" if p["prob_fake_pct"] >= 50.0 else "[REAL]"
                box_str = f"({p['box'][0]}, {p['box'][1]}, {p['box'][2]}, {p['box'][3]})"
                logger.print(f"  {p['idx']:^8} | {box_str:^26} | {p['prob_real_pct']:^10.2f} | {p['prob_fake_pct']:^10.2f} | {p['high_freq_ratio']:^12.2f} % | {status:^10}")
            logger.print("  " + "-" * 96)

            # -----------------------------------------------------------------
            # PHẦN 4: HỒ SƠ ĐÁNH GIÁ PHÁP Y & KẾT LUẬN TOÀN CỤC
            # -----------------------------------------------------------------
            logger.print("\n[PHẦN 4: HỒ SƠ ĐÁNH GIÁ PHÁP Y & KẾT LUẬN TOÀN CỤC (FORENSIC VERDICT)]")
            logger.print("-" * 100)
            logger.print(f"  * KẾT LUẬN TOÀN CỤC       : [{verdict}]")
            logger.print(f"  * Mảnh nghi vấn Fake nhất : Mảnh #{worst_patch['idx']} với Fake = {max_fake_prob:.2f}% (Tọa độ: {worst_patch['box']})")
            logger.print(f"  * Số mảnh bị nghi ngờ Fake: {flagged_count} / {len(patch_results)} mảnh ({flagged_count/len(patch_results)*100.0:.1f}%)")
            logger.print(f"  * Xác suất Fake trung bình: {mean_fake_prob:.2f}%")
            logger.print(f"  * Logits Margin (Delta-z) : {worst_patch['logit_margin']:+.4f}")
            logger.print(f"  * Đóng góp thành phần     : CLIP S(i) Margin={worst_patch['clip_s_margin']:+.4f} | Adapter S'(i) Margin={worst_patch['adapter_s_prime_margin']:+.4f}")
            logger.print(f"  * Năng lượng sóng con DWT : Tỷ lệ tần số cao của mảnh đỉnh = {worst_patch['high_freq_ratio']:.2f}%")

            # Render Dashboard
            grid_report_path = os.path.join(run_dir, f"{image_stem}_grid_dashboard.png")
            render_grid_dashboard(raw_img, patch_results, summary_metrics, grid_report_path, check_number=check_number, grid_size=g)

            # Lưu file JSON
            json_path = os.path.join(run_dir, f"{image_stem}_grid_metrics.json")
            if args.save_json:
                json_data = {
                    "summary": summary_metrics,
                    "patches": [{k: v for k, v in p.items() if k != "heatmap"} for p in patch_results]
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False)

            # Ghi nhận vào file lịch sử toàn cục
            record_global_history(global_history_csv, {
                "session_id": session_id,
                "timestamp": now_str,
                "image_name": image_filename,
                "image_dims": f"{orig_w}x{orig_h}",
                "checkpoint_name": os.path.basename(ckpt_path),
                "model_type": model_type_str,
                "mode": "grid",
                "grid_size": f"{g}x{g}",
                "verdict": verdict,
                "max_fake_pct": f"{max_fake_prob:.2f}",
                "mean_fake_pct": f"{mean_fake_prob:.2f}",
                "flagged_count": flagged_count,
                "total_patches": len(patch_results),
                "run_dir": os.path.abspath(run_dir)
            })

            # -----------------------------------------------------------------
            # PHẦN 5: DANH MỤC THÀNH PHẨM & LƯU TRỮ VẾT
            # -----------------------------------------------------------------
            logger.print("\n[PHẦN 5: DANH MỤC THÀNH PHẨM & ARTIFACTS XUẤT RA (EXPORTED ARTIFACTS)]")
            logger.print("-" * 100)
            logger.print(f"  [✓] Ảnh Dashboard trực quan : {os.path.abspath(grid_report_path)}")
            logger.print(f"  [✓] Số liệu chi tiết JSON   : {os.path.abspath(json_path)}")
            logger.print(f"  [✓] Báo cáo văn bản phiên   : {os.path.abspath(session_log_path)}")
            logger.print(f"  [✓] Lịch sử phiên toàn cục  : {os.path.abspath(global_history_csv)}")
            logger.print("=" * 100 + "\n")
            logger.close()
            return

    # =========================================================================
    # CHẾ ĐỘ STANDARD HOẶC NATIVE CENTER
    # =========================================================================
    if args.mode == "standard":
        transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            norm_transform
        ])
        img_tensor = transform(raw_img).unsqueeze(0).to(device)
        mode_desc = "Standard Quy Chuẩn CVPR 2024 (Resize 256x256 -> CenterCrop 224x224)"
    else:  # native_center
        crop_x = max(0, (orig_w - 224) // 2)
        crop_y = max(0, (orig_h - 224) // 2)
        center_crop_img = raw_img.crop((crop_x, crop_y, crop_x + 224, crop_y + 224))
        img_tensor = norm_transform(center_crop_img).unsqueeze(0).to(device)
        mode_desc = "Native Center (1 mảnh 224x224 tại tâm gốc không Resize)"

    logger.print(f"  * Chế độ phân tích       : {mode_desc}")
    logger.print("  * Tiến trình suy luận     : Đang thực hiện Forward Pass...")

    dwt_metrics = compute_haar_dwt_energy(img_tensor)
    total_logits, logits_s, aug_logits_s_prime, heatmap_16x16 = inspect_forward(model, img_tensor)

    probs = F.softmax(total_logits, dim=-1).squeeze(0).tolist()
    prob_real, prob_fake = probs[0], probs[1]
    
    z_real = total_logits[0, 0].item()
    z_fake = total_logits[0, 1].item()
    logit_margin = z_fake - z_real

    eps = 1e-12
    entropy = - (prob_real * math.log2(prob_real + eps) + prob_fake * math.log2(prob_fake + eps))

    pred_label = "FAKE" if prob_fake >= 0.50 else "REAL"
    max_prob = max(prob_real, prob_fake)

    if max_prob >= 0.90:
        conf_label = "RẤT CAO (Very High Confidence)"
    elif max_prob >= 0.75:
        conf_label = "CAO (High Confidence)"
    elif max_prob >= 0.60:
        conf_label = "TRUNG BÌNH (Moderate Confidence)"
    else:
        conf_label = "PHÂN VÂN (Ambiguous / Boundary Case)"

    s_real, s_fake = logits_s[0, 0].item(), logits_s[0, 1].item()
    s_prime_real, s_prime_fake = aug_logits_s_prime[0, 0].item(), aug_logits_s_prime[0, 1].item()

    metrics_report = {
        "session_id": session_id,
        "timestamp": now_str,
        "image_path": os.path.abspath(image_path),
        "image_filename": image_filename,
        "image_dims": f"{orig_w}x{orig_h}",
        "checkpoint": os.path.abspath(ckpt_path),
        "model_type": model_type_str,
        "inspection_run_index": check_number,
        "inspection_folder": os.path.abspath(run_dir),
        "mode": args.mode,
        "prediction": pred_label,
        "confidence_label": conf_label,
        "prob_real_pct": prob_real * 100.0,
        "prob_fake_pct": prob_fake * 100.0,
        "logit_margin": logit_margin,
        "shannon_entropy": entropy,
        "raw_logits": {"real": z_real, "fake": z_fake},
        "architecture_decomposition": {
            "clip_s_real": s_real,
            "clip_s_fake": s_fake,
            "clip_s_margin": s_fake - s_real,
            "adapter_s_prime_real": s_prime_real,
            "adapter_s_prime_fake": s_prime_fake,
            "adapter_s_prime_margin": s_prime_fake - s_prime_real,
        },
        "haar_dwt_frequency": dwt_metrics
    }

    # -------------------------------------------------------------------------
    # PHẦN 4: HỒ SƠ ĐÁNH GIÁ PHÁP Y & KẾT LUẬN TOÀN CỤC
    # -------------------------------------------------------------------------
    logger.print("\n[PHẦN 4: HỒ SƠ ĐÁNH GIÁ PHÁP Y & KẾT LUẬN TOÀN CỤC (FORENSIC VERDICT)]")
    logger.print("-" * 100)
    logger.print(f"  * KẾT QUẢ DỰ ĐOÁN         : [{pred_label}]")
    logger.print(f"  * MỨC ĐỘ TIN CẬY          : {conf_label}")
    logger.print(f"  * Xác suất Ảnh Thật (Real): {prob_real * 100.0:6.2f} %")
    logger.print(f"  * Xác suất Ảnh Giả  (Fake): {prob_fake * 100.0:6.2f} %")
    logger.print(f"  * Độ bất định Entropy     : {entropy:.4f} / 1.0000")
    logger.print(f"  * Biên độ Logit (Dz)      : {logit_margin:+.4f}")
    logger.print("  * Phân rã kiến trúc       :")
    logger.print(f"      - Vanilla CLIP S(i)   : Real={s_real:6.2f} | Fake={s_fake:6.2f} | Margin={s_fake - s_real:+6.2f}")
    logger.print(f"      - Forgery Adapter S'(i) : Real={s_prime_real:6.2f} | Fake={s_prime_fake:6.2f} | Margin={s_prime_fake - s_prime_real:+6.2f}")
    logger.print("  * Tần số sóng con 2D DWT  :")
    logger.print(f"      - Tỷ lệ Năng lượng Tần số Cao: {dwt_metrics['high_freq_ratio_pct']:.2f} %")

    # Render Dashboard
    report_img_path = os.path.join(run_dir, f"{image_stem}_{args.mode}_dashboard.png")
    render_standard_dashboard(raw_img, heatmap_16x16, metrics_report, report_img_path, check_number=check_number, alpha=args.alpha)

    # Lưu JSON
    json_path = os.path.join(run_dir, f"{image_stem}_{args.mode}_metrics.json")
    if args.save_json:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_report, f, indent=4, ensure_ascii=False)

    # Ghi nhận vào file lịch sử toàn cục
    record_global_history(global_history_csv, {
        "session_id": session_id,
        "timestamp": now_str,
        "image_name": image_filename,
        "image_dims": f"{orig_w}x{orig_h}",
        "checkpoint_name": os.path.basename(ckpt_path),
        "model_type": model_type_str,
        "mode": args.mode,
        "grid_size": "1x1",
        "verdict": pred_label,
        "max_fake_pct": f"{prob_fake * 100.0:.2f}",
        "mean_fake_pct": f"{prob_fake * 100.0:.2f}",
        "flagged_count": 1 if pred_label == "FAKE" else 0,
        "total_patches": 1,
        "run_dir": os.path.abspath(run_dir)
    })

    # -------------------------------------------------------------------------
    # PHẦN 5: DANH MỤC THÀNH PHẨM & LƯU TRỮ VẾT
    # -------------------------------------------------------------------------
    logger.print("\n[PHẦN 5: DANH MỤC THÀNH PHẨM & ARTIFACTS XUẤT RA (EXPORTED ARTIFACTS)]")
    logger.print("-" * 100)
    logger.print(f"  [✓] Ảnh Dashboard trực quan : {os.path.abspath(report_img_path)}")
    logger.print(f"  [✓] Số liệu chi tiết JSON   : {os.path.abspath(json_path)}")
    logger.print(f"  [✓] Báo cáo văn bản phiên   : {os.path.abspath(session_log_path)}")
    logger.print(f"  [✓] Lịch sử phiên toàn cục  : {os.path.abspath(global_history_csv)}")
    logger.print("=" * 100 + "\n")
    logger.close()


if __name__ == "__main__":
    main()
