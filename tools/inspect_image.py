"""
================================================================================
FATFORMER FORENSIC INSPECTOR - PHÂN TÍCH VÀ ĐÁNH GIÁ PHÁP Y ẢNH ĐƠN LẺ
================================================================================
Công cụ kiểm thử pháp y cho 1 bức ảnh với FatFormer Checkpoint (CVPR 2024):
1. Chế độ Standard: Resize 256x256 -> CenterCrop 224x224 (Quy chuẩn tác giả).
2. Chế độ Grid Multi-Patch (1B): Cắt lưới đa mảnh ở độ phân giải gốc (Native Resolution),
   bảo toàn 100% vết vi sai tần số cao của AI, không bị làm mờ bởi Resize.
3. Chế độ Native Center: Cắt 1 mảnh 224x224 tại tâm gốc không qua Resize.
4. Trích xuất chỉ số xác suất, Logits Margin, Entropy, và Phân rã S(i) vs S'(i).
5. Phân tích phổ năng lượng sóng con 2D Haar DWT (High-Frequency Ratio).
6. Quản lý thư mục output chuyên nghiệp theo Phương Án A:
   - Thư mục gốc mặc định: DATASET/inspection_results/<tên_ảnh>/
   - Mỗi lần chạy tự động đóng vào thư mục riêng: check_01/, check_02/, ...
   - Tự động đếm và hiển thị: "LẦN KIỂM TRA THỨ: #N đối với ảnh <tên_ảnh>".
================================================================================
"""

import os
import sys
import json
import argparse
import math
import numpy as np
from PIL import Image

# Đảm bảo mã hóa UTF-8 cho Windows Terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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


def resolve_file_path(user_path, default_candidates):
    """Tìm đường dẫn tệp khả dụng từ tham số người dùng hoặc danh sách dự phòng."""
    if user_path and os.path.exists(user_path):
        return os.path.abspath(user_path)
    for cand in default_candidates:
        if cand and os.path.exists(cand):
            return os.path.abspath(cand)
    return None


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
    Chạy forward pass và bóc tách:
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


def render_standard_dashboard(orig_pil_img, heatmap_16x16, metrics, output_path, check_number=1, alpha=0.55):
    """Render và lưu Dashboard chuẩn 3 panel cho 1 ảnh."""
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

    conclusion_text = (
        f"PREDICTION: {metrics['prediction']} ({metrics['confidence_label']})\n"
        f"----------------------------------------------------------\n"
        f"• Inspection Run     : Check #{check_number}\n"
        f"• Shannon Entropy    : {metrics['shannon_entropy']:.4f} / 1.0000\n"
        f"• Logit Margin (Δz)  : {metrics['logit_margin']:+.4f} (z_fake: {metrics['raw_logits']['fake']:.2f}, z_real: {metrics['raw_logits']['real']:.2f})\n"
        f"• CLIP S(i) Margin   : {metrics['architecture_decomposition']['clip_s_margin']:+.4f}\n"
        f"• Adapter S' Margin  : {metrics['architecture_decomposition']['adapter_s_prime_margin']:+.4f}\n"
        f"• DWT High-Freq Ratio: {metrics['haar_dwt_frequency']['high_freq_ratio_pct']:.2f}%"
    )

    bbox_props = dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5", edgecolor=pred_color, linewidth=1.8)
    ax3.text(0.0, 0.48, conclusion_text, transform=ax3.transAxes, fontsize=10.2,
             family="monospace", va="top", bbox=bbox_props)

    plt.suptitle(f"FATFORMER FORENSIC REPORT: {os.path.basename(metrics['image_path'])} [CHECK #{check_number}]",
                 fontsize=14, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def render_grid_dashboard(orig_pil_img, patch_results, summary_metrics, output_path, check_number=1, grid_size=3):
    """
    Render Dashboard phân tích lưới đa mảnh ở độ phân giải gốc (Phương Án 1B).
    Không dùng ký tự có dấu trong biểu đồ Matplotlib để tránh cảnh báo font chữ.
    """
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
    ax1.set_title(f"Native Image ({orig_w}x{orig_h}) - Grid Patches", fontsize=12, fontweight="bold", pad=10)

    for p in patch_results:
        x1, y1, x2, y2 = p["box"]
        p_fake = p["prob_fake_pct"]
        is_p_fake = p_fake >= 50.0
        box_color = "#F44336" if is_p_fake else "#4CAF50"

        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.0,
                                  edgecolor=box_color, facecolor="none", alpha=0.9)
        ax1.add_patch(rect)
        ax1.text(x1 + 6, y1 + 22, f"#{p['idx']} ({p_fake:.1f}%)", color="white", fontsize=8,
                 fontweight="bold", bbox=dict(boxstyle="square,pad=0.2", facecolor=box_color, alpha=0.85))

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
                     color=text_color, fontweight="bold", fontsize=10)

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

    conclusion_text = (
        f"OVERALL VERDICT: {verdict}\n"
        f"----------------------------------------------------------\n"
        f"• Inspection Run Index    : Check #{check_number}\n"
        f"• Total Patches Scanned   : {len(patch_results)} (Native 224x224)\n"
        f"• Flagged Fake Patches    : {summary_metrics['flagged_fake_count']} / {len(patch_results)}\n"
        f"• Peak Fake Probability   : {summary_metrics['max_fake_prob']:.2f}% (Patch #{worst_p['idx']})\n"
        f"• Mean Fake Probability   : {summary_metrics['mean_fake_prob']:.2f}%\n"
        f"----------------------------------------------------------\n"
        f"MOST SUSPICIOUS PATCH (#{worst_p['idx']}):\n"
        f"• Box Coordinates         : {worst_p['box']}\n"
        f"• Logit Margin (Delta-z)  : {worst_p['logit_margin']:+.4f}\n"
        f"• CLIP S(i) Margin        : {worst_p['clip_s_margin']:+.4f}\n"
        f"• Adapter S'(i) Margin    : {worst_p['adapter_s_prime_margin']:+.4f}\n"
        f"• DWT High-Freq Ratio     : {worst_p['high_freq_ratio']:.2f}%"
    )

    bbox_props = dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5", edgecolor=verdict_color, linewidth=1.8)
    ax3.text(0.0, 0.48, conclusion_text, transform=ax3.transAxes, fontsize=9.6,
             family="monospace", va="top", bbox=bbox_props)

    plt.suptitle(f"FATFORMER NATIVE GRID FORENSIC DASHBOARD: {os.path.basename(summary_metrics['image_path'])} [CHECK #{check_number}]",
                 fontsize=13.5, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def setup_run_directory(base_output_dir, image_path):
    """
    Quản lý phân cấp thư mục theo Phương Án A:
    DATASET/inspection_results/<tên_ảnh>/check_XX/
    Tự động tăng số thứ tự check_XX và trả về: run_dir, check_number
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


def main():
    default_dataset_output = os.path.join(project_root, "DATASET", "inspection_results")

    parser = argparse.ArgumentParser("FatFormer Forensic Inspector (CVPR 2024)", add_help=True)
    parser.add_argument("--image", "-i", type=str, required=True, help="Đường dẫn đến file ảnh cần kiểm tra.")
    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "standard", "native_center"],
                        help="Chế độ kiểm tra: grid (Lưới đa mảnh độ phân giải gốc - Mặc định), standard (Resize 224), native_center (1 ô tâm).")
    parser.add_argument("--grid_size", type=int, default=4, help="Kích thước lưới khi chọn mode grid (mặc định 4: 4x4=16 mảnh).")
    parser.add_argument("--ckpt", type=str, default=None, help="Đường dẫn fatformer_4class_ckpt.pth.")
    parser.add_argument("--clip_path", type=str, default=None, help="Đường dẫn ViT-L-14.pt.")
    parser.add_argument("--output_dir", "-o", type=str, default=default_dataset_output,
                        help=f"Thư mục gốc lưu kết quả (Mặc định: DATASET/inspection_results).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--alpha", type=float, default=0.55, help="Độ trong suốt của Heatmap (0.0 đến 1.0).")
    parser.add_argument("--save_json", action="store_true", default=True, help="Lưu kết quả chi tiết ra JSON.")
    parser.add_argument("--use_srm", action="store_true", help="Kích hoạt khối SRM (nếu có)")
    parser.add_argument("--use_gating", action="store_true", help="Kích hoạt Dynamic Gating (nếu có)")
    args = parser.parse_args()

    print("=" * 85)
    print("      FATFORMER FORENSIC INSPECTOR - PHÂN TÍCH PHÁP Y THỊ GIÁC ĐA CHẾ ĐỘ")
    print("=" * 85)

    # 1. Kiểm tra ảnh đầu vào
    if not os.path.exists(args.image):
        print(f"❌ [LỖI] Không tìm thấy tệp ảnh tại: {args.image}")
        sys.exit(1)

    raw_img = Image.open(args.image).convert("RGB")
    orig_w, orig_h = raw_img.size
    image_filename = os.path.basename(args.image)

    # 2. Khởi tạo thư mục theo Phương Án A và xác định lần check thứ mấy
    run_dir, check_number, image_stem = setup_run_directory(args.output_dir, args.image)

    print(f"[*] Tệp ảnh kiểm thử  : {os.path.abspath(args.image)} ({orig_w}x{orig_h})")
    print(f"[*] Chế độ phân tích : {args.mode.upper()} (Grid={args.grid_size}x{args.grid_size} nếu mode grid)")
    print(f"[*] Thiết bị suy luận : {args.device}")
    print("-" * 85)
    print(f"▶ LẦN KIỂM TRA THỨ    : #{check_number} ĐỐI VỚI ẢNH [{image_filename}]")
    print(f"▶ Thư mục lưu kết quả : {os.path.abspath(run_dir)}")
    print("-" * 85)

    # 3. Định vị Checkpoint và CLIP
    ckpt_candidates = [
        args.ckpt,
        os.path.join(project_root, "DATASET", "pretrained", "fatformer_4class_ckpt.pth"),
        os.path.join(project_root, "fatformer_4class_ckpt.pth")
    ]
    ckpt_path = resolve_file_path(args.ckpt, ckpt_candidates)
    if not ckpt_path:
        print(f"❌ [LỖI] Không tìm thấy file checkpoint fatformer_4class_ckpt.pth!")
        sys.exit(1)

    clip_candidates = [
        args.clip_path,
        os.path.join(project_root, "DATASET", "pretrained", "ViT-L-14.pt"),
        os.path.join(project_root, "ViT-L-14.pt")
    ]
    clip_path = resolve_file_path(args.clip_path, clip_candidates)
    if not clip_path:
        print(f"❌ [LỖI] Không tìm thấy file backbone ViT-L-14.pt!")
        sys.exit(1)

    device = torch.device(args.device)

    # 4. Khởi tạo mô hình
    model_args = argparse.Namespace(
        backbone="CLIP:ViT-L/14",
        clip_path=clip_path,
        num_classes=2,
        num_vit_adapter=3,
        num_context_embedding=8,
        init_context_embedding="",
        hidden_dim=768,
        clip_vision_width=1024,
        frequency_encoder_layer=2,
        decoder_layer=4,
        num_heads=12,
        use_srm=args.use_srm,
        use_gating=args.use_gating
    )

    print("[*] Đang nạp mô hình FatFormer và Checkpoint tác giả...")
    model = build_model(model_args)
    model = model.to(device)

    strict = not (args.use_srm or args.use_gating)
    CheckpointManager.load(ckpt_path, model, device=device, strict=strict)
    print("  -> Mô hình đã sẵn sàng!\n")

    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # =========================================================================
    # CHẾ ĐỘ 1B: GRID MULTI-PATCH (QUÉT LƯỚI ĐỘ PHÂN GIẢI GỐC)
    # =========================================================================
    if args.mode == "grid":
        if orig_w < 224 or orig_h < 224:
            print(f"⚠️ [CẢNH BÁO] Ảnh ({orig_w}x{orig_h}) nhỏ hơn 224x224, tự động chuyển về mode standard.")
            args.mode = "standard"
        else:
            g = max(2, args.grid_size)
            print(f"[*] Đang chia lưới {g}x{g} ({g*g} mảnh 224x224 nguyên bản không qua Resize)...")
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
                "image_path": os.path.abspath(args.image),
                "image_filename": image_filename,
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

            print("=" * 85)
            print(f"       BẢNG KẾT QUẢ QUÉT LƯỚI ĐA MẢNH NGUYÊN BẢN (LẦN CHECK #{check_number})")
            print("=" * 85)
            print(f"{'Mảnh #':^8} | {'Tọa độ Box (x1, y1, x2, y2)':^26} | {'Real (%)':^10} | {'Fake (%)':^10} | {'DWT High-Freq':^14} | {'Đánh giá':^10}")
            print("-" * 85)
            for p in patch_results:
                status = "🔴 FAKE" if p["prob_fake_pct"] >= 50.0 else "🟢 REAL"
                box_str = f"({p['box'][0]}, {p['box'][1]}, {p['box'][2]}, {p['box'][3]})"
                print(f"{p['idx']:^8} | {box_str:^26} | {p['prob_real_pct']:^10.2f} | {p['prob_fake_pct']:^10.2f} | {p['high_freq_ratio']:^12.2f} % | {status:^10}")
            print("=" * 85)
            print(f"▶ LẦN KIỂM TRA THỨ      : #{check_number} đối với ảnh [{image_filename}]")
            print(f"▶ KẾT LUẬN TOÀN CỤC     : {verdict}")
            print(f"▶ Mảnh nghi vấn Fake nhất : Mảnh #{worst_patch['idx']} với Fake = {max_fake_prob:.2f}% (Tọa độ: {worst_patch['box']})")
            print(f"▶ Số mảnh bị nghi ngờ Fake: {flagged_count} / {len(patch_results)} mảnh")
            print("=" * 85 + "\n")

            # Render Dashboard Lưới
            grid_report_path = os.path.join(run_dir, f"{image_stem}_grid_dashboard.png")
            print("[*] Đang render Dashboard phân tích lưới đa mảnh...")
            render_grid_dashboard(raw_img, patch_results, summary_metrics, grid_report_path, check_number=check_number, grid_size=g)
            print(f"  -> Đã lưu ảnh Grid Dashboard tại: {grid_report_path}")

            if args.save_json:
                json_path = os.path.join(run_dir, f"{image_stem}_grid_metrics.json")
                json_data = {
                    "summary": summary_metrics,
                    "patches": [{k: v for k, v in p.items() if k != "heatmap"} for p in patch_results]
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False)
                print(f"  -> Đã lưu số liệu JSON tại: {json_path}")
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
    else:  # native_center
        crop_x = max(0, (orig_w - 224) // 2)
        crop_y = max(0, (orig_h - 224) // 2)
        center_crop_img = raw_img.crop((crop_x, crop_y, crop_x + 224, crop_y + 224))
        img_tensor = norm_transform(center_crop_img).unsqueeze(0).to(device)

    dwt_metrics = compute_haar_dwt_energy(img_tensor)
    print(f"[*] Đang thực hiện Forward Pass (Mode: {args.mode})...")
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
        "image_path": os.path.abspath(args.image),
        "image_filename": image_filename,
        "inspection_run_index": check_number,
        "inspection_folder": os.path.abspath(run_dir),
        "mode": args.mode,
        "image_dimensions": {"width": orig_w, "height": orig_h},
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

    report_img_path = os.path.join(run_dir, f"{image_stem}_{args.mode}_dashboard.png")
    print(f"[*] Đang render Dashboard trực quan hóa...")
    render_standard_dashboard(raw_img, heatmap_16x16, metrics_report, report_img_path, check_number=check_number, alpha=args.alpha)
    print(f"  -> Đã lưu ảnh Dashboard tại: {report_img_path}")

    if args.save_json:
        json_path = os.path.join(run_dir, f"{image_stem}_{args.mode}_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_report, f, indent=4, ensure_ascii=False)
        print(f"  -> Đã lưu số liệu JSON tại: {json_path}")

    print("\n" + "=" * 85)
    print(f"         HỒ SƠ ĐÁNH GIÁ PHÁP Y THỊ GIÁC ({args.mode.upper()} - LẦN CHECK #{check_number})")
    print("=" * 85)
    print(f"  ▶ LẦN KIỂM TRA THỨ      : #{check_number} đối với ảnh [{image_filename}]")
    print(f"  ▶ KẾT QUẢ DỰ ĐOÁN       : [{pred_label}]")
    print(f"  ▶ MỨC ĐỘ TIN CẬY        : {conf_label}")
    print(f"  ▶ Xác suất Ảnh Thật (Real): {prob_real * 100.0:6.2f} %")
    print(f"  ▶ Xác suất Ảnh Giả  (Fake): {prob_fake * 100.0:6.2f} %")
    print(f"  ▶ Độ bất định Entropy   : {entropy:.4f} / 1.0000")
    print(f"  ▶ Biên độ Logit (Δz)    : {logit_margin:+.4f}")
    print("-" * 85)
    print("  PHÂN RÃ ĐÓNG GÓP THÀNH PHẦN KIẾN TRÚC:")
    print(f"    • Vanilla CLIP S(i)   : Real={s_real:6.2f} | Fake={s_fake:6.2f} | Margin={s_fake - s_real:+6.2f}")
    print(f"    • Forgery Adapter S'(i) : Real={s_prime_real:6.2f} | Fake={s_prime_fake:6.2f} | Margin={s_prime_fake - s_prime_real:+6.2f}")
    print("-" * 85)
    print("  PHÂN TÍCH TẦN SỐ SÓNG CON 2D HAAR DWT:")
    print(f"    • Tỷ lệ Năng lượng Tần số Cao (High-Freq Ratio): {dwt_metrics['high_freq_ratio_pct']:.2f} %")
    print(f"    • Năng lượng Dải thấp E(LL)                   : {dwt_metrics['energy_low']:.2e}")
    print(f"    • Năng lượng Dải cao E(High)                  : {dwt_metrics['energy_high']:.2e}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
