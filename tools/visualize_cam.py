"""
tools/visualize_cam.py
======================
Trích xuất Grad-CAM ban đầu (Baseline Extraction) cho mô hình FatFormer.

MỤC TIÊU (Task 2.4):
  - Hook vào tầng `model.image_encoder.transformer.resblocks[-1]`
    (tầng cuối cùng của Visual Transformer ViT-L/14).
  - Tạo bản đồ kích hoạt nhiệt (heatmap) overlay lên ảnh gốc.
  - Hỗ trợ cả ảnh Clean và Degraded (JPEG Q=30).
  - Xuất 10 ảnh Grad-CAM vào thư mục output.

CÁCH SỬ DỤNG:
  # Chế độ thực (với dataset đã tải):
  python tools/visualize_cam.py \
      --checkpoint fatformer_4class_ckpt.pth \
      --input_dir datasets/ \
      --output_dir docs/assets/gradcam_baseline/

  # Chế độ demo (không cần dataset):
  python tools/visualize_cam.py \
      --checkpoint fatformer_4class_ckpt.pth \
      --demo \
      --output_dir docs/assets/gradcam_baseline/
"""

import os
import sys
import io
import argparse
import random
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

# ─── Đảm bảo import được package src ───────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Fix encoding cho Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.models import build_model
from src.training.checkpoint_manager import CheckpointManager


# ─── Cấu hình transform chuẩn CLIP ─────────────────────────────────────────
CLIP_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)),
])


def apply_jpeg_compression(pil_img: Image.Image, quality: int = 30) -> Image.Image:
    """Áp dụng nén JPEG với quality factor cho trước (mô phỏng suy thoái ảnh)."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


# ═══════════════════════════════════════════════════════════════════════════
#  CLASS GRAD-CAM
# ═══════════════════════════════════════════════════════════════════════════

class FatFormerGradCAM:
    """
    Grad-CAM cho nhánh Visual Transformer của FatFormer.
    Hook vào tầng target_layer để lưu activation và gradient.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._handles = []

        # Đăng ký forward hook và backward hook
        self._handles.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        self._handles.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        """Lưu activation map tại forward pass."""
        # output shape: [seq_len, batch, dim] (dạng LND của ViT)
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Lưu gradient tại backward pass."""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Tính bản đồ CAM cho ảnh đầu vào.

        Args:
            input_tensor: Tensor ảnh đã normalize, shape [1, 3, 224, 224].
            target_class: 0 = Real, 1 = Fake.

        Returns:
            cam: numpy array [224, 224] giá trị [0, 1].
        """
        self.model.zero_grad()
        self.gradients = None
        self.activations = None

        # Forward pass — cần enable grad để tính backward
        with torch.enable_grad():
            output = self.model(input_tensor)
            loss = output[:, target_class].sum()
            loss.backward()

        # Kiểm tra hook có thu được dữ liệu không
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Hook không thu được gradient/activation. "
                "Kiểm tra lại target_layer có nằm trong forward path không."
            )

        # grads & acts shape: [seq_len, batch, dim] = [L, B, D]
        grads = self.gradients   # [L, B, D]
        acts  = self.activations # [L, B, D]

        # Tính trọng số alpha bằng Global Average Pooling theo chiều dim
        weights = grads.mean(dim=-1, keepdim=True)  # [L, B, 1]
        cam = (weights * acts).sum(dim=-1)           # [L, B]
        cam = torch.relu(cam)                        # ReLU để loại kích hoạt âm

        # Lấy batch đầu tiên, bỏ token CLS [0]
        # ViT-L/14 với input 224×224: patch_size=14 → grid 16×16 = 256 patch tokens
        cam_seq = cam[1:, 0]  # [256] — bỏ token CLS
        n_patches = cam_seq.shape[0]
        grid_size = int(n_patches ** 0.5)  # 16 cho ViT-L/14

        cam_np = cam_seq.cpu().float().numpy().reshape(grid_size, grid_size)
        cam_np = cv2.resize(cam_np, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Normalize về [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        cam_np = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)
        return cam_np

    def remove_hooks(self):
        """Gỡ bỏ hook sau khi dùng xong để tránh memory leak."""
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ═══════════════════════════════════════════════════════════════════════════
#  HÀM TIỆN ÍCH
# ═══════════════════════════════════════════════════════════════════════════

def overlay_heatmap(original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay heatmap Grad-CAM lên ảnh gốc.

    Args:
        original_rgb: numpy [H, W, 3] uint8, ảnh gốc RGB.
        cam:          numpy [H, W] float [0,1], bản đồ kích hoạt.
        alpha:        Độ trong suốt của heatmap (0=chỉ ảnh gốc, 1=chỉ heatmap).

    Returns:
        Ảnh overlay RGB uint8.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # BGR
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Resize heatmap về đúng kích thước ảnh gốc nếu cần
    if heatmap_rgb.shape[:2] != original_rgb.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (original_rgb.shape[1], original_rgb.shape[0]))

    overlay = (alpha * heatmap_rgb + (1 - alpha) * original_rgb).astype(np.uint8)
    return overlay


def save_cam_result(
    original_img: Image.Image,
    cam: np.ndarray,
    save_path: str,
    label: str,
    condition: str,
    pred_real: float,
    pred_fake: float,
):
    """
    Lưu ảnh kết quả Grad-CAM dạng panel 3 cột: [Ảnh gốc | Heatmap | Overlay].
    Thêm chú thích nhãn, điều kiện và xác suất dự đoán.
    """
    orig_np = np.array(original_img.resize((224, 224)))  # [224, 224, 3] RGB

    # Heatmap thuần (colormap)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = overlay_heatmap(orig_np, cam, alpha=0.45)

    # Ghép 3 cột ngang
    panel = np.concatenate([orig_np, heatmap_rgb, overlay], axis=1)  # [224, 672, 3]

    # Thêm thanh tiêu đề bằng OpenCV
    title_bar = np.ones((40, panel.shape[1], 3), dtype=np.uint8) * 30  # nền tối
    title_text = f"[{label.upper()}] {condition} | P(real)={pred_real:.3f} | P(fake)={pred_fake:.3f}"
    cv2.putText(
        title_bar, title_text,
        org=(8, 28),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.55,
        color=(200, 230, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    final = np.concatenate([title_bar, panel], axis=0)  # [264, 672, 3]

    # Thêm nhãn cột
    col_labels = np.ones((24, final.shape[1], 3), dtype=np.uint8) * 50
    for i, col_name in enumerate(["Original", "Grad-CAM Heatmap", "Overlay"]):
        cv2.putText(
            col_labels, col_name,
            org=(i * 224 + 70, 17),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.45,
            color=(180, 180, 180),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    final = np.concatenate([final, col_labels], axis=0)

    # Lưu ảnh (OpenCV dùng BGR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print(f"  -> Đã lưu: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  THU THẬP ẢNH MẪU
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_CATEGORIES = ["car", "cat", "chair", "horse", "person"]  # 5 danh mục mẫu


def collect_sample_pairs(input_dir: str, n_samples: int = 5) -> list[dict]:
    """
    Thu thập n_samples cặp (real, fake) từ thư mục dataset.

    Cấu trúc kỳ vọng:
        input_dir/
          {category}/0_real/*.png
          {category}/1_fake/*.png

    Returns:
        Danh sách dict: {'real': path, 'fake': path, 'category': str}
    """
    pairs = []
    input_path = Path(input_dir)

    # Thử tìm cấu trúc thư mục
    candidates = []
    for cat in SAMPLE_CATEGORIES:
        real_dir = input_path / cat / "0_real"
        fake_dir = input_path / cat / "1_fake"
        if real_dir.exists() and fake_dir.exists():
            real_imgs = sorted(real_dir.glob("*.png")) + sorted(real_dir.glob("*.jpg"))
            fake_imgs = sorted(fake_dir.glob("*.png")) + sorted(fake_dir.glob("*.jpg"))
            if real_imgs and fake_imgs:
                candidates.append((cat, real_imgs, fake_imgs))

    # Nếu không tìm thấy đúng 5 danh mục kỳ vọng, lấy bất kỳ
    if not candidates:
        for cat_dir in sorted(input_path.iterdir()):
            if not cat_dir.is_dir():
                continue
            real_dir = cat_dir / "0_real"
            fake_dir = cat_dir / "1_fake"
            if real_dir.exists() and fake_dir.exists():
                real_imgs = sorted(real_dir.glob("*.png")) + sorted(real_dir.glob("*.jpg"))
                fake_imgs = sorted(fake_dir.glob("*.png")) + sorted(fake_dir.glob("*.jpg"))
                if real_imgs and fake_imgs:
                    candidates.append((cat_dir.name, real_imgs, fake_imgs))
            if len(candidates) >= n_samples:
                break

    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy cặp ảnh Real/Fake trong '{input_dir}'.\n"
            "Kiểm tra cấu trúc: input_dir/{category}/0_real/ và input_dir/{category}/1_fake/"
        )

    for i, (cat, real_imgs, fake_imgs) in enumerate(candidates[:n_samples]):
        pairs.append({
            "real": str(real_imgs[0]),
            "fake": str(fake_imgs[0]),
            "category": cat,
        })

    return pairs


# ═══════════════════════════════════════════════════════════════════════════
#  CHẠY GRAD-CAM CHO 1 ẢNH
# ═══════════════════════════════════════════════════════════════════════════

def run_gradcam_on_image(
    gradcam: FatFormerGradCAM,
    model: torch.nn.Module,
    img_path: str,
    label: str,
    condition: str,
    output_dir: str,
    idx: int,
    device: torch.device,
    jpeg_quality: int = None,
) -> dict:
    """
    Chạy Grad-CAM cho 1 ảnh và lưu kết quả.

    Args:
        jpeg_quality: Nếu không None, áp dụng nén JPEG trước khi inference.

    Returns:
        Dict chứa thông tin kết quả.
    """
    pil_img = Image.open(img_path).convert("RGB")
    display_img = pil_img.copy()

    # Áp dụng JPEG degradation nếu cần
    if jpeg_quality is not None:
        pil_img = apply_jpeg_compression(pil_img, quality=jpeg_quality)

    # Transform → tensor
    input_tensor = CLIP_TRANSFORM(pil_img).unsqueeze(0).to(device)

    # Tính xác suất dự đoán
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
    probs = logits.softmax(dim=1)[0]
    pred_real = probs[0].item()
    pred_fake = probs[1].item()
    pred_label = "REAL" if pred_real > pred_fake else "FAKE"

    # Tính Grad-CAM (cần grad → không dùng no_grad)
    input_tensor.requires_grad_(True)
    cam = gradcam.generate_cam(input_tensor, target_class=1)  # target = Fake class

    # Lưu ảnh kết quả
    filename = f"{idx:02d}_{label}_{condition}.png"
    save_path = os.path.join(output_dir, filename)
    save_cam_result(
        original_img=display_img,
        cam=cam,
        save_path=save_path,
        label=label,
        condition=condition,
        pred_real=pred_real,
        pred_fake=pred_fake,
    )

    return {
        "idx": idx,
        "label": label,
        "condition": condition,
        "img_path": img_path,
        "pred_label": pred_label,
        "pred_real": pred_real,
        "pred_fake": pred_fake,
        "output_path": save_path,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Trích xuất Grad-CAM baseline cho FatFormer (Task 2.4)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="fatformer_4class_ckpt.pth",
        help="Đường dẫn tới checkpoint FatFormer (mặc định: fatformer_4class_ckpt.pth)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets",
        help="Thư mục chứa ảnh dataset (cấu trúc: {category}/0_real/ & 1_fake/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/assets/gradcam_baseline",
        help="Thư mục lưu ảnh Grad-CAM đầu ra",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=30,
        help="Quality factor JPEG cho ảnh Degraded (mặc định: 30)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Số cặp ảnh mẫu (mặc định: 5)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Chế độ demo: tự sinh ảnh synthetic, không cần dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Thiết bị chạy inference (mặc định: cpu)",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Khởi tạo và nạp checkpoint FatFormer."""
    class Args:
        backbone             = "CLIP:ViT-L/14"
        clip_path            = os.path.join(project_root, "ViT-L-14.pt")
        num_classes          = 2
        num_vit_adapter      = 3
        num_context_embedding = 8
        init_context_embedding = ""
        hidden_dim           = 768
        clip_vision_width    = 1024
        frequency_encoder_layer = 2
        decoder_layer        = 4
        num_heads            = 12
        use_srm              = False
        use_gating           = False

    args = Args()
    print("  [1/3] Khởi tạo kiến trúc FatFormer...")
    model = build_model(args)
    model = model.to(device)

    print(f"  [2/3] Nạp checkpoint: {checkpoint_path}")
    CheckpointManager.load(checkpoint_path, model, device=device, strict=True)
    print("  [3/3] Checkpoint nạp thành công (strict=True, 100% key khớp).")

    model.eval()
    return model


def get_demo_samples(n_samples: int = 5) -> list[dict]:
    """
    Tạo ảnh synthetic demo khi không có dataset thực.
    Sử dụng noise pattern để mô phỏng Real/Fake.
    """
    import tempfile

    demo_dir = tempfile.mkdtemp(prefix="gradcam_demo_")
    samples = []
    categories = ["progan", "stylegan", "midjourney", "sd15", "deepfake"]

    for i in range(n_samples):
        cat = categories[i % len(categories)]
        # Ảnh Real: natural image noise (Gaussian)
        real_arr = np.clip(
            np.random.normal(128, 30, (224, 224, 3)), 0, 255
        ).astype(np.uint8)
        # Thêm gradient để tạo vùng sáng tối
        grad = np.linspace(80, 200, 224).astype(np.uint8)
        real_arr[:, :, 0] = np.clip(real_arr[:, :, 0] + grad[None, :], 0, 255)

        # Ảnh Fake: pattern với grid artifact (mô phỏng GAN artifact)
        fake_arr = np.random.randint(100, 180, (224, 224, 3), dtype=np.uint8)
        # Thêm grid 8×8 giả lập JPEG block artifact
        for r in range(0, 224, 8):
            fake_arr[r:r+1, :, :] = np.clip(fake_arr[r:r+1, :, :] + 40, 0, 255)
        for c in range(0, 224, 8):
            fake_arr[:, c:c+1, :] = np.clip(fake_arr[:, c:c+1, :] + 40, 0, 255)

        real_path = os.path.join(demo_dir, f"{cat}_real_{i}.png")
        fake_path = os.path.join(demo_dir, f"{cat}_fake_{i}.png")
        Image.fromarray(real_arr).save(real_path)
        Image.fromarray(fake_arr).save(fake_path)

        samples.append({
            "real": real_path,
            "fake": fake_path,
            "category": cat,
        })

    return samples


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print("=" * 70)
    print("TASK 2.4 — TRÍCH XUẤT GRAD-CAM BAN ĐẦU ĐỐI CHỨNG MỐC SÀN")
    print("=" * 70)
    print(f"Thiết bị  : {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"JPEG Q    : {args.jpeg_quality} (cho ảnh Degraded)")
    print()

    # ── Load model ──────────────────────────────────────────────────────────
    print("[BƯỚC 1] Nạp mô hình FatFormer...")
    model = load_model(args.checkpoint, device)

    # ── Xác định target layer ────────────────────────────────────────────────
    # Truy cập: model (CLIPModel) → image_encoder (VisionTransformer)
    #           → transformer (Transformer) → resblocks (Sequential) → [-1]
    target_layer = model.image_encoder.transformer.resblocks[-1]
    print(f"\n[BƯỚC 2] Target layer: image_encoder.transformer.resblocks[-1]")
    print(f"  Layer type: {type(target_layer).__name__}")

    # Khởi tạo Grad-CAM
    gradcam = FatFormerGradCAM(model, target_layer)

    # ── Thu thập ảnh mẫu ─────────────────────────────────────────────────────
    print(f"\n[BƯỚC 3] Thu thập ảnh mẫu...")
    if args.demo:
        print("  Chế độ DEMO: Sinh ảnh synthetic...")
        sample_pairs = get_demo_samples(args.n_samples)
    else:
        print(f"  Đọc dataset từ: {args.input_dir}")
        sample_pairs = collect_sample_pairs(args.input_dir, args.n_samples)

    print(f"  Thu thập được {len(sample_pairs)} cặp từ danh mục: "
          f"{[p['category'] for p in sample_pairs]}")

    # ── Tạo thư mục output ───────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Chạy Grad-CAM ────────────────────────────────────────────────────────
    print(f"\n[BƯỚC 4] Trích xuất Grad-CAM (10 ảnh = 5 Real + 5 Fake × Clean & Degraded)...")
    results = []
    idx = 1

    for pair in sample_pairs:
        cat = pair["category"]
        print(f"\n  --- [{cat.upper()}] ---")

        # Real Clean
        print(f"  [{idx:02d}] REAL — Clean")
        r = run_gradcam_on_image(
            gradcam, model, pair["real"], label="real", condition="clean",
            output_dir=args.output_dir, idx=idx, device=device
        )
        results.append(r)
        idx += 1

        # Fake Clean
        print(f"  [{idx:02d}] FAKE — Clean")
        r = run_gradcam_on_image(
            gradcam, model, pair["fake"], label="fake", condition="clean",
            output_dir=args.output_dir, idx=idx, device=device
        )
        results.append(r)
        idx += 1

    # Chạy thêm 5 ảnh Degraded (Q=30) — lấy từ 5 ảnh Real đầu tiên
    print(f"\n  --- [DEGRADED Q={args.jpeg_quality}] ---")
    degraded_idx = 1  # Bắt đầu từ ảnh số 1
    for pair in sample_pairs[:5]:
        cat = pair["category"]

        # Chọn real hoặc fake xen kẽ
        is_real = (degraded_idx % 2 == 1)
        img_path = pair["real"] if is_real else pair["fake"]
        label = "real" if is_real else "fake"

        print(f"  [{idx:02d}] {label.upper()} — Degraded Q={args.jpeg_quality} [{cat}]")
        r = run_gradcam_on_image(
            gradcam, model, img_path, label=label,
            condition=f"degraded_Q{args.jpeg_quality}",
            output_dir=args.output_dir, idx=idx, device=device,
            jpeg_quality=args.jpeg_quality,
        )
        results.append(r)
        idx += 1
        degraded_idx += 1

    # ── Gỡ hook ──────────────────────────────────────────────────────────────
    gradcam.remove_hooks()

    # ── In tổng kết ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KẾT QUẢ TỔNG KẾT")
    print("=" * 70)
    print(f"{'#':>3} | {'Label':^6} | {'Condition':^20} | {'Pred':^6} | {'P(real)':>8} | {'P(fake)':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['idx']:>3} | {r['label']:^6} | {r['condition']:^20} | "
            f"{r['pred_label']:^6} | {r['pred_real']:>8.4f} | {r['pred_fake']:>8.4f}"
        )

    correct_real = sum(1 for r in results if r["label"] == "real" and r["pred_label"] == "REAL")
    correct_fake = sum(1 for r in results if r["label"] == "fake" and r["pred_label"] == "FAKE")
    n_real = sum(1 for r in results if r["label"] == "real")
    n_fake = sum(1 for r in results if r["label"] == "fake")

    print("=" * 70)
    print(f"Real đúng: {correct_real}/{n_real} | Fake đúng: {correct_fake}/{n_fake}")
    print(f"Tổng ảnh đã xuất: {len(results)} ảnh → {args.output_dir}/")
    print("=" * 70)
    print("\nTASK 2.4 HOÀN THÀNH! Kiểm tra ảnh tại:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
