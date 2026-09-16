"""
Kiểm thử toàn diện (Smoke Test) cho kiến trúc mới dưới src/:
1. Kiểm tra build_model với CLIP:ViT-L/14 (tự động nạp ViT-L-14.pt).
2. Nạp fatformer_4class_ckpt.pth với strict=True (phải khớp 100% không thiếu/thừa key nào).
3. Chạy forward pass thử nghiệm trên dummy input (1, 3, 224, 224).
4. Kiểm tra mô hình mở rộng với SRM 3-Kernels và Dynamic Frequency Gating.
"""

import os
import sys
import argparse
import torch

# Đảm bảo import được package src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.models import build_model
from src.training.checkpoint_manager import CheckpointManager


def test_baseline_loading():
    print("=" * 80)
    print("TEST 1: KIỂM TRA NẠP BASELINE MODEL & CHECKPOINT GỐC (strict=True)")
    print("=" * 80)

    class Args:
        backbone = "CLIP:ViT-L/14"
        clip_path = os.path.join(project_root, "ViT-L-14.pt")
        num_classes = 2
        num_vit_adapter = 3
        num_context_embedding = 8
        init_context_embedding = ""
        hidden_dim = 768
        clip_vision_width = 1024
        frequency_encoder_layer = 2
        decoder_layer = 4
        num_heads = 12
        use_srm = False
        use_gating = False

    args = Args()
    print(f"- Đang khởi tạo mô hình FatFormer (use_srm=False, use_gating=False)...")
    model = build_model(args)
    print("  -> Khởi tạo thành công!")

    ckpt_path = os.path.join(project_root, "fatformer_4class_ckpt.pth")
    print(f"- Đang nạp checkpoint tác giả: {os.path.basename(ckpt_path)}...")
    
    ckpt = CheckpointManager.load(ckpt_path, model, device=torch.device("cpu"), strict=True)
    print("  -> Nạp thành công với strict=True! 100% key khớp hoàn toàn.")

    # Forward pass
    print("- Đang chạy Forward Pass với Dummy Tensor (1, 3, 224, 224)...")
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        prob = output.softmax(dim=1)
    
    print(f"  -> Output shape: {output.shape} (Kỳ vọng: [1, 2])")
    print(f"  -> Raw Logits: {output.tolist()}")
    print(f"  -> Softmax Probabilities: Real={prob[0, 0]:.4f}, Fake={prob[0, 1]:.4f}")
    assert output.shape == (1, 2), f"Lỗi shape đầu ra: {output.shape}"
    print("  -> TEST 1 PASSED: Baseline hoạt động hoàn hảo!\n")


def test_enhanced_model():
    print("=" * 80)
    print("TEST 2: KIỂM TRA MÔ HÌNH CẢI TIẾN (SRM 3-Kernels + Dynamic Frequency Gating)")
    print("=" * 80)

    class Args:
        backbone = "CLIP:ViT-L/14"
        clip_path = os.path.join(project_root, "ViT-L-14.pt")
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

    args = Args()
    print(f"- Đang khởi tạo mô hình cải tiến (use_srm=True, use_gating=True)...")
    model = build_model(args)
    print("  -> Khởi tạo thành công!")

    ckpt_path = os.path.join(project_root, "fatformer_4class_ckpt.pth")
    print(f"- Nạp pre-trained weights tác giả với strict=False (giữ nguyên SRM và Gating)...")
    ckpt = CheckpointManager.load(ckpt_path, model, device=torch.device("cpu"), strict=False)

    print("- Đang chạy Forward Pass với Dummy Tensor (1, 3, 224, 224)...")
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        prob = output.softmax(dim=1)

    print(f"  -> Output shape: {output.shape} (Kỳ vọng: [1, 2])")
    print(f"  -> Raw Logits: {output.tolist()}")
    print(f"  -> Softmax Probabilities: Real={prob[0, 0]:.4f}, Fake={prob[0, 1]:.4f}")
    assert output.shape == (1, 2), f"Lỗi shape đầu ra: {output.shape}"
    print("  -> TEST 2 PASSED: Mô hình cải tiến SRM + Gating hoạt động hoàn hảo!\n")


if __name__ == "__main__":
    test_baseline_loading()
    test_enhanced_model()
    print("=" * 80)
    print("TẤT CẢ CÁC BÀI KIỂM THỬ SMOKE TEST ĐÃ HOÀN TẤT THÀNH CÔNG RỰC RỠ!")
    print("=" * 80)
