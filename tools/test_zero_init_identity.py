# -*- coding: utf-8 -*-
"""
FatFormer-XLA: Kiểm thử Đồng nhất Bước 0 (Step 0 Zero-Init Identity Proof)
-------------------------------------------------------------------------
Chiến dịch Khôi phục Hiệu năng Đỉnh cao (Phase 6-R: High-Accuracy Anchored Recovery)
Người chủ trì: Thành viên B (Architecture & Loss Lead)
Tham chiếu: Task 6.2-R (Zero-Init Identity Architecture & Proof)

Mục tiêu khoa học:
  Chứng minh bằng toán học và thực nghiệm rằng:
  1. Module vết dư không gian SRM (SpatialResidualBlock) có đầu ra triệt tiêu hoàn toàn
     tại Step 0 (f_srm == 0.0) nhờ khởi tạo Zero-Init ở tầng Conv chiếu cuối cùng.
  2. Module Cổng Tần số Động (DynamicFrequencyGating) có hệ số điều chế trung tính tuyệt đối
     tại Step 0 (lambda(x) == 1.0) với mọi đầu vào x nhờ khởi tạo Zero-Init ở tầng Linear cuối.
  3. Cơ chế nhân tích lũy (Multiplicative Modulation): scale = freq_scale * lambda(x) bảo toàn
     100% hệ số freq_scale đã học của FatFormer gốc tại Step 0.
  4. Mô hình FatFormer nâng cấp (use_srm=True, use_gating=True) nạp checkpoint gốc
     sản sinh ra logit đồng nhất bit-perfect với mô hình FatFormer nguyên bản:
     max |S_upgraded(x) - S_vanilla(x)| < 1e-5.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
from src.models.srm import SpatialResidualBlock
from src.models.gating import DynamicFrequencyGating
from src.training.checkpoint_manager import CheckpointManager


def find_checkpoint_path() -> str:
    """Tìm đường dẫn checkpoint tác giả hợp lệ."""
    candidates = [
        os.path.join(PROJECT_ROOT, "DATASET", "pretrained", "fatformer_4class_ckpt.pth"),
        os.path.join(PROJECT_ROOT, "pretrained", "fatformer_4class_ckpt.pth"),
        os.path.join(PROJECT_ROOT, "fatformer_4class_ckpt.pth"),
        "DATASET/pretrained/fatformer_4class_ckpt.pth",
        "pretrained/fatformer_4class_ckpt.pth",
        "fatformer_4class_ckpt.pth",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return ""


def test_isolated_srm(device: torch.device) -> bool:
    print("\n" + "=" * 78)
    print("TEST 1: KIỂM THỬ KHỐI SRM CÔ LẬP (SpatialResidualBlock Zero-Init)")
    print("=" * 78)

    srm = SpatialResidualBlock(in_channels=3, out_dim=1024).to(device)
    srm.eval()

    # Kiểm tra tensor trọng số tầng cuối
    last_conv = srm.proj[-1]
    weight_norm = last_conv.weight.abs().max().item()
    bias_norm = last_conv.bias.abs().max().item() if last_conv.bias is not None else 0.0

    print(f"[*] Trọng số proj[-1].weight max abs: {weight_norm:.8f} (Kỳ vọng: 0.00000000)")
    print(f"[*] Trọng số proj[-1].bias max abs  : {bias_norm:.8f} (Kỳ vọng: 0.00000000)")

    # Test forward pass với các giá trị đầu vào đa dạng
    shapes_to_test = [(1, 3, 224, 224), (4, 3, 224, 224)]
    passed = True

    for bs_shape in shapes_to_test:
        x_rand = torch.randn(*bs_shape, device=device)
        x_ones = torch.ones(*bs_shape, device=device)
        x_zeros = torch.zeros(*bs_shape, device=device)

        with torch.no_grad():
            out_rand = srm(x_rand)
            out_ones = srm(x_ones)
            out_zeros = srm(x_zeros)

        diff_rand = out_rand.abs().max().item()
        diff_ones = out_ones.abs().max().item()
        diff_zeros = out_zeros.abs().max().item()

        print(f"[*] Forward pass batch {bs_shape}:")
        print(f"    - Random input: max abs output = {diff_rand:.10e}")
        print(f"    - All-ones    : max abs output = {diff_ones:.10e}")
        print(f"    - All-zeros   : max abs output = {diff_zeros:.10e}")

        if diff_rand > 1e-7 or diff_ones > 1e-7 or diff_zeros > 1e-7:
            passed = False

    if passed and weight_norm == 0.0 and bias_norm == 0.0:
        print("[✓] TEST 1: PASS — SpatialResidualBlock triệt tiêu vết dư hoàn hảo về 0!")
        return True
    else:
        print("[✗] TEST 1: FAIL — Vết dư SRM không triệt tiêu về 0 tại Step 0.")
        return False


def test_isolated_gating(device: torch.device) -> bool:
    print("\n" + "=" * 78)
    print("TEST 2: KIỂM THỬ KHỐI GATING CÔ LẬP (DynamicFrequencyGating Neutrality)")
    print("=" * 78)

    gating = DynamicFrequencyGating(d_model=1024, hidden_dim=64).to(device)
    gating.eval()

    # Kiểm tra tầng linear cuối (mlp[2])
    last_linear = gating.mlp[2]
    w_norm = last_linear.weight.abs().max().item()
    b_norm = last_linear.bias.abs().max().item() if last_linear.bias is not None else 0.0

    print(f"[*] Trọng số mlp[2].weight max abs: {w_norm:.8f} (Kỳ vọng: 0.00000000)")
    print(f"[*] Trọng số mlp[2].bias max abs  : {b_norm:.8f} (Kỳ vọng: 0.00000000)")

    passed = True
    # Test forward pass với các tensor đặc trưng token dạng [N, B, D]
    for shape in [(256, 2, 1024), (257, 4, 1024)]:
        x_rand = torch.randn(*shape, device=device)
        x_large = torch.randn(*shape, device=device) * 50.0

        with torch.no_grad():
            lambda_rand = gating(x_rand)
            lambda_large = gating(x_large)

        # Vì weight=0, bias=0 -> Linear=0 -> Sigmoid(0)=0.5 -> 0.5 * 2.0 = 1.0 chính xác
        diff_rand = (lambda_rand - 1.0).abs().max().item()
        diff_large = (lambda_large - 1.0).abs().max().item()

        print(f"[*] Forward pass tensor tokens {shape}:")
        print(f"    - Normal tokens: max |lambda(x) - 1.0| = {diff_rand:.10e}")
        print(f"    - Large tokens : max |lambda(x) - 1.0| = {diff_large:.10e}")

        if diff_rand > 1e-7 or diff_large > 1e-7:
            passed = False

    if passed and w_norm == 0.0 and b_norm == 0.0:
        print("[✓] TEST 2: PASS — DynamicFrequencyGating trung tính tuyệt đối (lambda == 1.0)!")
        return True
    else:
        print("[✗] TEST 2: FAIL — lambda(x) không đạt 1.0 chính xác.")
        return False


def test_full_model_identity(ckpt_path: str, device: torch.device) -> bool:
    print("\n" + "=" * 78)
    print("TEST 3: KIỂM THỬ ĐỒNG NHẤT TOÀN DIỆN MÔ HÌNH FATFORMER (Step 0 Identity Proof)")
    print("=" * 78)

    class VanillaArgs:
        backbone = "CLIP:ViT-L/14"
        clip_path = os.path.join(PROJECT_ROOT, "DATASET", "pretrained", "ViT-L-14.pt")
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

    class UpgradedArgs:
        backbone = "CLIP:ViT-L/14"
        clip_path = os.path.join(PROJECT_ROOT, "DATASET", "pretrained", "ViT-L-14.pt")
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

    print(f"[*] Đang nạp mô hình Vanilla (use_srm=False, use_gating=False)...")
    model_vanilla = build_model(VanillaArgs()).to(device)
    CheckpointManager.load(ckpt_path, model_vanilla, device=device, strict=True)
    model_vanilla.eval()

    print(f"[*] Đang nạp mô hình Upgraded (use_srm=True, use_gating=True) với Zero-Init...")
    model_upgraded = build_model(UpgradedArgs()).to(device)
    # Nạp với strict=False vì checkpoint gốc không chứa tham số SRM/Gating (những tham số này giữ zero-init)
    CheckpointManager.load(ckpt_path, model_upgraded, device=device, strict=False)
    model_upgraded.eval()

    passed = True
    torch.manual_seed(42)

    # Thử nghiệm qua 4 batch đầu vào khác nhau
    test_batches = [
        ("Batch ngẫu nhiên chuẩn [2, 3, 224, 224]", torch.randn(2, 3, 224, 224, device=device)),
        ("Batch ngẫu nhiên chuẩn [4, 3, 224, 224]", torch.randn(4, 3, 224, 224, device=device)),
        ("Batch ảnh trắng tinh (All 1.0)", torch.ones(2, 3, 224, 224, device=device)),
        ("Batch ảnh đen tinh   (All 0.0)", torch.zeros(2, 3, 224, 224, device=device)),
    ]

    for desc, inp in test_batches:
        with torch.no_grad():
            out_vanilla = model_vanilla(inp)
            out_upgraded = model_upgraded(inp)

            # Test dual stream nếu có
            dual_v1, dual_v2 = model_vanilla(inp, return_dual=True)
            dual_u1, dual_u2 = model_upgraded(inp, return_dual=True)

        diff_total = (out_upgraded - out_vanilla).abs().max().item()
        diff_s1 = (dual_u1 - dual_v1).abs().max().item()
        diff_s2 = (dual_u2 - dual_v2).abs().max().item()

        print(f"\n[*] {desc}:")
        print(f"    • Vanilla Logits: {out_vanilla[0].tolist()}")
        print(f"    • Upgraded Logits: {out_upgraded[0].tolist()}")
        print(f"    • Max Diff (Sum Logits)  : {diff_total:.10e}")
        print(f"    • Max Diff (Stream 1 S)  : {diff_s1:.10e}")
        print(f"    • Max Diff (Stream 2 S') : {diff_s2:.10e}")

        # Ngưỡng chấp nhận sai số dấu phẩy động FP32: < 1e-5
        if diff_total > 1e-5 or diff_s1 > 1e-5 or diff_s2 > 1e-5:
            print(f"    [!] CẢNH BÁO: Sai số vượt ngưỡng 1e-5!")
            passed = False
        else:
            print(f"    [✓] Đạt chuẩn đồng nhất Bit-Perfect (< 1e-5)!")

    if passed:
        print("\n" + "=" * 78)
        print("🏆 KẾT LUẬN: ĐỒNG NHẤT TOÁN HỌC STEP 0 ĐƯỢC CHỨNG MINH 100% HOÀN HẢO!")
        print("   Tại Step 0, mô hình nâng cấp tái hiện CHÍNH XÁC 100% logit của FatFormer gốc.")
        print("   -> Triệt tiêu hoàn toàn nguy cơ sụt giảm Clean ACC khi bổ sung SRM & Gating.")
        print("=" * 78)
        return True
    else:
        print("\n[✗] TEST 3 THẤT BẠI: Mô hình nâng cấp bị lệch logit so với bản gốc!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Step 0 Zero-Init Identity Proof for FatFormer-XLA")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Thiết bị chạy test")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn fatformer_4class_ckpt.pth")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[*] Thiết bị kiểm thử: {device}")

    # 1. Test SRM cô lập
    ok_srm = test_isolated_srm(device)

    # 2. Test Gating cô lập
    ok_gating = test_isolated_gating(device)

    # 3. Test Full Model
    ckpt_path = args.checkpoint or find_checkpoint_path()
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[!] Không tìm thấy file fatformer_4class_ckpt.pth. Bỏ qua TEST 3.")
        ok_full = False
    else:
        print(f"[*] Checkpoint gốc: {ckpt_path}")
        ok_full = test_full_model_identity(ckpt_path, device)

    print("\n" + "=" * 78)
    print("TỔNG HỢP KẾT QUẢ NGHIỆM THU TASK 6.2-R:")
    print(f"  1. SRM Zero-Init Isolation Test        : {'[✓] PASS' if ok_srm else '[✗] FAIL'}")
    print(f"  2. Dynamic Gating Neutrality Test     : {'[✓] PASS' if ok_gating else '[✗] FAIL'}")
    print(f"  3. Full Model Bit-Perfect Logit Identity: {'[✓] PASS' if ok_full else '[✗] FAIL'}")
    print("=" * 78 + "\n")

    if ok_srm and ok_gating and ok_full:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
