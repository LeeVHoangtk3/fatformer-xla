"""
==============================================================================
FATFORMER-XLA: TIỆN ÍCH ĐÓNG BĂNG THAM SỐ VÀ KIỂM TOÁN PEFT (TASK 3.4)
PARAMETER-EFFICIENT FINE-TUNING (PEFT) FREEZE UTILITIES
==============================================================================
Tác giả: Thành viên B (Architecture & Loss Lead)
Mục đích:
  - Đóng băng 94.1% Backbone CLIP ViT-L/14, bảo toàn tri thức nền tảng
    và giữ vững trần độ chính xác Baseline Clean ACC = 96.45%.
  - Chỉ mở khóa có chọn lọc ~5.9% tham số thích ứng (FAA, SRM projection,
    Dynamic Frequency Gating, LGA, và soft prompt vector ctx).
  - Kiểm tra bắt buộc các kernel vi sai SRM không bị rò rỉ gradient.
==============================================================================
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


def count_trainable_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Thống kê chi tiết số lượng tham số của mô hình:
    - Tổng số tham số (Total)
    - Tham số đóng băng (Frozen)
    - Tham số cần tối ưu (Trainable)
    - Tỷ lệ tham số huấn luyện (Trainable Ratio)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_ratio = trainable_params / max(1, total_params)

    return {
        "total_params": total_params,
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "trainable_percent": trainable_ratio * 100.0,
    }


def assert_srm_kernels_frozen(model: nn.Module):
    """
    Kiểm tra chốt chặn an toàn: Bắt buộc toàn bộ các ma trận lọc vi sai SRM
    (k1, k2, k3) phải có requires_grad = False (tiêu thụ 0 tham số tối ưu).
    """
    for name, param in model.named_parameters():
        if "srm" in name.lower() and ("k1" in name.lower() or "k2" in name.lower() or "k3" in name.lower()):
            assert not param.requires_grad, (
                f"VI PHẠM BỘ LỌC SRM: Tham số {name} có requires_grad=True! "
                f"Các bộ lọc vi sai SRM bắt buộc phải là hằng số cố định (requires_grad=False)."
            )

    # Kiểm tra buffers nếu được đăng ký qua register_buffer
    for name, buf in model.named_buffers():
        if "srm" in name.lower() and ("k1" in name.lower() or "k2" in name.lower() or "k3" in name.lower()):
            assert not buf.requires_grad, f"VI PHẠM BUFFER SRM: Buffer {name} có requires_grad=True!"


def freeze_clip_backbone(
    model: nn.Module,
    trainable_module_prefixes: Optional[list] = None,
    target_trainable_ratio_max: float = 0.08,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Thực thi thuật toán đóng băng 2 bước (Parameter-Efficient Fine-Tuning):
    Bước 1: Đóng băng toàn bộ mô hình (requires_grad = False).
    Bước 2: Mở khóa có chọn lọc theo danh sách trắng (whitelist) các module thích ứng.

    Kế thừa từ configs/train_config.yaml:
      trainable_modules:
        - "language_guided_alignment"
        - "text_guided_interactor"
        - "norm1", "norm2", "linear1", "linear2", "activation"
        - "srm" (chỉ tầng proj học được)
        - "gating" (Dynamic Gating MLP)
        - "ctx" (Soft prompt vector)
    """
    if trainable_module_prefixes is None:
        trainable_module_prefixes = [
            "language_guided_alignment",
            "text_guided_interactor",
            "norm1",
            "norm2",
            "linear1",
            "linear2",
            "activation",
            "srm.proj",
            "gating",
            "ctx",
            "forgery_aware_adapter",
        ]

    # Bước 1: Đóng băng TOÀN BỘ mô hình
    for param in model.parameters():
        param.requires_grad = False

    # Bước 2: Mở khóa CÓ CHỌN LỌC theo danh sách whitelist
    for name, param in model.named_parameters():
        # Kiểm tra nếu tên tham số chứa bất kỳ tiền tố whitelist nào
        is_trainable = any(prefix in name for prefix in trainable_module_prefixes)
        
        # Ngoại trừ tuyệt đối: các kernel vi sai SRM bắt buộc phải frozen
        if "srm" in name.lower() and any(k in name.lower() for k in ["k1", "k2", "k3"]):
            is_trainable = False

        if is_trainable:
            param.requires_grad = True

    # Bước 3: Kiểm tra an toàn bắt buộc
    assert_srm_kernels_frozen(model)

    stats = count_trainable_parameters(model)

    if verbose:
        print("=" * 70)
        print("KẾT QUẢ ĐÓNG BĂNG THAM SỐ CLIP BACKBONE (PEFT STANDARDIZED):")
        print(f"  • Tổng số tham số:        {stats['total_params']:,} ({stats['total_params']/1e6:.2f} M)")
        print(f"  • Tham số đóng băng:      {stats['frozen_params']:,} ({stats['frozen_params']/1e6:.2f} M - {100 - stats['trainable_percent']:.2f}%)")
        print(f"  • Tham số cần tối ưu:     {stats['trainable_params']:,} ({stats['trainable_params']/1e6:.2f} M - {stats['trainable_percent']:.2f}%)")
        print("=" * 70)

    # Xác nhận ngưỡng an toàn: tỷ lệ trainable không được vượt ngưỡng tối đa (mặc định < 8%)
    if stats["total_params"] > 0:
        assert stats["trainable_ratio"] <= target_trainable_ratio_max, (
            f"VI PHẠM NGUYÊN TẮC PEFT: Tỷ lệ tham số mở khóa ({stats['trainable_percent']:.2f}%) "
            f"vượt quá ngưỡng cho phép ({target_trainable_ratio_max * 100:.1f}%)!"
        )
        assert stats["frozen_params"] > 0, "LỖI AN TOÀN: Backbone chưa được đóng băng bất kỳ tham số nào!"

    return stats
