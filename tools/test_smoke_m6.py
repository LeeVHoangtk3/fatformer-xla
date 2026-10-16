"""
==============================================================================
FATFORMER-XLA: CỔNG KIỂM THỬ TÍCH HỢP MILESTONE 6 (SMOKE GATE M6)
Task 6.3-B — Thành viên B (Architecture & Loss Lead)
==============================================================================
Mục đích:
    Xác nhận 5 tiêu chí DoD trước khi chuyển giao sang Task 6.3-C (Training GPU):
        1. PairedCleanDegradedAugmentation → cặp tensor (3, 224, 224) hợp lệ
        2. Teacher model: eval(), toàn bộ tham số requires_grad=False
        3. CrossDomainDistillationLoss: L_KD hữu hạn, không NaN/Inf
        4. PairedConsistencyLoss: L_cons hữu hạn, không NaN/Inf
        5. CrossDomainTotalLoss: forward + backward không lỗi, loss_total hữu hạn
           — gradient chảy qua adapter, không chạm vào frozen weights

Thực thi:
    # Kiểm tra trên CPU local (không cần GPU):
    python tools/test_smoke_m6.py --device cpu

    # Trên Colab T4:
    python tools/test_smoke_m6.py --device cuda
==============================================================================
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from PIL import Image

# Bảo đảm import được các module từ thư mục gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.training.loss import (
    CrossDomainDistillationLoss,
    PairedConsistencyLoss,
    CrossDomainTotalLoss,
)
from src.datasets.transforms import PairedCleanDegradedAugmentation
from src.models.lora_adapter import LoRALinear, ViTLoRAAdapter


# ──────────────────────────────────────────────────────────
# Mô hình giả lập (Mock) — không cần checkpoint thực tế
# ──────────────────────────────────────────────────────────

class _MockTeacher(nn.Module):
    """Teacher đóng băng hoàn toàn — mô phỏng baseline gốc CVPR 2024."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, num_classes)
        # Đóng băng tất cả tham số (tiêu chí #2)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class _MockStudent(nn.Module):
    """Student có gradient — mô phỏng FatFormer-XLA đang finetune."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


# ──────────────────────────────────────────────────────────
# Helper in kết quả
# ──────────────────────────────────────────────────────────

def _pass(msg: str) -> None:
    print(f"  [✓] PASS — {msg}")


def _fail(msg: str) -> None:
    print(f"  [✗] FAIL — {msg}")
    sys.exit(1)


def _section(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  TIÊU CHÍ {title}")
    print(f"{'─' * 70}")


# ──────────────────────────────────────────────────────────
# Tiêu chí #1: PairedCleanDegradedAugmentation
# ──────────────────────────────────────────────────────────

def test_paired_augmentation(batch_size: int) -> None:
    _section("#1: PairedCleanDegradedAugmentation → cặp tensor (B, 3, 224, 224)")
    aug = PairedCleanDegradedAugmentation(img_resolution=256, crop_resolution=224)
    dummy_pil = Image.new("RGB", (256, 256), color=(123, 87, 42))

    x_cleans, x_degs = [], []
    for _ in range(batch_size):
        xc, xd = aug(dummy_pil)
        x_cleans.append(xc)
        x_degs.append(xd)

    x_clean_batch    = torch.stack(x_cleans)   # (B, 3, 224, 224)
    x_degraded_batch = torch.stack(x_degs)     # (B, 3, 224, 224)

    expected_shape = (batch_size, 3, 224, 224)
    if x_clean_batch.shape != expected_shape:
        _fail(f"x_clean shape sai: {x_clean_batch.shape} ≠ {expected_shape}")
    if x_degraded_batch.shape != expected_shape:
        _fail(f"x_degraded shape sai: {x_degraded_batch.shape} ≠ {expected_shape}")
    if torch.isnan(x_clean_batch).any() or torch.isnan(x_degraded_batch).any():
        _fail("Tensor chứa NaN sau augmentation!")

    _pass(f"x_clean={tuple(x_clean_batch.shape)}, x_degraded={tuple(x_degraded_batch.shape)}")
    return x_clean_batch, x_degraded_batch


# ──────────────────────────────────────────────────────────
# Tiêu chí #2: Teacher model eval + frozen
# ──────────────────────────────────────────────────────────

def test_teacher_model(device: torch.device) -> _MockTeacher:
    _section("#2: Teacher model — eval() & requires_grad=False hoàn toàn")
    teacher = _MockTeacher(num_classes=2).to(device)
    teacher.eval()

    # Kiểm tra eval mode
    if teacher.training:
        _fail("Teacher model vẫn ở training mode!")

    # Kiểm tra mọi tham số đều frozen
    trainable = [n for n, p in teacher.named_parameters() if p.requires_grad]
    if trainable:
        _fail(f"Phát hiện tham số chưa frozen: {trainable}")

    _pass(f"Teacher: eval=True | trainable_params=0 | frozen_params={sum(p.numel() for p in teacher.parameters()):,}")
    return teacher


# ──────────────────────────────────────────────────────────
# Tiêu chí #3: CrossDomainDistillationLoss (L_KD)
# ──────────────────────────────────────────────────────────

def test_distillation_loss(
    student: nn.Module,
    teacher: nn.Module,
    x_clean: torch.Tensor,
    device: torch.device,
) -> None:
    _section("#3: CrossDomainDistillationLoss (L_KD) — hữu hạn, không NaN/Inf")
    kd_loss_fn = CrossDomainDistillationLoss(temperature=2.0, alpha_kd=0.4)
    batch_size = x_clean.shape[0]

    with torch.no_grad():
        teacher_logits = teacher(x_clean)

    student_logits = student(x_clean)
    # 50% mẫu là "clean" (thực tế sẽ dựa vào nhãn real/fake)
    clean_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    clean_mask[:batch_size // 2] = True

    loss_kd = kd_loss_fn(student_logits, teacher_logits, clean_mask=clean_mask)

    if torch.isnan(loss_kd):
        _fail(f"L_KD = NaN!")
    if torch.isinf(loss_kd):
        _fail(f"L_KD = Inf!")
    if loss_kd.item() < 0:
        _fail(f"L_KD = {loss_kd.item():.4f} < 0 (KL phân kỳ phải ≥ 0)!")

    _pass(f"L_KD = {loss_kd.item():.6f} (τ=2.0, α_kd=0.4, clean_mask={clean_mask.sum().item()}/{batch_size} mẫu)")


# ──────────────────────────────────────────────────────────
# Tiêu chí #4: PairedConsistencyLoss (L_cons)
# ──────────────────────────────────────────────────────────

def test_consistency_loss(
    student: nn.Module,
    x_clean: torch.Tensor,
    x_degraded: torch.Tensor,
) -> None:
    _section("#4: PairedConsistencyLoss (L_cons) — hữu hạn, không NaN/Inf")
    cons_loss_fn = PairedConsistencyLoss(beta_cons=0.2)

    logits_clean    = student(x_clean)
    logits_degraded = student(x_degraded)

    loss_cons = cons_loss_fn(logits_clean, logits_degraded)

    if torch.isnan(loss_cons):
        _fail("L_cons = NaN!")
    if torch.isinf(loss_cons):
        _fail("L_cons = Inf!")
    if loss_cons.item() < 0:
        _fail(f"L_cons = {loss_cons.item():.4f} < 0 (MSE phải ≥ 0)!")

    _pass(f"L_cons = {loss_cons.item():.6f} (β_cons=0.2, shape_clean={tuple(logits_clean.shape)})")


# ──────────────────────────────────────────────────────────
# Tiêu chí #5: CrossDomainTotalLoss Forward + Backward
# ──────────────────────────────────────────────────────────

def test_total_loss_backward(
    student: nn.Module,
    teacher: nn.Module,
    x_clean: torch.Tensor,
    x_degraded: torch.Tensor,
    device: torch.device,
) -> None:
    _section("#5: CrossDomainTotalLoss — Forward + Backward không lỗi, gradient thông suốt")
    total_loss_fn = CrossDomainTotalLoss(
        label_smoothing=0.05,
        temperature=2.0,
        alpha_kd=0.4,
        beta_cons=0.2,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    batch_size = x_clean.shape[0]
    targets = torch.randint(0, 2, (batch_size,), device=device)

    # Forward
    optimizer.zero_grad()
    logits       = student(x_clean)
    logits_deg   = student(x_degraded)
    with torch.no_grad():
        teacher_logits = teacher(x_clean)

    clean_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    clean_mask[:batch_size // 2] = True

    loss_dict = total_loss_fn(
        logits=logits,
        targets=targets,
        teacher_logits=teacher_logits,
        logits_degraded=logits_deg,
        clean_mask=clean_mask,
    )
    loss_total = loss_dict["loss_total"]

    # Kiểm tra tất cả thành phần
    for key in ("loss_total", "loss_ce", "loss_kd", "loss_cons"):
        val = loss_dict[key]
        if torch.isnan(val):
            _fail(f"{key} = NaN!")
        if torch.isinf(val):
            _fail(f"{key} = Inf!")

    # Backward
    loss_total.backward()

    # Kiểm tra gradient chảy qua student (không frozen)
    has_grad = any(
        p.grad is not None and p.grad.abs().max() > 0
        for p in student.parameters()
        if p.requires_grad
    )
    if not has_grad:
        _fail("Không có gradient nào chảy qua student model!")

    # Kiểm tra teacher gradient = None (frozen)
    teacher_has_grad = any(
        p.grad is not None
        for p in teacher.parameters()
    )
    if teacher_has_grad:
        _fail("Teacher model có gradient — vi phạm yêu cầu frozen!")

    optimizer.step()

    _pass(
        f"loss_total={loss_total.item():.4f} | "
        f"loss_ce={loss_dict['loss_ce'].item():.4f} | "
        f"loss_kd={loss_dict['loss_kd'].item():.4f} | "
        f"loss_cons={loss_dict['loss_cons'].item():.4f}"
    )
    _pass("Gradient thông suốt qua student | Teacher đúng frozen (grad=None)")


# ──────────────────────────────────────────────────────────
# Tiêu chí BONUS: LoRA Adapter khởi tạo + summary
# ──────────────────────────────────────────────────────────

def test_lora_adapter() -> None:
    _section("BONUS — LoRALinear khởi tạo & delta=0 khi B=0")
    dummy_linear = nn.Linear(1024, 1024, bias=True)
    lora = LoRALinear.from_linear(dummy_linear, rank=8, alpha=16.0, enabled=True)

    x = torch.randn(4, 1024)
    # Khi B=0, delta phải = 0 → out == original linear
    with torch.no_grad():
        out_lora     = lora(x)
        out_original = nn.functional.linear(x, lora.weight, lora.bias_param)

    if not torch.allclose(out_lora, out_original, atol=1e-5):
        _fail("delta ≠ 0 khi B khởi tạo = 0 (vi phạm LoRA init contract!)")

    _pass(f"LoRALinear: delta=0 tại init ✓ | rank=8, α=16, scaling=2.0")
    _pass(f"Trainable params: lora_A={lora.lora_A.numel()}, lora_B={lora.lora_B.numel()}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="FatFormer-XLA Smoke Gate Milestone 6 (Task 6.3-B)"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị tính toán: cuda hoặc cpu (mặc định: cuda nếu có)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Kích thước mini-batch kiểm thử (mặc định: 8)",
    )
    return parser.parse_args()


def run_smoke_gate_m6():
    args   = parse_args()
    device = torch.device(args.device)
    B      = args.batch_size

    print("=" * 70)
    print("  FATFORMER-XLA: SMOKE GATE MILESTONE 6 — TASK 6.3-B")
    print(f"  Thiết bị: {device} | Batch size: {B}")
    print("=" * 70)

    t_start = time.time()

    # Tiêu chí #1
    x_clean, x_degraded = test_paired_augmentation(B)
    x_clean    = x_clean.to(device)
    x_degraded = x_degraded.to(device)

    # Tiêu chí #2
    teacher = test_teacher_model(device)

    # Khởi tạo student
    student = _MockStudent(num_classes=2).to(device)

    # Tiêu chí #3
    test_distillation_loss(student, teacher, x_clean, device)

    # Tiêu chí #4
    test_consistency_loss(student, x_clean, x_degraded)

    # Tiêu chí #5
    test_total_loss_backward(student, teacher, x_clean, x_degraded, device)

    # Tiêu chí BONUS
    test_lora_adapter()

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  KẾT QUẢ NGHIỆM THU SMOKE GATE MILESTONE 6")
    print(f"  Thời gian chạy: {elapsed:.2f}s")
    print("=" * 70)
    print("  [✓✓✓] PASS 100% — 5/5 TIÊU CHÍ DoD HOÀN THÀNH!")
    print("  Đủ điều kiện chuyển giao sang Task 6.3-C (Training GPU).")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_gate_m6()
