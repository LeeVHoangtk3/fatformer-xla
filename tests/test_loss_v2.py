"""
tests/test_loss_v2.py — Task 7.9 Definition of Done
Kiểm thử: RealGuardLoss, FeatureConsistencyLoss, CrossDomainTotalLoss tích hợp.
Không cần checkpoint hay GPU — chỉ dùng tensor giả.
"""
import pytest
import torch
import torch.nn as nn
from src.training.loss import (
    CrossDomainTotalLoss,
    RealGuardLoss,
    FeatureConsistencyLoss,
)


# ---------------------------------------------------------------------------
# Test (a): weight_feat=weight_guard=0 → legacy_mode → allclose với bản cũ
# ---------------------------------------------------------------------------
def test_a_backward_compat_allclose():
    """Khi cả hai weight mới = 0, loss_total phải khớp hoàn toàn công thức cũ."""
    torch.manual_seed(0)
    loss_new = CrossDomainTotalLoss(weight_ce=0.7, alpha_kd=0.3, beta_cons=0.2)
    loss_old = CrossDomainTotalLoss(weight_ce=0.7, alpha_kd=0.3, beta_cons=0.2)

    for seed in range(3):
        torch.manual_seed(seed)
        B = 8
        logits_c = torch.randn(B, 2)
        logits_d = torch.randn(B, 2)
        targets  = torch.randint(0, 2, (B,))
        teacher  = torch.randn(B, 2)

        out_new = loss_new(
            logits=logits_c, targets=targets,
            teacher_logits=teacher, logits_degraded=logits_d,
            weight_feat=0.0, weight_guard=0.0,
        )
        out_old = loss_old(
            logits=logits_c, targets=targets,
            teacher_logits=teacher, logits_degraded=logits_d,
        )
        assert torch.allclose(out_new["loss_total"], out_old["loss_total"], atol=1e-6), (
            f"seed={seed}: loss_total mới={out_new['loss_total'].item():.6f} "
            f"!= cũ={out_old['loss_total'].item():.6f}"
        )


# ---------------------------------------------------------------------------
# Test (b): Real có z_s < z_t → guard = 0, gradient z_s^fake = 0
# ---------------------------------------------------------------------------
def test_b_guard_zero_when_student_below_teacher():
    """Khi student còn ngoan hơn teacher (z_s < z_t), guard phải = 0."""
    guard_fn = RealGuardLoss(fake_index=1)
    B = 4
    # z_s^fake < z_t^fake → excess = relu(z_s - z_t) = 0
    z_s = torch.tensor([[0.1, -0.5],  # fake logit = -0.5
                         [0.2, -0.3],
                         [0.1, -0.4],
                         [0.0, -0.6]], requires_grad=True)
    z_t = torch.tensor([[0.1,  0.8],  # fake logit = 0.8 > -0.5 ✓
                         [0.2,  0.5],
                         [0.1,  0.7],
                         [0.0,  0.9]])
    is_real = torch.tensor([True, True, True, True])

    loss, info = guard_fn(z_s, z_t, is_real)
    assert loss.item() == pytest.approx(0.0, abs=1e-7), f"guard phải = 0 khi z_s < z_t, nhận {loss.item()}"
    loss.backward()
    assert z_s.grad is not None
    assert z_s.grad[:, 1].abs().max().item() == pytest.approx(0.0, abs=1e-7), (
        "gradient logit Fake phải = 0 khi student không vượt teacher"
    )


# ---------------------------------------------------------------------------
# Test (c): Real có z_s > z_t → guard > 0, ∂guard/∂z_s^fake > 0
# ---------------------------------------------------------------------------
def test_c_guard_positive_and_gradient_pulls_down():
    """Khi student vượt teacher (z_s > z_t), guard > 0 và gradient kéo logit Fake xuống."""
    guard_fn = RealGuardLoss(fake_index=1)
    B = 4
    # z_s^fake > z_t^fake → excess > 0
    z_s = torch.tensor([[0.1,  1.5],  # fake logit = 1.5
                         [0.2,  1.2],
                         [0.1,  1.8],
                         [0.0,  2.0]], requires_grad=True)
    z_t = torch.tensor([[0.1,  0.3],  # fake logit = 0.3 < 1.5 ✓
                         [0.2,  0.2],
                         [0.1,  0.5],
                         [0.0,  0.4]])
    is_real = torch.tensor([True, True, True, True])

    loss, info = guard_fn(z_s, z_t, is_real)
    assert loss.item() > 0, f"guard phải > 0 khi z_s > z_t, nhận {loss.item()}"
    assert info["guard_active"] is True
    loss.backward()
    assert z_s.grad is not None
    # gradient dương → optimizer sẽ kéo logit Fake xuống
    assert (z_s.grad[:, 1] > 0).all(), (
        "∂guard/∂z_s^fake phải > 0 để kéo logit xuống"
    )


# ---------------------------------------------------------------------------
# Test (d): Batch toàn Fake → guard = 0, không NaN, guard_active = False
# ---------------------------------------------------------------------------
def test_d_all_fake_no_nan():
    """Batch không có mẫu Real: guard = 0, không NaN, guard_active = False."""
    guard_fn = RealGuardLoss(fake_index=1)
    B = 8
    z_s = torch.randn(B, 2, requires_grad=True)
    z_t = torch.randn(B, 2)
    is_real = torch.zeros(B, dtype=torch.bool)  # toàn Fake

    loss, info = guard_fn(z_s, z_t, is_real)
    assert not torch.isnan(loss), "guard không được NaN khi batch toàn Fake"
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert info["guard_active"] is False
    assert info["n_real"] == 0

    # Tích hợp vào CrossDomainTotalLoss: cũng không được NaN
    loss_fn = CrossDomainTotalLoss()
    logits_deg = torch.randn(B, 2)
    teacher    = torch.randn(B, 2)
    targets    = torch.ones(B, dtype=torch.long)  # toàn Fake (label=1)
    out = loss_fn(
        logits=logits_deg, targets=targets,
        teacher_logits=teacher, logits_degraded=logits_deg,
        weight_guard=0.5,
    )
    assert not torch.isnan(out["loss_total"]), "loss_total không được NaN khi batch toàn Fake"
    assert out["guard_active"] is False
