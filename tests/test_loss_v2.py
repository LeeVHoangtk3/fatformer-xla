"""
tests/test_loss_v2.py — Task 7.9 Definition of Done
Kiểm thử: RealGuardLoss, FeatureConsistencyLoss, CrossDomainTotalLoss tích hợp.
Không cần checkpoint hay GPU — chỉ dùng tensor giả.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # Independent frozen formula, not a second call to the implementation.
        expected = (0.7 * (0.7 * F.cross_entropy(logits_c, targets, label_smoothing=.05)
                           + 0.3 * F.cross_entropy(logits_d, targets, label_smoothing=.05))
                    + .3 * reference_kd(logits_c, teacher, targets)
                    + .2 * .5 * F.mse_loss(logits_d, logits_c.detach()))
        assert torch.allclose(out_new["loss_total"], expected, atol=1e-6), (
            f"seed={seed}: loss_total mới={out_new['loss_total'].item():.6f} "
            f"!= cũ={expected.item():.6f}"
        )


def reference_kd(student, teacher, targets, temperature=2.0):
    per_sample = F.kl_div(F.log_softmax(student / temperature, dim=-1),
                          F.softmax(teacher.detach() / temperature, dim=-1),
                          reduction="none").sum(-1)
    correct = teacher.argmax(-1) == targets
    real, fake = correct & (targets == 0), correct & (targets == 1)
    if real.any() and fake.any():
        raw = .5 * (per_sample[real].mean() + per_sample[fake].mean())
    elif real.any():
        raw = .25 * per_sample[real].mean()
    elif fake.any():
        raw = per_sample[fake].mean()
    else:
        raw = student.sum() * 0
    return raw * temperature ** 2


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


@pytest.mark.parametrize("weights", [(.3, .3, .7, .2, .3, .5), (.6, .8, .4, .6, .2, .1)])
def test_v2_formula_applies_configured_weights_once(weights):
    ce, ce_deg, kd, cons, feat, guard = weights
    torch.manual_seed(7)
    x, d, t = (torch.randn(6, 2, requires_grad=True) for _ in range(3))
    fc, fd = (torch.randn(6, 4, requires_grad=True) for _ in range(2))
    targets = torch.tensor([0, 1, 0, 1, 0, 1])
    criterion = CrossDomainTotalLoss(weight_ce=ce, weight_ce_deg=ce_deg, alpha_kd=kd, beta_cons=cons)
    out = criterion(x, targets, teacher_logits=t, logits_degraded=d,
                    features_clean=fc, features_deg=fd, weight_feat=feat, weight_guard=guard)
    feat_raw = ((1 - F.cosine_similarity(fd, fc.detach(), dim=-1, eps=1e-8)).mean()
                + F.kl_div(F.log_softmax(d / 2, dim=-1), F.softmax(x.detach() / 2, dim=-1),
                           reduction="batchmean") * 4)
    guard_raw = F.relu(d[targets == 0, 1] - t.detach()[targets == 0, 1]).square().mean()
    expected = (ce * F.cross_entropy(x, targets, label_smoothing=.05)
                + ce_deg * F.cross_entropy(d, targets, label_smoothing=.05)
                + kd * reference_kd(x, t, targets) + cons * .5 * F.mse_loss(d, x.detach())
                + feat * feat_raw + guard * guard_raw)
    assert torch.allclose(out["loss_total"], expected, atol=1e-6)
    assert {"loss_total", "ce_clean", "ce_deg", "kd", "cons", "feat", "guard"} <= out.keys()
    out["loss_total"].backward()
    assert t.grad is None


def test_feature_loss_detaches_clean_and_scales_by_batchmean():
    torch.manual_seed(5)
    fd, fc = (torch.randn(3, 4, requires_grad=True) for _ in range(2))
    ld, lc = (torch.randn(3, 2, requires_grad=True) for _ in range(2))
    criterion = FeatureConsistencyLoss()
    loss, _ = criterion(fd, fc, ld, lc)
    duplicated, _ = criterion(fd.repeat(2, 1), fc.repeat(2, 1), ld.repeat(2, 1), lc.repeat(2, 1))
    assert torch.allclose(loss, duplicated, atol=1e-6)
    loss.backward()
    assert fc.grad is None and lc.grad is None
    assert torch.isfinite(fd.grad).all() and torch.isfinite(ld.grad).all()


def test_fp16_feature_zero_norm_and_guard_square_are_finite():
    fd = torch.zeros(2, 4, dtype=torch.float16, requires_grad=True)
    fc = torch.zeros_like(fd, requires_grad=True)
    ld = torch.zeros(2, 2, dtype=torch.float16, requires_grad=True)
    lc = torch.zeros_like(ld, requires_grad=True)
    feature, _ = FeatureConsistencyLoss()(fd, fc, ld, lc)
    feature.backward()
    assert torch.isfinite(feature) and torch.isfinite(fd.grad).all()
    assert fc.grad is None and lc.grad is None
    s = torch.tensor([[0., 400.]], dtype=torch.float16, requires_grad=True)
    t = torch.zeros_like(s, requires_grad=True)
    guard, _ = RealGuardLoss()(s, t, torch.tensor([True]))
    guard.backward()
    assert guard.item() == pytest.approx(160000.)
    assert torch.isfinite(s.grad).all() and t.grad is None


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_feature_temperature_validation(temperature):
    with pytest.raises(ValueError, match="temperature"):
        FeatureConsistencyLoss(temperature)


@pytest.mark.parametrize("kwargs, message", [({"weight_feat": .3}, "features"),
                                           ({"weight_guard": .5}, "teacher"),
                                           ({"weight_feat": -.3}, "weight_feat")])
def test_v2_rejects_missing_inputs_and_invalid_weights(kwargs, message):
    x = torch.randn(2, 2)
    with pytest.raises(ValueError, match=message):
        CrossDomainTotalLoss()(x, torch.tensor([0, 1]), logits_degraded=x, **kwargs)


def test_guard_rejects_invalid_class_index_even_on_all_fake_batch():
    with pytest.raises(ValueError, match="fake_index"):
        RealGuardLoss(fake_index=2)(torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, dtype=torch.bool))


def test_phase7_keyword_aliases_match_legacy_call():
    x, d = torch.randn(4, 2), torch.randn(4, 2)
    targets = torch.tensor([0, 1, 0, 1])
    criterion = CrossDomainTotalLoss()
    old = criterion(logits=x, logits_degraded=d, targets=targets)
    new = criterion(logits_clean=x, logits_deg=d, targets=targets)
    assert torch.equal(old["loss_total"], new["loss_total"])
    with pytest.raises(ValueError, match="not both"):
        criterion(logits=x, logits_clean=x, targets=targets)
