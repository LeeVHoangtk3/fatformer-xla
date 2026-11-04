"""
tests/test_degradation_v2.py — Task 7.8
5 test kiểm tra phân phối và tính đúng đắn của DegradationPipelineV2.

Chạy: pytest -q tests/test_degradation_v2.py
"""
import random
import io
import pytest
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from src.datasets.transforms import DegradationPipelineV2


# ---------------------------------------------------------------------------
# Fixture: ảnh giả 224x224 RGB
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_img():
    """Ảnh màu ngẫu nhiên 224x224 — đủ để test pipeline."""
    rng = random.Random(0)
    pixels = [(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
              for _ in range(224 * 224)]
    img = Image.new("RGB", (224, 224))
    img.putdata(pixels)
    return img


@pytest.fixture
def pipeline():
    """Pipeline v2 với config mặc định."""
    return DegradationPipelineV2()


# ---------------------------------------------------------------------------
# T1 — Clean branch bất biến
# ---------------------------------------------------------------------------

def test_T1_clean_branch_unchanged(sample_img):
    """
    Nhánh clean phải cho kết quả giống hệt pipeline chuẩn
    Resize(224) → CenterCrop(224) → ToTensor → Normalize.
    """
    _mean = [0.485, 0.456, 0.406]
    _std  = [0.229, 0.224, 0.225]
    ref = T.Compose([
        T.Resize((224, 224)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=_mean, std=_std),
    ])(sample_img)

    pipe = DegradationPipelineV2(config={"clean_prob": 1.0})  # luôn clean
    x_clean, x_degraded = pipe(sample_img)

    assert x_clean.shape == (3, 224, 224)
    assert x_degraded.shape == (3, 224, 224)
    assert torch.allclose(x_clean, ref, atol=1e-5), "Clean branch sai so với pipeline chuẩn"
    assert torch.allclose(x_clean, x_degraded, atol=1e-5), \
        "Khi clean_prob=1.0, x_degraded phải == x_clean"


# ---------------------------------------------------------------------------
# T2 — Tỷ lệ clean ≈ 0.60 ± 0.02 (10.000 mẫu)
# ---------------------------------------------------------------------------

def test_T2_clean_ratio(sample_img):
    """10.000 mẫu: tỷ lệ clean ≈ 0.60 ± 0.02."""
    random.seed(42)
    pipe = DegradationPipelineV2()
    n = 10_000
    n_clean = sum(
        1 for _ in range(n)
        if (lambda r: torch.allclose(r[0], r[1], atol=1e-5))(pipe(sample_img))
    )
    ratio = n_clean / n
    assert 0.58 <= ratio <= 0.62, f"Tỷ lệ clean thực tế = {ratio:.3f}, kỳ vọng 0.60 ± 0.02"


# ---------------------------------------------------------------------------
# T3 — Nhóm nặng JPEG [30,45] ≥ 35% ± 3pp (10.000 mẫu)
# ---------------------------------------------------------------------------

def test_T3_jpeg_heavy_share(sample_img):
    """
    Trong các mẫu bị JPEG, q ∈ [30,45] chiếm ≥ 35% ± 3 pp.
    Dùng config clean_prob=0, comp_prob=0, chỉ JPEG để đo thuần.
    """
    random.seed(42)
    # Buộc chỉ dùng JPEG đơn, tắt clean và composite
    cfg = {
        "clean_prob": 0.0,
        "jpeg":        {"prob": 1.0, "heavy_share": 0.35,
                        "heavy_range": [30, 45], "light_range": [45, 85]},
        "webp":        {"prob": 0.0, "q_range": [30, 70]},
        "double_jpeg": {"prob": 0.0, "q1_range": [70, 90], "q2_range": [30, 50]},
        "blur":        {"prob": 0.0, "sigma_range": [0.5, 2.0]},
        "down_up":     {"prob": 0.0, "ratio_range": [0.50, 0.75]},
        "composite":   {"prob": 0.0, "pairs": []},
    }
    pipe = DegradationPipelineV2(config=cfg)
    n = 10_000
    n_heavy = 0
    for _ in range(n):
        pipe(sample_img)
        tag = pipe.last_tag[0] if pipe.last_tag else ""
        if tag.startswith("jpeg:"):
            q = int(tag.split(":")[1])
            if 30 <= q <= 45:
                n_heavy += 1
    ratio = n_heavy / n
    assert 0.32 <= ratio <= 0.38, \
        f"Heavy JPEG share = {ratio:.3f}, kỳ vọng 0.35 ± 0.03"


# ---------------------------------------------------------------------------
# T4 — Composite ≈ 0.25 ± 0.02 trong nhóm bị suy biến (10.000 mẫu)
# ---------------------------------------------------------------------------

def test_T4_composite_ratio(sample_img):
    """
    Trong nhóm bị suy biến (không clean), composite (>= 2 tag) ≈ 25% ± 2pp.
    """
    random.seed(42)
    # Tắt clean hoàn toàn để chỉ đo tỷ lệ composite
    cfg = {"clean_prob": 0.0}
    pipe = DegradationPipelineV2(config=cfg)
    n = 10_000
    n_composite = 0
    for _ in range(n):
        pipe(sample_img)
        if len(pipe.last_tag) >= 2:
            n_composite += 1
    ratio = n_composite / n
    assert 0.23 <= ratio <= 0.27, \
        f"Composite ratio = {ratio:.3f}, kỳ vọng 0.25 ± 0.02"


# ---------------------------------------------------------------------------
# T5 — Tất định: cùng seed → cùng tag + cùng tensor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degradation", DegradationPipelineV2._SINGLE_TYPES)
def test_T5_deterministic(sample_img, degradation):
    """Cùng seed → cùng last_tag và cùng tensor (allclose)."""
    cfg = {"clean_prob": 0.0, "composite": {"prob": 0.0}}
    cfg.update({name: {"prob": float(name == degradation)}
                for name in DegradationPipelineV2._SINGLE_TYPES})
    pipe = DegradationPipelineV2(cfg)
    seed = 123

    c1, d1 = pipe(sample_img, seed=seed)
    tag1 = list(pipe.last_tag)

    c2, d2 = pipe(sample_img, seed=seed)
    tag2 = list(pipe.last_tag)

    assert tag1 == tag2, f"Tag khác nhau: {tag1} vs {tag2}"
    assert len(tag1) == 1 and tag1[0].split(":")[0] == degradation
    assert torch.allclose(c1, c2, atol=1e-6), "x_clean không tất định"
    assert torch.equal(c1, pipe.clean_transform(sample_img)), "Nhánh clean bị thay đổi khi suy biến"
    assert torch.allclose(d1, d2, atol=1e-6), "x_degraded không tất định"
