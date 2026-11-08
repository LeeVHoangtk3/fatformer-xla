"""Unit tests toàn diện cho Task 7.15 (Unified Packaging & Inference CLI).

Bao quát các khía cạnh:
1. Đóng gói bảo toàn 100% trọng số và nhúng metadata.
2. Từ chối đóng gói layout không đầy đủ (partial=True).
3. Bắt buộc có calibration hợp lệ và từ chối calibration sai SHA-256.
4. CLI từ chối nạp checkpoint thiếu calibration (không tự tạo số giả).
5. CLI từ chối calibration truyền từ ngoài nếu không khớp SHA-256.
6. Luồng nạp package và gọi predict() trên tensor/ảnh mẫu (End-to-End).
7. Đo độ trễ trên CPU không tự gán overhead_meets_target=True.
8. Trích xuất logits dummy tạo NPZ đầy đủ trường metadata.
"""
import json
import os
import tempfile
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

import src.models.canonical_loader as canonical_loader
from tools.package_unified_fatformer import (
    package_checkpoint,
    compute_file_sha256,
    validate_calibration_against_checkpoint
)
from tools.infer_unified import (
    load_unified_model,
    collect_image_paths,
    get_default_transform,
    benchmark_latency
)
from tools.extract_calibration_logits import extract_logits


# ==============================================================================
# MOCK MODEL DÙNG CHO KIỂM THỬ KHÔNG CẦN TẢI WEIGHTS CLIP 2GB
# ==============================================================================

class _DummyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, 32))


class _DummyBlock(nn.Module):
    def __init__(self, has_adapter=False):
        super().__init__()
        if has_adapter:
            self.forgery_aware_adapter = _DummyAdapter()


class _DummyToyBackbone(nn.Module):
    """Backbone thu nhỏ mô phỏng cấu trúc 24 blocks của ViT-L/14."""
    def __init__(self, adapter_indices=(7, 15, 23)):
        super().__init__()
        self.adapter_block_indices = tuple(adapter_indices)
        self.clip_model = nn.Module()
        self.clip_model.visual = nn.Module()
        self.clip_model.visual.transformer = nn.Module()
        self.clip_model.visual.transformer.resblocks = nn.ModuleList([
            _DummyBlock(has_adapter=(i in set(adapter_indices))) for i in range(24)
        ])
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        # x: [B, 3, 224, 224] -> trả về logits [B, 2]
        b = x.shape[0]
        return torch.randn(b, 2)


def _install_dummy_builder(monkeypatch):
    def fake_build_model(args):
        indices = getattr(args, "adapter_block_indices", [7, 15, 23])
        return _DummyToyBackbone(indices)

    monkeypatch.setattr(canonical_loader, "build_model", fake_build_model, raising=False)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def dummy_checkpoint_and_calib():
    """Tạo checkpoint fixture dummy và calibration khớp SHA-256."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "dummy_model.pth")
        calib_path = os.path.join(tmpdir, "dummy_calib.json")
        out_pkg_path = os.path.join(tmpdir, "dummy_packaged.pth")
        meta_out_path = os.path.join(tmpdir, "dummy_metadata.json")

        # Tạo dummy state_dict khớp layout n=3 {7, 15, 23}
        toy = _DummyToyBackbone([7, 15, 23])
        sd = toy.state_dict()
        torch.save(sd, ckpt_path)
        actual_sha = compute_file_sha256(ckpt_path)

        calib_data = {
            "schema": "calibration/v1",
            "evidence_level": "measured",
            "checkpoint": {"sha256": actual_sha},
            "T": 1.5000,
            "T_star": 0.8500
        }
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(calib_data, f)

        yield {
            "tmpdir": tmpdir,
            "ckpt_path": ckpt_path,
            "calib_path": calib_path,
            "out_pkg_path": out_pkg_path,
            "meta_out_path": meta_out_path,
            "actual_sha": actual_sha,
            "state_dict": sd
        }


# ==============================================================================
# TESTS ĐÓNG GÓI (PACKAGING TESTS)
# ==============================================================================

def test_packaging_matches_weights_and_metadata(dummy_checkpoint_and_calib):
    """Kiểm tra đóng gói bảo toàn 100% trọng số và nhúng metadata chính xác."""
    info = dummy_checkpoint_and_calib
    meta = package_checkpoint(
        checkpoint_path=info["ckpt_path"],
        output_path=info["out_pkg_path"],
        calibration_path=info["calib_path"],
        metadata_json_path=info["meta_out_path"]
    )

    assert meta["source_checkpoint_sha256"] == info["actual_sha"]
    assert meta["layout"]["n"] == 3
    assert meta["layout"]["indices"] == [7, 15, 23]
    assert meta["layout"]["partial"] is False
    assert meta["calibration"]["T"] == 1.5000
    assert os.path.exists(info["out_pkg_path"])
    assert os.path.exists(info["meta_out_path"])

    # Nạp lại kiểm tra tensor
    reloaded = torch.load(info["out_pkg_path"], map_location="cpu")
    assert "model" in reloaded
    assert "metadata" in reloaded
    for k, v in info["state_dict"].items():
        assert torch.equal(reloaded["model"][k], v)


def test_packaging_rejects_partial_layout(dummy_checkpoint_and_calib):
    """Kiểm tra từ chối đóng gói layout không đầy đủ (partial=True)."""
    info = dummy_checkpoint_and_calib
    partial_ckpt = os.path.join(info["tmpdir"], "partial_model.pth")
    # Chỉ có 1 block thay vì đủ 3
    toy_partial = _DummyToyBackbone([7])
    torch.save(toy_partial.state_dict(), partial_ckpt)

    with pytest.raises(ValueError, match="partial=True"):
        package_checkpoint(
            checkpoint_path=partial_ckpt,
            output_path=os.path.join(info["tmpdir"], "partial_out.pth"),
            calibration_path=info["calib_path"],
            allow_partial=False
        )


def test_packaging_requires_calibration(dummy_checkpoint_and_calib):
    """Kiểm tra bắt buộc phải có calibration hợp lệ khi đóng gói."""
    info = dummy_checkpoint_and_calib
    with pytest.raises(ValueError, match="bắt buộc phải truyền --calibration"):
        package_checkpoint(
            checkpoint_path=info["ckpt_path"],
            output_path=info["out_pkg_path"],
            calibration_path=None,
            allow_uncalibrated_debug=False
        )


def test_packaging_rejects_mismatched_calibration_sha256(dummy_checkpoint_and_calib):
    """Kiểm tra từ chối calibration có SHA-256 không khớp."""
    info = dummy_checkpoint_and_calib
    wrong_calib = {
        "schema": "calibration/v1",
        "evidence_level": "measured",
        "checkpoint": {"sha256": "wrong_hash_000000000000000000000000000000000000000000000000000000000"},
        "T": 1.6402,
        "T_star": 0.9127
    }
    with pytest.raises(ValueError, match="Calibration thuộc checkpoint khác"):
        validate_calibration_against_checkpoint(wrong_calib, info["actual_sha"])


# ==============================================================================
# TESTS CLI SUY LUẬN (INFERENCE CLI TESTS)
# ==============================================================================

def test_cli_rejects_uncalibrated_checkpoint(monkeypatch, dummy_checkpoint_and_calib):
    """CLI từ chối nạp checkpoint thiếu calibration (không tự tạo số giả)."""
    _install_dummy_builder(monkeypatch)
    info = dummy_checkpoint_and_calib
    # Đóng gói debug không kèm calibration
    uncalib_pkg = os.path.join(info["tmpdir"], "uncalib_pkg.pth")
    package_checkpoint(
        checkpoint_path=info["ckpt_path"],
        output_path=uncalib_pkg,
        calibration_path=None,
        allow_uncalibrated_debug=True
    )

    with pytest.raises(ValueError, match="không chứa calibration trong metadata"):
        load_unified_model(uncalib_pkg, device="cpu")


def test_cli_rejects_mismatched_external_calibration(monkeypatch, dummy_checkpoint_and_calib):
    """CLI từ chối tệp calibration truyền từ ngoài nếu không khớp SHA-256."""
    _install_dummy_builder(monkeypatch)
    info = dummy_checkpoint_and_calib
    uncalib_pkg = os.path.join(info["tmpdir"], "uncalib_pkg.pth")
    package_checkpoint(
        checkpoint_path=info["ckpt_path"],
        output_path=uncalib_pkg,
        calibration_path=None,
        allow_uncalibrated_debug=True
    )

    fake_calib_path = os.path.join(info["tmpdir"], "fake_calib.json")
    with open(fake_calib_path, "w") as f:
        json.dump({"schema": "calibration/v1", "checkpoint": {"sha256": "wrong_sha_xyz"}}, f)

    with pytest.raises(ValueError, match="Tệp calibration thuộc checkpoint khác"):
        load_unified_model(uncalib_pkg, calibration_path=fake_calib_path, device="cpu")


def test_load_and_predict_packaged_model_end_to_end(monkeypatch, dummy_checkpoint_and_calib):
    """Kiểm tra toàn bộ luồng nạp package và gọi predict() trên tensor."""
    _install_dummy_builder(monkeypatch)
    info = dummy_checkpoint_and_calib

    package_checkpoint(
        checkpoint_path=info["ckpt_path"],
        output_path=info["out_pkg_path"],
        calibration_path=info["calib_path"]
    )

    model, meta = load_unified_model(info["out_pkg_path"], device="cpu")
    assert meta["layout"]["n"] == 3

    # Tạo batch giả lập [2, 3, 224, 224]
    x = torch.randn(2, 3, 224, 224)
    outputs = model.predict(x)

    assert "prob_fake" in outputs
    assert "label" in outputs
    assert "w_deg" in outputs
    assert "threshold" in outputs
    assert outputs["prob_fake"].shape == (2,)
    assert outputs["label"].shape == (2,)


def test_benchmark_latency_cpu_behavior(monkeypatch, dummy_checkpoint_and_calib):
    """Kiểm tra benchmark_latency trên CPU không tự gán overhead_meets_target=True."""
    _install_dummy_builder(monkeypatch)
    info = dummy_checkpoint_and_calib

    package_checkpoint(
        checkpoint_path=info["ckpt_path"],
        output_path=info["out_pkg_path"],
        calibration_path=info["calib_path"]
    )

    model, _ = load_unified_model(info["out_pkg_path"], device="cpu")
    res = benchmark_latency(model, device=torch.device("cpu"), batch_sizes=[1], n_warmup=1, n_steps=2)

    assert res["overhead_meets_target"] is None
    assert "notes" in res


def test_extract_calibration_logits_dummy(tmp_path):
    """Kiểm tra extract_calibration_logits tạo file NPZ đầy đủ trường."""
    out_npz = str(tmp_path / "test_calib.npz")
    extract_logits(
        checkpoint_path="dummy_ckpt.pth",
        data_dir=str(tmp_path),
        output_npz=out_npz,
        test_dummy=True
    )

    assert os.path.exists(out_npz)
    data = np.load(out_npz)
    assert "logits" in data
    assert "targets" in data
    assert "sample_ids" in data
    assert "checkpoint_sha256" in data
    assert data["logits"].shape == (400, 2)
    assert len(data["targets"]) == 400
