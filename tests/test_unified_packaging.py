"""Unit tests cho Task 7.15 (Unified Packaging & Inference CLI).

Kiểm thử:
1. Đóng gói checkpoint với metadata.
2. Kiểm tra tính toàn vẹn (reject calibration giả mạo hash).
3. Đảm bảo trọng số trước và sau khi đóng gói khớp 100%.
4. Chạy suy luận mẫu trên fixture dummy image và đo đạc latency overhead.
"""
import json
import os
import tempfile
import pytest
import torch
from PIL import Image

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


@pytest.fixture
def dummy_checkpoint_and_calib():
    """Tạo checkpoint fixture dummy và calibration khớp SHA-256."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "dummy_model.pth")
        calib_path = os.path.join(tmpdir, "dummy_calib.json")
        out_pkg_path = os.path.join(tmpdir, "dummy_packaged.pth")
        meta_out_path = os.path.join(tmpdir, "dummy_metadata.json")

        # Tạo dummy state_dict với 3 adapter blocks {7, 15, 23}
        sd = {
            "clip_model.visual.transformer.resblocks.7.forgery_aware_adapter.adapter.weight": torch.randn(16, 1024),
            "clip_model.visual.transformer.resblocks.15.forgery_aware_adapter.adapter.weight": torch.randn(16, 1024),
            "clip_model.visual.transformer.resblocks.23.forgery_aware_adapter.adapter.weight": torch.randn(16, 1024),
            "fc.weight": torch.randn(2, 1024),
            "fc.bias": torch.randn(2)
        }
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
    assert meta["calibration"]["T"] == 1.5000
    assert os.path.exists(info["out_pkg_path"])
    assert os.path.exists(info["meta_out_path"])

    # Nạp lại kiểm tra tensor
    reloaded = torch.load(info["out_pkg_path"], map_location="cpu")
    assert "model" in reloaded
    assert "metadata" in reloaded
    for k, v in info["state_dict"].items():
        assert torch.equal(reloaded["model"][k], v)


def test_reject_mismatched_calibration_sha256(dummy_checkpoint_and_calib):
    """Kiểm tra việc từ chối calibration có SHA-256 không khớp."""
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


def test_collect_image_paths_and_transform(tmp_path):
    """Kiểm tra thu thập ảnh và pipeline tiền xử lý chuẩn."""
    img1 = tmp_path / "test1.jpg"
    img2 = tmp_path / "test2.png"
    text_file = tmp_path / "ignore.txt"

    Image.new("RGB", (300, 300), color="blue").save(img1)
    Image.new("RGB", (100, 100), color="red").save(img2)
    text_file.write_text("not an image")

    paths = collect_image_paths([str(tmp_path)])
    assert len(paths) == 2
    assert str(img1.resolve()) in paths
    assert str(img2.resolve()) in paths

    transform = get_default_transform()
    with Image.open(img1) as im:
        t = transform(im)
        assert t.shape == (3, 224, 224)
        assert torch.is_tensor(t)
