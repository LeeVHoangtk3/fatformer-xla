import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from tools.preflight_phase7_train import Preflight, run_checks, check_gpu_environment, ALL_TEST_SUBSETS
from src.training.runtime_state import sha256_file


@pytest.fixture
def staged(tmp_path):
    data = tmp_path / "ssd"
    data.mkdir()
    rows = []
    for i, split in enumerate(["train", "train", "val_calib", "val_calib", "val_select", "val_select"]):
        path = data / f"image_{i}.png"
        Image.new("RGB", (16, 16), (i * 30, 40, 70)).save(path)
        rows.append({"path": path.name, "sha256": sha256_file(path), "label": i % 2, "split": split})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"metadata": {"n_records": 6}, "records": rows}))
    data_config = tmp_path / "data.yaml"
    data_config.write_text("seed: 42\n")
    report = tmp_path / "audit.json"
    audit = {"pass": True, "violations": 0, "integrity": {"failures": 0},
             "coverage": {"complete": True, "expected_sets": ALL_TEST_SUBSETS, "observed_sets": ALL_TEST_SUBSETS},
             "metadata": {"manifest_sha256": sha256_file(manifest), "config_sha256": sha256_file(data_config),
                          "references": {k: {"sha256": "a" * 64, "n_images": 1} for k in ("cfe", "test18")}},
             "pairs": {k: {"violations": 0, "n_checked": 6, "n_reference_images": 1}
                       for k in ("v2_vs_cfe_test", "v2_vs_task51_18sets")}}
    audit["pairs"]["internal_splits"] = {"violations": 0, "pairs": {k: {} for k in ("train__vs__calib", "train__vs__select", "calib__vs__select")}}
    report.write_text(json.dumps(audit))
    degradation = tmp_path / "degradation.yaml"
    degradation.write_text("clean_prob: 0.60\n")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"evidence_level": "measured", "T": 2., "T_star_default": .9}))
    baseline = tmp_path / "baseline.pth"
    baseline.write_bytes(b"baseline")
    calibration.write_text(json.dumps({"evidence_level": "measured", "T": 2., "T_star_default": .9,
                                       "checkpoint": {"sha256": sha256_file(baseline)}}))
    backbone = tmp_path / "backbone.pt"
    backbone.write_bytes(b"backbone")
    config = {"dataset": {"manifest": str(manifest), "data_root": str(data), "val_split": "val_select",
                           "isolation_report": str(report), "audit_config": str(data_config),
                           "degradation_config": str(degradation)},
              "training": {"calibration_file": str(calibration)},
              "model": {"checkpoint_init": str(baseline), "teacher_checkpoint": str(baseline), "clip_path": str(backbone)}}
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(config))
    return Preflight(path, repo_root=tmp_path, min_records=2)


def test_staged_images_and_bound_audit_pass(staged):
    assert "6 staged images" in staged.check_images()
    assert "exact manifest" in staged.check_isolation()
    assert "disjoint" in staged.check_splits()
    assert "calibration" in staged.check_transforms()


@pytest.mark.parametrize("change", ["empty", "stale_manifest", "stale_config", "incomplete", "missing_pair"])
def test_empty_or_stale_audit_never_passes(staged, change):
    report = json.loads(staged.report.read_text())
    if change == "empty":
        report = {}
    elif change == "stale_manifest":
        report["metadata"]["manifest_sha256"] = "b" * 64
    elif change == "stale_config":
        staged.data_config.write_text("seed: 43\n")
    elif change == "incomplete":
        report["coverage"]["observed_sets"] = ["progan"]
    else:
        del report["pairs"]["v2_vs_cfe_test"]
    staged.report.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        staged.check_isolation()


def test_missing_and_corrupt_images_fail(staged):
    image = staged.data_root / "image_0.png"
    image.unlink()
    with pytest.raises(ValueError, match="staged images failed"):
        staged.check_images()
    image.write_bytes(b"not an image")
    data = json.loads(staged.manifest.read_text())
    data["records"][0]["sha256"] = sha256_file(image)
    staged.manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="staged images failed"):
        staged.check_images()


def test_same_content_different_paths_is_split_leakage(staged):
    data = json.loads(staged.manifest.read_text())
    data["records"][4]["sha256"] = data["records"][2]["sha256"]
    staged.manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA256|sha256"):
        staged.check_splits()


def test_cpu_is_not_gpu_pass_and_vram_limit_is_14_gib(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert check_gpu_environment()[0] is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test GPU")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (int(13.5 * 1024**3), 16 * 1024**3))
    assert check_gpu_environment()[0] is False
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (14 * 1024**3, 16 * 1024**3))
    assert check_gpu_environment()[0] is True


def test_audit_mode_skips_are_explicit_and_train_mode_fails_without_gpu(staged, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(staged, "check_canonical", lambda: "strict test evidence")
    monkeypatch.setattr(staged, "check_tests", lambda files: "passed test evidence")
    audit = run_checks(staged, mode="audit", skip_tests=True)
    assert audit["gpu_14_gib_free"]["status"] == "SKIP"
    assert audit["selection_tests"]["status"] == "SKIP"
    train = run_checks(staged)
    assert train["gpu_14_gib_free"]["status"] == "FAIL"


def test_canonical_missing_report_keys_are_not_assumed_zero(staged, monkeypatch):
    import src.models.canonical_loader as loader
    def strict(path, args, allow_partial):
        assert allow_partial is False
        return torch.nn.Linear(1, 1), {}
    monkeypatch.setattr(loader, "load_canonical", strict)
    with pytest.raises(ValueError, match="did not prove"):
        staged.check_canonical()


def test_cli_audit_never_declares_training_ready_and_skips_cannot_bypass_train(staged, monkeypatch, tmp_path):
    import sys
    import tools.preflight_phase7_train as module
    monkeypatch.setattr(module, "Preflight", lambda *a, **kw: staged)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["preflight", "--mode", "audit", "--skip-model-load", "--skip-tests", "--output", str(output)])
    assert module.main() == 0
    report = json.loads(output.read_text())
    assert report["ready_to_train"] is False
    assert report["checks"]["gpu_14_gib_free"]["status"] == "SKIP"
    monkeypatch.setattr(sys, "argv", ["preflight", "--skip-model-load"])
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 2


def test_gpu_driver_error_is_a_structured_failure(staged, monkeypatch):
    monkeypatch.setattr(staged, "check_canonical", lambda: "strict test evidence")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    def broken():
        raise RuntimeError("driver unavailable")
    monkeypatch.setattr(torch.cuda, "mem_get_info", broken)
    checks = run_checks(staged, mode="audit", skip_tests=True)
    assert checks["gpu_14_gib_free"]["status"] == "FAIL"
    assert "driver unavailable" in checks["gpu_14_gib_free"]["message"]
