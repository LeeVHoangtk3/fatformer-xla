"""Fail-closed Phase 7.11 preflight. Audit mode never declares training ready."""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.training.runtime_state import sha256_file, image_path, source_sha256
from src.training.selection import OperatingPoint
from src.datasets.dataset import ALL_TEST_SUBSETS


class Preflight:
    def __init__(self, config, data_root=None, repo_root=PROJECT_ROOT, min_records=12000):
        self.root = Path(repo_root).resolve()
        self.config_path = Path(config).resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        ds = self.config["dataset"]
        self.data_root = Path(data_root or ds.get("data_root") or self.root).resolve()
        self.manifest = self.path(ds["manifest"])
        self.report = self.path(ds.get("isolation_report", "DATASET/logs/phase7/data_isolation_report.json"))
        self.data_config = self.path(ds.get("audit_config", "configs/phase7_data_v2.runtime.yaml"))
        self.min_records = min_records

    def path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def records(self):
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        rows = data["records"]
        if len(rows) < self.min_records or data["metadata"]["n_records"] != len(rows):
            raise ValueError("Manifest count is too small or inconsistent with metadata")
        for row in rows:
            if row.get("label") not in (0, 1) or not row.get("path"):
                raise ValueError("Missing image path or invalid label")
            if not re.fullmatch("[0-9a-f]{64}", row.get("sha256", "")):
                raise ValueError("Missing/invalid image SHA256")
        return rows

    def check_images(self):
        rows = self.records()
        failures, examples = 0, []
        for row in rows:
            try:
                path = image_path(row["path"], self.data_root)
                if sha256_file(path) != row["sha256"]:
                    raise ValueError("image SHA256 mismatch")
                with Image.open(path) as image:
                    image.convert("RGB").load()
            except Exception as exc:
                failures += 1
                if len(examples) < 3:
                    examples.append(f"{row['path']}: {exc}")
        if failures:
            raise ValueError(f"{failures}/{len(rows)} staged images failed: {examples}")
        return f"{len(rows)} staged images decoded and matched manifest SHA256"

    def check_isolation(self):
        audit = json.loads(self.report.read_text(encoding="utf-8"))
        if (audit.get("pass") is not True or audit.get("violations") != 0
                or audit.get("integrity", {}).get("failures") != 0
                or audit.get("coverage", {}).get("complete") is not True):
            raise ValueError("Audit must be explicitly passing, complete, and have zero violations/integrity failures")
        meta = audit["metadata"]
        if meta.get("manifest_sha256") != sha256_file(self.manifest):
            raise ValueError("Audit refers to a different manifest")
        if meta.get("config_sha256") != sha256_file(self.data_config):
            raise ValueError("Audit data-config SHA256 mismatch; stage the original config from the audited archive")
        pairs = audit["pairs"]
        required = ("v2_vs_cfe_test", "v2_vs_task51_18sets", "internal_splits")
        if any(name not in pairs or pairs[name].get("violations") != 0 for name in required):
            raise ValueError("Audit lacks required zero-violation comparisons")
        n = len(self.records())
        for name in required[:2]:
            if pairs[name].get("n_checked") != n or pairs[name].get("n_reference_images", 0) <= 0:
                raise ValueError("Incomplete external audit coverage")
        internal = pairs["internal_splits"].get("pairs", {})
        if len([k for k in internal if "__vs__" in k]) != 3:
            raise ValueError("Audit lacks all three internal split comparisons")
        refs = meta.get("references", {})
        if not {"cfe", "test18"} <= set(refs):
            raise ValueError("Missing reference fingerprints")
        for ref in refs.values():
            if not re.fullmatch("[0-9a-f]{64}", ref.get("sha256", "")) or ref.get("n_images", 0) <= 0:
                raise ValueError("Invalid reference fingerprint")
        coverage = audit["coverage"]
        expected = set(coverage.get("expected_sets", []))
        if not set(ALL_TEST_SUBSETS) <= expected or not expected <= set(coverage.get("observed_sets", [])):
            raise ValueError("18-set coverage incomplete")
        return "Passing audit bound to exact manifest and original data config; external reference digests recorded"

    def check_transforms(self):
        ds = self.config["dataset"]
        from src.datasets.preprocessing import PROFILES
        if ds.get("preprocessing", "legacy_imagenet") not in PROFILES or ds.get("validation_preprocessing") not in (*PROFILES, None):
            raise ValueError("Unknown training/validation preprocessing profile")
        cfg = yaml.safe_load(self.path(ds["degradation_config"]).read_text(encoding="utf-8"))
        if cfg.get("clean_prob") != .60 or ds.get("val_split") != "val_select":
            raise ValueError("Require clean_prob=.60 and val_split=val_select")
        if ds.get("train_dir") or ds.get("val_dir"):
            raise ValueError("Phase 7 must use the audited manifest, not override folders")
        run = self.config["training"]
        OperatingPoint(self.path(run["calibration_file"]), run.get("selection_threshold"))
        return "v2 clean_prob=.60; manifest val_select; measured frozen calibration is valid"

    def check_splits(self):
        groups = {name: [] for name in ("train", "val_calib", "val_select")}
        for row in self.records():
            if row.get("split") not in groups:
                raise ValueError("Unknown manifest split")
            groups[row["split"]].append(row)
        for name, rows in groups.items():
            if not rows or {r["label"] for r in rows} != {0, 1}:
                raise ValueError(f"{name} is empty or missing a label")
        names = list(groups)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                for key in ("path", "sha256"):
                    if {r[key] for r in groups[first]} & {r[key] for r in groups[second]}:
                        raise ValueError(f"Cross-split {key} leakage: {first}/{second}")
        return str({name: len(rows) for name, rows in groups.items()}) + "; disjoint paths and SHA256"

    def check_canonical(self):
        from src.models.canonical_loader import load_canonical
        model = self.config["model"]
        checkpoint, clip = self.path(model["checkpoint_init"]), self.path(model["clip_path"])
        if not checkpoint.is_file() or not clip.is_file():
            raise FileNotFoundError("Baseline checkpoint or CLIP backbone is missing")
        baseline_sha = sha256_file(checkpoint)
        teacher_path = self.path(model["teacher_checkpoint"])
        if baseline_sha != (baseline_sha if checkpoint.resolve() == teacher_path.resolve() else sha256_file(teacher_path)):
            raise ValueError("Phase 7 warm start must match the baseline teacher")
        calibration = json.loads(self.path(self.config["training"]["calibration_file"]).read_text(encoding="utf-8"))
        if calibration.get("checkpoint", {}).get("sha256") != baseline_sha:
            raise ValueError("Frozen calibration is not bound to the baseline checkpoint")
        args = SimpleNamespace(backbone="CLIP:ViT-L/14", clip_path=str(clip), num_classes=2,
                               num_vit_adapter=3, num_context_embedding=8, init_context_embedding="",
                               hidden_dim=768, clip_vision_width=1024, frequency_encoder_layer=2,
                               decoder_layer=4, num_heads=12, use_srm=False, use_gating=False)
        loaded, report = load_canonical(str(checkpoint), args, allow_partial=False)
        if report.get("missing") != 0 or report.get("unexpected") != 0:
            raise ValueError("Canonical load did not prove missing=0/unexpected=0")
        del loaded
        return f"Strict canonical load 0/0; baseline_sha256={baseline_sha}"

    def check_tests(self, files):
        scratch = self.root / "scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch, prefix="preflight-tests-") as root:
            command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", root, *files]
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            result = subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if result.returncode:
                raise ValueError(f"Tests failed ({result.returncode}): {(result.stdout + result.stderr)[-3000:]}")
        return result.stdout.strip()[-1000:]


def check_gpu_environment(min_free_gib=14.0):
    if not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError("GPU minimum free memory must be nonnegative")
    if not torch.cuda.is_available():
        return False, "CUDA unavailable; GPU training gate is NOT satisfied"
    free, total = torch.cuda.mem_get_info()
    if free < min_free_gib * 1024 ** 3:
        return False, f"Free VRAM {free / 1024**3:.2f} GiB < {min_free_gib:g} GiB (total {total / 1024**3:.2f})"
    return True, f"{torch.cuda.get_device_name()}: {free / 1024**3:.2f} GiB free"


def run_checks(preflight, mode="train", skip_tests=False, skip_model_load=False):
    cases = [
        ("manifest_and_staged_images", preflight.check_images),
        ("bound_isolation_audit", preflight.check_isolation),
        ("degradation_and_calibration", preflight.check_transforms),
        ("loss_tests", lambda: preflight.check_tests(["tests/test_loss_v2.py", "tests/test_smoke_loss_v2_integration.py"])),
        ("selection_tests", lambda: preflight.check_tests(["tests/test_selection_metric.py", "tests/test_smoke_selection_v2_integration.py", "tests/test_selection_regressions.py"])),
        ("update_evidence_tests", lambda: preflight.check_tests(["tests/test_training_update_evidence.py", "tests/test_phase7_training_pilot.py", "tests/test_phase7_git_source.py"])),
        ("baseline_canonical", preflight.check_canonical),
        ("split_path_and_hash_isolation", preflight.check_splits),
    ]
    results = {}
    for name, call in cases:
        if (skip_tests and name.endswith("_tests")) or (skip_model_load and name == "baseline_canonical"):
            results[name] = {"status": "SKIP", "message": "Not executed; audit mode only"}
            continue
        try:
            results[name] = {"status": "PASS", "message": call()}
        except Exception as exc:
            results[name] = {"status": "FAIL", "message": str(exc)}
    try:
        minimum = preflight.config.get("training", {}).get("preflight_min_free_gib", 14.0)
        passed, message = check_gpu_environment(float(minimum))
    except Exception as exc:
        passed, message = False, f"GPU probe failed: {exc}"
    status = "PASS" if passed else "SKIP" if mode == "audit" and not torch.cuda.is_available() else "FAIL"
    key = "gpu_14_gib_free" if preflight.config.get("training", {}).get("preflight_min_free_gib", 14.0) == 14.0 else "gpu_memory_availability"
    results[key] = {"status": status, "message": message}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_phase7_anchored_v2.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--mode", choices=["train", "audit"], default="train")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument("--output", default="DATASET/logs/phase7/preflight_train.json")
    args = parser.parse_args()
    if (args.skip_tests or args.skip_model_load) and args.mode != "audit":
        parser.error("Skipping gates is only permitted in audit mode")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Invalidate stale readiness before expensive checks, including failed init.
    path.write_text(json.dumps({"schema": "phase7-preflight/v2", "mode": args.mode,
                                "ready_to_train": False, "state": "running"}) + "\n", encoding="utf-8")
    try:
        preflight = Preflight(args.config, args.data_root)
    except Exception as exc:
        path.write_text(json.dumps({"schema": "phase7-preflight/v2", "mode": args.mode,
                                    "ready_to_train": False, "initialization_error": str(exc)}) + "\n", encoding="utf-8")
        print(f"Preflight initialization failed: {exc}")
        return 1
    checks = run_checks(preflight, args.mode, args.skip_tests, args.skip_model_load)
    ready = args.mode == "train" and all(row["status"] == "PASS" for row in checks.values())
    report = {"schema": "phase7-preflight/v2", "mode": args.mode, "ready_to_train": ready,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": str(preflight.config_path),
              "config_sha256": sha256_file(preflight.config_path), "data_root": str(preflight.data_root),
              "source_sha256": source_sha256(preflight.root),
              "preflight_tool_sha256": sha256_file(__file__),
              "python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "checks": checks}
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, row in checks.items():
        print(f"[{row['status']}] {name}: {row['message']}")
    print(f"ready_to_train={ready}; report={path}")
    return 0 if ready or (args.mode == "audit" and not any(r["status"] == "FAIL" for r in checks.values())) else 1


if __name__ == "__main__":
    raise SystemExit(main())
