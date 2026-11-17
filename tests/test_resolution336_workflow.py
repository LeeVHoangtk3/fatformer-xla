import ast
import copy
from contextlib import nullcontext
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_336_training_changes_resolution_and_microbatch_without_changing_losses():
    original = yaml.safe_load((ROOT / "configs/train_phase7_anchored_v2.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "configs/train_anchored_336.yaml").read_text(encoding="utf-8"))
    assert config["loss"] == original["loss"]
    assert config["training"]["epochs"] == original["training"]["epochs"] == 5
    assert config["training"]["batch_size"] * config["training"]["grad_accum_steps"] == 32
    assert config["dataset"]["img_resolution"] == config["dataset"]["crop_resolution"] == 336
    assert config["dataset"]["preprocessing"] == "legacy_imagenet"
    degradation = yaml.safe_load((ROOT / config["dataset"]["degradation_config"]).read_text(encoding="utf-8"))
    old_degradation = yaml.safe_load((ROOT / "configs/degradation_v2.yaml").read_text(encoding="utf-8"))
    assert degradation.pop("img_resolution") == degradation.pop("crop_resolution") == 336
    old_degradation.pop("img_resolution")
    old_degradation.pop("crop_resolution")
    assert degradation == old_degradation


def test_colab_336_runs_git_training_calibration_then_complete_master_audit():
    notebook = json.loads((ROOT / "notebooks/train_336_master_audit_colab.ipynb").read_text(encoding="utf-8"))
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        ast.parse(source)
        assert cell["outputs"] == [] and cell["execution_count"] is None
        code.append(source)
    joined = "\n".join(code)
    assert "'hoang'" not in joined and "expected_branch='test'" in joined
    assert "configs/train_anchored_336.yaml" in joined
    assert "configs/degradation_336.yaml" in joined
    assert "--input_resolution" in joined and "'336'" in joined
    assert "extract_calibration_logits" in joined and "verify_unified_package" in joined
    assert "evaluate_archive(" in joined and "audit_catalog(" in joined
    assert "package_sha256=PACKAGE_SHA" in joined
    assert "len(expected_jobs) == 37" in joined
    assert "len(reports) != len(expected_jobs)" in joined
    assert "sum(r['pooled']['total_samples']" in joined
    assert "phase7_training_pilot" not in joined
    assert "epochs'] = 4" not in joined
    assert "248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0" not in joined


def test_notebook_comparison_reports_actual_deltas_without_promoting_candidate(tmp_path):
    import pandas as pd
    from tools.calibration_data import sha256_file
    from tools.phase7_corrected import write_json

    notebook = json.loads((ROOT / "notebooks/train_336_master_audit_colab.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code" and "with step('comparison_with_224')" in "".join(c["source"]))
    path = ROOT / "DATASET/logs/phase7/master_audit/runs/master_audit_20261115_040751_7beca5/raw/benchmark_summary.json"
    current = copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))
    current["cfe"]["clean"]["pooled"]["auc"] += 1.5
    namespace = dict(REPO=ROOT, OUTPUT=tmp_path, summary=current,
        gates={"training_selection": {"status": "PASS"}}, PACKAGE_SHA="a" * 64,
        CONTRACT=dict(train_batch=4, train_grad_accum_steps=8, inference_batch=4),
        pd=pd, json=json, sha256_file=sha256_file, write_json=write_json, step=lambda _: nullcontext())
    exec(compile(source, "<notebook-comparison-unit-test>", "exec"), namespace)
    rows = pd.read_csv(tmp_path / "comparison_224_vs_336.csv")
    assert len(rows) == 77
    changed = rows[rows.delta_percentage_points != 0]
    assert len(changed) == 1 and changed.iloc[0].delta_percentage_points == 1.5
    conclusion = json.loads((tmp_path / "comparison_conclusion.json").read_text())
    assert "no automatic replacement" in conclusion["acceptance"]
