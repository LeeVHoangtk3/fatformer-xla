"""Regression tests for fair ablation and fail-closed evidence handling (synthetic fixtures)."""
import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.training.loss import CrossDomainTotalLoss
from src.training.runtime_state import sha256_file
from tools.ablation_v2_common import load_partition, filter_records, validate_history, NAMES, WEIGHTS
from tools.audit_ablation_configs import audit_configs
from tools.evaluate_ablation_v2 import (parse_cfe_eval_json, evaluate_delta_verdict, contrasts,
                                       main as summarize, build_summary, write_outputs)
from tools.run_ablation_v2 import restore_latest, verify_run


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def csv_file(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@pytest.mark.parametrize("weights", WEIGHTS)
def test_ablation_ce_is_identical_with_all_four_switch_combinations(weights):
    torch.manual_seed(13)
    clean, degraded, teacher = (torch.randn(6, 2, requires_grad=True) for _ in range(3))
    fc, fd = torch.randn(6, 4), torch.randn(6, 4)
    targets = torch.tensor([0, 1, 0, 1, 0, 1])
    loss = CrossDomainTotalLoss(weight_ce=.3, weight_ce_deg=.3, loss_mode="v2")
    output = loss(clean, targets, teacher_logits=teacher, logits_degraded=degraded,
                  features_clean=fc, features_deg=fd, weight_feat=weights[0], weight_guard=weights[1])
    expected = .3*torch.nn.functional.cross_entropy(clean, targets, label_smoothing=.05)+.3*torch.nn.functional.cross_entropy(degraded, targets, label_smoothing=.05)
    assert torch.allclose(output["loss_ce"], expected)
    output["loss_total"].backward()
    assert torch.isfinite(clean.grad).all() and teacher.grad is None


def test_config_audit_requires_v2_and_detects_non_loss_drift(tmp_path):
    configs = audit_configs()
    for name, cfg in configs.items():
        (tmp_path/f"{name}.yaml").write_text(yaml.safe_dump(cfg))
    cfg = configs[NAMES[0]]; cfg["training"]["epochs"] += 1
    (tmp_path/f"{NAMES[0]}.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="differ"):
        audit_configs(tmp_path)
    cfg["training"]["epochs"] -= 1; cfg["loss"]["mode"] = "auto"
    (tmp_path/f"{NAMES[0]}.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="mode"):
        audit_configs(tmp_path)


def eval_document(condition):
    return {"report_schema": "v3", "evidence_level": "measured", "gate_eligible": True, "git_dirty": False,
            "degrade_order": "post_resize", "dataset": {"condition": condition, "n": 20},
            "partition": {"split": "evaluation", "n": 20},
            "metrics": {"auc": 79.27, "fake_at_real90": 39.1, "ece": 13.23,
                        "ci95": {"auc": [75., 82.], "fake_at_real90": [30., 48.]}}}


@pytest.fixture
def eval_pair(tmp_path):
    paths = [tmp_path/"clean.json", tmp_path/"q30.json"]
    for path, condition in zip(paths, ("clean", "q30")):
        dump(path, eval_document(condition))
    return paths


def test_real_schema_and_percent_units_are_preserved(eval_pair):
    result = parse_cfe_eval_json(*eval_pair)
    assert result["auc_clean"] == 79.27
    assert result["auc_q30_ci95"] == [75., 82.]
    assert result["fake_at_real90_q30"] == 39.1
    assert result["ece_q30_frozen_temperature_pct"] == 13.23


@pytest.mark.parametrize("mutation", ["synthetic", "missing_ci", "legacy_schema", "no_partition", "not_eligible", "nonfinite"])
def test_evaluation_rejects_missing_or_untrusted_evidence(eval_pair, mutation):
    doc = eval_document("clean")
    if mutation == "synthetic":doc["evidence_level"] = "synthetic"
    if mutation == "missing_ci":doc["metrics"]["ci95"]["auc"] = [None, None]
    if mutation == "legacy_schema":doc["metrics_pooled"] = doc.pop("metrics")
    if mutation == "no_partition":doc.pop("partition")
    if mutation == "not_eligible":doc["gate_eligible"] = False
    if mutation == "nonfinite":doc["metrics"]["auc"] = float("nan")
    dump(eval_pair[0], doc)
    with pytest.raises((ValueError, KeyError)):
        parse_cfe_eval_json(*eval_pair)


def test_missing_eval_files_are_errors_and_no_dummy_mode(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_cfe_eval_json(tmp_path/"missing", tmp_path/"also_missing")
    with pytest.raises(SystemExit):
        summarize(["--test_dummy", "--experiment_dir", str(tmp_path), "--output_dir", str(tmp_path/"output")])
    assert not (tmp_path/"output").exists()
    with pytest.raises(FileNotFoundError):
        summarize(["--experiment_dir", str(tmp_path), "--output_dir", str(tmp_path/"output")])
    assert not (tmp_path/"output").exists()


def test_delta_requires_actual_delta_ci_and_interaction_contrast():
    assert "Chưa" in evaluate_delta_verdict(15.)
    assert "Chưa" in evaluate_delta_verdict(15., [-1., 20.])
    assert "Có" in evaluate_delta_verdict(3., [1., 5.])
    assert "giảm" in evaluate_delta_verdict(-3., [-5., -1.])
    c = contrasts([10., 30., 20., 35.])
    assert c["interaction_2x2"] == -5. and c["full_over_best_single"] == 5.


@pytest.fixture
def partition(tmp_path):
    c, e, a = tmp_path/"calib.npz", tmp_path/"eval.npz", tmp_path/"calibration.json"
    np.savez(c, split="calibration", sample_ids=np.array(["cal-r", "cal-f"]), targets=[0, 1])
    np.savez(e, split="evaluation", sample_ids=np.array(["s/0/r.png", "s/1/f.png"]), targets=[0, 1])
    dump(a, {"schema": "calibration/v1", "evidence_level": "measured", "n_calib": 2,
             "calibration_source_sha256": sha256_file(c)})
    return e, c, a


def test_original_partition_binding_and_exact_record_filter(partition):
    ids, meta = load_partition(*partition)
    records = {"s": [{"subset": "s", "label": 0, "fname": "r.png"},
                     {"subset": "s", "label": 1, "fname": "f.png"},
                     {"subset": "s", "label": 0, "fname": "calibration.png"}]}
    assert len(filter_records(records, ids)["s"]) == 2 and meta["n"] == 2
    with pytest.raises(ValueError, match="missing"):
        filter_records({"s": records["s"][:1]}, ids)


def test_partition_rejects_overlap_and_modified_calibration(partition):
    e, c, a = partition
    np.savez(e, split="evaluation", sample_ids=["cal-r", "other"], targets=[0, 1])
    with pytest.raises(ValueError, match="overlap"):
        load_partition(e, c, a)
    c.write_bytes(b"modified")
    with pytest.raises(ValueError, match="source"):
        load_partition(e, c, a)


def history_row(epoch):
    return {"epoch": epoch, "train_steps": 10, "loss_total": .4, "grad_norm": .2,
            "clean_r": .95, "clean_f": .6, "q30_r": .9, "q30_f": .4}


def test_one_epoch_is_not_complete_and_nonfinite_history_is_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        validate_history([history_row(1)], 5)
    rows = [history_row(i) for i in range(1, 6)]
    assert validate_history(rows, 5) == 10
    rows[3]["grad_norm"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        validate_history(rows, 5)


def test_partial_snapshot_is_resumed_not_mistaken_for_finished(tmp_path):
    local, remote = tmp_path/"local", tmp_path/"remote"
    local.mkdir(); remote.mkdir()
    context = {"epochs": 5, "loss_mode": "v2"}
    state = {"schema": "phase7-epoch-resume/v1", "metadata": {"epoch": 1},
             "resume_state": {"epoch_complete": True, "context": context}}
    path = remote/"checkpoint_latest.pth"
    torch.save(state, path); Path(str(path)+".sha256").write_text(sha256_file(path))
    (remote/"best.pth").write_bytes(b"best alone cannot establish completion")
    result = restore_latest(local, remote, context)
    assert result == local/"checkpoint_latest.pth"
    assert torch.load(result, weights_only=False)["metadata"]["epoch"] == 1
    with pytest.raises(ValueError, match="horizon"):
        verify_run(local, context, "source")
    Path(str(path)+".sha256").write_text("wrong")
    with pytest.raises(ValueError, match="checksum"):
        restore_latest(local, remote, context)


def make_run(folder, weights):
    """Tiny synthetic weights in temporary directories; never production artifacts."""
    folder.mkdir()
    context = {"epochs": 2, "seed": 42, "loss_mode": "v2", "weight_feat": weights[0],
               "weight_guard": weights[1], "calibration_file_sha256": "calibration"}
    metrics = {"clean_r": .95, "clean_f": .6, "q30_r": .9, "q30_f": .4, "worst_group": .4}
    metadata = {"epoch": 1, "is_best": True, "calibration_sha256": "calibration", **metrics}
    torch.save({"model": {"tiny": torch.tensor([1.])}, "metadata": metadata}, folder/"best.pth")
    history = [history_row(1), history_row(2)]
    validation = [{"epoch": i, "metrics": {**metrics, "q30_f": 0. if i == 0 else .4,
                                            "worst_group": 0. if i == 0 else .4}}
                  for i in range(3)]
    torch.save({"schema": "phase7-epoch-resume/v1", "metadata": {"epoch": 2},
                "resume_state": {"context": context, "epoch_complete": True, "training_history": history,
                                 "validation_history": validation}}, folder/"checkpoint_latest.pth")
    csv_file(folder/"train_history.csv", history)
    csv_file(folder/"val_groups.csv", [{"epoch": row["epoch"],
        **{k:v for k,v in row["metrics"].items() if k != "worst_group"}, "worst": row["metrics"]["worst_group"]}
                                     for row in validation])
    proof = verify_run(folder, context, "source")
    assert proof["epochs"] == 2 and proof["selected_epoch"] == 1
    return proof


def test_verified_checkpoint_cannot_be_replaced_by_mismatched_metadata(tmp_path):
    folder = tmp_path/"run"
    proof = make_run(folder, WEIGHTS[0])
    data = torch.load(folder/"best.pth", weights_only=False)
    data["metadata"]["q30_f"] = .5
    torch.save(data, folder/"best.pth")
    with pytest.raises(ValueError, match="metadata"):
        verify_run(folder, proof["context"], "source")


def test_end_to_end_summary_uses_heldout_predictions_and_detects_tampering(tmp_path):
    from tools.eval_communityforensics import calculate_metrics_v3
    for name, weights in zip(NAMES, WEIGHTS):
        folder = tmp_path/name
        proof = make_run(folder, weights)
        y = np.array([0]*10+[1]*10)
        logits = np.column_stack([1-y, y]).astype(np.float32)
        p = torch.softmax(torch.from_numpy(logits), -1)[:, 1].numpy()
        for condition in ("clean", "q30"):
            doc = eval_document(condition)
            doc["git_commit"] = "source"
            doc["partition"]["calibration_sha256"] = "calibration"
            doc["metrics"] = calculate_metrics_v3(p, y)
            doc["metrics"]["ci95"] = {"auc": [100., 100.], "fake_at_real90": [100., 100.]}
            doc["checkpoints"] = [{"load_report": {"sha256": proof["files"]["best.pth"], "missing": 0,
                                                 "unexpected": 0, "layout": {"n": 3, "dropped": []}}}]
            doc["calibration"] = {"T_star": .5, "T": 1.}
            np.savez(folder/f"logits_{name}_{condition}.npz", logits=logits, targets=y,
                     sample_ids=np.array([f"id{i}" for i in range(20)]), temperature=1.)
            doc["predictions"] = {"file": f"logits_{name}_{condition}.npz",
                                  "sha256": sha256_file(folder/f"logits_{name}_{condition}.npz")}
            dump(folder/f"eval_v3_{name}_{condition}.json", doc)
    result = build_summary(tmp_path, n_boot=100)
    assert result["evidence_level"] == "measured" and result["gate_eligible"] is False
    assert result["configurations"][NAMES[0]]["auc_clean"] == 100.
    assert result["primary_driver"] is None
    assert result["paired_cfe_contrast_ci95"]["auc_q30"]["guard"] == [0., 0.]
    write_outputs(result, tmp_path/"summary")
    assert (tmp_path/"summary/m7_ablation.png").stat().st_size > 1000
    with (tmp_path/"summary/ablation_mini_v2.csv").open(encoding="utf-8") as handle:
        saved_rows = list(csv.DictReader(handle))
    assert len(saved_rows) == 4 and saved_rows[0]["epochs"] == "2"
    # A changed best cannot be legitimized merely by retaining its old CSV.
    (tmp_path/NAMES[0]/"best.pth").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="changed"):
        build_summary(tmp_path, n_boot=100)


def test_v2_nonfinite_gradient_stops_before_optimizer_update(tmp_path):
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.training.train_crossdomain import CrossDomainTrainer
    model = nn.Linear(2, 2)
    model.weight.register_hook(lambda grad: grad*float("nan"))
    class Paired(torch.utils.data.Dataset):
        def __len__(self):return 2
        def __getitem__(self, i):return (torch.ones(2), torch.ones(2)), i
    trainer = CrossDomainTrainer(model, None, DataLoader(Paired(), batch_size=2), None,
                                CrossDomainTotalLoss(loss_mode="v2"), torch.optim.SGD(model.parameters(), lr=.1),
                                save_dir=str(tmp_path))
    before = model.weight.detach().clone()
    with pytest.raises(FloatingPointError, match="gradient"):
        trainer.train_epoch(1)
    assert torch.equal(model.weight, before)


def test_command_failure_is_not_reported_as_success_and_log_is_backed_up(tmp_path):
    import sys
    import subprocess
    from tools.run_ablation_v2 import run_logged
    folder, backup = tmp_path/"run", tmp_path/"backup"
    folder.mkdir(); backup.mkdir()
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_logged([sys.executable, "-c", "print('failure evidence', flush=True); raise SystemExit(7)"], folder, backup)
    assert error.value.returncode == 7
    assert "failure evidence" in (folder/"commands.log").read_text()
    assert (backup/"commands.log").read_bytes() == (folder/"commands.log").read_bytes()


def test_verify_run_rejects_incomplete_validation_log(tmp_path):
    folder = tmp_path/"run"
    proof = make_run(folder, WEIGHTS[0])
    with (folder/"val_groups.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    csv_file(folder/"val_groups.csv", rows[1:])
    with pytest.raises(ValueError, match="validation epochs"):
        verify_run(folder, proof["context"], "source")


def test_training_cli_reads_explicit_v2_and_notebook_cells_compile(monkeypatch):
    import sys
    from src.training.train_crossdomain import parse_arguments
    monkeypatch.setattr(sys, "argv", ["train", "--config", "configs/ablation_v2/a1_data.yaml", "--device", "cpu"])
    args = parse_arguments()
    assert args.loss_mode == "v2" and args.weight_guard == args.weight_feat == 0
    data = json.loads(Path("notebooks/task/task_7.12/task_7.12_ablation_mini_colab.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code"]
    for source in sources:ast.parse(source)
    joined = "\n".join(sources)
    assert "check=True" in joined and "run_ablation_v2.py" in joined
    assert "anchored_v2_run1" not in joined and "remote_best_ckpt.exists" not in joined

def test_evaluator_filters_original_partition_before_inference(tmp_path, monkeypatch):
    import sys
    import tools.eval_communityforensics as evaluator
    subset = evaluator.CFE_SUBSETS[0]
    rows = [{"subset": subset, "label": i%2, "fname": f"image{i}.png", "path": f"unused{i}"} for i in range(22)]
    ids = [f"{r['subset']}/{r['label']}/{r['fname']}" for r in rows]
    calibration_npz, evaluation_npz = tmp_path/"calibration.npz", tmp_path/"evaluation.npz"
    np.savez(calibration_npz, split="calibration", sample_ids=ids[:2], targets=[0, 1])
    np.savez(evaluation_npz, split="evaluation", sample_ids=ids[2:], targets=[r["label"] for r in rows[2:]])
    artifact = tmp_path/"calibration.json"
    dump(artifact, {"schema": "calibration/v1", "evidence_level": "measured", "T": 1., "T_star_default": .5,
                    "n_calib": 2, "calibration_source_sha256": sha256_file(calibration_npz)})
    monkeypatch.setattr(evaluator, "get_git_info", lambda: ("test-fixture", False))
    monkeypatch.setattr(evaluator, "run_mock_array_tests", lambda: None)
    monkeypatch.setattr(evaluator, "scan_cfe_dataset", lambda _: {subset: rows})
    monkeypatch.setattr(evaluator, "build_cfe_model_v3", lambda **kw: (torch.nn.Identity(), {"missing": 0, "unexpected": 0}))
    seen = []
    def infer(*, records, threshold, **kwargs):
        seen.extend([r["fname"] for r in records])
        y = torch.tensor([r["label"] for r in records], dtype=torch.long)
        logits = torch.stack([1-y, y], -1).float()
        p = torch.softmax(logits, -1)
        metrics = evaluator.calculate_metrics_v3(p[:, 1].numpy(), y.numpy(), threshold)
        return metrics, p, y, torch.zeros(len(y)), logits
    monkeypatch.setattr(evaluator, "infer_records_v3", infer)
    original_bootstrap = evaluator.compute_bootstrap_ci
    monkeypatch.setattr(evaluator, "compute_bootstrap_ci", lambda p,y,**kw: original_bootstrap(p,y,n_boot=100))
    monkeypatch.setattr(sys, "argv", ["eval", "--data_dir", "unused", "--checkpoint", "fixture.pth",
        "--device", "cpu", "--model_name", "fixture", "--condition", "all", "--output_dir", str(tmp_path/"output"),
        "--calibration", str(artifact), "--calibration_split", str(calibration_npz), "--evaluation_split", str(evaluation_npz)])
    assert evaluator.main() == 0
    assert "image0.png" not in seen and "image1.png" not in seen and len(seen) == 40
    for condition in ("clean", "q30"):
        doc = json.loads((tmp_path/"output"/f"eval_v3_fixture_{condition}.json").read_text(encoding="utf-8"))
        assert doc["dataset"]["n"] == 20 and doc["partition"]["split"] == "evaluation"
        assert doc["metrics"]["ci95"]["fake_at_real90"] == [100., 100.]
        assert sha256_file(tmp_path/"output"/doc["predictions"]["file"]) == doc["predictions"]["sha256"]
        with np.load(tmp_path/"output"/f"logits_fixture_{condition}.npz") as data:
            assert set(data["sample_ids"].tolist()) == set(ids[2:])
