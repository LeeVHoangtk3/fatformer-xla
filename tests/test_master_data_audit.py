import ast
import hashlib
import io
import json
import sqlite3
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.unified_fatformer import UnifiedFatFormerXLA
from tools.master_data_audit import (audit_archive, catalog, check_overlap, evaluate_archive,
    extract_audited, final_metrics, identify, jobs_for, open_inventory, safe_member, summarize_predictions)
from tools.calibration_data import sha256_file


def image_bytes(color):
    stream = io.BytesIO()
    Image.new("RGB", (32, 40), color).save(stream, format="PNG")
    return stream.getvalue()


def make_archive(path, members):
    with tarfile.open(path, "w") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


@pytest.fixture
def fixture_archive(tmp_path):
    path = tmp_path/"samples.tar"
    make_archive(path, [("./root/0_real/real.png", image_bytes("black")),
                        ("./root/1_fake/fake.png", image_bytes("white"))])
    return dict(name="sample", kind="train", path=str(path), expected_images=2)


def test_actual_bytes_are_decoded_hashed_extracted_and_counted(tmp_path, fixture_archive):
    connection = open_inventory(tmp_path/"inventory.sqlite")
    report = audit_archive(fixture_archive, connection, {}, tmp_path/"audit.json")
    assert report["images"] == 2 and report["status"] == "completed"
    assert report["by_split"] == {"train": 2}
    records = extract_audited(fixture_archive, connection, tmp_path/"stage")
    assert {r["label"] for r in records} == {0,1}
    for row in records:
        assert sha256_file(row["path"]) == connection.execute(
            "SELECT sha256 FROM images WHERE member=?", (row["member"],)).fetchone()[0]
    connection.close()


@pytest.mark.parametrize("bad", ["../../escape.png", "/absolute.png", "C:/escape.png", "a\\escape.png"])
def test_unsafe_archive_names_are_rejected(bad):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_member(bad)


def test_corrupt_images_do_not_become_zero_tensors(tmp_path, fixture_archive):
    make_archive(Path(fixture_archive["path"]), [("0_real/broken.png", b"broken")])
    connection = open_inventory(tmp_path/"inventory.sqlite")
    with pytest.raises(ValueError, match="Cannot decode sample:0_real/broken.png"):
        audit_archive(fixture_archive, connection, {}, tmp_path/"audit.json")
    connection.close()


def test_missing_and_duplicate_members_cannot_pass_counts(tmp_path, fixture_archive):
    connection = open_inventory(tmp_path/"inventory.sqlite")
    fixture_archive["expected_images"] = 3
    with pytest.raises(ValueError, match="2/3"):
        audit_archive(fixture_archive, connection, {}, tmp_path/"audit.json")
    connection.close()
    connection = open_inventory(tmp_path/"duplicate.sqlite")
    data = image_bytes("blue")
    make_archive(Path(fixture_archive["path"]), [("./0_real/x.png", data), ("0_real/x.png", data)])
    with pytest.raises(sqlite3.IntegrityError):
        audit_archive(fixture_archive, connection, {}, tmp_path/"duplicate.json")
    connection.close()


def test_v2_uses_manifest_hash_and_split_not_filename_guess():
    data = image_bytes("red")
    digest = hashlib.sha256(data).hexdigest()
    maps = dict(v2={digest: dict(label=1, split="val_calib", path="DATASET/raw_v2/generator/x.png")})
    assert identify(dict(name="v2", kind="v2_manifest"), "raw_v2/generator/x.png", digest, maps) == (
        1, "val_calib", "data_v2", "DATASET/raw_v2/generator/x.png")
    with pytest.raises(ValueError, match="absent"):
        identify(dict(name="v2", kind="v2_manifest"), "x.png", "wrong_hash", maps)


def test_content_leakage_prevents_inference(tmp_path):
    connection = open_inventory(tmp_path/"inventory.sqlite")
    connection.executemany("INSERT INTO images VALUES (?,?,?,?,?,?,?,?)", [
        ("data_v2", "train.png", "same", 0, "train", "v2", "train-id", 10),
        ("cfe", "eval.png", "same", 0, "evaluation", "cfe", "eval-id", 10)])
    with pytest.raises(ValueError, match="leakage"):
        check_overlap(connection, tmp_path/"overlap.json")
    assert len(json.loads((tmp_path/"overlap.json").read_text())["exact_v2_vs_test_collisions"]) == 1
    connection.close()


def test_catalog_and_job_roles_do_not_mix_training_into_evaluation():
    items = catalog("/drive/Fatformer")
    assert len(items) == 8
    assert next(s for s in items if s["name"] == "test_gans")["expected_images"] == 90329
    records = [dict(split="calibration"), dict(split="evaluation")]
    jobs = list(jobs_for(dict(kind="cfe"), records))
    assert len(jobs) == 8
    assert len([j for j in jobs if j[0] == "calibration"]) == 1
    assert len([j for j in jobs if j[0] == "evaluation"]) == 7
    assert list(jobs_for(dict(kind="stored_degraded"), [dict(split="jpeg_q30")]))[0][1] == "clean"


class ToyBackbone(torch.nn.Module):
    def forward(self, x):
        value = x.mean((1,2,3))
        return torch.stack([-value, value], dim=1)


def test_measured_predictions_are_saved_and_metrics_recomputed(tmp_path, fixture_archive):
    connection = open_inventory(tmp_path/"inventory.sqlite")
    report = audit_archive(fixture_archive, connection, {}, tmp_path/"audit.json")
    fixture_archive["audited_sha256"] = report["archive_sha256"]
    calibration = dict(schema="calibration/v1", T=1.0, T_star=0.7, fpr_target=0.05,
                       evidence_level="measured", checkpoint=dict(sha256="a"*64))
    model = UnifiedFatFormerXLA(ToyBackbone(), calibration, checkpoint_sha256="a"*64).eval()
    reports = evaluate_archive(fixture_archive, connection, tmp_path/"stage", tmp_path/"predictions",
                               model, torch.device("cpu"), batch_size=1, bootstrap=0)
    assert len(reports) == 1 and reports[0]["role"] == "diagnostic_only_no_refit"
    with np.load(tmp_path/"predictions"/reports[0]["prediction_file"], allow_pickle=False) as values:
        assert len(values["prob_fake"]) == 2
        assert reports[0]["pooled"]["accuracy"] == float((values["labels"] == values["targets"]).mean()*100)
        assert all(str(p).startswith("sample:") for p in values["paths"])
    connection.close()


def test_single_class_metrics_are_not_fabricated():
    with pytest.raises(ValueError):
        summarize_predictions(dict(prob_fake=np.array([0.4]), targets=np.array([0]), subsets=np.array(["x"])), 0.7)


def test_notebook_is_executable_source_without_historical_metric_tables():
    notebook = json.loads(Path("notebooks/master_full_data_audit_colab.ipynb").read_text(encoding="utf-8"))
    source = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert not cell["outputs"] and cell["execution_count"] is None
            source.append("".join(cell["source"]))
    joined = "\n".join(source)
    assert "audit_archive(" in joined and "evaluate_archive(" in joined and "load_unified_model(" in joined
    assert "cfe_summary_data" not in joined and "indist_summary_data" not in joined
    assert "checkpoint['model']" not in joined and "PASS IN PACKAGING" not in joined
    assert "bootstrap" in joined and "content_overlap" in joined
    assert "'git', 'clone'" in joined and "'checkout', '--detach'" in joined
    assert "git_source_identity" in joined and "configs/master_audit" in joined
    assert "files.upload" not in joined and "ZipFile" not in joined


def test_final_summary_excludes_supplemental_and_requires_all_conditions(tmp_path, monkeypatch):
    import tools.master_data_audit as audit
    monkeypatch.setattr(audit, "INDIST_COUNTS", {"progan": 2, "guided": 2})
    reports = []
    for condition in audit.CONDITIONS:
        for archive, group in (("test_gans", "progan"), ("test_diffusion", "guided")):
            path = tmp_path/f"{archive}_{condition}.npz"
            np.savez(path, prob_fake=np.array([0.1,0.9,0.9,0.1]), targets=np.array([0,1,0,1]),
                     subsets=np.array([group,group,"crn","crn"]))
            reports.append(dict(archive=archive, condition=condition, prediction_file=path.name))
        reports.append(dict(archive="cfe", split="evaluation", condition=condition,
            pooled=dict(auc=90.,fake_at_real90=70.,ece=4.),
            per_subset={s:dict(fake_acc=60.) for s in ["1_commercial", "2_latent_diffusion", "3_pixel_diffusion", "4_modern_gans", "5_other_autoregressive"]}))
    result = final_metrics(reports, tmp_path, 0.7, bootstrap=0)
    assert result["indist_core18"]["clean"]["pooled"]["total_samples"] == 4
    assert result["indist_core18"]["clean"]["macro_balanced_acc"] == 100
    assert set(result["indist_core18"]["clean"]["per_subset"]) == {"progan", "guided"}
    with pytest.raises(ValueError, match="Both clean test archives"):
        final_metrics([r for r in reports if not (r["archive"] == "test_gans" and r["condition"] == "q30")], tmp_path, 0.7, bootstrap=0)


def test_running_finalize_manually_after_failure_cannot_claim_completion():
    from contextlib import contextmanager
    notebook = json.loads(Path("notebooks/master_full_data_audit_colab.ipynb").read_text(encoding="utf-8"))
    final_cell = next("".join(c["source"]) for c in notebook["cells"]
                      if c["cell_type"] == "code" and "with step('finalize')" in "".join(c["source"]))
    state = dict(steps=[dict(name="git_source", status="failed")])
    @contextmanager
    def step(name):
        state["steps"].append(dict(name=name, status="running"))
        yield
    with pytest.raises(RuntimeError, match="finalize"):
        exec(final_cell, dict(step=step, state=state))


def test_git_tracked_json_partitions_preserve_frozen_counts_and_labels():
    from tools.calibration_data import read_partition
    from tools.master_data_audit import manifest_maps
    root = Path(__file__).resolve().parents[1]
    maps = manifest_maps(root)
    assert len(maps['v2']) == 14400 and len(maps['legacy']) == 3600 and len(maps['cfe']) == 2000
    calib, cy = read_partition(root/'configs/master_audit/calibration_split.json', 'calibration')
    evaluation, ey = read_partition(root/'configs/master_audit/evaluation_split.json', 'evaluation')
    assert len(calib) == 400 and len(evaluation) == 1600 and not set(calib) & set(evaluation)
    assert np.bincount(cy).tolist() == [200,200] and np.bincount(ey).tolist() == [800,800]
    for split, expected_sha in [('calibration','128176bd4f914ce188be05ef964e05a81802cbf41b140bc99200a4d59646b70c'),
                                ('evaluation','ed5af91047320147517c7a2a1283e242aaecb0cf9a91351a469902158b8bc4e9')]:
        payload = json.loads((root/f'configs/master_audit/{split}_split.json').read_text(encoding='utf-8'))
        assert payload['source_npz_sha256'] == expected_sha
