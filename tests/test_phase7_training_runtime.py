import copy
import csv
import json
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.training.loss import CrossDomainTotalLoss
from src.training.train_crossdomain import CrossDomainTrainer, set_seed
from src.training.runtime_state import sha256_file


class NoisyImages(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, i):
        noise = (random.random() + np.random.random() + torch.rand(()).item()) * .01
        return torch.full((3, 4, 4), i / 6 + noise), i % 2


def trainer(path, seed=123, backup=None):
    set_seed(seed)
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, 8), nn.ReLU(), nn.Dropout(.3), nn.Linear(8, 2))
    loader = DataLoader(NoisyImages(), batch_size=2, shuffle=True,
                        generator=torch.Generator().manual_seed(77))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    value = CrossDomainTrainer(model, None, loader, None, CrossDomainTotalLoss(), optimizer,
                                scheduler=scheduler, save_dir=str(path), backup_dir=str(backup) if backup else None)
    value.run_context = {"experiment": "test", "epochs": 2}
    return value


def equal_state(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_state(x, y)
    else:
        assert a == b


def test_resume_matches_uninterrupted_training_and_all_rng(tmp_path):
    continuous = trainer(tmp_path / "continuous")
    continuous.train_epoch(1)
    continuous.save_checkpoint(1)
    checkpoint = tmp_path / "continuous" / "checkpoint_latest.pth"
    continuous.train_epoch(2)
    expected_model = copy.deepcopy(continuous.student.state_dict())
    expected_optimizer = copy.deepcopy(continuous.optimizer.state_dict())
    expected_scheduler = continuous.scheduler.state_dict()
    expected_random = (random.random(), float(np.random.random()), torch.rand(3))
    restarted = trainer(tmp_path / "restarted", seed=999)
    assert restarted.load_resume(checkpoint) == 2
    restarted.train_epoch(2)
    equal_state(expected_model, restarted.student.state_dict())
    equal_state(expected_optimizer, restarted.optimizer.state_dict())
    equal_state(expected_scheduler, restarted.scheduler.state_dict())
    actual_random = (random.random(), float(np.random.random()), torch.rand(3))
    equal_state(expected_random, actual_random)
    assert restarted.global_step == continuous.global_step == 6


def test_model_only_partial_epoch_and_wrong_context_are_rejected(tmp_path):
    value = trainer(tmp_path / "run")
    model_only = tmp_path / "model.pth"
    torch.save({"model": value.student.state_dict()}, model_only)
    with pytest.raises(ValueError, match="full resume"):
        value.load_resume(model_only)
    value.train_epoch(1, max_steps=1)
    value.save_checkpoint(1)
    path = tmp_path / "run" / "checkpoint_latest.pth"
    with pytest.raises(ValueError, match="completed-epoch"):
        value.load_resume(path)
    value.train_epoch(1)
    value.save_checkpoint(1)
    value.run_context["epochs"] = 3
    with pytest.raises(ValueError, match="context differs"):
        value.load_resume(path)


def test_scaler_state_is_part_of_resume_contract(tmp_path):
    class Scaler:
        def __init__(self, scale):
            self.scale = scale

        def state_dict(self):
            return {"scale": self.scale}

        def load_state_dict(self, state):
            self.scale = state["scale"]
    original = trainer(tmp_path / "original")
    original.scaler = Scaler(8192)
    original.save_checkpoint(0)
    restarted = trainer(tmp_path / "restart")
    restarted.scaler = Scaler(1)
    restarted.load_resume(tmp_path / "original" / "checkpoint_latest.pth")
    assert restarted.scaler.scale == 8192


def test_atomic_latest_backup_and_log_repair(tmp_path, monkeypatch):
    value = trainer(tmp_path / "run", backup=tmp_path / "backup")
    metrics = {"clean_r": .94, "clean_f": .99, "q30_r": .7, "q30_f": .8, "worst_group": .7}
    value.record_validation(0, metrics)
    value.train_epoch(1)
    value.training_history.append({"epoch": 1, "loss_total": .5})
    value.record_validation(1, metrics)
    latest = tmp_path / "run" / "checkpoint_latest.pth"
    digest = sha256_file(latest)
    backup = tmp_path / "backup" / "checkpoint_latest.pth"
    assert sha256_file(backup) == digest
    assert backup.with_suffix(".pth.sha256").read_text().strip() == digest
    def interrupted(*args, **kwargs):
        raise OSError("interrupted write")
    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            value.save_checkpoint(2)
    assert sha256_file(latest) == digest
    (tmp_path / "run" / "val_groups.csv").write_text("broken")
    (tmp_path / "run" / "train_history.csv").write_text("broken")
    value.load_resume(latest)
    with (tmp_path / "run" / "val_groups.csv").open() as f:
        assert [row["epoch"] for row in csv.DictReader(f)] == ["0", "1"]
    with (tmp_path / "run" / "train_history.csv").open() as f:
        assert [row["epoch"] for row in csv.DictReader(f)] == ["1"]


def test_benchmark_cli_isolated_from_training_outputs(tmp_path, monkeypatch):
    import src.training.train_crossdomain as module
    run = tmp_path / "run"
    output = tmp_path / "bench" / "result.json"
    monkeypatch.setattr(sys, "argv", ["train", "--config", "configs/train_phase7_anchored_v2.yaml",
                                      "--dry_run", "--device", "cpu", "--batch_size", "4",
                                      "--benchmark_steps", "2", "--benchmark_warmup", "1",
                                      "--checkpoint_init", str(tmp_path / "missing.pth"),
                                      "--teacher_checkpoint", str(tmp_path / "missing.pth"),
                                      "--output", str(output), "--save_dir", str(run)])
    monkeypatch.setattr(module, "find_first_existing", lambda _: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("Benchmark cannot validate or save training checkpoints")
    monkeypatch.setattr(module.CrossDomainTrainer, "save_checkpoint", forbidden)
    monkeypatch.setattr(module.CrossDomainTrainer, "validate", forbidden)
    result = module.run_training_pipeline(module.parse_arguments())
    assert result["steps"] == 2 and result["ms_per_step"] > 0
    assert result["peak_vram_gib"] is None and result["evidence_level"] == "mock"
    assert result["est_total_minutes"] > 0
    assert output.exists() and not list(run.glob("*.pth")) and not list(run.glob("*.csv"))


def test_notebook_python_cells_compile_and_use_checked_commands():
    import ast
    from pathlib import Path
    data = json.loads(Path("notebooks/task/task_7.11/task_7.11_train_anchored_v2_colab.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code"]
    for source in sources:
        ast.parse(source)
    joined = "\n".join(sources)
    assert "check=True" in joined and "--benchmark_steps" in joined and "--resume_from" in joined
    assert "--max_steps 100" not in joined


def test_pipeline_resume_continues_next_epoch_without_resetting_anchor(tmp_path, monkeypatch):
    import src.training.train_crossdomain as module
    import src.models
    import src.training.freeze_utils
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 8)
            self.gating = nn.Linear(8, 8)
            self.head = nn.Linear(8, 2)

        def forward(self, image, return_features=False):
            feature = torch.relu(self.fc(image.mean((-2, -1))))
            logits = self.head(feature) + self.gating(feature)[:, :2] * 0
            return (logits, feature) if return_features else logits
    baseline = tmp_path / "baseline.pth"
    torch.save({"model": TinyModel().state_dict()}, baseline)
    backbone = tmp_path / "backbone.pt"
    backbone.write_bytes(b"tiny mocked backbone")
    monkeypatch.setattr(src.models, "build_model", lambda args: TinyModel())
    monkeypatch.setattr(src.training.freeze_utils, "freeze_clip_backbone", lambda *a, **kw: None)
    monkeypatch.setattr(module, "PROJECT_ROOT", str(tmp_path))
    class PairedImages(NoisyImages):
        def __getitem__(self, i):
            image, label = super().__getitem__(i)
            return (image, image + .01), label
    def loaders(args):
        return DataLoader(PairedImages(), batch_size=2), DataLoader(NoisyImages(), batch_size=2)
    monkeypatch.setattr(module, "build_crossdomain_dataloaders", loaders)
    root = tmp_path / "run"
    argv = ["train", "--config", "configs/train_phase7_anchored_v2.yaml", "--device", "cpu",
            "--checkpoint_init", str(baseline), "--teacher_checkpoint", str(baseline),
            "--save_dir", str(root), "--epochs", "2", "--batch_size", "2", "--clip_path", str(backbone)]
    monkeypatch.setattr(sys, "argv", argv + ["--max_steps", "3"])
    module.run_training_pipeline(module.parse_arguments())
    latest = root / "checkpoint_latest.pth"
    assert torch.load(latest, weights_only=False)["metadata"]["epoch"] == 1
    monkeypatch.setattr(sys, "argv", argv + ["--resume_from", str(latest)])
    module.run_training_pipeline(module.parse_arguments())
    state = torch.load(latest, weights_only=False)
    assert state["metadata"]["epoch"] == 2
    assert state["resume_state"]["global_step"] == 6
    with (root / "val_groups.csv").open() as f:
        assert [r["epoch"] for r in csv.DictReader(f)] == ["0", "1", "2"]
    with (root / "train_history.csv").open() as f:
        assert [r["epoch"] for r in csv.DictReader(f)] == ["1", "2"]
