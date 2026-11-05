import csv
import io
import json
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from PIL import Image
import torchvision.transforms as T

from src.datasets.transforms import get_eval_transforms
from src.training.loss import CrossDomainTotalLoss
from src.training.metrics import worst_group_score
from src.training.selection import OperatingPoint, build_q30_cache, checkpoint_decision
from src.training.train_crossdomain import CrossDomainTrainer, CrossDomainDataset
from src.training.checkpoint_manager import CheckpointManager


class SortedImages(Dataset):
    def __init__(self, n_real=300, n_fake=20):
        self.labels = [0] * n_real + [1] * n_fake

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.full((3, 16, 16), float(self.labels[i])), self.labels[i]


def make_trainer(path, dataset=None, **kwargs):
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.copy_(torch.tensor([0., .6]))
    loader = DataLoader(dataset, batch_size=32) if dataset is not None else None
    return CrossDomainTrainer(model, None, loader, loader, CrossDomainTotalLoss(),
                               torch.optim.SGD(model.parameters(), lr=.01), save_dir=str(path), **kwargs)


def metrics(real=.92, worst=.7):
    return {"clean_r": real, "clean_f": .99, "q30_r": worst, "q30_f": .95, "worst_group": worst}


def test_sorted_generic_data_is_stratified_and_missing_cache_raises(tmp_path):
    trainer = make_trainer(tmp_path, SortedImages())
    assert sum(y == 0 for _, y in trainer.val_q30_cache) == 150
    assert sum(y == 1 for _, y in trainer.val_q30_cache) == 20
    trainer.val_q30_cache = []
    with pytest.raises(ValueError, match="nan"):
        trainer.validate()


def test_one_class_cannot_create_a_selection_cache(tmp_path):
    with pytest.raises(ValueError, match="both Real and Fake"):
        make_trainer(tmp_path, SortedImages(8, 0))


def test_records_q30_matches_post_resize_and_is_deterministic(tmp_path):
    image_path = tmp_path / "noise.png"
    Image.effect_noise((400, 320), 80).convert("RGB").save(image_path)
    ds = CrossDomainDataset([{"path": str(image_path), "label": y} for y in (0, 1)],
                            paired=False, img_resolution=224)
    cache = build_q30_cache(ds)
    expected = get_eval_transforms(224, 224, {"type": "jpeg", "quality": 30}, "post_resize")
    with Image.open(image_path) as image:
        reference = expected(image.convert("RGB"))
    assert all(torch.equal(x, reference) for x, _ in cache)
    repeat = build_q30_cache(ds)
    assert all(y == z and torch.equal(a, b) for (a, y), (b, z) in zip(cache, repeat))
    sub = Subset(ds, [1, 0])
    assert sorted(y for _, y in build_q30_cache(sub)) == [0, 1]
    image_path.write_bytes(b"corrupt")
    with pytest.raises(Exception):
        build_q30_cache(ds)


def test_calibration_changes_both_conditions_and_is_logged(tmp_path):
    artifact = tmp_path / "calibration.json"
    artifact.write_text(json.dumps({"evidence_level": "measured", "T": 2., "T_star_default": .9}))
    dataset = SortedImages(2, 2)
    default = make_trainer(tmp_path / "default", dataset).validate()
    trainer = make_trainer(tmp_path / "calibrated", dataset, calibration_file=str(artifact))
    calibrated = trainer.validate()
    assert default["clean_r"] == default["q30_r"] == 0
    assert calibrated["clean_r"] == calibrated["q30_r"] == 1
    assert calibrated["clean_f"] == calibrated["q30_f"] == 0
    trainer.record_validation(0, calibrated)
    with (tmp_path / "calibrated" / "val_groups.csv").open(newline="") as f:
        row = next(csv.DictReader(f))
    assert float(row["threshold"]) == .9 and float(row["temperature"]) == 2
    assert int(row["n_eval_q30"]) == 4
    state = torch.load(tmp_path / "calibrated" / "best.pth", weights_only=False)
    assert state["metadata"]["threshold"] == .9
    assert state["metadata"]["calibration_sha256"]


def test_independent_records_ties_and_epoch_zero_guard(tmp_path):
    trainer = make_trainer(tmp_path)
    assert trainer.record_validation(0, metrics(.85, .8)) == (False, True)
    assert not (tmp_path / "best.pth").exists()
    assert trainer.record_validation(1, metrics(.92, .7)) == (True, False)
    saved = (tmp_path / "best.pth").read_bytes()
    assert trainer.record_validation(2, metrics(.99, .7)) == (False, False)
    assert trainer.record_validation(3, metrics(.84, .81)) == (False, True)
    assert (tmp_path / "best.pth").read_bytes() == saved
    with pytest.raises(ValueError, match="Real guard"):
        trainer.save_checkpoint(4, is_best=True, val_metrics=metrics(.85, .8))


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("filename", [None, "custom.pth"])
def test_checkpoint_manager_modes_and_backup(tmp_path, optimized, filename):
    model = nn.Linear(3, 2)
    manager = CheckpointManager(str(tmp_path / "local"), str(tmp_path / "backup"),
                                save_only_latest_and_best=optimized)
    path = manager.save(model, epoch=1, is_best=True, filename=filename)
    assert path
    best = (tmp_path / "local" / "best.pth").read_bytes()
    manager.save(model, epoch=2, val_metrics={"is_best_effort": True})
    assert (tmp_path / "local" / "best.pth").read_bytes() == best
    assert (tmp_path / "backup" / "best_effort.pth").exists()
    with pytest.raises(ValueError, match="Real guard"):
        manager.save(model, is_best=True, val_metrics=metrics(.85, .8))


def test_standard_trainer_legacy_and_real_q30_pair(tmp_path, monkeypatch):
    import src.training.trainer as module
    def loader():
        return DataLoader(TensorDataset(torch.zeros(4, 3), torch.tensor([0, 0, 1, 1])), batch_size=2)
    clean, q30, q50, diffusion = loader(), loader(), loader(), loader()
    model = nn.Linear(3, 2)
    def evaluate(model, data, device, operating_point=None):
        scores = [.1, .1, .8, .8] if data is clean or data is q50 else [.1, .9, .8, .8]
        return {"acc": 75., "ap": 90., "auc": 90., "r_acc": 50., "f_acc": 100.,
                "y_true": [0, 0, 1, 1], "y_score": scores,
                "y_pred_selection": (operating_point or OperatingPoint()).from_scores(scores).tolist()}
    monkeypatch.setattr(module, "evaluate_dataloader", evaluate)
    trainer = module.Trainer(model, clean, {"progan_clean": clean, "progan_degraded_q50": q50,
                                           "progan_degraded_q30": q30, "diffusion_clean": diffusion},
                             device=torch.device("cpu"), use_amp=False,
                             checkpoint_manager=CheckpointManager(str(tmp_path / "dict")))
    monkeypatch.setattr(trainer, "train_epoch", lambda epoch: 0.)
    trainer.fit(epochs=1)
    with (tmp_path / "dict" / "val_groups.csv").open() as f:
        row = next(csv.DictReader(f))
    assert float(row["q30_r"]) == .5
    single = module.Trainer(nn.Linear(3, 2), clean, clean, device=torch.device("cpu"), use_amp=False,
                            checkpoint_manager=CheckpointManager(str(tmp_path / "single")))
    monkeypatch.setattr(single, "train_epoch", lambda epoch: 0.)
    with pytest.warns(RuntimeWarning, match="legacy"):
        single.fit(epochs=1)
    assert (tmp_path / "single" / "best.pth").exists()
    assert not (tmp_path / "single" / "val_groups.csv").exists()
    with pytest.raises(ValueError, match="Q50/deg"):
        module.Trainer(nn.Linear(3, 2), clean, {"clean": clean, "deg_q50": q50},
                       device=torch.device("cpu"), checkpoint_manager=CheckpointManager(str(tmp_path / "bad")))


def test_real_pipeline_never_promotes_ineligible_epoch_zero_or_last(tmp_path, monkeypatch):
    import src.training.train_crossdomain as module
    import src.models
    import src.training.freeze_utils
    baseline = tmp_path / "baseline.pth"
    torch.save({"model": module._MockBackbone().state_dict()}, baseline)
    monkeypatch.setattr(sys, "argv", ["train", "--config", "configs/train_phase7_anchored_v2.yaml",
                                      "--device", "cpu", "--checkpoint_init", str(baseline),
                                      "--teacher_checkpoint", str(baseline), "--epochs", "1", "--max_steps", "1",
                                      "--save_dir", str(tmp_path / "run")])
    args = module.parse_arguments()
    monkeypatch.setattr(src.models, "build_model", lambda _: module._MockBackbone())
    monkeypatch.setattr(src.training.freeze_utils, "freeze_clip_backbone", lambda *a, **kw: None)
    train = DataLoader(module._DummyDataset(4, paired=True), batch_size=2)
    val = DataLoader(module._DummyDataset(4, paired=False), batch_size=2)
    monkeypatch.setattr(module, "build_crossdomain_dataloaders", lambda _: (train, val))
    monkeypatch.setattr(module, "PROJECT_ROOT", str(tmp_path))
    result = {**metrics(.85, .8), "val_acc": 90., "val_r_acc": 85., "val_f_acc": 99., "val_composite": 88.}
    monkeypatch.setattr(module.CrossDomainTrainer, "validate", lambda _: result)
    module.run_training_pipeline(args)
    assert not (tmp_path / "run" / args.save_name).exists()
    assert not (tmp_path / "run" / "best.pth").exists()
    assert (tmp_path / "run" / "best_effort.pth").exists()
    assert (tmp_path / "run" / "checkpoint_latest.pth").exists()


@pytest.mark.parametrize("bad", [float("inf"), -.1, 1.1, float("nan")])
def test_invalid_metrics_are_rejected(bad):
    with pytest.raises(ValueError):
        worst_group_score({**metrics(), "q30_f": bad})


@pytest.mark.parametrize("change", [{"evidence_level": "mock"}, {"T": 0},
                                   {"T_star_default": 1}, {"T": float("nan")}])
def test_invalid_calibration_rejected(tmp_path, change):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"evidence_level": "measured", "T": 2., "T_star_default": .8, **change}))
    with pytest.raises(ValueError):
        OperatingPoint(str(path))


def test_nonfinite_model_output_is_not_a_real_prediction():
    with pytest.raises(ValueError, match="finite"):
        OperatingPoint().predictions(torch.tensor([[0., float("nan")]]))
    with pytest.raises(ValueError, match="finite"):
        OperatingPoint().from_scores([float("nan")])


def test_sampler_restriction_is_respected(tmp_path):
    from torch.utils.data import SubsetRandomSampler
    ds = SortedImages()
    trainer = make_trainer(tmp_path)
    trainer.val_loader = DataLoader(ds, batch_size=2, sampler=SubsetRandomSampler([0, 300, 301]))
    cache = trainer._build_val_q30_cache()
    assert sorted(y for _, y in cache) == [0, 1, 1]


def test_temperature_uses_logits_before_probability_saturation(tmp_path):
    from src.evaluation.fast_eval import evaluate_dataloader
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"evidence_level": "measured", "T": 100., "T_star_default": .57}))
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([0., 20.]))
    loader = DataLoader(TensorDataset(torch.zeros(2, 3), torch.tensor([0, 1])), batch_size=2)
    output = evaluate_dataloader(model, loader, torch.device("cpu"), operating_point=OperatingPoint(str(path)))
    assert output["y_score"] == [1., 1.]
    assert output["y_pred_selection"] == [0, 0]  # sigmoid(20/100) < .57
