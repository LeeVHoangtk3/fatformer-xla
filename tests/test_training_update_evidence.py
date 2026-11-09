"""Behavior checks: actual parameter updates, overflow skips and deployed geometry."""
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.training.loss import CrossDomainTotalLoss
from src.training.train_crossdomain import CrossDomainTrainer, CrossDomainDataset
from src.training.selection import build_q30_cache


def tiny_trainer(tmp_path, accum=2):
    torch.manual_seed(12)
    images = torch.randn(10, 3, 4, 4)
    samples = [((x, x + .01), i % 2) for i, x in enumerate(images)]
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, 2))
    return CrossDomainTrainer(model, None, DataLoader(samples, batch_size=2), None,
        CrossDomainTotalLoss(loss_mode="v2"), torch.optim.SGD(model.parameters(), lr=.01),
        save_dir=str(tmp_path), grad_accum_steps=accum)


class SimulatedOverflowScaler:
    """A CPU fixture that mimics scaler skip semantics; never evidence of CUDA health."""
    def __init__(self, overflow_attempts):
        self.overflows = set(overflow_attempts)
        self.attempt = 0
        self.value = 2048.

    def get_scale(self):
        return self.value

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        self.attempt += 1
        if self.attempt in self.overflows:
            optimizer.param_groups[0]["params"][0].grad.fill_(float("inf"))

    def step(self, optimizer):
        if self.attempt not in self.overflows:
            optimizer.step()

    def update(self):
        if self.attempt in self.overflows:
            self.value /= 2


def test_accumulation_counts_actual_weight_updates_including_partial_window(tmp_path):
    value = tiny_trainer(tmp_path)
    before = [p.detach().clone() for p in value.student.parameters()]
    # SGD.step returns None even when it really updates the model.
    metrics = value.train_epoch(1)
    assert metrics['optimizer_steps_expected'] == metrics['optimizer_steps_applied'] == 3
    assert metrics['optimizer_steps_skipped'] == metrics['nonfinite_grad_events'] == 0
    assert any(not torch.equal(a, b) for a, b in zip(before, value.student.parameters()))
    events = [json.loads(x) for x in (tmp_path/'optimizer_updates.jsonl').read_text().splitlines()]
    assert [e['mini_batch'] for e in events] == [2, 4, 5]
    assert all(e['applied'] and e['grad_norm_finite'] for e in events)


def test_scaler_skip_is_not_hidden_by_finite_average(tmp_path):
    value = tiny_trainer(tmp_path)
    value.scaler = SimulatedOverflowScaler({1})
    metrics = value.train_epoch(1)
    assert metrics['optimizer_steps_attempted'] == 3
    assert metrics['optimizer_steps_applied'] == 2
    assert metrics['optimizer_steps_skipped'] == metrics['nonfinite_grad_events'] == 1
    assert metrics['grad_norm_finite_count'] == 2
    assert np.isfinite(metrics['grad_norm'])
    first = json.loads((tmp_path/'optimizer_updates.jsonl').read_text().splitlines()[0])
    assert first['amp_skipped'] and first['grad_norm'] is None
    assert first['amp_scale_after'] == first['amp_scale_before'] / 2


def test_all_skipped_epoch_fails_and_keeps_trace_without_fake_zero(tmp_path):
    value = tiny_trainer(tmp_path)
    before = [p.detach().clone() for p in value.student.parameters()]
    value.scaler = SimulatedOverflowScaler({1, 2, 3})
    with pytest.raises(FloatingPointError, match='no usable optimizer updates'):
        value.train_epoch(1)
    assert not value.epoch_complete
    assert all(torch.equal(a, b) for a, b in zip(before, value.student.parameters()))
    assert np.isnan(value.last_optimization_metrics['grad_norm'])
    assert len((tmp_path/'optimizer_updates.jsonl').read_text().splitlines()) == 3


def test_initial_amp_warmup_can_recover_before_measured_smoke(tmp_path):
    value = tiny_trainer(tmp_path)
    value.scaler = SimulatedOverflowScaler({1})
    warmup = value.train_epoch(0,max_steps=2,advance_scheduler=False,trace_phase='warmup')
    assert warmup['optimizer_steps_applied']==0 and np.isnan(warmup['grad_norm'])
    measured = value.train_epoch(0,max_steps=2,advance_scheduler=False,trace_phase='benchmark')
    assert measured['optimizer_steps_applied']==1 and np.isfinite(measured['grad_norm'])


def test_non_amp_nonfinite_gradient_stops_before_mutating_weights(tmp_path):
    value = tiny_trainer(tmp_path)
    before = [p.detach().clone() for p in value.student.parameters()]
    next(value.student.parameters()).register_hook(lambda g: torch.full_like(g, float('nan')))
    with pytest.raises(FloatingPointError, match='Non-finite gradient'):
        value.train_epoch(1)
    assert all(torch.equal(a, b) for a, b in zip(before, value.student.parameters()))
    event = json.loads((tmp_path/'optimizer_updates.jsonl').read_text().splitlines()[0])
    assert event['unrecoverable'] and not event['applied']


def test_clip_clean_and_q30_match_independent_pil_reference(tmp_path):
    # A non-square, textured image catches stretch-vs-short-edge errors.
    image = Image.fromarray(np.random.default_rng(8).integers(0, 256, (240, 410, 3), dtype=np.uint8))
    path = tmp_path/'image.png'; image.save(path)
    rows = [dict(path=str(path),label=i) for i in (0,1)]
    ds = CrossDomainDataset(rows, paired=False, img_resolution=224, preprocessing='clip')
    from src.models.clip.clip import _transform
    reference = _transform(224)
    assert torch.equal(ds[0][0], reference(image))
    crop = T.CenterCrop(224)(T.Resize(224, interpolation=T.InterpolationMode.BICUBIC)(image))
    buf = io.BytesIO(); crop.save(buf,format='JPEG',quality=30); buf.seek(0)
    with Image.open(buf) as jpg:
        # Geometry is already 224x224, so reference only normalizes here.
        expected = reference(jpg.convert('RGB'))
    assert all(torch.equal(x, expected) for x, _ in build_q30_cache(ds, max_samples=2))
    pair_ds = CrossDomainDataset(rows, paired=True, img_resolution=224, preprocessing='clip',
        degradation_config={'img_resolution':224,'clean_prob':1.})
    (clean, degraded), _ = pair_ds[0]
    assert torch.equal(clean, reference(image)) and torch.equal(clean, degraded)


def test_legacy_profile_preserves_old_tensor_values(tmp_path):
    path = tmp_path/'image.png'; image = Image.new('RGB',(330,240),(40,80,150)); image.save(path)
    ds = CrossDomainDataset([dict(path=str(path),label=0)], paired=False, img_resolution=224)
    expected = T.Compose([T.Resize((224,224)),T.CenterCrop(224),T.ToTensor(),
                          T.Normalize([.485,.456,.406],[.229,.224,.225])])(image)
    assert torch.equal(ds[0][0],expected)
