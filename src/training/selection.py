"""Shared, read-only operating point and checkpoint selection for Phase 7."""
import csv
import bisect
import hashlib
import io
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset, ConcatDataset
import torchvision.transforms as T

from .metrics import worst_group_score


class OperatingPoint:
    def __init__(self, calibration_file=None, threshold=None):
        self.temperature, self.threshold = 1.0, 0.5
        self.source, self.sha256 = "fixed_default", None
        if calibration_file:
            path = Path(calibration_file)
            raw = path.read_bytes()
            data = json.loads(raw)
            if data.get("evidence_level") != "measured":
                raise ValueError("Selection calibration must have measured evidence")
            self.temperature = float(data["T"])
            self.threshold = float(data["T_star_default"])
            self.source, self.sha256 = str(path), hashlib.sha256(raw).hexdigest()
        if threshold is not None:
            self.threshold = float(threshold)
            self.source += "+threshold_override"
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Calibration temperature must be finite and > 0")
        if not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("Selection threshold must be finite and in (0, 1)")

    def predictions(self, logits):
        if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
            raise ValueError("Selection requires finite [batch, 2] logits")
        return (torch.softmax(logits.float() / self.temperature, dim=-1)[:, 1] >= self.threshold).long()

    def from_scores(self, scores):
        p = np.asarray(scores, dtype=float)
        if p.ndim != 1 or not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
            raise ValueError("Selection requires finite probabilities in [0, 1]")
        p = np.clip(p, 1e-15, 1 - 1e-15)
        log_odds = (np.log(p) - np.log1p(-p)) / self.temperature
        probability = 1 / (1 + np.exp(-np.clip(log_odds, -700, 700)))
        return (probability >= self.threshold).astype(int)


def checkpoint_decision(metrics, best, best_effort, real_min=0.90):
    if not math.isfinite(real_min) or not .90 <= real_min <= 1:
        raise ValueError("real_guard_threshold must be in [0.90, 1]")
    worst = worst_group_score(metrics)
    if "worst_group" in metrics and not math.isclose(worst, metrics["worst_group"], abs_tol=1e-12):
        raise ValueError("worst_group must equal the minimum of all four recalls")
    eligible = metrics["clean_r"] >= real_min
    return bool(eligible and worst > best), bool(not eligible and worst > best_effort)


def selection_loader_names(loaders):
    clean = [k for k in loaders if re.search(r"(^|[_-])clean($|[_-])", k.lower())]
    q30 = [k for k in loaders if re.search(r"(^|[_-])q30($|[_-])", k.lower())]
    if len(q30) == 1 and len(clean) > 1:
        # Multi-domain legacy runs may also log diffusion_clean. Use the clean
        # partner of the Q30 domain, never an unrelated clean population.
        clean = [k for k in clean if q30[0].lower().startswith(k.lower().split("clean")[0])]
    if len(clean) != 1 or len(q30) != 1:
        raise ValueError("Selection v2 needs an unambiguous clean/Q30 loader pair; Q50/deg is insufficient")
    return clean[0], q30[0]


def write_group_log(path, epoch, metrics, is_best_effort, operating_point):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"epoch": epoch, **{k: metrics[k] for k in ("clean_r", "clean_f", "q30_r", "q30_f")},
           "worst": metrics["worst_group"], "best_effort": is_best_effort,
           "threshold": operating_point.threshold, "temperature": operating_point.temperature,
           "calibration_source": operating_point.source,
           "n_eval_clean": metrics.get("n_eval_clean", ""), "n_eval_q30": metrics.get("n_eval_q30", ""),
           "q30_source": metrics.get("q30_source", "loader_jpeg30")}
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(encoding="utf-8", newline="") as f:
            if next(csv.reader(f)) != list(row):
                raise ValueError(f"Incompatible validation CSV schema; use a new run directory: {path}")
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    groups = ", ".join(f"{name}={metrics[name]:.4f}" for name in ("clean_r", "clean_f", "q30_r", "q30_f", "worst_group"))
    print(f"[SELECTION] epoch={epoch}, {groups}, best_effort={is_best_effort}, threshold={operating_point.threshold}, T={operating_point.temperature}")


def _source(dataset, index):
    while isinstance(dataset, (Subset, ConcatDataset)):
        if isinstance(dataset, Subset):
            index = int(dataset.indices[index])
            dataset = dataset.dataset
        else:
            part = bisect.bisect_right(dataset.cumulative_sizes, index)
            index -= dataset.cumulative_sizes[part - 1] if part else 0
            dataset = dataset.datasets[part]
    records = getattr(dataset, "records", None)
    if records is None and hasattr(dataset, "samples"):
        path, label = dataset.samples[index]
        return dataset, {"path": path, "label": label}
    return dataset, records[index] if records is not None else None


def build_q30_cache(dataset, max_samples=300, seed=42):
    """Sample both labels, then encode JPEG30 AFTER the clean resize/crop.

    CrossDomainDataset supplies its explicit geometry and normalization profile.
    Generic tensor datasets default to ImageNet normalization unless they expose
    normalization_mean/std. Subset indices are resolved before using records.
    Missing/corrupt images and missing labels raise; no synthetic-Q30 fallback.
    """
    if not 2 <= max_samples <= 300:
        raise ValueError("q30_max_samples must be in [2, 300]")
    groups = {0: [], 1: []}
    for i in range(len(dataset)):
        source, record = _source(dataset, i)
        if record is not None:
            label = int(record["label"])
        elif hasattr(source, "targets"):
            # Use dataset[i] for a Subset to preserve its mapping.
            label = int(dataset[i][1]) if isinstance(dataset, Subset) else int(source.targets[i])
        else:
            label = int(dataset[i][1])
        if label not in groups:
            raise ValueError("Validation labels must be REAL=0, FAKE=1")
        groups[label].append(i)
    if not all(groups.values()):
        raise ValueError("Q30 validation requires both Real and Fake samples")
    rng = random.Random(seed)
    selected = []
    for indices in groups.values():
        rng.shuffle(indices)
        selected.extend(indices[:max_samples // 2])
    cached = []
    for index in selected:
        source, record = _source(dataset, index)
        mean = torch.tensor(getattr(source, "normalization_mean", [.485, .456, .406])).view(3, 1, 1)
        std = torch.tensor(getattr(source, "normalization_std", [.229, .224, .225])).view(3, 1, 1)
        normalize = T.Compose([T.ToTensor(), T.Normalize(mean.flatten().tolist(), std.flatten().tolist())])
        path = record.get("file_path", record.get("path", "")) if record is not None else ""
        if record is not None and not getattr(source, "allow_dummy", False):
            # CrossDomainDataset's geometry is deterministic Resize -> CenterCrop.
            if hasattr(source, "eval_geometry"):
                geometry = source.eval_geometry
            elif hasattr(source, "img_resolution"):
                geometry = T.Compose([T.Resize((source.img_resolution, source.img_resolution)),
                                      T.CenterCrop(source.crop_resolution)])
            else:
                steps = getattr(getattr(source, "transform", None), "transforms", [])
                if len(steps) < 2 or not isinstance(steps[0], T.Resize) or not isinstance(steps[1], T.CenterCrop):
                    raise ValueError("Record dataset must expose deterministic Resize/CenterCrop geometry")
                geometry = T.Compose(steps[:2])
            with Image.open(path) as image:
                image = geometry(image.convert("RGB"))
            label = int(record["label"])
        else:
            x, label = dataset[index]
            if isinstance(x, (list, tuple)):
                x = x[0]
            image = T.ToPILImage()((x.detach().cpu() * std + mean).clamp(0, 1))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=30)
        buf.seek(0)
        with Image.open(buf) as decoded:
            cached.append((normalize(decoded.convert("RGB")), int(label)))
    return cached
