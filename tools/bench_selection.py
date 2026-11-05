"""Measure warm clean-only vs clean+cached JPEG30 validation on the same corpus.

python -m tools.bench_selection --output DATASET/logs/phase7/selection_benchmark.json
This CPU synthetic benchmark does not establish CLIP/T4 performance.
"""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.training.train_crossdomain import CrossDomainDataset, CrossDomainTrainer
from src.training.loss import CrossDomainTotalLoss


@torch.no_grad()
def clean_only(trainer):
    trainer.student.eval()
    correct = real_correct = fake_correct = n = 0
    for images, targets in trainer.val_loader:
        pred = trainer.operating_point.predictions(trainer.student(images))
        correct += int((pred == targets).sum())
        real_correct += int(((pred == targets) & (targets == 0)).sum())
        fake_correct += int(((pred == targets) & (targets == 1)).sum())
        n += len(targets)
    return correct, real_correct, fake_correct, n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="DATASET/logs/phase7/selection_benchmark.json")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be >= 3")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as root:
        rng = np.random.default_rng(42)
        paths = []
        for i in range(32):
            path = Path(root) / f"{i}.png"
            Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(path)
            paths.append(str(path))
        rows = [{"path": paths[i % 32], "label": i % 2} for i in range(400)]
        ds = CrossDomainDataset(rows, paired=False, img_resolution=224)
        loader = DataLoader(ds, batch_size=32, num_workers=0)
        model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))
        start = time.perf_counter()
        trainer = CrossDomainTrainer(model, None, loader, loader, CrossDomainTotalLoss(),
                                     torch.optim.SGD(model.parameters(), lr=.01), save_dir=root)
        startup = time.perf_counter() - start
        for _ in range(2):
            clean_only(trainer)
            trainer.validate()
        times = {"clean": [], "clean_plus_q30": []}
        for repeat in range(args.repeats):
            calls = [("clean", lambda: clean_only(trainer)), ("clean_plus_q30", trainer.validate)]
            if repeat % 2:
                calls.reverse()
            for name, call in calls:
                start = time.perf_counter()
                call()
                times[name].append((time.perf_counter() - start) * 1000)
    clean, full = (statistics.median(times[k]) for k in times)
    result = {"source": "synthetic:32 seeded PNG RGB noise images reused across 400 records",
              "model": "CPU pooled RGB linear classifier, not CLIP", "seed": 42,
              "n_eval_clean": 400, "n_eval_q30": 300, "q30_real": 150, "q30_fake": 150,
              "batch_size": 32, "num_workers": 0, "parent_torch_threads": 1,
              "method": "2 warmups; alternating order; median; one-time cache setup excluded",
              "setup_ms": startup * 1000, "runs_ms": times,
              "clean_median_ms": clean, "clean_plus_q30_median_ms": full,
              "ratio": full / clean, "limit_ratio": 2, "passed": full <= 2 * clean,
              "torch": torch.__version__}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
