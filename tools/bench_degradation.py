"""Compare paired v1/v2 with real DataLoaders and identical preprocessing.

python -m tools.bench_degradation --batches 200 --num_workers 4
Use --image-dir to measure a local dataset; otherwise use a seeded synthetic corpus.
"""
import argparse
import json
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.datasets.transforms import DegradationPipelineV2, PairedCleanDegradedAugmentation


class BenchmarkDataset(Dataset):
    """Same source images and sample order for both pipelines (spawn-safe)."""

    def __init__(self, pipeline, sample_count, image_paths=None):
        self.pipeline = pipeline
        self.sample_count = sample_count
        self.image_paths = image_paths
        self.images = None
        if not image_paths:
            rng = np.random.default_rng(42)
            self.images = [Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
                           for _ in range(32)]

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        if self.image_paths:
            with Image.open(self.image_paths[index % len(self.image_paths)]) as source:
                img = source.convert("RGB")
        else:
            img = self.images[index % len(self.images)]
        return self.pipeline(img)


def _seed_worker(worker_id):
    random.seed(torch.initial_seed() % (2 ** 32))


def bench_one(pipeline, batches, batch_size, num_workers, warmup, seed, image_paths=None):
    random.seed(seed)
    torch.manual_seed(seed)
    loader = DataLoader(
        BenchmarkDataset(pipeline, (batches + warmup) * batch_size, image_paths),
        batch_size=batch_size, num_workers=num_workers, shuffle=False,
        worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(seed),
        pin_memory=False,
    )
    # Exclude worker startup; warmup consumes complete collated batches.
    iterator = iter(loader)
    for _ in range(warmup):
        next(iterator)
    start = time.perf_counter()
    for _ in range(batches):
        next(iterator)
    ms_per_batch = (time.perf_counter() - start) * 1000 / batches
    # Exhaust outside the timer to exclude worker shutdown too.
    for _ in iterator:
        pass
    return ms_per_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/degradation_v2.yaml")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output", default="DATASET/logs/phase7/degradation_benchmark.json")
    args = parser.parse_args()
    if min(args.batches, args.batch_size, args.warmup, args.repeats) < 1 or args.num_workers < 0:
        parser.error("batches, batch_size, warmup, repeats must be positive; num_workers >= 0")

    paths = None
    if args.image_dir:
        paths = sorted(str(p.resolve()) for p in args.image_dir.rglob("*")
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not paths:
            parser.error("--image-dir contains no supported images")

    torch.set_num_threads(1)
    v2 = DegradationPipelineV2.from_config(args.config, return_pair=True)
    v1 = PairedCleanDegradedAugmentation(
        img_resolution=v2.img_resolution, crop_resolution=v2.crop_resolution)
    timings = {"v1": [], "v2": []}
    for repeat in range(args.repeats):
        order = [("v1", v1), ("v2", v2)]
        if repeat % 2:
            order.reverse()
        for name, pipeline in order:
            timing = bench_one(pipeline, args.batches, args.batch_size, args.num_workers,
                               args.warmup, args.seed + repeat, paths)
            timings[name].append(timing)
            print(f"repeat {repeat + 1}: {name} {timing:.2f} ms/batch", flush=True)

    old, new = (statistics.median(timings[name]) for name in ("v1", "v2"))
    overhead = (new / old - 1) * 100
    result = {
        "method": "DataLoader; median of repeated runs; alternating order; startup/shutdown excluded",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.image_dir.resolve()) if paths else "synthetic:32 seeded RGB noise images, 256x256",
        "source_image_count": len(paths) if paths else 32,
        "v1_ms_per_batch": old, "v2_ms_per_batch": new,
        "pct_overhead": overhead, "threshold_pct": 15.0, "passed": overhead <= 15,
        "batches": args.batches, "batch_size": args.batch_size, "num_workers": args.num_workers,
        "warmup_batches": args.warmup, "repeats": args.repeats, "seed": args.seed,
        "img_resolution": v2.img_resolution, "crop_resolution": v2.crop_resolution,
        "return_pair": True, "config": args.config, "runs_ms_per_batch": timings,
        "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(), "parent_torch_threads": torch.get_num_threads(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{'PASS' if result['passed'] else 'FAIL'}: overhead {overhead:+.2f}% (limit +15%)")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
