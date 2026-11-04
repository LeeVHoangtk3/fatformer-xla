"""
tools/bench_degradation.py — Task 7.8
Đo tốc độ pipeline suy biến trước/sau v2.

Chạy:
    python -m tools.bench_degradation --config configs/degradation_v2.yaml \
        --batches 200 --num_workers 4 --output logs/phase7/degradation_benchmark.json
"""
import argparse
import json
import os
import time
import random
from pathlib import Path

from PIL import Image


def _make_dummy_img(size=224):
    """Sinh ảnh dummy RGB."""
    rng = random.Random(42)
    pix = [(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
           for _ in range(size * size)]
    img = Image.new("RGB", (size, size))
    img.putdata(pix)
    return img


def _bench_one(pipeline, n_batches: int, batch_size: int) -> float:
    """
    Đo tổng thời gian (ms) cho n_batches batch, trả về ms/batch.
    lean: chạy single-process (không DataLoader thật), đủ để so sánh tương đối.
    """
    img = _make_dummy_img()
    start = time.perf_counter()
    for _ in range(n_batches):
        for _ in range(batch_size):
            pipeline(img)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms / n_batches


def main():
    parser = argparse.ArgumentParser(description="Bench degradation pipeline v2")
    parser.add_argument("--config",      default="configs/degradation_v2.yaml")
    parser.add_argument("--batches",     type=int, default=200)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Chỉ ghi vào JSON (benchmark chạy single-proc để so sánh công bằng)")
    parser.add_argument("--output",      default="logs/phase7/degradation_benchmark.json")
    args = parser.parse_args()

    from src.datasets.transforms import PairedCleanDegradedAugmentation, DegradationPipelineV2

    print(f"[bench] Đo v1 (PairedCleanDegradedAugmentation)...")
    pipe_v1 = PairedCleanDegradedAugmentation()
    ms_v1 = _bench_one(pipe_v1, args.batches, args.batch_size)

    print(f"[bench] Đo v2 (DegradationPipelineV2 từ {args.config})...")
    pipe_v2 = DegradationPipelineV2.from_config(args.config) \
        if os.path.exists(args.config) else DegradationPipelineV2()
    ms_v2 = _bench_one(pipe_v2, args.batches, args.batch_size)

    pct_diff = (ms_v2 - ms_v1) / ms_v1 * 100
    passed   = pct_diff <= 15.0

    result = {
        "v1_ms_per_batch": round(ms_v1, 2),
        "v2_ms_per_batch": round(ms_v2, 2),
        "pct_overhead":    round(pct_diff, 2),
        "threshold_pct":   15.0,
        "passed":          passed,
        "batches":         args.batches,
        "batch_size":      args.batch_size,
        "num_workers":     args.num_workers,
        "config":          args.config,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'✅ PASS' if passed else '❌ FAIL'} — v1: {ms_v1:.1f} ms/batch | "
          f"v2: {ms_v2:.1f} ms/batch | overhead: {pct_diff:+.1f}% (ngưỡng ≤15%)")
    print(f"Kết quả: {args.output}")


if __name__ == "__main__":
    main()
