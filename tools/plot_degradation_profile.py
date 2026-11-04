"""
tools/plot_degradation_profile.py — Task 7.8
Sinh logs/phase7/degradation_profile.png với 4 panel:
  (a) Histogram Q JPEG
  (b) Tỷ lệ theo loại suy biến
  (c) Histogram σ blur
  (d) Tỷ lệ composite

Chạy: python -m tools.plot_degradation_profile --config configs/degradation_v2.yaml --n 10000
"""
import argparse
import random
import os
from collections import Counter
from pathlib import Path
from PIL import Image


def _make_img(size=64):
    """Ảnh dummy nhỏ để đo nhanh phân phối."""
    rng = random.Random(42)
    img = Image.new("RGB", (size, size), color=(rng.randint(0,255), rng.randint(0,255), rng.randint(0,255)))
    return img


def sample_distribution(pipe, n=10_000):
    """Lấy mẫu n lần, trả về danh sách tags."""
    img = _make_img()
    tags_all = []
    for _ in range(n):
        pipe(img)
        tags_all.append(list(pipe.last_tag))
    return tags_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/degradation_v2.yaml")
    parser.add_argument("--n",      type=int, default=10_000)
    parser.add_argument("--output", default="logs/phase7/degradation_profile.png")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib chưa cài. Bỏ qua vẽ histogram.")
        return

    from src.datasets.transforms import DegradationPipelineV2

    random.seed(42)
    pipe = DegradationPipelineV2.from_config(args.config) \
        if os.path.exists(args.config) else DegradationPipelineV2()

    print(f"[plot] Lấy mẫu {args.n} lần...")
    tags_all = sample_distribution(pipe, args.n)

    # Phân tích
    jpeg_qs      = []
    blur_sigmas  = []
    type_counts  = Counter()
    n_composite  = 0
    n_clean      = 0

    for tags in tags_all:
        if not tags:
            n_clean += 1
            type_counts["clean"] += 1
            continue
        if len(tags) >= 2:
            n_composite += 1
        for t in tags:
            if t.startswith("jpeg:"):
                q = int(t.split(":")[1])
                jpeg_qs.append(q)
                type_counts["jpeg"] += 1
            elif t.startswith("blur:"):
                s = float(t.split(":")[1])
                blur_sigmas.append(s)
                type_counts["blur"] += 1
            elif t == "webp":
                type_counts["webp"] += 1
            elif t == "double_jpeg":
                type_counts["double_jpeg"] += 1
            elif t.startswith("down_up:"):
                type_counts["down_up"] += 1

    n_degraded = args.n - n_clean
    comp_ratio = n_composite / n_degraded if n_degraded > 0 else 0
    heavy_count = sum(1 for q in jpeg_qs if 30 <= q <= 45)
    heavy_ratio = heavy_count / len(jpeg_qs) if jpeg_qs else 0

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Degradation Profile v2 — Task 7.8", fontsize=14, fontweight="bold")

    # (a) Histogram Q JPEG
    ax = axes[0, 0]
    if jpeg_qs:
        ax.hist(jpeg_qs, bins=range(30, 86, 2), color="#2196F3", edgecolor="white", alpha=0.85)
        ax.axvspan(30, 45, alpha=0.15, color="red", label=f"Heavy [30-45] = {heavy_ratio:.1%}")
        ax.set_title("(a) JPEG Quality Distribution")
        ax.set_xlabel("Quality Q")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No JPEG samples", ha="center", va="center", transform=ax.transAxes)

    # (b) Tỷ lệ theo loại
    ax = axes[0, 1]
    labels = [k for k in type_counts if k != "clean"]
    vals   = [type_counts[k] for k in labels]
    total  = sum(type_counts.values())
    pcts   = [v / total * 100 for v in vals]
    colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50", "#F44336", "#795548"]
    ax.bar(labels, pcts, color=colors[:len(labels)], edgecolor="white")
    ax.bar("clean", n_clean / total * 100, color="#9E9E9E", edgecolor="white")
    ax.set_title("(b) Degradation Type Ratio (%)")
    ax.set_xlabel("Type")
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=30)

    # (c) Histogram σ blur
    ax = axes[1, 0]
    if blur_sigmas:
        ax.hist(blur_sigmas, bins=20, color="#4CAF50", edgecolor="white", alpha=0.85)
        ax.set_title("(c) Gaussian Blur σ Distribution")
        ax.set_xlabel("σ")
        ax.set_ylabel("Count")
    else:
        ax.text(0.5, 0.5, "No Blur samples", ha="center", va="center", transform=ax.transAxes)

    # (d) Composite ratio
    ax = axes[1, 1]
    ax.pie(
        [n_composite, n_degraded - n_composite, n_clean],
        labels=[f"Composite\n{comp_ratio:.1%}", f"Single\n{(n_degraded-n_composite)/args.n:.1%}",
                f"Clean\n{n_clean/args.n:.1%}"],
        colors=["#FF5722", "#2196F3", "#9E9E9E"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("(d) Composite vs Single vs Clean")

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved: {args.output}")
    print(f"  Clean: {n_clean/args.n:.3f} | Composite: {comp_ratio:.3f} | Heavy JPEG: {heavy_ratio:.3f}")


if __name__ == "__main__":
    main()
