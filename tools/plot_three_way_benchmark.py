# -*- coding: utf-8 -*-
"""
FatFormer-XLA: Ve Bieu Do Doi Chieu 3 Ben (Three-Way Comparative Benchmark) (Task 6.6)
So sanh truc dien:
1. Baseline Goc CVPR 2024 (M1)
2. Mo hinh Finetune Plan A (Task 5.1 / M2)
3. Mo hinh Da Mien Toan Dien (Milestone 6 / M4 Fallback)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_CHART_PATH = os.path.join(PROJECT_ROOT, "DATASET", "logs", "task_6.6_three_way_comparison.png")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

    # 1. Subplot 1: Bar Chart so sanh 5 chi so cot loi
    categories = [
        "Clean Macro ACC\n(Tổng thể)",
        "Clean Fake ACC\n(Nhận diện Diffusion)",
        "Clean Real ACC\n(Bảo toàn mỏ neo)",
        "Clean ROC-AUC\n(Ranh giới phân tách)",
        "Độ Tin Cậy ECE\n(Càng thấp càng tốt)",
    ]

    m1_scores = [50.00, 0.00, 100.00, 51.86, 48.17]   # ECE scaled x100 de ve chung thang do %
    m2_scores = [89.26, 82.22, 96.30, 95.81, 5.56]
    m4_scores = [92.22, 89.31, 94.96, 96.85, 2.99]

    x = np.arange(len(categories))
    width = 0.25

    rects1 = ax1.bar(x - width, m1_scores, width, label="CVPR 2024 Baseline (M1)", color="#8c8c8c", edgecolor="#555555")
    rects2 = ax1.bar(x, m2_scores, width, label="Plan A Finetune (Task 5.1 / M2)", color="#3182bd", edgecolor="#08519c")
    rects3 = ax1.bar(x + width, m4_scores, width, label="FatFormer-XLA M4 (+Distill +Cons)", color="#e66101", edgecolor="#b30000")

    # In gia tri tren dinh cot
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax1.annotate(f"{h:.1f}%",
                         xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax1.set_title("Đối Chiếu Trực Diện 3 Bên: Baseline vs Plan A vs M4 Khóa Sổ", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Tỷ Lệ (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # 2. Subplot 2: Radar Chart bieu dien buoc nhay nang luc
    ax2.remove()
    ax2 = fig.add_subplot(1, 2, 2, polar=True)

    radar_cats = [
        "Clean Macro ACC",
        "Diffusion Fake ACC",
        "Clean ROC-AUC",
        "Clean AP",
        "Real Preservation",
        "Confidence (1-ECE)",
    ]
    N = len(radar_cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Normalize metrics to 0-100
    r_m1 = [50.00, 0.00, 51.86, 52.81, 100.00, (1 - 0.4817) * 100]
    r_m2 = [89.26, 82.22, 95.81, 96.28, 96.30, (1 - 0.0556) * 100]
    r_m4 = [92.22, 89.31, 96.85, 96.82, 94.96, (1 - 0.0299) * 100]

    r_m1 += r_m1[:1]
    r_m2 += r_m2[:1]
    r_m4 += r_m4[:1]

    ax2.plot(angles, r_m1, linewidth=1.5, linestyle=":", color="#8c8c8c", label="CVPR 2024 (M1)")
    ax2.plot(angles, r_m2, linewidth=1.8, linestyle="--", color="#3182bd", label="Plan A (M2)")
    ax2.plot(angles, r_m4, linewidth=2.8, linestyle="-", color="#e66101", label="M4 Khóa Sổ (Fallback)")
    ax2.fill(angles, r_m4, color="#e66101", alpha=0.12)

    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(radar_cats, fontsize=9.5, fontweight="bold")
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color="#666666", size=8)
    ax2.set_ylim(0, 105)
    ax2.set_title("Biểu Đồ Radar Bao Quát Năng Lực Sau Hiệu Chuẩn", fontsize=12, fontweight="bold", pad=18)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_CHART_PATH), exist_ok=True)
    plt.savefig(OUT_CHART_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Da xuat bieu do doi chieu 3 ben ra: {OUT_CHART_PATH}")


if __name__ == "__main__":
    main()
