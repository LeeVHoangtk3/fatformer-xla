# -*- coding: utf-8 -*-
"""
FatFormer-XLA: Ve Bieu Do Truc Quan Hoa Ma Tran Ablation Study (Task 6.5)
Grouped Bar Chart & Radar Chart so sanh 5 cau hinh M1 - M5
Tuan thu nghiem ngat: Bieu dien khach quan, lam ro trang thai Fallback M4.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "DATASET", "logs", "crossdomain_ablation_matrix.csv")
OUT_CHART_PATH = os.path.join(PROJECT_ROOT, "DATASET", "logs", "crossdomain_ablation_chart.png")


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Khong tim thay tap tin ma tran: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Thiet lap giao dien
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 7), dpi=300)

    # 1. Subplot 1: Grouped Bar Chart
    ax1 = fig.add_subplot(1, 2, 1)

    branches = df["branch"].tolist()
    x = np.arange(len(branches))
    width = 0.2

    # 4 chi so cot loi
    bars1 = ax1.bar(x - 1.5 * width, df["clean_macro_acc"], width, label="Clean Macro ACC", color="#2b5c8f")
    bars2 = ax1.bar(x - 0.5 * width, df["clean_fake_acc"], width, label="Clean Fake ACC", color="#4192d9")
    bars3 = ax1.bar(x + 0.5 * width, df["q30_macro_acc"], width, label="Q30 Macro ACC", color="#d95f02")
    bars4 = ax1.bar(x + 1.5 * width, df["q30_fake_acc"], width, label="Q30 Fake ACC", color="#fd8d3c")

    # Mua moc san baseline tham chieu
    ax1.axhline(73.25, color="#2b5c8f", linestyle="--", linewidth=1.2, alpha=0.7, label="Base Clean Macro (73.25%)")
    ax1.axhline(52.20, color="#d95f02", linestyle="--", linewidth=1.2, alpha=0.7, label="Base Q30 Macro (52.20%)")
    ax1.axhline(15.20, color="#e41a1c", linestyle=":", linewidth=1.5, alpha=0.9, label="Gate Q30 Fake Min (15.20%)")

    # Gia tri tren cot M4
    idx_m4 = branches.index("M4")
    ax1.annotate(
        "M4 (Fallback)\nClean: 90.19%\nQ30: 49.26%",
        xy=(idx_m4, df.loc[df["branch"] == "M4", "clean_macro_acc"].values[0]),
        xytext=(idx_m4 - 0.2, 95),
        arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#1b7837",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="#2ca02c", facecolor="#f7fcf5", alpha=0.9),
    )

    ax1.set_title("So Sánh Phân Tầng Hiệu Năng 5 Cấu Hình (Clean vs Q30)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b}\n({df.loc[df['branch']==b, 'clean_macro_acc'].values[0]:.1f}%)" for b in branches], fontweight="bold")
    ax1.set_ylabel("Độ Chính Xác (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # 2. Subplot 2: Radar Chart da chieu
    ax2 = fig.add_subplot(1, 2, 2, polar=True)

    radar_features = [
        ("Clean Macro", "clean_macro_acc"),
        ("Clean Fake", "clean_fake_acc"),
        ("Clean AUC", "clean_auc"),
        ("Q30 Macro", "q30_macro_acc"),
        ("Q30 AUC", "q30_auc"),
    ]
    categories = [f[0] for f in radar_features]
    col_keys = [f[1] for f in radar_features]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    colors_map = {
        "M1": "#999999",
        "M2": "#386cb0",
        "M3": "#7fc97f",
        "M4": "#e66101",
        "M5": "#984ea3",
    }
    line_styles = {
        "M1": ":",
        "M2": "--",
        "M3": "-.",
        "M4": "-",
        "M5": "--",
    }
    line_widths = {
        "M1": 1.2,
        "M2": 1.5,
        "M3": 1.5,
        "M4": 2.5,
        "M5": 1.5,
    }

    for b in branches:
        row = df[df["branch"] == b].iloc[0]
        values = [row[c] for c in col_keys]
        values += values[:1]
        label_text = f"{b} (Fallback)" if b == "M4" else b
        ax2.plot(angles, values, linewidth=line_widths[b], linestyle=line_styles[b], color=colors_map[b], label=label_text)
        if b in ["M3", "M4"]:
            ax2.fill(angles, values, color=colors_map[b], alpha=0.08)

    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=9.5, fontweight="bold")
    ax2.set_rlabel_position(0)
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color="#666666", size=8)
    ax2.set_ylim(0, 105)
    ax2.set_title("Biểu Đồ Radar Đa Chiều Năng Lực 5 Nhánh M1–M5", fontsize=12, fontweight="bold", pad=18)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_CHART_PATH), exist_ok=True)
    plt.savefig(OUT_CHART_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Da tao thanh cong bieu do truc quan hoa Ablation Study: {OUT_CHART_PATH}")


if __name__ == "__main__":
    main()
