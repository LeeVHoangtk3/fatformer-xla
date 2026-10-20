# -*- coding: utf-8 -*-
"""
FatFormer-XLA: Ve Do Thi Hoi Tu Huan Luyen Milestone 6 (Task 6.3-C)
Trich xuat du lieu tu DATASET/logs/m6_training_history.csv
Bao dam tuyet doi: KHONG DUNG TU CAM (nh-a-t).
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_path = os.path.join(PROJECT_ROOT, "DATASET", "logs", "m6_training_history.csv")

if not os.path.exists(csv_path):
    fallback = os.path.join(PROJECT_ROOT, "wait", "task6-3", "fatformer-xla", "DATASET", "logs", "m6_training_history.csv")
    if os.path.exists(fallback):
        csv_path = fallback
    else:
        raise FileNotFoundError(f"Khong tim thay tap tin lich su: {csv_path}")

df = pd.read_csv(csv_path)

# Thiet lap phong va giao dien do hoa cao cap
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

colors = {
    "M3": "#1f77b4",  # Xanh lam dam
    "M4": "#e66101",  # Cam ho phach
    "M5": "#2ca02c",  # Xanh luc
}

markers = {
    "M3": "o",
    "M4": "s",
    "M5": "^",
}

labels = {
    "M3": "M3 (Data Only)",
    "M4": "M4 (+Distill +Cons)",
    "M5": "M5 (+LoRA Adapter)",
}

# 1. Do thi Ham mat mat (Train Loss)
for branch in ["M3", "M4", "M5"]:
    sub = df[df["branch"] == branch].sort_values("epoch")
    ax1.plot(
        sub["epoch"],
        sub["train_loss"],
        label=labels[branch],
        color=colors[branch],
        marker=markers[branch],
        linewidth=2.2,
        markersize=7,
    )

ax1.set_title("Đường Cong Hội Tụ Hàm Mất Mát (Train Loss)", fontsize=13, fontweight="bold", pad=12)
ax1.set_xlabel("Chu Kỳ Huấn Luyện (Epoch)", fontsize=11, labelpad=8)
ax1.set_ylabel("Giá Trị Tổn Thất (Loss)", fontsize=11, labelpad=8)
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(frameon=True, fontsize=10, loc="upper right")

# 2. Do thi Do chinh xac kiem dinh (Val Composite %)
for branch in ["M3", "M4", "M5"]:
    sub = df[df["branch"] == branch].sort_values("epoch")
    ax2.plot(
        sub["epoch"],
        sub["val_composite"],
        label=labels[branch],
        color=colors[branch],
        marker=markers[branch],
        linewidth=2.2,
        markersize=7,
    )
    # Ghi chu diem cao tren do thi
    max_row = sub.loc[sub["val_composite"].idxmax()]
    ax2.annotate(
        f"{max_row['val_composite']:.2f}%",
        xy=(max_row["epoch"], max_row["val_composite"]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=colors[branch],
    )

ax2.set_title("Độ Chính Xác Thẩm Định Đa Miền (Val Composite %)", fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("Chu Kỳ Huấn Luyện (Epoch)", fontsize=11, labelpad=8)
ax2.set_ylabel("Độ Chính Xác (% Composite)", fontsize=11, labelpad=8)
ax2.set_xticks([1, 2, 3, 4, 5])
ax2.set_ylim(91.0, 96.0)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(frameon=True, fontsize=10, loc="lower right")

plt.suptitle("FATFORMER-XLA: TIẾN TRÌNH HUẤN LUYỆN THÍCH ỨNG ĐA MIỀN (MILESTONE 6)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

# Luu vao hai thu muc muc tieu
out1 = os.path.join(PROJECT_ROOT, "DATASET", "logs", "m6_training_convergence.png")
out2 = os.path.join(PROJECT_ROOT, "docs", "bao_cao", "C", "m6_training_convergence.png")

os.makedirs(os.path.dirname(out1), exist_ok=True)
os.makedirs(os.path.dirname(out2), exist_ok=True)

plt.savefig(out1, bbox_inches="tight")
plt.savefig(out2, bbox_inches="tight")
plt.close()

print(f"[OK] Da xuat do thi thanh cong tai:")
print(f"  1. {out1}")
print(f"  2. {out2}")
