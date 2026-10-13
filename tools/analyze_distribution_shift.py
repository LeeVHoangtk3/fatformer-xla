"""
Script phan tich distribution shift va phan cum loi ky thuat tren 18 bo sinh.
Du lieu dau vao: DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv
Phuc vu nghiem thu Task 6.1 (Milestone 6).
"""

import os
import pandas as pd
import numpy as np

def run_analysis():
    csv_path = os.path.join("DATASET", "logs", "m4_full_benchmark", "final_full_benchmark_matrix.csv")
    if not os.path.exists(csv_path):
        print(f"Loi: Khong tim thay tep {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Loc rieng du lieu bien the clean cho 18 bo sinh
    excluded_subsets = ["MEAN_GANS", "POOLED_GANS", "MEAN_DIFFUSION", "POOLED_DIFFUSION", "MEAN_OVERALL", "POOLED_OVERALL"]
    df_clean = df[(df["variant"] == "clean") & (~df["subset"].isin(excluded_subsets))].copy()

    # Ep kieu du lieu so
    numeric_cols = ["acc", "ap", "auc", "r_acc", "f_acc"]
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Tinh toan do lech r_acc va f_acc (Real - Fake Bias)
    df_clean["rf_bias"] = df_clean["r_acc"] - df_clean["f_acc"]

    # Lay them thong so bien the jpeg_q30 de do do nhay nen
    df_q30 = df[(df["variant"] == "jpeg_q30") & (~df["subset"].isin(excluded_subsets))].copy()
    for col in ["acc", "ap", "auc", "r_acc", "f_acc", "delta_drop"]:
        df_q30[col] = pd.to_numeric(df_q30[col], errors="coerce")

    q30_map = df_q30.set_index("subset")[["acc", "f_acc", "delta_drop"]].to_dict("index")

    df_clean["q30_acc"] = df_clean["subset"].map(lambda s: q30_map.get(s, {}).get("acc", np.nan))
    df_clean["q30_f_acc"] = df_clean["subset"].map(lambda s: q30_map.get(s, {}).get("f_acc", np.nan))
    df_clean["q30_drop"] = df_clean["subset"].map(lambda s: q30_map.get(s, {}).get("delta_drop", np.nan))

    print("=" * 80)
    print("KET QUA PHAN TICH CHI TIET 18 BO SINH (TASK 6.1)")
    print("=" * 80)
    print(df_clean[["group", "subset", "acc", "ap", "auc", "r_acc", "f_acc", "rf_bias", "q30_acc", "q30_f_acc"]].to_string(index=False))

    # Phan cum loi ky thuat dua tren tieu chi ro rang:
    # 1. Cum Mu Dac Trung (Feature Blind Spot): AP < 60% va AUC < 60%
    # 2. Cum Lech Diem Cat (Threshold Misalignment): AP >= 70% va AUC >= 70% nhung f_acc < 65% (rf_bias cao)
    # 3. Cum Loi Ben Vung (Robust Core): AP >= 85% va clean ACC >= 80% va f_acc >= 65%
    # 4. Cum Do Dang / Trung Gian: Cac truong hop con lai va nhay cam nen sau

    blind_spot = []
    misalignment = []
    robust_core = []
    compression_sensitive = []

    for _, row in df_clean.iterrows():
        s = row["subset"]
        ap = row["ap"]
        auc = row["auc"]
        acc = row["acc"]
        r_acc = row["r_acc"]
        f_acc = row["f_acc"]
        q30_f = row["q30_f_acc"]
        bias = row["rf_bias"]

        # Kiem tra nhay cam nen sau truoc neu f_acc bi sup do nghiem trong duoi q30
        if f_acc >= 30.0 and q30_f <= 25.0:
            compression_sensitive.append((s, row["group"], f_acc, q30_f, f_acc - q30_f))

        if ap < 60.0 and auc < 60.0:
            blind_spot.append((s, row["group"], ap, auc, acc, f_acc))
        elif (ap >= 70.0 or auc >= 70.0) and f_acc < 65.0:
            misalignment.append((s, row["group"], ap, auc, acc, r_acc, f_acc, bias))
        elif ap >= 85.0 and acc >= 80.0 and f_acc >= 65.0:
            robust_core.append((s, row["group"], ap, auc, acc, r_acc, f_acc))

    print("\n" + "=" * 80)
    print("PHAN TANG 4 CUM LOI KY THUAT")
    print("=" * 80)

    print("\n1. CUM MU DAC TRUNG (Feature Blind Spot - Khong nhan dien duoc dau vet sinh):")
    for item in blind_spot:
        print(f" - {item[0]} ({item[1]}): AP={item[2]:.2f}%, AUC={item[3]:.2f}%, ACC={item[4]:.2f}%, Fake_ACC={item[5]:.2f}%")

    print("\n2. CUM LECH DIEM CAT PHAN LOP (Threshold Misalignment - AP/AUC kha nhung f_acc thap do bias lech Real):")
    for item in misalignment:
        print(f" - {item[0]} ({item[1]}): AP={item[2]:.2f}%, AUC={item[3]:.2f}%, ACC={item[4]:.2f}%, Real_ACC={item[5]:.2f}%, Fake_ACC={item[6]:.2f}%, Bias={item[7]:.2f}%")

    print("\n3. CUM NHAY CAM NEN SAU (Compression Sensitivity - Fake ACC sup do duoi Q=30):")
    for item in compression_sensitive:
        print(f" - {item[0]} ({item[1]}): Fake_ACC Clean={item[2]:.2f}% -> Q30={item[3]:.2f}% (Giam={item[4]:.2f}%)")

    print("\n4. CUM LOI BEN VUNG (Robust Core - Mo hinh nam bat vung vang):")
    for item in robust_core:
        print(f" - {item[0]} ({item[1]}): AP={item[2]:.2f}%, AUC={item[3]:.2f}%, ACC={item[4]:.2f}%, Real_ACC={item[5]:.2f}%, Fake_ACC={item[6]:.2f}%")

if __name__ == "__main__":
    run_analysis()
