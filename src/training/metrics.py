"""
src/training/metrics.py — Selection Metric v2 (Task 7.10)
Định nghĩa:
  - group_recalls(y_true, y_pred, condition): tính recall cho 4 nhóm {clean, q30} x {real, fake}
  - worst_group_score(recalls, nan_policy="raise"): trả min recall của 4 nhóm, fail-loud nếu có nhóm rỗng
"""
from typing import Dict, Union
import numpy as np


def group_recalls(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    condition: Union[np.ndarray, list]
) -> Dict[str, float]:
    """
    Tính recall riêng biệt cho 4 nhóm:
      - condition: {0: clean, 1: q30}
      - y: {0: real, 1: fake}

    Trả về dict:
      {"clean_r": float, "clean_f": float, "q30_r": float, "q30_f": float}
    Nếu một nhóm không có mẫu nào, trả np.nan (không trả 0 để tránh che giấu dữ liệu lỗi).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    cond_arr = np.asarray(condition)
    if any(a.ndim != 1 for a in (y_true_arr, y_pred_arr, cond_arr)):
        raise ValueError("Labels, predictions and conditions must be one-dimensional")
    if any(not np.isin(a, [0, 1]).all() for a in (y_true_arr, y_pred_arr, cond_arr)):
        raise ValueError("Labels, predictions and conditions must be binary")

    if not (len(y_true_arr) == len(y_pred_arr) == len(cond_arr)):
        raise ValueError("y_true, y_pred và condition phải có cùng độ dài.")

    recalls = {}
    groups = [
        ("clean_r", 0, 0),
        ("clean_f", 0, 1),
        ("q30_r", 1, 0),
        ("q30_f", 1, 1),
    ]

    for name, cond_val, label_val in groups:
        mask = (cond_arr == cond_val) & (y_true_arr == label_val)
        total = int(np.sum(mask))
        if total == 0:
            recalls[name] = float(np.nan)
        else:
            correct = int(np.sum((y_pred_arr[mask] == label_val)))
            recalls[name] = float(correct / total)

    return recalls


def worst_group_score(recalls: Dict[str, float], nan_policy: str = "raise") -> float:
    """
    Lấy giá trị nhỏ nhất (min) trong 4 nhóm recall.
    - nan_policy:
        "raise": ném ValueError nếu bất kỳ nhóm nào có giá trị np.nan (nhóm thiếu dữ liệu).
        "omit": bỏ qua np.nan khi tính min.
    """
    required_keys = ["clean_r", "clean_f", "q30_r", "q30_f"]
    if nan_policy not in ("raise", "omit"):
        raise ValueError("nan_policy must be 'raise' or 'omit'")
    for k in required_keys:
        if k not in recalls:
            raise KeyError(f"Thiếu chỉ số '{k}' trong recalls dict.")

    vals = [recalls[k] for k in required_keys]
    if any(np.isinf(v) or (np.isfinite(v) and not 0 <= v <= 1) for v in vals):
        raise ValueError("Recalls must be finite values in [0, 1], or NaN for an empty group")

    if any(np.isnan(v) for v in vals):
        if nan_policy == "raise":
            missing = [k for k in required_keys if np.isnan(recalls[k])]
            raise ValueError(f"Dữ liệu đánh giá có nhóm rỗng (nan): {missing}. Kiểm tra lại validation split.")
        elif nan_policy == "omit":
            valid_vals = [v for v in vals if not np.isnan(v)]
            if not valid_vals:
                return float(np.nan)
            return float(np.min(valid_vals))
        else:
            raise ValueError(f"nan_policy '{nan_policy}' không hợp lệ. Chọn 'raise' hoặc 'omit'.")

    return float(np.min(vals))

