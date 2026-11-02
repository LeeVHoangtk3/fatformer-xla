"""
tests/test_selection_metric.py — Task 7.10 Definition of Done
Kiểm thử 5 kịch bản bắt buộc:
  T1: Mảng giả 4 nhóm đã biết -> worst đúng giá trị min
  T2: Nhóm rỗng -> raise ValueError (không được trả 0)
  T3: worst tăng nhưng clean_r = 0.85 -> is_best=False (best_effort)
  T4: worst tăng và clean_r = 0.92 -> is_best=True (best.pth)
  T5: Loader đơn (không dict) -> không crash, fallback an toàn
"""
import pytest
import numpy as np
from src.training.metrics import group_recalls, worst_group_score


def test_t1_worst_group_correct_min():
    """T1: Mảng giả 4 nhóm đã biết -> worst đúng min, không làm tròn."""
    # Condition: 0=clean, 1=q30. Target: 0=real, 1=fake.
    # Nhóm clean_r (cond=0, y=0): 10 mẫu, 9 đúng -> 0.90
    # Nhóm clean_f (cond=0, y=1): 10 mẫu, 10 đúng -> 1.00
    # Nhóm q30_r (cond=1, y=0): 10 mẫu, 7 đúng -> 0.70
    # Nhóm q30_f (cond=1, y=1): 10 mẫu, 8 đúng -> 0.80
    y_true = np.array([0]*10 + [1]*10 + [0]*10 + [1]*10)
    cond = np.array([0]*20 + [1]*20)
    
    y_pred = np.array(
        [0]*9 + [1]*1 +    # clean_r: 9/10
        [1]*10 +           # clean_f: 10/10
        [0]*7 + [1]*3 +    # q30_r: 7/10
        [1]*8 + [0]*2      # q30_f: 8/10
    )

    recalls = group_recalls(y_true, y_pred, cond)
    assert np.isclose(recalls["clean_r"], 0.90)
    assert np.isclose(recalls["clean_f"], 1.00)
    assert np.isclose(recalls["q30_r"], 0.70)
    assert np.isclose(recalls["q30_f"], 0.80)

    worst = worst_group_score(recalls)
    assert np.isclose(worst, 0.70)


def test_t2_empty_group_raises_error():
    """T2: Nhóm rỗng -> raise ValueError (nan_policy='raise'), không trả 0."""
    # Thiếu nhóm q30_f (cond=1, y=1)
    y_true = np.array([0]*10 + [1]*10 + [0]*10)
    cond = np.array([0]*20 + [1]*10)
    y_pred = y_true.copy()

    recalls = group_recalls(y_true, y_pred, cond)
    assert np.isnan(recalls["q30_f"])

    with pytest.raises(ValueError, match="Dữ liệu đánh giá có nhóm rỗng"):
        worst_group_score(recalls, nan_policy="raise")


def test_t3_worst_improved_but_real_low_triggers_best_effort():
    """T3: worst tăng nhưng clean_r = 0.85 -> passes_real_guard=False (best_effort)."""
    recalls = {
        "clean_r": 0.85,
        "clean_f": 0.98,
        "q30_r": 0.75,
        "q30_f": 0.82
    }
    worst = worst_group_score(recalls)
    assert np.isclose(worst, 0.75)

    passes_real_guard = (recalls["clean_r"] >= 0.90)
    assert passes_real_guard is False


def test_t4_worst_improved_and_real_high_triggers_best_official():
    """T4: worst tăng và clean_r = 0.92 -> passes_real_guard=True (best.pth)."""
    recalls = {
        "clean_r": 0.92,
        "clean_f": 0.95,
        "q30_r": 0.78,
        "q30_f": 0.85
    }
    worst = worst_group_score(recalls)
    assert np.isclose(worst, 0.78)

    passes_real_guard = (recalls["clean_r"] >= 0.90)
    assert passes_real_guard is True


def test_t5_single_loader_fallback_safe():
    """T5: Loader đơn không dict -> omit nan_policy tính min an toàn không crash."""
    val_r = 0.88
    val_f = 0.94
    recalls_dict = {"clean_r": val_r, "clean_f": val_f, "q30_r": val_r, "q30_f": val_f}
    worst = worst_group_score(recalls_dict, nan_policy="omit")
    assert np.isclose(worst, 0.88)

