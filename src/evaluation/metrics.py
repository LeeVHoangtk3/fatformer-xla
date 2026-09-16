import numpy as np
from typing import Dict, Tuple, Union

try:
    from sklearn.metrics import accuracy_score, average_precision_score
except ImportError:
    # Fallback thuần numpy nếu môi trường thiếu scikit-learn
    def accuracy_score(y_true, y_pred):
        return (np.array(y_true) == np.array(y_pred)).mean()

    def average_precision_score(y_true, y_score):
        # Tính xấp xỉ AP theo công thức cơ bản
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        order = np.argsort(-y_score)
        y_true = y_true[order]
        cumsum = np.cumsum(y_true)
        precision = cumsum / (np.arange(len(y_true)) + 1)
        recall = cumsum / max(1, np.sum(y_true))
        return np.sum(precision[y_true == 1]) / max(1, np.sum(y_true))


def compute_binary_metrics(
    y_true: Union[list, np.ndarray], 
    y_score: Union[list, np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Tính toán các chỉ số đánh giá tiêu chuẩn của bài báo FatFormer:
    - acc: Độ chính xác tổng hợp (Accuracy)
    - ap: Độ chính xác trung bình (Average Precision)
    - r_acc: Độ chính xác trên mẫu thật (Real Accuracy)
    - f_acc: Độ chính xác trên mẫu giả (Fake Accuracy)
    """
    y_true = np.array(y_true).flatten()
    y_score = np.array(y_score).flatten()
    y_pred = (y_score > threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    
    # Xử lý trường hợp nếu tập test chỉ có toàn ảnh thật hoặc toàn ảnh giả
    if len(np.unique(y_true)) > 1:
        ap = float(average_precision_score(y_true, y_score))
    else:
        ap = acc

    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)

    r_acc = float(accuracy_score(y_true[real_mask], y_pred[real_mask])) if np.sum(real_mask) > 0 else 0.0
    f_acc = float(accuracy_score(y_true[fake_mask], y_pred[fake_mask])) if np.sum(fake_mask) > 0 else 0.0

    return {
        "acc": acc * 100.0,
        "ap": ap * 100.0,
        "r_acc": r_acc * 100.0,
        "f_acc": f_acc * 100.0,
    }


def format_evaluation_summary(
    results_by_subset: Dict[str, Dict[str, float]]
) -> Tuple[str, Dict[str, float]]:
    """
    Định dạng bảng tổng kết kết quả đánh giá cho nhiều subset và tính giá trị trung bình (mean).
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Index':<6} {'Subset':<16} {'ACC (%)':<12} {'AP (%)':<12} {'Real ACC (%)':<15} {'Fake ACC (%)':<15}")
    lines.append("-" * 80)

    accs, aps, r_accs, f_accs = [], [], [], []

    for idx, (name, metrics) in enumerate(results_by_subset.items()):
        accs.append(metrics["acc"])
        aps.append(metrics["ap"])
        r_accs.append(metrics["r_acc"])
        f_accs.append(metrics["f_acc"])

        lines.append(
            f"({idx:<3})  {name:<16} {metrics['acc']:<12.2f} {metrics['ap']:<12.2f} "
            f"{metrics['r_acc']:<15.2f} {metrics['f_acc']:<15.2f}"
        )

    mean_metrics = {
        "acc": float(np.mean(accs)) if accs else 0.0,
        "ap": float(np.mean(aps)) if aps else 0.0,
        "r_acc": float(np.mean(r_accs)) if r_accs else 0.0,
        "f_acc": float(np.mean(f_accs)) if f_accs else 0.0,
    }

    lines.append("-" * 80)
    lines.append(
        f"{'MEAN':<23} {mean_metrics['acc']:<12.2f} {mean_metrics['ap']:<12.2f} "
        f"{mean_metrics['r_acc']:<15.2f} {mean_metrics['f_acc']:<15.2f}"
    )
    lines.append("=" * 80)

    return "\n".join(lines), mean_metrics
