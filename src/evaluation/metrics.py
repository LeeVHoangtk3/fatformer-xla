import numpy as np
from typing import Dict, Tuple, Union

try:
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
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

    def roc_auc_score(y_true, y_score):
        # Tính xấp xỉ ROC-AUC bằng Mann-Whitney U statistic (thuần numpy)
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        # So sánh cặp
        diff = pos[:, None] - neg[None, :]
        return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


def compute_binary_metrics(
    y_true: Union[list, np.ndarray], 
    y_score: Union[list, np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Tính toán các chỉ số đánh giá tiêu chuẩn của bài báo FatFormer:
    - acc: Độ chính xác tổng hợp (Accuracy)
    - ap: Độ chính xác trung bình (Average Precision)
    - auc: Diện tích dưới đường cong ROC (ROC-AUC)
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
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = 0.5
    else:
        ap = acc
        auc = 0.5

    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)

    r_acc = float(accuracy_score(y_true[real_mask], y_pred[real_mask])) if np.sum(real_mask) > 0 else 0.0
    f_acc = float(accuracy_score(y_true[fake_mask], y_pred[fake_mask])) if np.sum(fake_mask) > 0 else 0.0

    num_samples = int(len(y_true))
    num_real = int(np.sum(real_mask))
    num_fake = int(np.sum(fake_mask))
    num_correct = int(np.sum(y_true == y_pred))
    num_real_correct = int(np.sum((y_true == 0) & (y_pred == 0)))
    num_fake_correct = int(np.sum((y_true == 1) & (y_pred == 1)))

    return {
        "acc": acc * 100.0,
        "ap": ap * 100.0,
        "auc": auc * 100.0,
        "r_acc": r_acc * 100.0,
        "f_acc": f_acc * 100.0,
        "num_samples": num_samples,
        "num_real": num_real,
        "num_fake": num_fake,
        "num_correct": num_correct,
        "num_real_correct": num_real_correct,
        "num_fake_correct": num_fake_correct,
    }


def format_evaluation_summary(
    results_by_subset: Dict[str, Dict[str, float]]
) -> Tuple[str, Dict[str, Any]]:
    """
    Định dạng bảng tổng kết kết quả đánh giá cho nhiều subset:
    Báo cáo song song cả Macro-Average (trung bình đồng đều) và Sample-Weighted Micro-Average (theo số lượng mẫu).
    """
    lines = []
    lines.append("=" * 86)
    lines.append(f"{'Index':<6} {'Subset':<16} {'ACC (%)':<10} {'AP (%)':<10} {'AUC (%)':<10} {'Real ACC (%)':<14} {'Fake ACC (%)':<14}")
    lines.append("-" * 86)

    accs, aps, aucs, r_accs, f_accs = [], [], [], [], []

    for idx, (name, metrics) in enumerate(results_by_subset.items()):
        accs.append(metrics["acc"])
        aps.append(metrics["ap"])
        aucs.append(metrics.get("auc", 0.0))
        r_accs.append(metrics["r_acc"])
        f_accs.append(metrics["f_acc"])

        lines.append(
            f"({idx:<3})  {name:<16} {metrics['acc']:<10.2f} {metrics['ap']:<10.2f} "
            f"{metrics.get('auc', 0.0):<10.2f} {metrics['r_acc']:<14.2f} {metrics['f_acc']:<14.2f}"
        )

    macro_metrics = {
        "acc": float(np.mean(accs)) if accs else 0.0,
        "ap": float(np.mean(aps)) if aps else 0.0,
        "auc": float(np.mean(aucs)) if aucs else 0.0,
        "r_acc": float(np.mean(r_accs)) if r_accs else 0.0,
        "f_acc": float(np.mean(f_accs)) if f_accs else 0.0,
    }

    # Tính Sample-Weighted (Micro-Average)
    total_samples = sum(m.get("num_samples", 0) for m in results_by_subset.values())
    total_correct = sum(m.get("num_correct", 0) for m in results_by_subset.values())
    total_real = sum(m.get("num_real", 0) for m in results_by_subset.values())
    total_real_correct = sum(m.get("num_real_correct", 0) for m in results_by_subset.values())
    total_fake = sum(m.get("num_fake", 0) for m in results_by_subset.values())
    total_fake_correct = sum(m.get("num_fake_correct", 0) for m in results_by_subset.values())

    if total_samples > 0:
        weighted_acc = (total_correct / total_samples) * 100.0
        weighted_r_acc = (total_real_correct / max(1, total_real)) * 100.0
        weighted_f_acc = (total_fake_correct / max(1, total_fake)) * 100.0
    else:
        weighted_acc = macro_metrics["acc"]
        weighted_r_acc = macro_metrics["r_acc"]
        weighted_f_acc = macro_metrics["f_acc"]

    weighted_metrics = {
        "acc": weighted_acc,
        "ap": macro_metrics["ap"],
        "auc": macro_metrics["auc"],
        "r_acc": weighted_r_acc,
        "f_acc": weighted_f_acc,
        "total_samples": total_samples,
    }

    lines.append("-" * 86)
    lines.append(
        f"{'MACRO-AVERAGE':<23} {macro_metrics['acc']:<10.2f} {macro_metrics['ap']:<10.2f} "
        f"{macro_metrics['auc']:<10.2f} {macro_metrics['r_acc']:<14.2f} {macro_metrics['f_acc']:<14.2f}"
    )
    lines.append(
        f"{'SAMPLE-WEIGHTED (MICRO)':<23} {weighted_metrics['acc']:<10.2f} {weighted_metrics['ap']:<10.2f} "
        f"{weighted_metrics['auc']:<10.2f} {weighted_metrics['r_acc']:<14.2f} {weighted_metrics['f_acc']:<14.2f}"
    )
    lines.append("=" * 86)

    combined_summary = {
        "macro_metrics": macro_metrics,
        "weighted_metrics": weighted_metrics,
        "mean_metrics": macro_metrics  # Tương thích ngược
    }
    return "\n".join(lines), combined_summary
