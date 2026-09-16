import torch
import torch.nn as nn
import torch.nn.functional as F


class FatFormerLoss(nn.Module):
    """
    Hàm mất mát cho FatFormer dựa theo Equation (10) trong bài báo:
    Sử dụng Cross-Entropy Loss trên tổng điểm tương đồng S(i) + S'(i).
    Hỗ trợ Label Smoothing để tăng tính tổng quát hóa và chống overfitting.
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor kích thước (B, 2) biểu thị [S_real + S'_real, S_fake + S'_fake].
            targets: Tensor nhãn ground truth kích thước (B,) với giá trị {0, 1}.
        """
        return self.criterion(logits, targets)
