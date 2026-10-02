import torch
import torch.nn as nn
import torch.nn.functional as F


class FatFormerLoss(nn.Module):
    """
    Hàm mất mát gốc của FatFormer dựa theo Equation (10) trong bài báo:
    Sử dụng Cross-Entropy Loss trên tổng điểm tương đồng S(i) + S'(i).
    Hỗ trợ Label Smoothing để tăng tính tổng quát hóa và chống overfitting.
    Giữ nguyên để backward-compatible với pipeline eval Baseline (Task 2.2, 2.3).
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


class DualStreamFocalLoss(nn.Module):
    """
    Hàm mất mát thích ứng cho FatFormer-XLA (Task 3.3 - Thành viên B):

    Công thức: L_total = L_focal(S, y) + L_focal(S', y)
      - S:  logits từ nhánh Visual (FAA - Frequency-Aware Adapter)
      - S': logits từ nhánh Alignment (LGA - Language-Guided Alignment)
      - α = 0.25, γ = 2.0 (chuẩn theo Lin et al., 2017 RetinaNet Focal Loss)

    Cơ chế kháng nén: Focal Loss đè bẹp gradient của các mẫu dễ (p_t ≈ 0.99),
    tập trung 100% năng lượng kéo gradient vào các mẫu khó bị nén sâu (Q=30),
    giải quyết mất cân bằng cực đoan (Fake ACC = 5.78% tại Q=30).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Tính Focal Loss trên một luồng đầu ra.
        Args:
            logits:  Tensor (B, num_classes)
            targets: Tensor (B,) với giá trị {0, 1}
        Returns:
            Scalar loss value
        """
        # Tính Cross-Entropy theo từng mẫu để giữ lại xác suất p_t
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [B]
        # p_t = xác suất dự đoán đúng nhãn ground-truth
        p_t = torch.exp(-ce_loss)  # [B], ∈ (0, 1]
        # Trọng số điều chỉnh lớp: đè bẹp mẫu dễ, nhấn mạnh mẫu khó
        focal_weight = self.alpha * ((1.0 - p_t) ** self.gamma)  # [B]
        focal_loss = focal_weight * ce_loss  # [B]
        return focal_loss.mean()

    def forward(
        self,
        logits_visual: torch.Tensor,
        logits_align: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits_visual: Logits từ nhánh FAA (Visual), shape (B, 2)
            logits_align:  Logits từ nhánh LGA (Alignment), shape (B, 2)
            targets:       Nhãn ground-truth, shape (B,), giá trị {0=real, 1=fake}
        Returns:
            L_total = L_focal(S, y) + L_focal(S', y)  [scalar]
        """
        loss_visual = self._focal_loss(logits_visual, targets)
        loss_align  = self._focal_loss(logits_align, targets)
        return loss_visual + loss_align

