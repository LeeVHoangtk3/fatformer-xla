from typing import Optional

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


class DualStreamCELoss(nn.Module):
    """
    Hàm mất mát Dual-Stream Cross-Entropy với Label Smoothing (Task Finetune - Phương Án A).
    L_total = CE(logits_visual + logits_align, targets) + 0.5 * (CE(logits_visual, targets) + CE(logits_align, targets))
    - Bảo toàn gradient cho cả hai nhánh Visual (FAA/SRM) và Alignment (LGA).
    - Tránh hiện tượng triệt tiêu logit của Focal Loss khi số lượng mẫu dữ liệu còn hạn chế.
    - Label Smoothing 0.1 chống over-fitting và over-confidence trên miền dữ liệu nhỏ.
    """
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits_visual: torch.Tensor,
        logits_align: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits_visual: Logits từ nhánh FAA/SRM (Visual), shape (B, 2)
            logits_align:  Logits từ nhánh LGA (Alignment), shape (B, 2)
            targets:       Nhãn ground-truth, shape (B,), {0=real, 1=fake}
        """
        loss_combined = self.criterion(logits_visual + logits_align, targets)
        loss_v = self.criterion(logits_visual, targets)
        loss_a = self.criterion(logits_align, targets)
        return loss_combined + 0.5 * (loss_v + loss_a)


# ============================================================
# MILESTONE 6 — Task 6.3-B (Thành viên B)
# CrossDomainDistillationLoss + PairedConsistencyLoss
# ============================================================

class CrossDomainDistillationLoss(nn.Module):
    """
    Hàm chưng cất tri thức (Knowledge Distillation) đa miền — Task 6.3-B, Milestone 6.

    Ngăn chặn trôi dạt biểu diễn (representation drift) trên ảnh sạch bằng cách neo logit
    student vào logit của checkpoint baseline gốc (teacher đóng băng hoàn toàn).

    Công thức:
        L_KD = τ² · KL( Softmax(z_student / τ) ‖ Softmax(z_teacher / τ) )

    Trong đó:
        - z_student: logit của mô hình đang huấn luyện (B, C)
        - z_teacher: logit từ teacher model đã đóng băng (B, C), không tính gradient
        - τ (temperature): làm mịn phân phối xác suất, mặc định τ = 2.0
        - α_kd: trọng số hàm mất mát, mặc định 0.4

    Chỉ áp dụng trên batch con ảnh sạch (clean_mask) để bảo vệ độ chính xác lab.
    """

    def __init__(self, temperature: float = 2.0, alpha_kd: float = 0.4):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature phải > 0, nhận được {temperature}")
        if not (0.0 <= alpha_kd <= 1.0):
            raise ValueError(f"alpha_kd phải ∈ [0, 1], nhận được {alpha_kd}")
        self.temperature = temperature
        self.alpha_kd = alpha_kd
        # KLDivLoss mong đợi log-prob ở input, prob ở target — reduction='batchmean' chuẩn hoá theo batch
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        clean_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Class-Balanced Selective Teacher-Correct Knowledge Distillation:
        - Chỉ chưng cất trên các mẫu mà Teacher dự đoán ĐÚNG với nhãn thật `labels`.
        - Cân bằng 50/50 trọng số KL giữa nhóm Real đúng và nhóm Fake đúng để chống thiên lệch lớp Real.
        """
        if self.alpha_kd == 0.0:
            return student_logits.sum() * 0.0

        tau = self.temperature
        student_log_prob = F.log_softmax(student_logits / tau, dim=-1)  # (B, C)
        teacher_prob = F.softmax(teacher_logits.detach() / tau, dim=-1).to(student_log_prob.dtype)  # (B, C)

        # Tính KL divergence cho từng mẫu trong batch: shape (B,)
        kl_per_sample = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)

        if labels is not None:
            teacher_preds = teacher_logits.detach().argmax(dim=-1)
            correct_mask = (teacher_preds == labels)
            if clean_mask is not None:
                correct_mask = correct_mask & clean_mask.bool()

            real_correct_mask = correct_mask & (labels == 0)
            fake_correct_mask = correct_mask & (labels == 1)

            has_real = bool(real_correct_mask.any().item())
            has_fake = bool(fake_correct_mask.any().item())

            if has_real and has_fake:
                kl_real = kl_per_sample[real_correct_mask].mean()
                kl_fake = kl_per_sample[fake_correct_mask].mean()
                kl = 0.5 * (kl_real + kl_fake)
            elif has_real:
                # Giảm chấn lực kéo một chiều nếu trong batch không có mẫu Fake nào Teacher đoán đúng
                kl = 0.25 * kl_per_sample[real_correct_mask].mean()
            elif has_fake:
                kl = kl_per_sample[fake_correct_mask].mean()
            else:
                return student_logits.sum() * 0.0
        else:
            if clean_mask is not None:
                kl_per_sample = kl_per_sample[clean_mask.bool()]
            if kl_per_sample.numel() == 0:
                return student_logits.sum() * 0.0
            kl = kl_per_sample.mean()

        return self.alpha_kd * (tau ** 2) * kl




class PairedConsistencyLoss(nn.Module):
    """
    Hàm mất mát đồng bộ logit cặp sạch-suy biến — Task 6.3-B, Milestone 6.

    Ràng buộc mô hình duy trì quyết định phân loại đồng đều xuyên suốt giữa ảnh sạch và
    ảnh bị suy biến vật lý (JPEG nén / Gaussian Blur), chống nhiễu loạn ma trận
    lượng tử hóa khối 8×8.

    Công thức:
        L_cons = (1/2) · ‖ z(x_clean) − z(x_degraded) ‖₂²

    Trong đó:
        - z(x_clean):    logit tổng hợp (visual + align) của ảnh sạch (B, C)
        - z(x_degraded): logit tổng hợp của ảnh suy biến tương ứng (B, C)
        - β_cons: trọng số mặc định 0.2
    """

    def __init__(self, beta_cons: float = 0.2):
        super().__init__()
        if beta_cons < 0:
            raise ValueError(f"beta_cons phải ≥ 0, nhận được {beta_cons}")
        self.beta_cons = beta_cons

    def forward(
        self,
        logits_clean: torch.Tensor,
        logits_degraded: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits_clean:    Logit tổng hợp ảnh sạch (B, C)
            logits_degraded: Logit tổng hợp ảnh suy biến (B, C) — cùng shape
        Returns:
            Scalar: β_cons · (1/2) · MSE(logits_clean, logits_degraded)
        """
        if logits_clean.shape != logits_degraded.shape:
            raise ValueError(
                f"Shape không khớp: logits_clean {logits_clean.shape} "
                f"vs logits_degraded {logits_degraded.shape}"
            )
        mse = F.mse_loss(logits_clean, logits_degraded.to(logits_clean.dtype), reduction="mean")
        return self.beta_cons * 0.5 * mse


class CrossDomainTotalLoss(nn.Module):
    """
    Hàm mất mát đa mục tiêu neo chặt Teacher (Anchored Recovery Loss) — Task 6.3-B & Task 6.3-R.

    Công thức chuẩn hóa Phase 6-R (Cân bằng & Selective Teacher-Correct):
        L_total = weight_ce · L_AugCE + α_kd · L_KD(Teacher-Correct) + β_cons · L_cons
                = 0.5 · L_AugCE + 0.5 · L_KD + 0.2 · L_cons

    Kết hợp:
        1. L_AugCE: CrossEntropyLoss với Label Smoothing 0.05 (trọng số 0.5)
        2. L_KD   : Chưng cất tri thức KL Selective Teacher-Correct (trọng số 0.5, T=2.0)
        3. L_cons : PairedConsistencyLoss đồng bộ logit clean <-> degraded (trọng số 0.2)
    """

    def __init__(
        self,
        weight_ce: float = 0.5,
        label_smoothing: float = 0.05,
        temperature: float = 2.0,
        alpha_kd: float = 0.5,
        beta_cons: float = 0.2,
    ):
        super().__init__()
        self.weight_ce = weight_ce
        self.ce_loss   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kd_loss   = CrossDomainDistillationLoss(temperature=temperature, alpha_kd=alpha_kd)
        self.cons_loss = PairedConsistencyLoss(beta_cons=beta_cons)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        logits_degraded: Optional[torch.Tensor] = None,
        clean_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            logits:           Logit student tổng hợp (B, C)
            targets:          Nhãn ground-truth (B,)
            teacher_logits:   Logit teacher đóng băng (B, C). None → bỏ qua L_KD.
            logits_degraded:  Logit ảnh suy biến (B, C). None → bỏ qua L_cons.
            clean_mask:       Boolean mask (B,) xác định mẫu sạch cho L_KD.
        Returns:
            dict với keys: 'loss_total', 'loss_ce', 'loss_kd', 'loss_cons'
        """
        loss_ce_raw = self.ce_loss(logits, targets)
        loss_ce     = self.weight_ce * loss_ce_raw
        loss_kd     = torch.tensor(0.0, device=logits.device)
        loss_cons   = torch.tensor(0.0, device=logits.device)

        if teacher_logits is not None and self.kd_loss.alpha_kd > 0.0:
            loss_kd = self.kd_loss(logits, teacher_logits, labels=targets, clean_mask=clean_mask)

        if logits_degraded is not None and self.cons_loss.beta_cons > 0.0:
            loss_cons = self.cons_loss(logits, logits_degraded)

        loss_total = loss_ce + loss_kd + loss_cons
        return {
            "loss_total": loss_total,
            "loss_ce":    loss_ce,
            "loss_kd":    loss_kd,
            "loss_cons":  loss_cons,
        }

