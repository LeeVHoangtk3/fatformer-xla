from typing import Optional
import math

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
        if logits_clean.shape != logits_degraded.shape:
            raise ValueError(
                f"Shape không khớp: logits_clean {logits_clean.shape} "
                f"vs logits_degraded {logits_degraded.shape}"
            )
        # BẮT BUỘC detach() logits_clean: Chỉ ép nhánh suy biến học theo nhánh sạch,
        # tuyệt đối không để nhánh suy biến kéo lệch logit của nhánh ảnh sạch!
        target_clean = logits_clean.detach().to(logits_degraded.dtype)
        mse = F.mse_loss(logits_degraded, target_clean, reduction="mean")
        return self.beta_cons * 0.5 * mse


# ============================================================
# PHASE 7 — Task 7.9 (Thành viên B)
# RealGuardLoss + FeatureConsistencyLoss
# ============================================================

class RealGuardLoss(nn.Module):
    """Chỉ phạt khi student vượt teacher trên logit Fake của mẫu Real.

    Dùng ReLU để chỉ tạo gradient khi z_s > z_t — không kéo logit xuống vô hạn.
    Nếu batch không có mẫu Real: trả 0 kèm guard_active=False, tuyệt đối không NaN.
    """
    def __init__(self, fake_index: int = 1):
        super().__init__()
        if type(fake_index) is not int or fake_index < 0:
            raise ValueError("fake_index must be a nonnegative integer")
        self.fake_index = fake_index

    def forward(self, logits_student_deg, logits_teacher_clean, is_real):
        if logits_student_deg.ndim != 2 or logits_student_deg.shape != logits_teacher_clean.shape:
            raise ValueError("Guard requires matching [batch, classes] logits")
        if self.fake_index >= logits_student_deg.shape[1]:
            raise ValueError("fake_index is outside the logits class dimension")
        if is_real.dtype != torch.bool or is_real.shape != logits_student_deg.shape[:1]:
            raise ValueError("is_real must be a boolean mask of shape [batch]")
        # AMP inputs must not square/reduce in FP16 (overflow even at z=400).
        student = logits_student_deg.float() if logits_student_deg.dtype in (torch.float16, torch.bfloat16) else logits_student_deg
        teacher = logits_teacher_clean.detach().to(student.dtype)
        if is_real.sum() == 0:
            return student.sum() * 0.0, {"guard_active": False, "n_real": 0,
                                        "guard_excess_mean": student.new_zeros(())}
        z_s = student[is_real, self.fake_index]
        z_t = teacher[is_real, self.fake_index]
        excess = torch.relu(z_s - z_t)
        loss = (excess ** 2).mean()
        return loss, {"guard_active": True, "n_real": int(is_real.sum()),
                      "guard_excess_mean": excess.mean().detach()}


class FeatureConsistencyLoss(nn.Module):
    """Đồng nhất biểu diễn clean↔degraded bằng cosine + KL divergence.

    stop-gradient trên nhánh clean để nhánh deg học theo clean, không kéo ngược.
    Nhân T² theo chuẩn Hinton để bù KL khi T > 1.
    """
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and > 0")
        self.T = temperature

    def forward(self, f_deg, f_clean, logits_deg, logits_clean):
        if f_deg.ndim != 2 or f_deg.shape != f_clean.shape:
            raise ValueError("Features must have matching [batch, features] shapes")
        if logits_deg.shape != logits_clean.shape or logits_deg.shape[0] != f_deg.shape[0]:
            raise ValueError("Feature and logit batch shapes must match")
        # eps=1e-8 underflows in FP16. Perform new losses in FP32 under AMP.
        f_deg = f_deg.float() if f_deg.dtype in (torch.float16, torch.bfloat16) else f_deg
        f_clean = f_clean.detach().to(f_deg.dtype)
        logits_deg = logits_deg.float() if logits_deg.dtype in (torch.float16, torch.bfloat16) else logits_deg
        cos = F.cosine_similarity(f_deg, f_clean, dim=-1, eps=1e-8)
        loss_cos = (1.0 - cos).mean()
        p_clean  = F.softmax(logits_clean.detach().to(logits_deg.dtype) / self.T, dim=-1)
        log_p_deg = F.log_softmax(logits_deg / self.T, dim=-1)
        loss_kl  = F.kl_div(log_p_deg, p_clean, reduction="batchmean") * (self.T ** 2)
        return loss_cos + loss_kl, {"feat_cos": cos.mean().detach(), "feat_kl": loss_kl.detach()}


class CrossDomainTotalLoss(nn.Module):
    """
    Hàm mất mát đa mục tiêu neo chặt Teacher (Anchored Recovery Loss) — Task 6.3-B & Task 6.3-R.

    Công thức chuẩn hóa Phase 6-R (Cân bằng & Selective Teacher-Correct):
        L_total = weight_ce · L_AugCE + α_kd · L_KD(Teacher-Correct) + β_cons · L_cons
                = 0.7 · [0.7 · CE(clean) + 0.3 · CE(degraded)] + 0.3 · L_KD + 0.2 · L_cons

    Kết hợp:
        1. L_AugCE: CrossEntropyLoss với Label Smoothing 0.05 (trọng số 0.7, 70% clean + 30% degraded)
        2. L_KD   : Chưng cất tri thức KL Selective Teacher-Correct (trọng số 0.3, T=2.0)
        3. L_cons : PairedConsistencyLoss đồng bộ logit clean (detached) <-> degraded (trọng số 0.2)
    """

    def __init__(
        self,
        weight_ce: float = 0.7,
        label_smoothing: float = 0.05,
        temperature: float = 2.0,
        alpha_kd: float = 0.3,
        beta_cons: float = 0.2,
        fake_index: int = 1,
        weight_ce_deg: float = 0.3,
        loss_mode: str = "auto",
    ):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_ce_deg = weight_ce_deg
        if loss_mode not in ("auto", "legacy", "v2"):
            raise ValueError("loss_mode must be auto, legacy or v2")
        self.loss_mode = loss_mode
        for name, value in (("weight_ce", weight_ce), ("weight_ce_deg", weight_ce_deg)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        self.ce_loss   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kd_loss   = CrossDomainDistillationLoss(temperature=temperature, alpha_kd=alpha_kd)
        self.cons_loss = PairedConsistencyLoss(beta_cons=beta_cons)
        # Task 7.9: mặc định weight=0 → không ảnh hưởng run cũ
        self.feat_loss  = FeatureConsistencyLoss(temperature=temperature)
        self.guard_loss = RealGuardLoss(fake_index=fake_index)

    def forward(
        self,
        logits: Optional[torch.Tensor] = None,  # legacy name
        targets: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        logits_degraded: Optional[torch.Tensor] = None,
        clean_mask: Optional[torch.Tensor] = None,
        features_clean: Optional[torch.Tensor] = None,   # Task 7.9: aug_image_features nhánh clean
        features_deg: Optional[torch.Tensor] = None,     # Task 7.9: aug_image_features nhánh deg
        weight_feat: float = 0.0,
        weight_guard: float = 0.0,
        logits_clean: Optional[torch.Tensor] = None,
        logits_deg: Optional[torch.Tensor] = None,
    ) -> dict:
        if logits_clean is not None:
            if logits is not None:
                raise ValueError("Use either logits or logits_clean, not both")
            logits = logits_clean
        if logits_deg is not None:
            if logits_degraded is not None:
                raise ValueError("Use either logits_degraded or logits_deg, not both")
            logits_degraded = logits_deg
        if logits is None or targets is None or targets.numel() == 0:
            raise ValueError("Nonempty logits and targets are required")
        # lean: legacy_mode giữ công thức cũ → allclose test (a) pass
        legacy_mode = self.loss_mode == "legacy" or (
            self.loss_mode == "auto" and weight_feat == 0.0 and weight_guard == 0.0)
        if self.loss_mode == "legacy" and (weight_feat or weight_guard):
            raise ValueError("legacy loss_mode cannot enable v2 losses")
        for name, value in (("weight_feat", weight_feat), ("weight_guard", weight_guard)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if not legacy_mode and logits_degraded is None:
            raise ValueError("Loss v2 requires paired clean/degraded logits")
        if weight_feat > 0 and (features_clean is None or features_deg is None):
            raise ValueError("weight_feat > 0 requires clean and degraded features")
        if weight_guard > 0 and teacher_logits is None:
            raise ValueError("weight_guard > 0 requires clean teacher logits")

        ce_clean = self.ce_loss(logits, targets)
        ce_deg   = self.ce_loss(logits_degraded, targets) if logits_degraded is not None else None

        if legacy_mode:
            ce_raw = (0.7 * ce_clean + 0.3 * ce_deg) if ce_deg is not None else ce_clean
            loss_ce = self.weight_ce * ce_raw
        else:
            loss_ce = self.weight_ce * ce_clean + self.weight_ce_deg * ce_deg

        zero = (logits if legacy_mode else logits.float()).sum() * 0.0
        loss_kd   = zero
        loss_cons = zero

        if teacher_logits is not None and self.kd_loss.alpha_kd > 0.0:
            loss_kd = self.kd_loss(logits, teacher_logits, labels=targets, clean_mask=clean_mask)

        if logits_degraded is not None and self.cons_loss.beta_cons > 0.0:
            loss_cons = self.cons_loss(logits, logits_degraded)

        loss_feat  = zero
        loss_guard = zero
        guard_active = False
        feat_info = {}
        g_info = {"n_real": 0}

        if weight_feat > 0.0 and features_clean is not None and features_deg is not None:
            lf_input = logits_degraded if logits_degraded is not None else logits
            loss_feat, feat_info = self.feat_loss(features_deg, features_clean, lf_input, logits)

        if weight_guard > 0.0 and teacher_logits is not None and logits_degraded is not None:
            loss_guard, g_info = self.guard_loss(logits_degraded, teacher_logits, is_real=(targets == 0))
            guard_active = g_info.get("guard_active", False)

        # kd_loss/cons_loss already apply alpha_kd/beta_cons. Never weight twice.
        loss_total = (loss_ce + loss_kd + loss_cons
                      + weight_feat * loss_feat + weight_guard * loss_guard)

        return {
            "loss_total":  loss_total,
            "loss_ce":     loss_ce,
            "loss_kd":     loss_kd,
            "loss_cons":   loss_cons,
            "loss_feat":   loss_feat,
            "loss_guard":  loss_guard,
            "guard_active": guard_active,
            "ce_clean": ce_clean,
            "ce_deg": ce_deg if ce_deg is not None else logits.new_zeros(()),
            "kd": loss_kd / self.kd_loss.alpha_kd if self.kd_loss.alpha_kd else loss_kd,
            "cons": loss_cons / self.cons_loss.beta_cons if self.cons_loss.beta_cons else loss_cons,
            "feat": loss_feat,
            "guard": loss_guard,
            "weighted_feat": weight_feat * loss_feat,
            "weighted_guard": weight_guard * loss_guard,
            **feat_info,
            **{k: v for k, v in g_info.items() if k != "guard_active"},
        }

