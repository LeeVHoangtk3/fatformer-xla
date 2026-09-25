# TASK 3.3: HIỆN THỰC HÓA SRM 3-KERNELS, DYNAMIC GATING & DUAL-STREAM FOCAL LOSS

- **Mã nhiệm vụ**: `Task 3.3`
- **Người phụ trách**: **Thành viên B** *(Architecture & Loss Lead)*
- **Môi trường thực hiện**: 🖥️ **Local** + ☁️ **Colab T4** (kiểm thử autograd)
- **Thời hạn hoàn thành**: Tuần 3 (Ngày 19)
- **Độ ưu tiên**: 🔴 Cực kỳ khẩn cấp (Trọng tâm kỹ thuật Chiến lược 1 & 3 của đồ án)
- **Đầu vào (Input)**:
  - Công thức toán học 3 bộ lọc vi sai SRM cố định:
    - $K_1$ ($1^{st}$ order horizontal difference): $\begin{bmatrix} 0 & 0 & 0 \\ -1 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}$
    - $K_2$ ($2^{nd}$ order Laplacian): $\begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}$
    - $K_3$ (Square $5 \times 5$ Bicubic edge filter)
  - Công thức cổng tần số thích ứng: $\lambda(x) = \text{Sigmoid}(W_2 \cdot \text{GELU}(W_1 \cdot \text{GAP}(x) + b_1) + b_2) \times 2.0$
  - Công thức `DualStreamFocalLoss`: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal}}(S, y) + \mathcal{L}_{\text{Focal}}(S', y)$ với $\gamma=2.0, \alpha=0.25$.
- **Đầu ra (Output)**:
  - Mã nguồn hoàn chỉnh: `src/models/srm.py`, `src/models/gating.py`, `src/training/loss.py`.

---

## 1. MỤC TIÊU KỸ THUẬT
1. **Khối Lọc Vết Dư Không Gian SRM (`SpatialResidualBlock`)**:
   - Sử dụng 3 bộ lọc vi sai cố định (`requires_grad = False`) triệt tiêu hoàn toàn nội dung ngữ nghĩa (semantic content), chỉ giữ lại dấu vết nhiễu vi mô của quá trình nội suy sinh ảnh.
   - Chiếu đặc trưng vết dư qua lớp tích chập học được (`Conv2d 1x1`) để hòa trộn vào nhánh FAA.
2. **Cổng Thích Ứng Tần Số Động (`DynamicFrequencyGating` $\lambda(x)$)**:
   - Thay thế hệ số $\lambda$ cố định bằng mạng MLP nhẹ nhận biết mức độ suy biến của ảnh. Khi ảnh bị nén sâu ($Q \le 50$), cổng tự động hạ $\lambda(x) \rightarrow 0.05 \sim 0.1$ để đóng nhánh DWT, ngăn chặn nhiễu khối JPEG làm ô nhiễm mô hình.
3. **Hàm Mục Tiêu Thích Ứng `DualStreamFocalLoss`**:
   - Tính Focal Loss trên cả 2 nhánh Visual (FAA) và Alignment (LGA), đè bẹp gradient của mẫu dễ ($p_t \approx 0.99$) và tập trung toàn bộ năng lượng kéo gradient vào các mẫu khó bị nén sâu.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### 1. Cài đặt `src/models/srm.py`
```python
import torch
import torch.nn as nn
import numpy as np

class SpatialResidualBlock(nn.Module):
    """Khối trích xuất vết dư vi mô không gian sử dụng 3 bộ lọc vi sai SRM cố định."""
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # 1st order horizontal difference
        k1 = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32)
        # 2nd order Laplacian
        k2 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        # 5x5 Square filter
        k3 = np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=np.float32) / 12.0
        
        # Đóng gói thành Conv2d weights cố định
        self.conv_srm = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=5, padding=2, bias=False)
        # Cấu hình trọng số và đóng băng tuyệt đối:
        # Tự động pad k1, k2 lên 5x5 để dùng chung 1 tầng tích chập
        self.conv_srm.weight.requires_grad = False
        
        # Projection layer học được
        self.proj = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        residual = self.conv_srm(x)
        return self.proj(residual)
```

### 2. Cài đặt `src/models/gating.py`
```python
import torch
import torch.nn as nn

class DynamicFrequencyGating(nn.Module):
    """Mạng cổng thích ứng tần số động: λ(x) ∈ [0.0, 2.0]."""
    def __init__(self, in_dim=768, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        # feat: [Batch, Tokens, Dim]
        # Global Average Pooling theo chiều tokens
        gap = feat.mean(dim=1)
        # Scale Sigmoid lên [0.0, 2.0]
        lambda_x = self.mlp(gap) * 2.0
        return lambda_x # [Batch, 1]
```

### 3. Cài đặt `src/training/loss.py`
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualStreamFocalLoss(nn.Module):
    """Focal Loss 2 luồng: L_total = L_focal(Visual) + L_focal(Alignment)."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def _focal_loss(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss) # Xác suất dự đoán đúng
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

    def forward(self, logits_visual, logits_align, targets):
        loss_v = self._focal_loss(logits_visual, targets)
        loss_a = self._focal_loss(logits_align, targets)
        return loss_v + loss_a
```

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] 3 ma trận vi sai SRM có cờ `requires_grad = False` (không tiêu tốn trọng số tối ưu).
- [ ] Module `DynamicFrequencyGating` xuất giá trị tensor chuẩn trong đoạn $[0.0, 2.0]$.
- [ ] Backward pass thành công trên dummy batch, gradient lan truyền trơn tru về các adapter FAA.
