# BÁO CÁO NGHIỆM THU: TASK 3.3
## Hiện thực hóa SRM 3-Kernels, Dynamic Gating & DualStream Focal Loss

> **Dự án**: FatFormer-XLA  
> **Người thực hiện**: **Thành viên B** *(Architecture & Loss Lead)*  
> **Ngày hoàn thành**: 2026-10-03  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH`

---

## I. ĐẦU RA NGHIỆM THU

| File | Mô Tả |
|:---|:---|
| `src/models/srm.py` | Khối SRM 3-Kernels cố định + Projection layer học được |
| `src/models/gating.py` | Dynamic Frequency Gating $\lambda(x) \in [0, 2]$ |
| `src/training/loss.py` | Bổ sung `DualStreamFocalLoss` ($\alpha=0.25, \gamma=2.0$) |

---

## II. CHI TIẾT KỸ THUẬT

### 1. `src/models/srm.py` — SpatialResidualBlock

3 bộ lọc vi sai pháp y cố định (`register_buffer`, `requires_grad=False`):

| Kernel | Kích thước | Mục đích |
|:---|:---:|:---|
| $K_1$ — First-order horizontal difference | 3×3 | Dò vết sai phân pixel liền kề |
| $K_2$ — Second-order Laplacian | 3×3 | Dò vi sai độ cong bề mặt |
| $K_3$ — Square 5×5 Steganalysis filter | 5×5 | Dò vết nội suy bicubic AI upsampler |

Pipeline chiếu: `Depthwise Conv(3 kernels) → [B,9,H,W] → Conv2d(9→64) → BN → ReLU → AdaptivePool(16×16) → Conv2d(64→1024) → Permute → [256,B,1024]`

Output `[256, B, 1024]` khớp hoàn toàn token format của CLIP ViT-L/14.

### 2. `src/models/gating.py` — DynamicFrequencyGating

$$\lambda(x) = \sigma\!\left(W_2 \cdot \text{GELU}(W_1 \cdot \text{GAP}(x))\right) \times 2.0 \in [0.0, 2.0]$$

- MLP nhẹ: `Linear(1024→64) → GELU → Linear(64→1) → Sigmoid → ×2.0`
- Output: `[1, B, 1]` — broadcast trực tiếp nhân với token stream `[N, B, D]`

### 3. `src/training/loss.py` — DualStreamFocalLoss

$$\mathcal{L}_{\text{XLA}} = \mathcal{L}_{\text{Focal}}(S, y) + \mathcal{L}_{\text{Focal}}(S', y)$$
$$\mathcal{L}_{\text{Focal}}(p_t) = -0.25 \times (1 - p_t)^{2.0} \times \log(p_t)$$

- `FatFormerLoss` gốc giữ nguyên (backward-compatible với pipeline eval Baseline)

---

## III. KIỂM TOÁN DOD (DEFINITION OF DONE)

- [x] Kernel `k1`, `k2`, `k3` có `requires_grad=False` (đăng ký qua `register_buffer`)
- [x] `DynamicFrequencyGating` xuất tensor trong $[0.0, 2.0]$
- [x] `DualStreamFocalLoss.forward()` nhận 2 luồng logit riêng biệt
- [ ] **Pending (Task 3.6)**: Backward pass dummy batch — Lead C xác nhận tại Smoke Gate

---

## IV. GHI CHÚ BÀNG GIAO CHO LEAD C (Task 3.6)

Interface tích hợp SRM + Gating vào `CLIPModel.forward()`:
```python
f_srm    = self.srm(image)            # [256, B, 1024]
lambda_v = self.gating(f_srm)         # [1, B, 1]
f_faa    = f_clip_tokens + lambda_v * f_srm
```
