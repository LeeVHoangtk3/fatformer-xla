# BÁO CÁO NGHIỆM THU: TASK 3.4
## Cấu hình Đóng Băng 94.1% CLIP ViT & File Thiết Lập Huấn Luyện

> **Dự án**: FatFormer-XLA  
> **Người thực hiện**: **Thành viên B** *(Architecture & Loss Lead)*  
> **Ngày hoàn thành**: 2026-10-03  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH`

---

## I. ĐẦU RA NGHIỆM THU

| File | Mô Tả |
|:---|:---|
| `src/training/freeze_utils.py` | Hàm `freeze_clip_backbone()` + kiểm toán tham số |
| `src/training/__init__.py` | Cập nhật exports (thêm `DualStreamFocalLoss`, freeze utilities) |
| `configs/train_config.yaml` | File cấu hình huấn luyện chuẩn hóa 3 giai đoạn Curriculum |

---

## II. CHI TIẾT KỸ THUẬT

### 1. `src/training/freeze_utils.py` — PEFT Freeze Utilities

**Thuật toán 2 bước:**
1. Đóng băng TOÀN BỘ: `for param in model.parameters(): param.requires_grad = False`
2. Mở khóa CÓ CHỌN LỌC theo danh sách trắng (prefix matching)

**Modules được mở khóa (~5.9% tham số):**

| Module | Vai trò |
|:---|:---|
| `language_guided_alignment` | LGA: patch_basaed_enhancer, norm, FFN |
| `text_guided_interactor` | Text-guided interactor |
| `norm1`, `norm2`, `linear1`, `linear2` | FFN layers trong CLIPModel |
| `srm` | SRM projection layer (kernel cố định, proj học được) |
| `gating` | Dynamic Gating MLP $\lambda(x)$ |
| `language_guided_alignment.ctx` | Soft prompt vector |

**Utilities đi kèm:**
- `assert_srm_kernels_frozen(model)` — Kiểm tra SRM kernels bắt buộc frozen
- `count_trainable_parameters(model)` — Đếm nhanh tham số trainable

### 2. `configs/train_config.yaml`

Kế thừa thông số thực tế từ biên bản Milestone 1 (2026-09-30):

| Thông số | Giá trị | Nguồn |
|:---|:---:|:---|
| Total epochs | 8 | Task 3.4 spec |
| Batch size / Effective | 32 / 64 | Task 1.3 nghiệm thu (Peak VRAM 9.05GB T4) |
| LR adapter | 1e-4 | Task 3.4 spec |
| LR gating | 1e-3 | Task 3.4 spec |
| Phase 1 JPEG | Q ∈ [70, 90] | Curriculum spec |
| Phase 2 JPEG | Q ∈ [45, 70] | Curriculum spec |
| Phase 3 JPEG | Q ∈ [30, 50] | Curriculum spec (tử huyệt Q=30) |

---

## III. KIỂM TOÁN DOD (DEFINITION OF DONE)

- [x] `freeze_clip_backbone()` có assertion kiểm tra ngưỡng `< 8%` trainable
- [x] `assert_srm_kernels_frozen()` kiểm tra bắt buộc kernel SRM không bị mở gradient
- [x] `configs/train_config.yaml` sẵn sàng cho Lead C nạp vào `train.ipynb`
- [ ] **Pending (Task 3.6)**: Xác nhận trainable ratio thực tế ∈ [5.5%, 6.0%] sau tích hợp đầy đủ

---

## IV. GHI CHÚ BÀNG GIAO CHO LEAD C (Task 3.5 & 3.6)

> ⚠️ **Thứ tự gọi bắt buộc**: `freeze_clip_backbone()` phải được gọi **SAU** `model.load_state_dict(strict=True)`, không được gọi trước.

```python
# Đúng thứ tự trong train.ipynb:
model.load_state_dict(ckpt['model'], strict=True)   # Task 1.2
freeze_clip_backbone(model, verbose=True)             # Task 3.4
assert_srm_kernels_frozen(model)                      # Kiểm tra SRM
```
