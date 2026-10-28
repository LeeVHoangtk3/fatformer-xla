# BÁO CÁO KẾT QUẢ KIỂM ĐỊNH NGOẠI VI: COMMUNITYFORENSICS-EVAL (CVPR 2025) - v3.0
> **Dự án**: FatFormer-XLA | **Thời gian**: 2026-10-28T07:44:39.448231+00:00
> **Git Commit**: `c7daac66060e5743af61304f9207ddd0f38431ce` *(Dirty)* | **Thứ tự nén**: `post_resize` | **Ngưỡng T***: `0.5`
> **Quy chuẩn**: Bắt buộc kèm 95% CI và đầy đủ TPR@FPR 1/5/10% theo Phụ lục A.

---

### Bảng A: Kết Quả Kiểm Chuẩn Trên Miền Ảnh Sạch (Clean CFE)

| Họ Kiến Trúc / Tập Con | Số Mẫu | Macro ACC | Real ACC | Fake ACC | ROC-AUC | AP | TPR@1% | TPR@5% | **TPR@10%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Commercial (MJ v6, DALL-E 3)** | 8 | 100.00% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| **LatDiff (SDXL, Flux.1)** | 8 | 100.00% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| **PixDiff (DeepFloyd, ADM)** | 8 | 100.00% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| **GANs (StyleGAN3, GigaGAN)** | 8 | 100.00% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Other / Autoregressive (Muse, LlamaGen)** | 8 | 100.00% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| 🏆 **OVERALL POOLED** | 40 | **100.00%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 100.0% | 100.0% | **100.0%** |

> **Khoảng tin cậy 95% CI (Bootstrap 1.000)**: AUC: `[100.0, 100.0]` | AP: `[100.0, 100.0]` | TPR@5%: `[100.0, 100.0]`
> **Trạng thái nạp Checkpoint**: `ok` (0 missing, 0 unexpected keys)

---