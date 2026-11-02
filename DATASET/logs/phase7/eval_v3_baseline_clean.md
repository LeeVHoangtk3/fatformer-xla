# BÁO CÁO KẾT QUẢ KIỂM ĐỊNH NGOẠI VI: COMMUNITYFORENSICS-EVAL (CVPR 2025) - v3.0
> **Dự án**: FatFormer-XLA | **Thời gian**: 2026-11-01T05:13:12.936190+00:00
> **Git Commit**: `851be5ec5e0bab73413deb202290a8a514bac0fc` *(Clean)* | **Thứ tự nén**: `post_resize` | **Ngưỡng T***: `0.5`
> **Quy chuẩn**: Bắt buộc kèm 95% CI và đầy đủ TPR@FPR 1/5/10% theo Phụ lục A.

---

### Bảng A: Kết Quả Kiểm Chuẩn Trên Miền Ảnh Sạch (Clean CFE)

| Họ Kiến Trúc / Tập Con | Số Mẫu | Macro ACC | Real ACC | Fake ACC | ROC-AUC | AP | TPR@1% | TPR@5% | **TPR@10%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Commercial (MJ v6, DALL-E 3)** | 600 | 56.17% | 94.33% | 18.0% | 75.52% | 72.51% | 5.33% | 14.33% | **34.67%** |
| **LatDiff (SDXL, Flux.1)** | 600 | 64.50% | 94.67% | 34.33% | 73.64% | 75.58% | 13.0% | 32.67% | **45.67%** |
| **PixDiff (DeepFloyd, ADM)** | 400 | 65.00% | 93.5% | 36.5% | 74.55% | 76.64% | 18.5% | 34.0% | **39.5%** |
| **GANs (StyleGAN3, GigaGAN)** | 200 | 95.00% | 92.0% | 98.0% | 99.5% | 99.6% | 98.0% | 98.0% | **98.0%** |
| **Other / Autoregressive (Muse, LlamaGen)** | 200 | 59.50% | 94.0% | 25.0% | 72.96% | 73.14% | 10.0% | 22.0% | **40.0%** |
| 🏆 **OVERALL POOLED** | 2000 | **64.65%** | **94.0%** | **35.3%** | **76.21%** | **78.22%** | 19.3% | 31.3% | **45.2%** |

> **Khoảng tin cậy 95% CI (Bootstrap 1.000)**: AUC: `[74.15, 78.23]` | AP: `[75.73, 80.65]` | TPR@5%: `[27.17, 37.51]`
> **Trạng thái nạp Checkpoint**: `ok` (0 missing, 0 unexpected keys)

---