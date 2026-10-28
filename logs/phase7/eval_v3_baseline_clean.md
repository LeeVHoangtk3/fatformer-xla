# BÁO CÁO KẾT QUẢ KIỂM ĐỊNH NGOẠI VI: COMMUNITYFORENSICS-EVAL (CVPR 2025) - v3.0
> **Dự án**: FatFormer-XLA | **Thời gian**: 2026-10-28T09:16:13.710030+00:00
> **Git Commit**: `f1e6543a6c07db397feee4c4ff258bc482f897b7` *(Dirty)* | **Thứ tự nén**: `post_resize` | **Ngưỡng T***: `0.8468`
> **Quy chuẩn**: Bắt buộc kèm 95% CI và đầy đủ TPR@FPR 1/5/10% theo Phụ lục A.

---

### Bảng A: Kết Quả Kiểm Chuẩn Trên Miền Ảnh Sạch (Clean CFE)

| Họ Kiến Trúc / Tập Con | Số Mẫu | Macro ACC | Real ACC | Fake ACC | ROC-AUC | AP | TPR@1% | TPR@5% | **TPR@10%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Commercial (MJ v6, DALL-E 3)** | 8 | 50.00% | 100.0% | 0.0% | 43.75% | 50.12% | 0.0% | 0.0% | **0.0%** |
| **LatDiff (SDXL, Flux.1)** | 8 | 50.00% | 100.0% | 0.0% | 43.75% | 50.12% | 0.0% | 0.0% | **0.0%** |
| **PixDiff (DeepFloyd, ADM)** | 8 | 50.00% | 100.0% | 0.0% | 43.75% | 50.12% | 0.0% | 0.0% | **0.0%** |
| **GANs (StyleGAN3, GigaGAN)** | 8 | 50.00% | 100.0% | 0.0% | 43.75% | 50.12% | 0.0% | 0.0% | **0.0%** |
| **Other / Autoregressive (Muse, LlamaGen)** | 8 | 50.00% | 100.0% | 0.0% | 43.75% | 50.12% | 0.0% | 0.0% | **0.0%** |
| 🏆 **OVERALL POOLED** | 40 | **50.00%** | **100.0%** | **0.0%** | **43.75%** | **50.12%** | 0.0% | 0.0% | **0.0%** |

> **Khoảng tin cậy 95% CI (Bootstrap 1.000)**: AUC: `[23.72, 65.87]` | AP: `[35.84, 71.93]` | TPR@5%: `[0.0, 0.0]`
> **Trạng thái nạp Checkpoint**: `ok` (0 missing, 0 unexpected keys)

---