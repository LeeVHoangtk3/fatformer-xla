# BÁO CÁO DUMMY — KHÔNG PHẢI BENCHMARK COMMUNITYFORENSICS
> **CẢNH BÁO**: 40 ảnh và model trong báo cáo này là synthetic; `evidence_level=synthetic`, `gate_eligible=false`.

## KIỂM THỬ SCHEMA/HARNESS v3.0
> **Dự án**: FatFormer-XLA | **Thời gian**: 2026-10-28T16:43:14.630057+00:00
> **Git Commit**: `79847cc6b0996028d50c7f0bcc7ef6485d0f7f1e` *(Dirty)* | **Thứ tự nén**: `post_resize` | **Ngưỡng T***: `0.5`
> **Quy chuẩn**: Bắt buộc kèm 95% CI và đầy đủ TPR@FPR 1/5/10% theo Phụ lục A.

---

### Bảng A: Kết Quả Kiểm Chuẩn Trên Miền Ảnh Sạch (Clean CFE)

| Họ Kiến Trúc / Tập Con | Số Mẫu | Macro ACC | Real ACC | Fake ACC | ROC-AUC | AP | TPR@1% | TPR@5% | **TPR@10%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Commercial (MJ v6, DALL-E 3)** | 8 | 50.00% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| **LatDiff (SDXL, Flux.1)** | 8 | 50.00% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| **PixDiff (DeepFloyd, ADM)** | 8 | 50.00% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| **GANs (StyleGAN3, GigaGAN)** | 8 | 50.00% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Other / Autoregressive (Muse, LlamaGen)** | 8 | 50.00% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| 🏆 **OVERALL POOLED** | 40 | **50.00%** | **0.0%** | **100.0%** | **100.0%** | **100.0%** | 100.0% | 100.0% | **100.0%** |

> **Khoảng tin cậy 95% CI (Bootstrap 1.000)**: AUC: `[100.0, 100.0]` | AP: `[100.0, 100.0]` | TPR@5%: `[100.0, 100.0]`
> **Trạng thái nạp Checkpoint**: `ok` (0 missing, 0 unexpected keys)

---
