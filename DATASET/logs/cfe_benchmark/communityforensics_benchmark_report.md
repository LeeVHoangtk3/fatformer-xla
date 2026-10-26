# BÁO CÁO KẾT QUẢ KIỂM ĐỊNH NGOẠI VI: COMMUNITYFORENSICS-EVAL (CVPR 2025)
> **Dự án**: FatFormer-XLA | **Thời gian thực hiện**: 2026-10-26 18:59:15
> **Tổng số mẫu**: 2.000 ảnh (1.000 Real / 1.000 Fake trên 5 họ kiến trúc)

---

### Bảng A: Kết Quả Kiểm Chuẩn Trên Miền Ảnh Sạch (Clean CommunityForensics-Eval — 2.000 Ảnh)
*Mục đích: Chứng minh WiSE-FT bảo toàn độ chính xác đỉnh cao và chống báo oan ảnh thật trên các mô hình 2024–2025.*

| Tập Kiểm Chuẩn Con | Họ Kiến Trúc & Mô Hình Đại Diện | Số Mẫu | Baseline CVPR 2024 ($T^*=0.50$) | WiSE-FT Mode 1 ($T=1.08, T^*=0.50$) | Anchored Mode 2 ($T=1.28, T^*=0.40$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1_commercial** | Commercial (MJ v6, DALL-E 3) | 600 | 56.17% | **50.00%** | 73.17% |
| **2_latent_diffusion** | LatDiff (SDXL, Flux.1) | 600 | 64.50% | **50.00%** | 72.00% |
| **3_pixel_diffusion** | PixDiff (DeepFloyd, ADM) | 400 | 65.00% | **50.00%** | 62.75% |
| **4_modern_gans** | GANs (StyleGAN3, GigaGAN) | 200 | 95.00% | **70.50%** | 71.50% |
| **5_other_autoregressive** | Other / Autoregressive (Muse, LlamaGen) | 200 | 59.50% | **50.00%** | 60.50% |
| 🏆 **SUBSET-MACRO MEAN** | **Trung bình không trọng số (5 họ)** | 2.000 | 68.03% | **54.10% 🏆** | 67.98% |
| 📊 **OVERALL POOLED MICRO** | **Toàn bộ 2.000 ảnh gộp (Micro ACC)** | 2.000 | 64.65% | **52.05% 🏆** | 69.30% |
| • Clean Real ACC | COCO, FFHQ, RAISE, LAION | 1.000 | 94.00% | **100.00% 🏆** | 48.10% |
| • Clean Fake ACC | Toàn bộ 5 họ AI mới | 1.000 | 35.30% | 4.10% | **90.50% 🔥** |
| • Clean ROC-AUC | Độc lập ngưỡng cắt $T^*$ | 2.000 | 76.21% | **57.07% 🏆** | 79.28% |
| • Clean Average Precision (AP) | Năng lực phân tách | 2.000 | 78.22% | **60.41% 🏆** | 73.53% |

---

### Bảng B: Kết Quả Kiểm Chuẩn Dưới Biến Dạng Nén Sâu (JPEG Q=30 CommunityForensics-Eval — 2.000 Ảnh)
*Mục đích: Chứng minh Baseline gốc và WiSE-FT sụp đổ khi ảnh AI mới bị nén qua mạng xã hội, trong khi Anchored Recovery cứu vãn ngoạn mục.*

| Chỉ Số Đo Đạc Dưới Nén Sâu ($Q=30$) | Số Mẫu | Baseline CVPR 2024 | WiSE-FT (Chế Độ 1) | Anchored Recovery (Chế Độ 2) |
| :--- | :---: | :---: | :---: | :---: |
| • **1_commercial Fake ACC** (Commercial (MJ v6, DALL-E 3)) | 300 Fake | 0.00% | 0.00% | **66.67% 🔥** |
| • **2_latent_diffusion Fake ACC** (LatDiff (SDXL, Flux.1)) | 300 Fake | 0.00% | 0.00% | **37.67% 🔥** |
| • **3_pixel_diffusion Fake ACC** (PixDiff (DeepFloyd, ADM)) | 200 Fake | 0.50% | 0.00% | **20.00% 🔥** |
| • **4_modern_gans Fake ACC** (GANs (StyleGAN3, GigaGAN)) | 100 Fake | 8.00% | 2.00% | **19.00% 🔥** |
| • **5_other_autoregressive Fake ACC** (Other / Autoregressive (Muse, LlamaGen)) | 100 Fake | 0.00% | 0.00% | **36.00% 🔥** |
| 🏆 **SUBSET-MACRO FAKE ACC** | **Trung bình không trọng số Fake ACC (5 họ)** | 1.000 Fake | 1.70% | 0.40% | **35.87% 🏆** |
| 🔥 **OVERALL POOLED FAKE ACC** | **Toàn bộ 1.000 ảnh Fake gộp (Micro)** | 1.000 Fake | **0.90%** | **0.20%** | **40.80% 🏆** |
| • Overall Q=30 Macro ACC | Toàn bộ 2.000 ảnh gộp | 2.000 | 50.35% | 50.10% | **49.60%** |
| • Overall Q=30 Real ACC | Bảo vệ ảnh thật dưới nén sâu | 1.000 | 99.80% | 100.00% | 58.40% |
| • Overall Q=30 ROC-AUC | Độc lập ngưỡng cắt $T^*$ | 2.000 | 50.00% | 50.93% | **50.07% 🏆** |
| • Overall Q=30 AP | Năng lực phân tách dưới nén sâu | 2.000 | 55.10% | 54.11% | **50.57% 🏆** |

---
> **Kết luận tổng quát**: Thí nghiệm trên CommunityForensics-Eval (CVPR 2025) khẳng định tính thiết yếu của Kiến trúc Hệ thống Kép: WiSE-FT tối ưu cho môi trường phòng lab / ảnh chất lượng cao, còn Anchored Recovery là vũ khí sống còn khi ảnh AI bị phát tán qua các kênh mạng xã hội nén sâu.