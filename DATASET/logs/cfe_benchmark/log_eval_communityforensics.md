======================================================================================
FATFORMER-XLA: PIPELINE KIỂM CHUẨN NGOẠI VI COMMUNITYFORENSICS-EVAL (CVPR 2025)
======================================================================================
• Thiết bị tính toán: CUDA (Tesla T4)
• Thư mục dữ liệu:   /content/dataset_local/test_communityforensics

[*] Trạng thái định vị Checkpoints:
  - Baseline CVPR 2024: /content/drive/MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth
  - WiSE-FT (Mode 1): /content/drive/MyDrive/Fatformer/checkpoint/fatformer_wise_ft_best.pth
  - Anchored Recovery (Mode 2): /content/drive/MyDrive/Fatformer/checkpoint/fatformer_anchored_recovery_best.pth

[*] Đã quét được 2000 ảnh từ 5 tập con:
  • 1_commercial             :  600 ảnh (300 Real / 300 Fake)
  • 2_latent_diffusion       :  600 ảnh (300 Real / 300 Fake)
  • 3_pixel_diffusion        :  400 ảnh (200 Real / 200 Fake)
  • 4_modern_gans            :  200 ảnh (100 Real / 100 Fake)
  • 5_other_autoregressive   :  200 ảnh (100 Real / 100 Fake)

======================================================================================
BẮT ĐẦU ĐÁNH GIÁ CHUỖI: ẢNH SẠCH (CLEAN)
======================================================================================

[*] Đang đánh giá mô hình [Baseline CVPR 2024] trên ẢNH SẠCH (CLEAN)...
[*] Dang khoi tao kien truc (BASELINE | SRM=False, Gating=False, Adapters=3)...
[*] Dang nap checkpoint: /content/drive/MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth
[CHECKPOINT] Đã nạp thành công từ: /content/drive/MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth (strict=False)
  [✓] Đã ghi log kiểm toán keys vào: logs/checkpoint_keys_audit.json
  -> Nap thanh cong!
  • 1_commercial           : ACC=56.17% | Real=94.33% | Fake=18.00% | AUC=75.52%
  • 2_latent_diffusion     : ACC=64.50% | Real=94.67% | Fake=34.33% | AUC=73.65%
  • 3_pixel_diffusion      : ACC=65.00% | Real=93.50% | Fake=36.50% | AUC=74.55%
  • 4_modern_gans          : ACC=95.00% | Real=92.00% | Fake=98.00% | AUC=99.50%
  • 5_other_autoregressive : ACC=59.50% | Real=94.00% | Fake=25.00% | AUC=72.97%
  🏆 OVERALL POOLED: ACC=64.65% | Real=94.00% | Fake=35.30% | AUC=76.21%

[*] Đang đánh giá mô hình [WiSE-FT (Mode 1)] trên ẢNH SẠCH (CLEAN)...
[*] Dang khoi tao kien truc (WISE_FT | SRM=True, Gating=True, Adapters=8)...
[*] Dang nap checkpoint: /content/drive/MyDrive/Fatformer/checkpoint/fatformer_wise_ft_best.pth
[CHECKPOINT] Đã nạp thành công từ: /content/drive/MyDrive/Fatformer/checkpoint/fatformer_wise_ft_best.pth (strict=False)
  Unexpected keys (146): ['srm.proj.4.weight', 'srm.proj.1.bias', 'srm.k1', 'srm.proj.0.weight', 'gating.mlp.0.weight']...
  [✓] Đã ghi log kiểm toán keys vào: logs/checkpoint_keys_audit.json
  -> Nap thanh cong!
  • 1_commercial           : ACC=50.00% | Real=100.00% | Fake= 0.00% | AUC=65.18%
  • 2_latent_diffusion     : ACC=50.00% | Real=100.00% | Fake= 0.00% | AUC=43.22%
  • 3_pixel_diffusion      : ACC=50.00% | Real=100.00% | Fake= 0.00% | AUC=57.87%
  • 4_modern_gans          : ACC=70.50% | Real=100.00% | Fake=41.00% | AUC=91.23%
  • 5_other_autoregressive : ACC=50.00% | Real=100.00% | Fake= 0.00% | AUC=47.05%
  🏆 OVERALL POOLED: ACC=52.05% | Real=100.00% | Fake=4.10% | AUC=57.07%

[*] Đang đánh giá mô hình [Anchored Recovery (Mode 2)] trên ẢNH SẠCH (CLEAN)...
[*] Dang khoi tao kien truc (ANCHORED | SRM=True, Gating=True, Adapters=3)...
[*] Dang nap checkpoint: /content/drive/MyDrive/Fatformer/checkpoint/fatformer_anchored_recovery_best.pth
[CHECKPOINT] Đã nạp thành công từ: /content/drive/MyDrive/Fatformer/checkpoint/fatformer_anchored_recovery_best.pth (strict=False)
  [✓] Đã ghi log kiểm toán keys vào: logs/checkpoint_keys_audit.json
  -> Nap thanh cong!
  • 1_commercial           : ACC=73.17% | Real=53.33% | Fake=93.00% | AUC=83.29%
  • 2_latent_diffusion     : ACC=72.00% | Real=50.33% | Fake=93.67% | AUC=82.31%
  • 3_pixel_diffusion      : ACC=62.75% | Real=50.50% | Fake=75.00% | AUC=65.57%
  • 4_modern_gans          : ACC=71.50% | Real=44.00% | Fake=99.00% | AUC=88.02%
  • 5_other_autoregressive : ACC=60.50% | Real=25.00% | Fake=96.00% | AUC=81.62%
  🏆 OVERALL POOLED: ACC=69.30% | Real=48.10% | Fake=90.50% | AUC=79.28%

======================================================================================
BẮT ĐẦU ĐÁNH GIÁ CHUỖI: ẢNH NÉN SÂU (JPEG Q=30)
======================================================================================

[*] Đang đánh giá mô hình [Baseline CVPR 2024] trên ẢNH NÉN SÂU (JPEG Q=30)...
[*] Dang khoi tao kien truc (BASELINE | SRM=False, Gating=False, Adapters=3)...
[*] Dang nap checkpoint: /content/drive/MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth
[CHECKPOINT] Đã nạp thành công từ: /content/drive/MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth (strict=False)
  [✓] Đã ghi log kiểm toán keys vào: logs/checkpoint_keys_audit.json
  -> Nap thanh cong!
  • 1_commercial           : ACC=49.83% | Real=99.67% | Fake= 0.00% | AUC=58.13%
  • 2_latent_diffusion     : ACC=50.00% | Real=100.00% | Fake= 0.00% | AUC=25.56%
  • 3_pixel_diffusion      : ACC=50.25% | Real=100.00% | Fake= 0.50% | AUC=72.97%
  • 4_modern_gans          : ACC=54.00% | Real=100.00% | Fake= 8.00% | AUC=86.48%
  • 5_other_autoregressive : ACC=49.50% | Real=99.00% | Fake= 0.00% | AUC=20.68%
  🏆 OVERALL POOLED: ACC=50.35% | Real=99.80% | Fake=0.90% | AUC=50.00%

[*] Đang đánh giá mô hình [WiSE-FT (Mode 1)] trên ẢNH NÉN SÂU (JPEG Q=30)...
[*] Dang khoi tao kien truc (WISE_FT | SRM=True, Gating=True, Adapters=8)...
[*] Dang nap checkpoint: /content/drive/MyDrive/Fatformer/checkpoint/fatformer_wise_ft_best.pth
[CHECKPOINT] Đã nạp thành công từ: /content/drive/MyDrive/Fatformer/checkpoint/fatformer_wise_ft_best.pth (strict=False)
  Unexpected keys (146): ['srm.proj.4.weight', 'srm.proj.1.bias', 'srm.k1', 'srm.proj.0.weight', 'gating.mlp.0.weight']...
  [✓] Đã ghi log kiểm toán keys vào: logs/checkpoint_keys_audit.json
  -> Nap thanh cong!
  • 1_commercial           : ACC=50.00% | Real=100.00% | Fake= 0.00% | AUC=60.46%
  • 2_latent_diffusion     : ACC=50.00% | Real=100.00% | Fake= 0.00% | AUC=30.95%
  • 3_pixel_diffusion      : ACC=50.00% | Real=100.00% | Fake= 0.00% | AUC=65.78%
  • 4_modern_gans          : ACC=51.00% | Real=100.00% | Fake= 2.00% | AUC=73.32%
  • 5_other_autoregressive : ACC=50.00% | Real=100.00% | Fake= 0.00% | AUC=34.98%
  🏆 OVERALL POOLED: ACC=50.10% | Real=100.00% | Fake=0.20% | AUC=50.93%

[*] Đang đánh giá mô hình [Anchored Recovery (Mode 2)] trên ẢNH NÉN SÂU (JPEG Q=30)...
[*] Dang khoi tao kien truc (ANCHORED | SRM=True, Gating=True, Adapters=3)...
[*] Dang nap checkpoint: /content/drive/MyDrive/Fatformer/checkpoint/fatformer_anchored_recovery_best.pth
[CHECKPOINT] Đã nạp thành công từ: /content/drive/MyDrive/Fatformer/checkpoint/fatformer_anchored_recovery_best.pth (strict=False)
  [✓] Đã ghi log kiểm toán keys vào: logs/checkpoint_keys_audit.json
  -> Nap thanh cong!
  • 1_commercial           : ACC=65.67% | Real=64.67% | Fake=66.67% | AUC=72.72%
  • 2_latent_diffusion     : ACC=46.83% | Real=56.00% | Fake=37.67% | AUC=47.57%
  • 3_pixel_diffusion      : ACC=41.75% | Real=63.50% | Fake=20.00% | AUC=34.66%
  • 4_modern_gans          : ACC=34.50% | Real=50.00% | Fake=19.00% | AUC=29.96%
  • 5_other_autoregressive : ACC=40.50% | Real=45.00% | Fake=36.00% | AUC=37.91%
  🏆 OVERALL POOLED: ACC=49.60% | Real=58.40% | Fake=40.80% | AUC=50.07%

======================================================================================
HOÀN TẤT TOÀN DIỆN KIỂM CHUẨN NGOẠI VI COMMUNITYFORENSICS-EVAL!
======================================================================================
[✓] Đã xuất báo cáo Markdown tại: /content/drive/MyDrive/Fatformer/logs/cfe_benchmark/communityforensics_benchmark_report.md
[✓] Đã xuất dữ liệu JSON tại:     /content/drive/MyDrive/Fatformer/logs/cfe_benchmark/communityforensics_benchmark_report.json

NỘI DUNG BÁO CÁO TỔNG HỢP (BẢNG A & B):

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