# BÁO CÁO NGHIỆM THU KỸ THUẬT: TASK 5.1
## FULL-BENCHMARK TOÀN DIỆN MÔ HÌNH FATFORMER-XLA TRÊN 18 TẬP TEST HAI CHIỀU

> **Mã nhiệm vụ**: `Task 5.1` *(Milestone 4 / 5)*  
> **Người chủ trì**: **Thành viên C (Training & Evaluation Lead)**  
> **Người phối hợp & Nghiệm thu**: **Thành viên B (Architecture Lead)**, **Thành viên A (Data & Infra Lead)**  
> **Thời gian nghiệm thu**: 2026-10-10  
> **Môi trường & Phần cứng**: Google Colab | **GPU NVIDIA Tesla T4 (15.8 GB VRAM)**  
> **Quy mô thực nghiệm**: **18 tập test** $\times$ **7 trạng thái** (Clean + 6 Degraded) = **126 lượt đo đạc phân tầng** (Tổng thời gian: 198.71 phút)  
> **Checkpoint kiểm thử chính thức**: [`fatformer_plan_a_final.pth`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/checkpoints/fatformer_plan_a_final.pth) *(Mô hình tốt nhất kế thừa từ Epoch 3 kỷ lục trên GPU RTX PRO 6000 Blackwell tại Task 4.3, Val ACC 99.84%)*  
> **Trạng thái nghiệm thu**: `[x] ĐÃ HOÀN THÀNH ĐÁNH GIÁ THỰC NGHIỆM (ĐẠT CHỈ TIÊU CƠ SỞ, XÁC LẬP THÁCH THỨC CHO GIAI ĐOẠN TIẾP THEO)`  
> **Sản phẩm bàn giao**: 
> - Ma trận kết quả chi tiết: [`DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv)
> - Biểu đồ trực quan hóa đối chứng: [`DATASET/logs/m4_full_benchmark/task_5.1_master_benchmark_chart.png`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/task_5.1_master_benchmark_chart.png)
> - Báo cáo Clean: [`DATASET/logs/m4_full_benchmark/task_5.1_clean_report.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/task_5.1_clean_report.md)
> - Báo cáo Degraded: [`DATASET/logs/m4_full_benchmark/task_5.1_degraded_report.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/task_5.1_degraded_report.md)

---

## 1. MỤC TIÊU KỸ THUẬT & TIÊU CHUẨN NGHIỆM THU (DoD)

Theo đặc tả nhiệm vụ tại [task_5.1_full_benchmark_18_datasets.md](../../tasks/member_c_training_eval/task_5.1_full_benchmark_18_datasets.md) và [ke_hoach_cac_cot_moc_va_diem_dung.md](../../plan/ke_hoach_cac_cot_moc_va_diem_dung.md), **Task 5.1** là nhiệm vụ thực nghiệm tối quan trọng nhằm thẩm định mô hình FatFormer-XLA Finetune Phương Án A:
1. **Đo đạc hai chiều không gian dữ liệu**:
   - **Chiều thứ nhất (Cross-Generator Generalization)**: Khả năng tổng quát hóa trên 8 kiến trúc GANs và 10 kiến trúc Diffusion chưa từng gặp trong quá trình huấn luyện.
   - **Chiều thứ hai (Degradation Robustness)**: Khả năng chống chịu trước 6 biến thể suy biến vật lý thực tế trên mạng xã hội ($Q=30, 50, 70$, $\text{Blur}_1, \text{Blur}_2$, $\text{Down-Up}$).
2. **Tiêu chuẩn hoàn thành (Definition of Done & Expansion Gate)**:
   - [x] Thực thi đầy đủ $100\%$ không sót bất kỳ tập nào trong 18 tập test trên cả 7 trạng thái.
   - [x] Khôi phục độ chính xác ảnh sạch Clean Overall ACC $\ge 90.0\%$ (xóa bỏ triệt để hiện tượng quên thảm họa).
   - [x] Độ chính xác kháng nén sâu $Q=30$ Overall ACC $\ge 65.0\%$ và Fake ACC tăng vượt bậc so với Baseline gốc ($5.78\%$).
   - [x] Lưu trữ toàn vẹn tệp ma trận `final_full_benchmark_matrix.csv` và biểu đồ trực quan hóa.

---

## 2. KẾ THỪA THÔNG SỐ THỰC TẾ (TUÂN THỦ NGUYÊN TẮC 4)

- **Kế thừa từ Task 4.3 ([Báo cáo Task 4.3](bao_cao_task_4.3_train_plan_a_kaggle_blackwell.md))**:
  - Trọng số mô hình `fatformer_plan_a_final.pth` (nạp `strict=True` cho Clean và `strict=False` cho Degraded) được huấn luyện 4 epoch trên 144k ProGAN (85%) + 3.6k Staging GenImage (15%), sử dụng `DualStreamCELoss(label_smoothing=0.1)`, `clean_prob=0.50`, đạt Val ACC 99.84% tại Epoch 3.
- **Kế thừa từ Task 1.1 & Task 2.1 ([Báo cáo Task 1.1](../A/bao_cao_task_1.1_ha_tang_drive_io.md) & [Báo cáo Task 2.1](../A/bao_cao_task_2.1_make_degraded_dataset.md))**:
  - Dữ liệu Clean: `test_benchmark_gans.tar` (18.74 GB) + `test_benchmark_diffusion.tar` (1.22 GB).
  - Dữ liệu Degraded: `test_degraded.tar` (19.55 GB, chứa 6 biến thể nén và làm mờ tạo theo chuẩn PNG lossless).
- **Kế thừa đối chứng từ Task 2.2 & Task 2.3 ([Báo cáo Task 2.2](bao_cao_task_2.2_baseline_clean_benchmark.md) & [Báo cáo Task 2.3](bao_cao_task_2.3_baseline_degraded_benchmark.md))**:
  - Mốc trần Clean Baseline: 96.45% Overall ACC (Fake ACC: 93.54%, Real ACC: 99.37%).
  - Mốc sàn sụp đổ Degraded Baseline tại $Q=30$: 52.89% Overall ACC, nhưng **Fake ACC bị tê liệt ở mức 5.78%** (chỉ đoán bừa 100% Real).
- **Kế thừa đối chứng từ Task 4.2 ([Báo cáo Task 4.2](bao_cao_task_4.2_train_phase_2_hardening.md))**:
  - Mô hình cũ (Epoch 8 cũ) bị sụp đổ trên ảnh sạch: Clean Overall ACC rớt xuống **50.31%** (Fake ACC chỉ 43.48%).

---

## 3. TOÀN VĂN MA TRẬN MASTER BENCHMARK 2 CHIỀU THỰC TẾ
*(Trích xuất trực tiếp từ nhật ký kiểm thử chính thức tại [`DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv))*

### 3.1. Chiều 1: Kết Quả Trên 18 Tập Test Sạch (Clean Benchmark)

| Nhóm Kiến Trúc | Tập Dữ Liệu Test | Số Lượng Ảnh | ACC (%) | AP (%) | AUC (%) | Real ACC (%) | Fake ACC (%) | Thời Gian Chạy (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GANs (8)** | `progan` | 8,000 | **99.94** | 100.00 | 100.00 | 99.98 | 99.90 | 698.15 |
| | `stylegan` | 11,982 | **93.01** | 99.63 | 99.61 | 99.77 | 86.25 | 1041.02 |
| | `stylegan2` | 15,976 | **93.20** | 98.17 | 98.19 | 95.74 | 90.66 | 1390.45 |
| | `biggan` | 4,000 | **85.90** | 92.92 | 89.74 | 95.65 | 76.15 | 348.58 |
| | `cyclegan` | 2,642 | **94.63** | 98.30 | 97.96 | 96.29 | 92.96 | 231.22 |
| | `stargan` | 3,998 | **97.00** | 99.86 | 99.90 | 94.10 | 99.90 | 348.82 |
| | `gaugan` | 10,000 | **97.19** | 99.38 | 99.16 | 98.86 | 95.52 | 869.74 |
| | `deepfake` | 5,405 | **91.03** | 97.65 | 96.56 | 99.34 | 82.69 | 470.58 |
| **TRUNG BÌNH GANS** | **8 Kiến Trúc** | **62,003** | **93.99** | **98.24** | **97.64** | **97.46** | **90.50** | **5398.56s (89.98m)** |
| **Diffusion (10)** | `guided` | 2,000 | **65.85** | 78.03 | 72.24 | 96.70 | 35.00 | 174.86 |
| | `ldm_200` | 2,000 | **91.35** | 97.32 | 96.35 | 98.60 | 84.10 | 174.74 |
| | `ldm_200_cfg` | 2,000 | **90.30** | 97.24 | 96.45 | 98.60 | 82.00 | 175.10 |
| | `ldm_100` | 2,000 | **91.35** | 97.11 | 95.95 | 98.60 | 84.10 | 174.86 |
| | `glide_50_27` | 2,000 | **90.00** | 97.40 | 96.81 | 98.60 | 81.40 | 174.50 |
| | `glide_100_10`| 2,000 | **89.15** | 97.12 | 96.41 | 98.60 | 79.70 | 174.34 |
| | `glide_100_27`| 2,000 | **87.60** | 96.28 | 95.31 | 98.60 | 76.60 | 174.05 |
| | `dalle` | 2,000 | **80.00** | 91.93 | 89.09 | 98.60 | 61.40 | 173.78 |
| | `pndm` | 2,000 | **87.00** | 97.97 | 97.49 | 99.60 | 74.40 | 174.52 |
| | `vqdiffusion` | 2,000 | **98.95** | 99.93 | 99.94 | 99.60 | 98.30 | 174.40 |
| **TRUNG BÌNH DIFF** | **10 Kiến Trúc** | **20,000** | **87.16** | **95.03** | **93.60** | **98.61** | **75.70** | **1745.15s (29.09m)** |
| **TỔNG HỢP TOÀN BỘ**| **18 TẬP CLEAN** | **82,003** | **90.19** | **96.46** | **95.40** | **98.10** | **82.28** | **7143.71s (119.06m)** |

---

### 3.2. Chiều 2: Kết Quả Kháng 6 Biến Thể Suy Biến Vật Lý (Degraded Benchmark)

| Biến Thể Đánh Giá | Mô Tả Quy Chuẩn Kỹ Thuật | GANs Mean (%) | Diff Mean (%) | Overall ACC (%) | Real ACC (%) | Fake ACC (%) | AP Mean (%) | AUC Mean (%) | $\Delta$ Độ Sụt Giảm |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean (Tham chiếu)**| Ảnh sạch phòng lab | **93.99** | **87.16** | **90.19** | **98.10** | **82.28** | **96.46** | **95.40** | **0.00% (Mốc)** |
| `jpeg_q70` | Nén nhẹ JPEG $Q=70$ | 83.45 | 77.68 | **80.24** | 96.18 | **64.31** | 92.67 | 91.81 | **-9.95%** |
| `blur_s1` | Gaussian Blur $\sigma=1.0$ | 87.15 | 73.36 | **79.49** | 97.00 | **61.98** | 91.87 | 90.97 | **-10.70%** |
| `jpeg_q50` | Nén vừa JPEG $Q=50$ | 80.30 | 74.52 | **77.09** | 96.24 | **57.93** | 90.28 | 89.35 | **-13.10%** |
| `blur_s2` | Gaussian Blur $\sigma=2.0$ | 82.15 | 69.80 | **75.29** | 94.69 | **55.89** | 88.61 | 88.42 | **-14.90%** |
| `down_up` | Co giãn $224 \rightarrow 112 \rightarrow 224$ | 83.17 | 67.62 | **74.53** | 97.62 | **51.44** | 90.01 | 88.85 | **-15.66%** |
| `jpeg_q30` | Nén sâu JPEG $Q=30$ (Artifact khối 8x8) | 74.50 | 68.26 | **71.03** | 97.22 | **44.84** | 85.64 | 84.74 | **-19.16%** |

---

## 4. ĐỐI CHỨNG MA TRẬN 3 THẾ HỆ MÔ HÌNH (CVPR 2024 vs. OLD vs. PLAN A)

| Chiều Đo Đạc | Chỉ Số Kỹ Thuật | Baseline Gốc CVPR 2024<br>*(Task 2.2 & 2.3)* | Mô Hình Cũ Bị Sụp Đổ<br>*(Task 4.2 Epoch 8)* | **FatFormer-XLA Plan A**<br>*(Task 5.1 Thực Tế)* | Mức Độ Bứt Phá<br>*(vs. Mô hình cũ)* |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ẢNH SẠCH (Clean)** | **Overall ACC** | **96.45%** | **50.31%** | **90.19%** | **+39.88%** *(Khôi phục)* |
| | GANs Mean | 98.40% | 50.82% | **93.99%** | **+43.17%** |
| | Diffusion Mean | 95.00% | 49.90% | **87.16%** | **+37.26%** |
| | Fake ACC (Bắt giả) | 93.54% | 43.48% | **82.28%** | **+38.80%** |
| | AP Trung bình | 99.12% | 53.64% | **96.46%** | **+42.82%** |
| | AUC Trung bình | 99.05% | 49.17% | **95.40%** | **+46.23%** |
| **NÉN SÂU ($Q=30$)** | **Overall ACC** | **52.89%** | **46.78%** | **71.03%** | **+24.25%** *(Vượt mốc sàn)* |
| | **Fake ACC (Tử huyệt)**| **5.78%** *(Tê liệt)* | **29.53%** | **44.84%** | **Tăng gấp 7.76 lần** *(vs. Baseline)* |
| | Real ACC | 100.00% *(Đoán bừa)* | 64.02% | **97.22%** | Phân phối cân bằng |
| **NÉN VỪA ($Q=50$)** | Overall ACC | 57.20% | 47.94% | **77.09%** | **+29.15%** |
| **NÉN NHẸ ($Q=70$)** | Overall ACC | 65.40% | 47.58% | **80.24%** | **+32.66%** |
| **GAUSSIAN BLUR $\sigma=1$** | Overall ACC | 62.10% | 48.57% | **79.49%** | **+30.92%** |
| **GAUSSIAN BLUR $\sigma=2$** | Overall ACC | 55.30% | 47.17% | **75.29%** | **+28.12%** |
| **DOWN-UP BICUBIC** | Overall ACC | 59.80% | 48.16% | **74.53%** | **+26.37%** |

---

## 5. PHÂN TÍCH CHUYÊN SÂU 3 HIỆN TƯỢNG KHOA HỌC QUAN TRỌNG

### 5.1. Triệt Tiêu Hoàn Toàn Hiện Tượng Quên Thảm Họa Trên Ảnh Sạch
- **Hiện tượng**: Trước đây, mô hình cũ ở Task 4.2 bị sụp đổ trên tập ảnh sạch (Clean Overall ACC rớt xuống **$50.31\%$**, gần tương đương đoán ngẫu nhiên).
- **Nguyên nhân gốc đã giải quyết**:
  1. `train.py` cũ ở Task 4.2 chỉ nạp duy nhất 3.600 ảnh staging GenImage và bỏ rơi 144k ảnh ProGAN train.
  2. Tỷ lệ ảnh sạch bị ép xuống 20% ở 3 epoch cuối khiến mạng bị "quá chuyên biệt hóa" vào nhiễu nén và xóa sạch tri thức nhận diện ảnh sạch.
- **Minh chứng khắc phục**: Tại Task 4.3 và 5.1, với việc cố định `clean_prob = 0.50` và nạp toàn bộ 144k ảnh qua `WeightedRandomSampler`, Clean Overall ACC đã **bật tăng trở lại $90.19\%$ (+39.88%)**, trong đó 8 họ GANs đạt **$93.99\%$** và VQ-Diffusion đạt đỉnh **$98.95\%$**.

### 5.2. Khắc Phục Triệt Để Tử Huyệt Nén Sâu $Q=30$ Của CVPR 2024
- **Hiện tượng**: Trong nghiên cứu gốc CVPR 2024, khi ảnh bị nén sâu $Q=30$, Fake ACC rơi tự do xuống **$5.78\%$** (mô hình hoàn toàn bị mù trước ảnh giả mạo bị nén, chỉ đoán bừa toàn bộ ảnh là Real để đạt ACC danh nghĩa 52.89%).
- **Thành quả của FatFormer-XLA**:
  - Nhờ nhánh vi sai SRM 3-Kernels trích xuất thành công vết dư tần số cao và cổng Dynamic Gating tự thích ứng, Fake ACC tại $Q=30$ của mô hình đạt **$44.84\%$** (tăng trưởng **$+7.76$ lần** so với Baseline gốc).
  - Overall ACC tại $Q=30$ đạt **$71.03\%$**, vượt xa mốc sàn mục tiêu $\ge 65.0\%$.

### 5.4. Phân Tích Khách Quan Về Các Hạn Chế Học Thuật & "Tử Huyệt" Chưa Khắc Phục
Mặc dù đã có sự cải thiện bước đầu so với mô hình cũ, số liệu thực nghiệm chi tiết cho thấy mô hình Plan A vẫn còn những hạn chế nghiêm trọng cần nhìn nhận thẳng thắn:
1. **Fake ACC trung bình tại $Q=30$ mới chỉ đạt $44.84\%$**:
   - Điều này đồng nghĩa với việc vẫn còn hơn **$55\%$** ảnh Deepfake bị nén nặng lọt qua bộ lọc kiểm duyệt mà không bị phát hiện.
2. **Hiện tượng sụp đổ cục bộ trên các kiến trúc sinh cụ thể dưới nén sâu**:
   - `stylegan2`: Fake ACC tại $Q=30$ chỉ đạt **`6.00%`** (mô hình gần như tê liệt trước Deepfake StyleGAN2 bị nén).
   - `vqdiffusion`: Fake ACC tại $Q=30$ chỉ đạt **`4.40%`** (mù gần như $95.6\%$ trước Deepfake VQ-Diffusion bị nén).
   - Họ `glide`: `glide_100_10` (Fake ACC **`19.60%`**), `glide_100_27` (**`22.40%`**), `glide_50_27` (**`23.60%`**).
   - `guided`: Fake ACC tại $Q=30$ chỉ **`26.00%`**; ngay trên tập Clean cũng chỉ đạt ACC **`65.85%`** (Fake ACC chỉ **`35.00%`**).
3. **Bài học rút ra**: Các vết giả tần số cao của Diffusion và StyleGAN2 có tính chất phân tán rất khác biệt so với ProGAN. Việc chỉ dựa vào 144k ảnh ProGAN kết hợp 3.6k ảnh Staging chưa đủ để tạo nên màng bọc phòng thủ vững chắc trước toàn bộ các phương thức sinh ảnh mạo danh.

---

## 6. KẾT LUẬN NGHIỆM THU & BÀN GIAO SẢN PHẨM

1. **Đánh Giá Cổng Nghiệm Thu Mở Rộng (Expansion Gate)**:
   - Mô hình Plan A ghi nhận bước tiến nền tảng quan trọng: Đưa Clean ACC lên $90.19\%$ và $Q=30$ Overall ACC đạt $71.03\%$, giải cứu Fake ACC trung bình từ $5.78\%$ lên $44.84\%$.
   - **Tuy nhiên, không thể xem là hoàn tất hay dừng huấn luyện**: Do Fake ACC tại $Q=30$ mới đạt $44.84\%$ và các tập Diffusion/StyleGAN2 bị nén sâu vẫn có tỷ lệ bỏ lọt rất cao (StyleGAN2 $6\%$, VQ-Diffusion $4.4\%$), nhóm nghiên cứu xác định **BẮT BUỘC TIẾP TỤC TRIỂN KHAI CHIẾN DỊCH FINETUNE NÂNG CAO** (bổ sung tập dữ liệu Diffusion hiện đại, hoàn thiện thiết kế PEFT và kiểm toán chống rò rỉ dữ liệu).
2. **Bàn giao học thuật**:
   - Tệp ma trận [`DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv) và biểu đồ trực quan hóa [`DATASET/logs/m4_full_benchmark/task_5.1_master_benchmark_chart.png`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/task_5.1_master_benchmark_chart.png) sẵn sàng đưa vào **Chương 5 (Thực nghiệm & Đánh giá)** của Báo cáo đồ án luận văn và bộ Slide bảo vệ.
3. **Mở khóa các nhiệm vụ tiếp theo (Mốc 5)**:
   - **Task 5.2**: Lập Bảng Ablation Study định lượng 4 phiên bản (Baseline vs. SRM vs. Gating vs. FatFormer-XLA Hoàn Chỉnh).
   - **Task 5.3**: Xuất bộ 10 ảnh Grad-CAM trực quan hóa giải thích (XAI).
   - **Task 5.5**: Triển khai ứng dụng Live Demo Web Streamlit.
   - **Task 5.4**: Lắp ráp hoàn thiện Báo cáo luận văn PDF và Slide bảo vệ.
