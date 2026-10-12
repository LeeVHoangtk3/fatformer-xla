# BÁO CÁO NGHIỆM THU KỸ THUẬT: TASK 5.1
## ĐÁNH GIÁ MÔ HÌNH FATFORMER-XLA TRÊN 18 TẬP TEST HAI CHIỀU

> **Mã nhiệm vụ**: `Task 5.1` *(Milestone 4 / 5)*  
> **Người chủ trì**: **Thành viên C (Training & Evaluation Lead)**  
> **Người phối hợp & Nghiệm thu**: **Thành viên B (Architecture Lead)**, **Thành viên A (Data & Infra Lead)**  
> **Thời gian nghiệm thu**: 2026-10-10  
> **Môi trường & Phần cứng**: Google Colab | **GPU NVIDIA Tesla T4 (15.8 GB VRAM)**  
> **Quy mô thực nghiệm**: **18 tập test** $\times$ **7 trạng thái** (Clean + 6 Degraded) = **126 lượt đo đạc phân tầng** (Tổng thời gian: 198.71 phút)  
> **Checkpoint kiểm thử**: [`fatformer_plan_a_final.pth`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/checkpoints/fatformer_plan_a_final.pth) *(Checkpoint trung gian kế thừa từ Epoch 3 của quá trình huấn luyện bị gián đoạn tại Task 4.3, chưa qua Phase 3 hardening)*  
> **Trạng thái nghiệm thu**: `[x] ĐÃ HOÀN THÀNH ĐÁNH GIÁ THỰC NGHIỆM (GHI NHẬN SỐ LIỆU THAM CHIẾU BƯỚC ĐẦU)`  
> **Sản phẩm bàn giao**: 
> - Ma trận kết quả chi tiết: [`DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv)
> - Biểu đồ đối chứng: [`DATASET/logs/m4_full_benchmark/task_5.1_master_benchmark_chart.png`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/task_5.1_master_benchmark_chart.png)
> - Báo cáo Clean: [`DATASET/logs/m4_full_benchmark/task_5.1_clean_report.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/task_5.1_clean_report.md)
> - Báo cáo Degraded: [`DATASET/logs/m4_full_benchmark/task_5.1_degraded_report.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/task_5.1_degraded_report.md)

---

## 1. MỤC TIÊU KỸ THUẬT & TIÊU CHUẨN NGHIỆM THU (DoD)

Theo đặc tả nhiệm vụ tại [task_5.1_full_benchmark_18_datasets.md](../../tasks/member_c_training_eval/task_5.1_full_benchmark_18_datasets.md) và [ke_hoach_cac_cot_moc_va_diem_dung.md](../../plan/ke_hoach_cac_cot_moc_va_diem_dung.md), **Task 5.1** thực hiện đo đạc thực nghiệm nhằm kiểm tra mô hình trên hai phương diện:
1. **Đo đạc hai chiều không gian dữ liệu**:
   - **Chiều thứ nhất (Cross-Generator Generalization)**: Khả năng nhận diện trên 8 kiến trúc GANs và 10 kiến trúc Diffusion chưa gặp trong quá trình huấn luyện.
   - **Chiều thứ hai (Degradation Robustness)**: Khả năng chống chịu trước 6 biến thể suy biến vật lý ($Q=30, 50, 70$, $\text{Blur}_1, \text{Blur}_2$, $\text{Down-Up}$).
2. **Tiêu chuẩn hoàn thành (Definition of Done & Expansion Gate)**:
   - [x] Thực thi đầy đủ $100\%$ không sót tập nào trong 18 tập test trên cả 7 trạng thái.
   - [!] Clean Overall ACC đạt $90.19\%$ (thấp hơn khoảng cách $-6.26\%$ so với Baseline gốc $96.45\%$; chưa khắc phục hoàn toàn hiện tượng quên tri thức).
   - [!] Độ chính xác kháng nén sâu $Q=30$ Overall ACC đạt $71.03\%$ và Fake ACC đạt $44.84\%$ (tăng so với $5.78\%$ của Baseline gốc, nhưng vẫn còn tỷ lệ bỏ sót $55.16\%$ đòi hỏi phải tiếp tục huấn luyện nâng cao).
   - [x] Lưu trữ tệp ma trận `final_full_benchmark_matrix.csv` và biểu đồ đối chứng.

---

## 2. KẾ THỪA THÔNG SỐ THỰC TẾ (TUÂN THỦ NGUYÊN TẮC 4)

- **Kế thừa từ Task 4.3 ([Báo cáo Task 4.3](bao_cao_task_4.3_finetune_plan_a_kaggle_blackwell.md))**:
   - Checkpoint kiểm thử `fatformer_plan_a_final.pth` là checkpoint trung gian lưu tại Epoch 3 của quá trình huấn luyện bị gián đoạn ở Epoch 4 do đầy bộ nhớ đĩa Kaggle. Mô hình chưa từng trải qua Phase 3 (Epoch 5 - Hardening với $Q \in [30, 50]$, learning rate $\times 0.5$).
   - Lần huấn luyện trước đó có tỷ lệ tham số trainable là 26.07% do whitelist chưa loại trừ các khối FFN trong ViT.
   - **Lưu ý phương pháp luận đánh giá cũ**: Đợt chạy thực nghiệm Task 5.1 lịch sử được thực hiện theo pipeline cũ (nhánh Degraded nạp với `strict=False`). Hiện tại mã nguồn đã được sửa đồng bộ bắt buộc `strict=True` và bổ sung cơ chế lưu log khóa không khớp. Toàn bộ ma trận số liệu này đóng vai trò số liệu tham chiếu trung gian và cần được đo đạc lại đầy đủ bằng quy trình chuẩn hóa mới khi hoàn thành lượt huấn luyện 5 epoch đầy đủ.
- **Kế thừa từ Task 1.1 & Task 2.1 ([Báo cáo Task 1.1](../A/bao_cao_task_1.1_ha_tang_drive_io.md) & [Báo cáo Task 2.1](../A/bao_cao_task_2.1_make_degraded_dataset.md))**:
  - Dữ liệu Clean: `test_benchmark_gans.tar` (18.74 GB) + `test_benchmark_diffusion.tar` (1.22 GB).
  - Dữ liệu Degraded: `test_degraded.tar` (19.55 GB, chứa 6 biến thể nén và làm mờ tạo theo chuẩn PNG lossless).
- **Kế thừa đối chứng từ Task 2.2 & Task 2.3 ([Báo cáo Task 2.2](bao_cao_task_2.2_baseline_clean_benchmark.md) & [Báo cáo Task 2.3](bao_cao_task_2.3_baseline_degraded_benchmark.md))**:
  - Mốc tham chiếu Clean Baseline CVPR 2024: 96.45% Overall ACC (Fake ACC: 93.54%, Real ACC: 99.37%).
  - Mốc tham chiếu Degraded Baseline tại $Q=30$: 52.89% Overall ACC, trong đó Fake ACC ở mức 5.78% (mô hình hầu như đoán Real).
- **Kế thừa đối chứng từ Task 4.2 ([Báo cáo Task 4.2](bao_cao_task_4.2_train_phase_2_hardening.md))**:
  - Mô hình cũ (Epoch 8 cũ) bị suy giảm trên ảnh sạch: Clean Overall ACC ở mức **50.31%** (Fake ACC 43.48%).

---

## 3. TOÀN VĂN MA TRẬN MASTER BENCHMARK 2 CHIỀU THỰC TẾ
*(Trích xuất trực tiếp từ nhật ký kiểm thử tại [`DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv))*

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
| `jpeg_q70` | Nén JPEG $Q=70$ | 83.45 | 77.68 | **80.24** | 96.18 | **64.31** | 92.67 | 91.81 | **-9.95%** |
| `blur_s1` | Gaussian Blur $\sigma=1.0$ | 87.15 | 73.36 | **79.49** | 97.00 | **61.98** | 91.87 | 90.97 | **-10.70%** |
| `jpeg_q50` | Nén JPEG $Q=50$ | 80.30 | 74.52 | **77.09** | 96.24 | **57.93** | 90.28 | 89.35 | **-13.10%** |
| `blur_s2` | Gaussian Blur $\sigma=2.0$ | 82.15 | 69.80 | **75.29** | 94.69 | **55.89** | 88.61 | 88.42 | **-14.90%** |
| `down_up` | Co giãn $224 \rightarrow 112 \rightarrow 224$ | 83.17 | 67.62 | **74.53** | 97.62 | **51.44** | 90.01 | 88.85 | **-15.66%** |
| `jpeg_q30` | Nén sâu JPEG $Q=30$ | 74.50 | 68.26 | **71.03** | 97.22 | **44.84** | 85.64 | 84.74 | **-19.16%** |

---

## 4. ĐỐI CHỨNG MA TRẬN 3 THẾ HỆ MÔ HÌNH (CVPR 2024 vs. OLD vs. PLAN A)

| Chiều Đo Đạc | Chỉ Số Kỹ Thuật | Baseline CVPR 2024<br>*(Task 2.2 & 2.3)* | Mô Hình Cũ Bị Sụt Giảm<br>*(Task 4.2 Epoch 8)* | **FatFormer-XLA Plan A**<br>*(Task 5.1 Đo đạc)* | Mức Độ Thay Đổi<br>*(vs. Mô hình cũ)* |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ẢNH SẠCH (Clean)** | **Overall ACC** | **96.45%** | **50.31%** | **90.19%** | **+39.88%** *(Cải thiện)* |
| | GANs Mean | 98.40% | 50.82% | **93.99%** | **+43.17%** |
| | Diffusion Mean | 95.00% | 49.90% | **87.16%** | **+37.26%** |
| | Fake ACC | 93.54% | 43.48% | **82.28%** | **+38.80%** |
| | AP Trung bình | 99.12% | 53.64% | **96.46%** | **+42.82%** |
| | AUC Trung bình | 99.05% | 49.17% | **95.40%** | **+46.23%** |
| **NÉN SÂU ($Q=30$)** | **Overall ACC** | **52.89%** | **46.78%** | **71.03%** | **+24.25%** |
| | **Fake ACC** | **5.78%** | **29.53%** | **44.84%** | **Tăng từ 5.78% lên 44.84%** |
| | Real ACC | 100.00% | 64.02% | **97.22%** | Cân bằng hơn |
| **NÉN VỪA ($Q=50$)** | Overall ACC | 57.20% | 47.94% | **77.09%** | **+29.15%** |
| **NÉN NHẸ ($Q=70$)** | Overall ACC | 65.40% | 47.58% | **80.24%** | **+32.66%** |
| **GAUSSIAN BLUR $\sigma=1$** | Overall ACC | 62.10% | 48.57% | **79.49%** | **+30.92%** |
| **GAUSSIAN BLUR $\sigma=2$** | Overall ACC | 55.30% | 47.17% | **75.29%** | **+28.12%** |
| **DOWN-UP BICUBIC** | Overall ACC | 59.80% | 48.16% | **74.53%** | **+26.37%** |

---

## 5. PHÂN TÍCH KỸ THUẬT CÁC HIỆN TƯỢNG QUAN SÁT ĐƯỢC

### 5.1. Đánh Giá Độ Chính Xác Trên Tập Ảnh Sạch
- **Hiện tượng**: Trước đây, mô hình cũ ở Task 4.2 bị sụt giảm trên tập ảnh sạch (Clean Overall ACC ở mức **$50.31\%$**).
- **Yếu tố kỹ thuật tác động**:
  1. Lần huấn luyện cũ ở Task 4.2 chỉ nạp 3.600 ảnh staging GenImage và không dùng 144k ảnh ProGAN train.
  2. Tỷ lệ ảnh sạch bị giảm xuống 20% ở các epoch sau.
- **Kết quả ghi nhận**: Tại Task 4.3 và 5.1, với việc giữ `clean_prob = 0.50` và bổ sung 144k ảnh qua `WeightedRandomSampler`, Clean Overall ACC đạt **$90.19\%$** (+39.88% so với mô hình cũ). Tuy nhiên, mức này vẫn còn khoảng cách $-6.26\%$ so với baseline ban đầu ($96.45\%$), cho thấy hiện tượng quên tri thức mới được khắc phục một phần.

### 5.2. Đánh Giá Khả Năng Nhận Diện Dưới Nén Sâu $Q=30$
- **Hiện tượng**: Trong mô hình gốc CVPR 2024, khi ảnh bị nén sâu $Q=30$, Fake ACC ghi nhận mức **$5.78\%$** (mô hình dự đoán phần lớn là Real).
- **Kết quả ghi nhận**:
  - Với việc bổ sung nhánh vi sai SRM 3-Kernels và cổng Dynamic Gating, Fake ACC tại $Q=30$ của mô hình đạt **$44.84\%$** (tăng so với mức $5.78\%$ của Baseline ban đầu).
  - Overall ACC tại $Q=30$ đạt **$71.03\%$**.
  - Tuy nhiên, mức Fake ACC 44.84% cho thấy vẫn còn hơn 55% ảnh giả bị bỏ lọt ở mức nén này.

### 5.3. Phân Tích Các Giới Hạn Kỹ Thuật Chưa Được Khắc Phục
Số liệu thực nghiệm chi tiết cho thấy mô hình Plan A vẫn còn các điểm hạn chế:
1. **Tỷ lệ nhận diện ảnh giả tại $Q=30$ đạt $44.84\%$**:
   - Vẫn còn **$55.16\%$** ảnh Deepfake bị nén chưa được nhận diện chính xác.
2. **Sự sụt giảm trên một số kiến trúc sinh cụ thể dưới nén sâu**:
   - `stylegan2`: Fake ACC tại $Q=30$ ghi nhận mức **`6.00%`**.
   - `vqdiffusion`: Fake ACC tại $Q=30$ ghi nhận mức **`4.40%`**.
   - Họ `glide`: `glide_100_10` (Fake ACC **`19.60%`**), `glide_100_27` (**`22.40%`**), `glide_50_27` (**`23.60%`**).
   - `guided`: Fake ACC tại $Q=30$ là **`26.00%`**; trên tập Clean đạt ACC **`65.85%`** (Fake ACC **`35.00%`**).
3. **Phân tích nguyên nhân**: Các đặc trưng tần số cao của Diffusion và StyleGAN2 có độ phân tán khác so với ProGAN. Việc tập dữ liệu huấn luyện chủ yếu gồm ProGAN kết hợp 3.6k ảnh Staging chưa bao quát đủ các dạng phân phối của các bộ sinh ảnh khác nhau.

---

## 6. KẾT LUẬN & ĐỀ XUẤT TIẾP THEO

1. **Đánh Giá Kết Quả Thực Nghiệm**:
   - Kết quả đo đạc ghi nhận Clean ACC đạt $90.19\%$ và $Q=30$ Overall ACC đạt $71.03\%$, Fake ACC tại $Q=30$ tăng từ $5.78\%$ lên $44.84\%$.
   - **Đánh giá về tính hoàn thiện**: Do Fake ACC tại $Q=30$ còn ở mức $44.84\%$ và một số tập như StyleGAN2 ($6\%$), VQ-Diffusion ($4.4\%$) bị sụt giảm sâu, các số liệu này chưa đủ để kết luận hoàn thành việc huấn luyện. Cần tiếp tục tiến hành các đợt huấn luyện tiếp theo với tập dữ liệu đa dạng hơn và hoàn thành đầy đủ Phase 3 hardening.
2. **Lưu trữ tài liệu tham chiếu**:
   - Tệp ma trận [`DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv) và biểu đồ đối chứng [`DATASET/logs/m4_full_benchmark/task_5.1_master_benchmark_chart.png`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/task_5.1_master_benchmark_chart.png) được lưu giữ phục vụ việc đối chiếu kết quả kỹ thuật.
3. **Kế hoạch tiếp theo**:
   - Rerun quy trình đánh giá trên pipeline chuẩn hóa (`strict=True`) sau khi hoàn thành lượt huấn luyện mới.
   - Triển khai các nhiệm vụ bổ trợ: Ablation Study (Task 5.2), Grad-CAM (Task 5.3) và hoàn thiện báo cáo đồ án.

---
*Báo cáo được lập bởi Lead C, lưu trữ tại `docs/bao_cao/C/bao_cao_task_5.1_full_benchmark_18_datasets.md`.*
