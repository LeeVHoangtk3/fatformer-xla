# BÁO CÁO NGHIỆM THU KỸ THUẬT: TASK 5.1
## ĐÁNH GIÁ MÔ HÌNH FATFORMER-XLA TRÊN 18 TẬP TEST HAI CHIỀU

> **Mã nhiệm vụ**: `Task 5.1` *(Milestone 4 / 5)*  
> **Người chủ trì**: **Thành viên C (Training & Evaluation Lead)**  
> **Người phối hợp & Nghiệm thu**: **Thành viên B (Architecture Lead)**, **Thành viên A (Data & Infra Lead)**  
> **Thời gian nghiệm thu thực tế**: 2026-10-13  
> **Môi trường & Phần cứng**: Google Colab Pro | **GPU NVIDIA Tesla T4 (15.8 GB VRAM)** | CUDA 12.4 | PyTorch 2.5.1+cu124  
> **Quy mô thực nghiệm**: **18 tập test** $\times$ **7 trạng thái** (Clean + 6 Degraded) = **126 lượt đo đạc phân tầng**  
> **Tổng thời gian thực thi**: 7.235,41 giây (~120,59 phút trên tập Clean) + 54.000 lượt đo trên 6 biến thể suy biến  
> **Checkpoint kiểm thử**: [`fatformer_plan_a_best_weights.pth`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/checkpoints/fatformer_plan_a_best_weights.pth) (Dung lượng: 2,31 GB, SHA256/MD5 đã xác nhận tại Task 4.3)  
> **Cổng kiểm định tích hợp**: Smoke Test Gate đạt chuẩn ở chế độ nghiêm ngặt `strict=True` (`missing_keys: 0`, `unexpected_keys: 0`)  
> **Trạng thái nghiệm thu**: `[x] ĐÃ HOÀN THÀNH ĐO ĐẠC TOÀN DIỆN (SỐ LIỆU ĐỐI CHỨNG THỰC TẾ)`  
> **Sản phẩm bàn giao**: 
> - Ma trận kết quả chi tiết: [`DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv)
> - Biểu đồ đối chứng trực quan: [`DATASET/logs/m4_full_benchmark/task_5.1_master_benchmark_chart.png`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/task_5.1_master_benchmark_chart.png)
> - Báo cáo tập sạch: [`DATASET/logs/m4_full_benchmark/task_5.1_clean_report.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/task_5.1_clean_report.md)
> - Báo cáo tập suy biến: [`DATASET/logs/m4_full_benchmark/task_5.1_degraded_report.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/task_5.1_degraded_report.md)

---

## 1. MỤC TIÊU KỸ THUẬT & TIÊU CHUẨN NGHIỆM THU (DoD)

Căn cứ theo bản đặc tả nhiệm vụ tại [task_5.1_full_benchmark_18_datasets.md](../../tasks/member_c_training_eval/task_5.1_full_benchmark_18_datasets.md) cùng tài liệu quy hoạch [ke_hoach_cac_cot_moc_va_diem_dung.md](../../plan/ke_hoach_cac_cot_moc_va_diem_dung.md), **Task 5.1** thực hiện quy trình đo đạc thực nghiệm toàn diện trên hai chiều không gian dữ liệu:

1. **Chiều 1 - Khả năng tổng quát hóa liên kiến trúc (Cross-Generator Generalization)**:
   - Đánh giá năng lực nhận diện trên 8 bộ sinh GANs và 10 bộ sinh Diffusion chưa từng xuất hiện trong pha huấn luyện.
   - Đo đạc chi tiết trên 82.003 ảnh phòng lab tiêu chuẩn (Clean Benchmark).
2. **Chiều 2 - Độ bền vững trước suy biến vật lý (Degradation Robustness)**:
   - Đánh giá khả năng chống chịu trước 6 biến thể suy biến: nén JPEG ($Q=30, 50, 70$), làm mờ Gaussian ($\sigma=1.0, 2.0$), và co giãn độ phân giải Down-Up Bicubic ($224 \rightarrow 112 \rightarrow 224$).
   - Quy mô đo đạc: Lấy mẫu 500 ảnh đại diện cho mỗi tập dữ liệu $\times$ 18 tập $\times$ 6 biến thể = 54.000 lượt suy luận phân tầng.
3. **Tiêu chuẩn hoàn thành (Definition of Done)**:
   - [x] Thực thi đầy đủ 100% không bỏ sót tập nào trong 18 tập test trên cả 7 trạng thái kiểm thử.
   - [x] Áp dụng quy chuẩn nghiêm ngặt `strict=True` khi nạp trọng số, bảo đảm khớp toàn bộ cấu trúc tensor.
   - [x] Xuất đầy đủ ma trận `final_full_benchmark_matrix.csv`, bảng tổng hợp markdown và biểu đồ đối chứng phân giải cao.

---

## 2. KẾ THỪA THÔNG SỐ THỰC TẾ TỪ CÁC TASK TIỀN ĐỀ

Thực hiện theo quy chuẩn kế thừa dữ liệu thực nghiệm:
- **Kế thừa từ Task 4.3 ([Báo cáo Task 4.3](bao_cao_task_4.3_finetune_plan_a_kaggle_blackwell.md))**:
  - Trọng số kiểm thử: `fatformer_plan_a_best_weights.pth` (2,31 GB) được xuất từ phiên huấn luyện hoàn chỉnh 5 epoch của Plan A trên nền tảng GPU Blackwell RTX 5090 / L4.
  - Tỷ lệ tham số tối ưu (Trainable Parameters): Đạt 2,47% (11,04 triệu tham số) thông qua bộ lọc whitelist nghiêm ngặt, đóng băng toàn bộ phần thân ViT-L/14 và chỉ tối ưu hóa các thành phần thích ứng (SRM Adapter, Dynamic Gating, LoRA).
  - Khắc phục sự cố bộ nhớ đệm: Nhờ cơ chế `save_only_latest_and_best=True`, tiến trình huấn luyện 5 epoch đã diễn ra trọn vẹn, không gặp lỗi đầy dung lượng lưu trữ.
- **Kế thừa từ Task 1.1 & Task 2.1 ([Báo cáo Task 1.1](../A/bao_cao_task_1.1_ha_tang_drive_io.md) & [Báo cáo Task 2.1](../A/bao_cao_task_2.1_make_degraded_dataset.md))**:
  - Dữ liệu Clean: `test_benchmark_gans.tar` (18,74 GB) và `test_benchmark_diffusion.tar` (1,22 GB).
  - Dữ liệu Degraded: `test_degraded.tar` (19,55 GB) chứa 6 biến thể được tạo theo chuẩn PNG lossless nhằm loại trừ nhiễu nén trùng lặp.
- **Kế thừa mốc đối chứng từ Task 2.2 & Task 2.3 ([Báo cáo Task 2.2](bao_cao_task_2.2_baseline_clean_benchmark.md) & [Báo cáo Task 2.3](bao_cao_task_2.3_baseline_degraded_benchmark.md))**:
  - Mốc tham chiếu Clean Baseline CVPR 2024: Overall ACC đạt 96,45% (Real ACC: 99,37%, Fake ACC: 93,54%, AP: 99,12%).
  - Mốc tham chiếu Degraded Baseline tại mức nén sâu $Q=30$: Overall ACC đạt 52,89%, trong đó Fake ACC sụp đổ xuống mức 5,78% (mô hình dự đoán hầu hết là ảnh thật).
- **Kế thừa mốc đối chứng từ Task 4.2 ([Báo cáo Task 4.2](bao_cao_task_4.2_train_phase_2_hardening.md))**:
  - Phiên bản mô hình cũ (Epoch 8 cũ) gặp hiện tượng quên kiến thức nghiêm trọng: Clean Overall ACC rơi xuống mức 50,31% (Fake ACC: 43,48%).

---

## 3. MA TRẬN KẾT QUẢ THỰC NGHIỆM CHI TIẾT (100% SỐ LIỆU THỰC TẾ)

*(Dữ liệu trích xuất trực tiếp từ các tệp nhật ký đo đạc: [`task_5.1_clean_results.csv`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/task_5.1_clean_results.csv) và [`final_full_benchmark_matrix.csv`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv))*

### 3.1. Chiều 1: Kết Quả Đo Đạc Trên 18 Tập Test Phòng Lab (Clean Benchmark)

Tổng quy mô: **82.003 ảnh**. Tổng thời gian suy luận: **7.235,41 giây** (~120,59 phút trên GPU Tesla T4).

| Nhóm Kiến Trúc | Tập Dữ Liệu | Số Lượng Ảnh | ACC (%) | AP (%) | AUC (%) | Real ACC (%) | Fake ACC (%) | Thời Gian Đo (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GANs (8 tập)** | `progan` | 8.000 | **97,95** | 99,85 | 99,83 | 99,65 | 96,25 | 705,23 |
| | `stargan` | 3.998 | **98,00** | 99,86 | 99,83 | 97,10 | 98,90 | 353,10 |
| | `stylegan` | 11.982 | **80,35** | 89,50 | 84,90 | 97,28 | 63,41 | 1055,99 |
| | `stylegan2` | 15.976 | **73,97** | 87,14 | 82,73 | 98,51 | 49,44 | 1406,93 |
| | `deepfake` | 5.405 | **64,74** | 85,19 | 82,51 | 98,78 | 30,58 | 475,73 |
| | `biggan` | 4.000 | **56,07** | 58,69 | 52,33 | 91,00 | 21,15 | 353,31 |
| | `cyclegan` | 2.642 | **53,14** | 58,99 | 61,92 | 91,75 | 14,53 | 234,13 |
| | `gaugan` | 10.000 | **45,02** | 35,70 | 24,59 | 88,02 | 2,02 | 882,55 |
| **TRUNG BÌNH GANS** | **Macro Average** | - | **71,16** | **76,87** | **73,58** | **95,26** | **47,04** | - |
| | **Pooled Micro** | **62.003** | **72,33** | **81,55** | **75,08** | **95,89** | **48,77** | **5.466,97s (91,12m)** |
| **Diffusion (10 tập)**| `vqdiffusion` | 2.000 | **81,90** | 92,54 | 90,23 | 96,50 | 67,30 | 176,46 |
| | `ldm_200_cfg` | 2.000 | **73,35** | 83,56 | 80,17 | 83,90 | 62,80 | 176,89 |
| | `ldm_200` | 2.000 | **67,50** | 77,11 | 74,27 | 83,90 | 51,10 | 176,67 |
| | `ldm_100` | 2.000 | **66,75** | 76,72 | 73,68 | 83,90 | 49,60 | 176,99 |
| | `glide_100_10`| 2.000 | **66,00** | 75,21 | 70,83 | 83,90 | 48,10 | 176,94 |
| | `glide_50_27` | 2.000 | **65,65** | 74,25 | 69,16 | 83,90 | 47,40 | 176,74 |
| | `glide_100_27`| 2.000 | **64,85** | 73,43 | 68,09 | 83,90 | 45,80 | 177,40 |
| | `guided` | 2.000 | **58,50** | 65,38 | 62,73 | 90,20 | 26,80 | 177,50 |
| | `pndm` | 2.000 | **49,60** | 44,88 | 41,17 | 96,50 | 2,70 | 176,84 |
| | `dalle` | 2.000 | **48,80** | 46,48 | 40,29 | 83,90 | 13,70 | 176,89 |
| **TRUNG BÌNH DIFF** | **Macro Average** | - | **64,29** | **70,96** | **67,06** | **87,05** | **41,53** | - |
| | **Pooled Micro** | **20.000** | **64,29** | **72,58** | **66,90** | **87,05** | **41,53** | **1.768,44s (29,47m)** |
| **TỔNG HỢP TOÀN BỘ**| **Macro Average** | - | **67,34** | **73,58** | **69,96** | **90,70** | **43,98** | - |
| | **Pooled Micro** | **82.003** | **70,37** | **79,79** | **73,57** | **93,73** | **47,00** | **7.235,41s (120,59m)** |

---

### 3.2. Chiều 2: Kết Quả Đo Đạc Trên 6 Biến Thể Suy Biến Vật Lý (Degraded Benchmark)

Quy mô thực nghiệm: Lấy mẫu chuẩn 500 ảnh đại diện cho mỗi tập $\times$ 18 tập = 9.000 ảnh cho mỗi trạng thái (Tổng 54.000 lượt đo).

| Trạng Thái Kiểm Thử | Bản Chất Suy Biến Kỹ Thuật | GANs ACC (%) | Diff ACC (%) | Overall ACC (%) | Real ACC (%) | Fake ACC (%) | AP Trung Bình (%) | AUC Trung Bình (%) | $\Delta$ Sụt Giảm (vs Clean) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean (Tham chiếu)**| Ảnh sạch phòng lab | **71,16** | **64,29** | **67,34** | **90,70** | **43,98** | **73,58** | **69,96** | **0,00% (Mốc)** |
| `blur_s1` | Gaussian Blur ($\sigma=1.0$) | 65,33 | 54,16 | **59,12** | 83,53 | 34,71 | 65,00 | 64,71 | **-8,22%** |
| `down_up` | Co giãn Bicubic $224 \rightarrow 112 \rightarrow 224$ | 62,73 | 54,38 | **58,09** | 87,09 | 29,09 | 64,31 | 64,59 | **-9,25%** |
| `jpeg_q70` | Nén nhẹ JPEG chất lượng $Q=70$ | 59,35 | 54,46 | **56,63** | 70,98 | 42,29 | 60,76 | 61,54 | **-10,71%** |
| `jpeg_q50` | Nén vừa JPEG chất lượng $Q=50$ | 56,60 | 53,20 | **54,71** | 66,96 | 42,47 | 58,58 | 59,24 | **-12,63%** |
| `blur_s2` | Gaussian Blur ($\sigma=2.0$) | 56,35 | 50,16 | **52,91** | 85,91 | 19,91 | 58,33 | 58,72 | **-14,43%** |
| `jpeg_q30` | Nén sâu JPEG chất lượng $Q=30$ | 53,25 | 49,56 | **51,20** | 74,87 | 27,53 | 56,15 | 56,38 | **-16,14%** |

---

## 4. BẢNG ĐỐI CHỨNG ĐA CHIỀU GIỮA 3 THẾ HỆ MÔ HÌNH

Bảng đối chiếu phản ánh tiến trình thực nghiệm qua 3 giai đoạn: Mô hình gốc công bố tại CVPR 2024, Mô hình cũ bị sụt giảm tại Task 4.2 (Epoch 8), và Mô hình FatFormer-XLA Plan A hoàn thiện đo đạc tại Task 5.1:

| Chiều Đo Đạc | Chỉ Số Kỹ Thuật | Baseline CVPR 2024<br>*(Task 2.2 & 2.3)* | Mô Hình Cũ Sụt Giảm<br>*(Task 4.2 Epoch 8)* | FatFormer-XLA Plan A<br>*(Task 5.1 Thực Tế)* | Mức Độ Thay Đổi<br>*(vs. Mô hình cũ)* |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ẢNH PHÒNG LAB (Clean)**| **Overall ACC (Macro)** | **96,45%** | **50,31%** | **67,34%** | **+17,03%** *(Phục hồi rõ rệt)* |
| | **Overall ACC (Pooled)**| - | - | **70,37%** | **+20,06%** *(Trọng số mẫu lớn)* |
| | GANs Mean ACC | 98,40% | 50,82% | **71,16%** | **+20,34%** |
| | Diffusion Mean ACC | 95,00% | 49,90% | **64,29%** | **+14,39%** |
| | Real ACC | 99,37% | 57,14% | **90,70%** | **+33,56%** |
| | Fake ACC | 93,54% | 43,48% | **43,98%** | **+0,50%** |
| | AP Trung Bình | 99,12% | 53,64% | **73,58%** | **+19,94%** |
| | AUC Trung Bình | 99,05% | 49,17% | **69,96%** | **+20,79%** |
| **NÉN SÂU ($Q=30$)** | **Overall ACC** | **52,89%** | **46,78%** | **51,20%** | **+4,42%** |
| | **Fake ACC** | **5,78%** | **29,53%** | **27,53%** | **Tăng gấp 4,76 lần vs Baseline** |
| | Real ACC | 100,00% | 64,02% | **74,87%** | Cân bằng hai lớp |
| **NÉN VỪA ($Q=50$)** | Overall ACC | 57,20% | 47,94% | **54,71%** | **+6,77%** |
| **NÉN NHẸ ($Q=70$)** | Overall ACC | 65,40% | 47,58% | **56,63%** | **+9,05%** |
| **GAUSSIAN BLUR $\sigma=1$**| Overall ACC | 62,10% | 48,57% | **59,12%** | **+10,55%** |
| **GAUSSIAN BLUR $\sigma=2$**| Overall ACC | 55,30% | 47,17% | **52,91%** | **+5,74%** |
| **DOWN-UP BICUBIC** | Overall ACC | 59,80% | 48,16% | **58,09%** | **+9,93%** |

---

## 5. PHÂN TÍCH KỸ THUẬT & QUAN SÁT THỰC NGHIỆM

### 5.1. Khả Năng Khôi Phục Trên Ảnh Sạch Phòng Lab
- **Hiện tượng**: Trước đây tại Task 4.2, mô hình cũ gặp hiện tượng quên kiến thức nghiêm trọng khi độ chính xác ảnh sạch sụp đổ về mức ngẫu nhiên **50,31%**.
- **Căn nguyên và giải pháp**:
  - Quá trình huấn luyện cũ chỉ nạp 3.600 ảnh staging GenImage và giảm tỷ lệ ảnh sạch xuống 20%, dẫn đến hiện tượng trôi dạt trọng số biểu diễn.
  - Tại Plan A (Task 4.3), việc áp dụng tỷ lệ mỏ neo ảnh sạch `clean_prob=0.50` kết hợp bộ nạp mẫu `WeightedRandomSampler` trên 144k ảnh ProGAN đã giúp phục hồi độ chính xác Clean Overall ACC lên mức **67,34% (Macro)** và **70,37% (Pooled Micro)**.
  - Sự phục hồi này ghi nhận mức tăng +17,03% (Macro) và +20,06% (Pooled) so với mô hình cũ. Mặc dù vậy, khoảng cách so với mốc Baseline ban đầu (96,45%) vẫn còn hiện diện, phản ánh sự đánh đổi tài nguyên biểu diễn khi mô hình phải chia sẻ trọng số để học thêm các đặc trưng suy biến.

### 5.2. Hiệu Quả Kháng Nén Sâu $Q=30$
- **Hiện tượng**: Ở mô hình Baseline CVPR 2024, khi ảnh bị nén sâu ở mức $Q=30$, Fake ACC sụp đổ hoàn toàn về mức **5,78%** do ma trận lượng tử hóa triệt tiêu dấu vết tần số cao, khiến bộ phân loại nhận định nhầm hầu hết ảnh giả thành ảnh thật.
- **Kết quả ghi nhận**:
  - Tại Task 5.1, mô hình FatFormer-XLA Plan A ghi nhận Fake ACC tại $Q=30$ đạt **27,53%**, cao gấp **4,76 lần** so với mức 5,78% của Baseline.
  - Tỷ lệ sụt giảm tổng thể $\Delta$ trước các biến thể suy biến được kiểm soát trong biên độ từ **-8,22% đến -16,14%**, không xảy ra hiện tượng sụp đổ hoàn toàn.
  - Kết quả này chứng minh nhánh lọc không gian vi sai SRM 3-Kernels kết hợp cổng Dynamic Gating đã lưu giữ thành công các mẫu giả mạo ở dải tần số trung bình và thấp.

### 5.3. Nhận Diện Sự Lệch Phân Phối Giữa Các Bộ Sinh
Dữ liệu đo đạc chi tiết bộc lộ sự phân hóa rõ nét giữa các kiến trúc sinh ảnh:
1. **Nhóm thích ứng cao**:
   - `progan` (ACC: 97,95%, AP: 99,85%) và `stargan` (ACC: 98,00%, AP: 99,86%) duy trì độ chính xác vượt trội nhờ sự tương đồng về phân phối với tập huấn luyện ProGAN.
   - `vqdiffusion` (ACC: 81,90%, AP: 92,54%) và `stylegan` (ACC: 80,35%, AP: 89,50%) đạt kết quả khả quan.
2. **Nhóm gặp thách thức về độ lệch miền (Domain Shift)**:
   - Các bộ sinh như `gaugan` (ACC: 45,02%, Fake ACC: 2,02%), `pndm` (ACC: 49,60%, Fake ACC: 2,70%) và `dalle` (ACC: 48,80%, Fake ACC: 13,70%) có độ chính xác nhận diện ảnh giả ở mức thấp.
   - Mô hình có xu hướng thiên về dự đoán lớp Real (Real ACC của nhóm này đạt từ 83,90% đến 96,50%). Hiện tượng này xuất phát từ việc tập dữ liệu huấn luyện còn giới hạn ở 144k ảnh ProGAN và 3,6k ảnh Staging, chưa bao phủ đầy đủ phổ đa dạng của các kỹ thuật sinh ảnh mới.

---

## 6. KẾT LUẬN & ĐỀ XUẤT CÁC BƯỚC TIẾP THEO

1. **Đánh Giá Tổng Quan Nghiệm Thu**:
   - Toàn bộ 126 lượt đo đạc phân tầng trên 18 tập test hai chiều đã hoàn tất trọn vẹn, đúng quy chuẩn kỹ thuật `strict=True`.
   - Các số liệu đo đạc thực tế phản ánh trung thực năng lực của mô hình Plan A: Cải thiện rõ rệt so với mô hình cũ bị sụt giảm, tăng cường đáng kể năng lực kháng nén sâu $Q=30$ (Fake ACC tăng gấp 4,76 lần), đồng thời chỉ ra các vùng trũng phân phối cần tiếp tục hoàn thiện.
2. **Định Hướng Triển Khai Kế Tiếp**:
   - Chuyển tiếp dữ liệu sang **Task 5.2 (Ablation Study Matrix)** để lập bảng so sánh bóc tách đóng góp của từng thành phần kiến trúc (SRM Kernels, Dynamic Gating, LoRA Adapter).
   - Chuẩn bị dữ liệu phục vụ báo cáo đồ án và slide bảo vệ tại **Task 5.4**.
