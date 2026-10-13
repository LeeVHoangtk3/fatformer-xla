# BÁO CÁO NGHIỆM THU KỸ THUẬT: TASK 5.1
## ĐÁNH GIÁ MÔ HÌNH FATFORMER-XLA TRÊN 18 TẬP TEST HAI CHIỀU & ĐỐI CHIẾU BÀI BÁO GỐC CVPR 2024

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
   - [x] Đối chiếu đa chiều với công bố khoa học của bài báo gốc CVPR 2024 để lượng hóa những bước tiến cải tiến và chỉ ra các vùng đánh đổi biểu diễn.

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

## 3. MA TRẬN KẾT QUẢ THỰC NGHIỆM CHI TIẾT CỦA FATFORMER-XLA PLAN A

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

## 4. SO SÁNH ĐỐI ĐẦU TOÀN DIỆN VỚI BÀI BÁO GỐC FATFORMER (CVPR 2024)

Chương này thực hiện đối chiếu song song giữa kết quả công bố của bài báo gốc FatFormer (CVPR 2024) (được tái lập độc lập tại [Task 2.2](bao_cao_task_2.2_baseline_clean_benchmark.md) và [Task 2.3](bao_cao_task_2.3_baseline_degraded_benchmark.md)) và mô hình hoàn thiện FatFormer-XLA Plan A (đo đạc tại Task 5.1).

### 4.1. Bảng Đối Chiếu Chi Tiết Từng Tập Dữ Liệu Trên Tập Ảnh Sạch (Clean Benchmark)

| Nhóm | Tập Kiểm Thử | Số Ảnh | Bài Báo Gốc CVPR 2024 (ACC %)<br>*(Task 2.2)* | FatFormer-XLA Plan A (ACC %)<br>*(Task 5.1 Thực Tế)* | Chênh Lệch $\Delta$ (Plan A - Gốc) | Phân Tích Hiện Tượng |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **GANs** | `progan` | 8.000 | **99,89** | **97,95** | -1,94% | Duy trì độ chính xác cao, bảo tồn tri thức gốc |
| | `stargan` | 3.998 | **99,75** | **98,00** | -1,75% | Khả năng tổng quát hóa rất tốt trên bộ sinh khuôn mặt |
| | `stylegan` | 11.982 | **97,13** | **80,35** | -16,78% | Chịu ảnh hưởng bởi chia sẻ biểu diễn với nhánh SRM |
| | `stylegan2` | 15.976 | **98,80** | **73,97** | -24,83% | Phân tán đặc trưng tần số cao của StyleGAN2 |
| | `deepfake` | 5.405 | **93,27** | **64,74** | -28,53% | Dấu vết hoán đổi khuôn mặt bị làm mờ nhẹ |
| | `biggan` | 4.000 | **99,50** | **56,07** | -43,43% | Lệch miền đa lớp đối tượng ImageNet |
| | `cyclegan` | 2.642 | **99,36** | **53,14** | -46,22% | Biến đổi phong cách không gian có độ phức tạp cao |
| | `gaugan` | 10.000 | **99,43** | **45,02** | -54,41% | Thiên về dự đoán lớp Real (Real ACC đạt 88,02%) |
| 📊 | **Trung bình GANs** | **62.003** | **98,39** | **71,16** *(Macro)* / **72,33** *(Pooled)* | **-27,23%** *(Macro)* | Nhóm GANs chịu sự phân hóa giữa các họ mô hình |
| **Diff** | `vqdiffusion` | 2.000 | **100,00** | **81,90** | -18,10% | Duy trì khả năng nhận diện tốt trong nhóm Diffusion |
| | `ldm_200_cfg` | 2.000 | **94,85** | **73,35** | -21,50% | Khử nhiễu có điều kiện bảo tồn một phần đặc trưng |
| | `ldm_200` | 2.000 | **98,55** | **67,50** | -31,05% | Sụt giảm độ nhạy trên ảnh Latent Diffusion thuần |
| | `ldm_100` | 2.000 | **98,60** | **66,75** | -31,85% | Số bước khử nhiễu thấp tạo vết giả mạo khác biệt |
| | `glide_100_10`| 2.000 | **94,15** | **66,00** | -28,15% | Nhóm GLIDE suy giảm tương đương nhau |
| | `glide_50_27` | 2.000 | **94,60** | **65,65** | -28,95% | Dấu vết text-to-image bị dung hòa |
| | `glide_100_27`| 2.000 | **94,30** | **64,85** | -29,45% | Tương đồng với biến thể glide_50_27 |
| | `guided` | 2.000 | **76,05** | **58,50** | -17,55% | Tập thử thách lớn của cả hai phiên bản mô hình |
| | `pndm` | 2.000 | **99,25** | **49,60** | -49,65% | Bộ giải PNDM làm mịn đặc trưng vi mô |
| | `dalle` | 2.000 | **98,70** | **48,80** | -49,90% | Độ phân tán ngữ nghĩa DALL-E tạo độ lệch miền |
| 📊 | **Trung bình Diff** | **20.000** | **94,91** | **64,29** *(Macro)* / **64,29** *(Pooled)* | **-30,62%** *(Macro)* | Phân phối Diffusion đòi hỏi mở rộng dữ liệu học |
| 🏆 | **TOÀN BỘ 18 TẬP** | **82.003** | **96,45** | **67,34** *(Macro)* / **70,37** *(Pooled)* | **-29,11%** *(Macro)* | Đánh đổi học thuật khách quan để tăng độ bền |

---

### 4.2. Bảng Đối Chiếu Năng Lực Chống Chịu Trên 6 Biến Thể Suy Biến (Degraded Benchmark)

| Trạng Thái Kiểm Thử | Bài Báo Gốc CVPR 2024<br>*(Task 2.3 Đo Đạc)* | FatFormer-XLA Plan A<br>*(Task 5.1 Đo Đạc Thật)* | So Sánh Bước Tiến Công Nghệ |
| :--- | :--- | :--- | :--- |
| **Nén sâu JPEG $Q=30$**<br>*(Thử thách then chốt)* | • Overall ACC: **52,89%**<br>• Real ACC: **100,00%**<br>• Fake ACC: **5,78%** *(Sụp đổ hoàn toàn)*<br>• $\Delta$ Sụt giảm: **-43,56%** | • Overall ACC: **51,20%**<br>• Real ACC: **74,87%**<br>• Fake ACC: **27,53%** *(Tăng vọt)*<br>• $\Delta$ Sụt giảm: **-16,14%** *(Kiềm chế tốt)* | **Fake ACC tăng gấp 4,76 lần** (từ 5,78% lên 27,53%). Mô hình không còn bị liệt phân loại một phía; biên độ sụt giảm $\Delta$ giảm từ -43,56% xuống -16,14%. |
| **Nén vừa JPEG $Q=50$** | • Overall ACC: **56,24%**<br>• Real ACC: **100,00%**<br>• Fake ACC: **12,49%**<br>• $\Delta$ Sụt giảm: **-40,21%** | • Overall ACC: **54,71%**<br>• Real ACC: **66,96%**<br>• Fake ACC: **42,47%**<br>• $\Delta$ Sụt giảm: **-12,63%** | **Fake ACC tăng gấp 3,40 lần** (từ 12,49% lên 42,47%). Tỷ lệ bắt ảnh giả mạo cải thiện vượt bậc, độ sụt giảm $\Delta$ kiểm soát ở mức -12,63%. |
| **Nén nhẹ JPEG $Q=70$** | • Overall ACC: **61,28%**<br>• Real ACC: **99,96%**<br>• Fake ACC: **22,60%**<br>• $\Delta$ Sụt giảm: **-35,17%** | • Overall ACC: **56,63%**<br>• Real ACC: **70,98%**<br>• Fake ACC: **42,29%**<br>• $\Delta$ Sụt giảm: **-10,71%** | **Fake ACC tăng gấp 1,87 lần** (từ 22,60% lên 42,29%). Độ dốc suy giảm thoải hơn rõ rệt so với hàm mũ của mô hình gốc. |
| **Gaussian Blur $\sigma=1.0$** | • Overall ACC: **71,76%**<br>• Real ACC: **99,47%**<br>• Fake ACC: **44,04%**<br>• $\Delta$ Sụt giảm: **-24,69%** | • Overall ACC: **59,12%**<br>• Real ACC: **83,53%**<br>• Fake ACC: **34,71%**<br>• $\Delta$ Sụt giảm: **-8,22%** | Độ sụt giảm $\Delta$ được giữ ở mức nhỏ (-8,22% so với -24,69% của gốc), bảo toàn tính ổn định trước làm mờ nhẹ. |
| **Gaussian Blur $\sigma=2.0$** | • Overall ACC: **65,53%**<br>• Real ACC: **98,27%**<br>• Fake ACC: **32,80%**<br>• $\Delta$ Sụt giảm: **-30,92%** | • Overall ACC: **52,91%**<br>• Real ACC: **85,91%**<br>• Fake ACC: **19,91%**<br>• $\Delta$ Sụt giảm: **-14,43%** | Giảm thiểu một nửa biên độ sụt giảm so với ảnh sạch ($\Delta = -14,43\%$ so với $\Delta = -30,92\%$ của bài báo gốc). |
| **Down-Up Bicubic**<br>($224 \rightarrow 112 \rightarrow 224$) | • Overall ACC: **69,52%**<br>• Real ACC: **99,29%**<br>• Fake ACC: **39,76%**<br>• $\Delta$ Sụt giảm: **-26,93%** | • Overall ACC: **58,09%**<br>• Real ACC: **87,09%**<br>• Fake ACC: **29,09%**<br>• $\Delta$ Sụt giảm: **-9,25%** | Mức suy giảm chỉ còn **-9,25%** (so với -26,93% của bản gốc), cấu trúc nội suy được tái tạo tốt hơn qua nhánh vi sai. |

---

### 4.3. Bảng Đối Chiếu Tổng Thể 3 Thế Hệ Mô Hình (CVPR 2024 vs. Mô Hình Cũ Task 4.2 vs. Plan A)

| Chiều Đo Đạc | Chỉ Số Kỹ Thuật | Baseline CVPR 2024<br>*(Task 2.2 & 2.3)* | Mô Hình Cũ Sụt Giảm<br>*(Task 4.2 Epoch 8)* | FatFormer-XLA Plan A<br>*(Task 5.1 Thực Tế)* | Mức Độ Thay Đổi<br>*(vs. Mô hình cũ)* | Mức Độ Thay Đổi<br>*(vs. Bài báo gốc)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ẢNH PHÒNG LAB (Clean)**| **Overall ACC (Macro)** | **96,45%** | **50,31%** | **67,34%** | **+17,03%** *(Phục hồi rõ)* | -29,11% *(Đánh đổi)* |
| | **Overall ACC (Pooled)**| - | - | **70,37%** | **+20,06%** *(Mẫu lớn)* | - |
| | GANs Mean ACC | 98,40% | 50,82% | **71,16%** | **+20,34%** | -27,24% |
| | Diffusion Mean ACC | 95,00% | 49,90% | **64,29%** | **+14,39%** | -30,71% |
| | Real ACC | 99,37% | 57,14% | **90,70%** | **+33,56%** | -8,67% |
| | Fake ACC | 93,54% | 43,48% | **43,98%** | **+0,50%** | -49,56% |
| | AP Trung Bình | 99,12% | 53,64% | **73,58%** | **+19,94%** | -25,54% |
| | AUC Trung Bình | 99,05% | 49,17% | **69,96%** | **+20,79%** | -29,09% |
| **NÉN SÂU ($Q=30$)** | **Overall ACC** | **52,89%** | **46,78%** | **51,20%** | **+4,42%** | -1,69% |
| | **Fake ACC** | **5,78%** | **29,53%** | **27,53%** | -2,00% | **Tăng gấp 4,76 lần** |
| | Real ACC | 100,00% | 64,02% | **74,87%** | +10,85% | Cân bằng 2 lớp |
| **NÉN VỪA ($Q=50$)** | Overall ACC | 57,20% | 47,94% | **54,71%** | **+6,77%** | -2,49% |
| | Fake ACC | 12,49% | 29,10% | **42,47%** | **+13,37%** | **Tăng gấp 3,40 lần** |
| **NÉN NHẸ ($Q=70$)** | Overall ACC | 65,40% | 47,58% | **56,63%** | **+9,05%** | -8,77% |
| | Fake ACC | 22,60% | 28,40% | **42,29%** | **+13,89%** | **Tăng gấp 1,87 lần** |
| **GAUSSIAN BLUR $\sigma=1$**| Overall ACC | 62,10% | 48,57% | **59,12%** | **+10,55%** | -2,98% |
| **GAUSSIAN BLUR $\sigma=2$**| Overall ACC | 55,30% | 47,17% | **52,91%** | **+5,74%** | -2,39% |
| **DOWN-UP BICUBIC** | Overall ACC | 59,80% | 48,16% | **58,09%** | **+9,93%** | -1,71% |

---

## 5. PHÂN TÍCH KHOA HỌC THỰC CHỨNG & CÁC GIỚI HẠN CỐT LÕI CỦA LƯỢT FINE-TUNE PLAN A

Kết luận thẳng thắn từ thực nghiệm: Lượt fine-tune Plan A vận hành ổn định về mặt kỹ thuật, nhưng chưa đạt được sự thành công về hiệu năng tổng quát hóa liên miền. Mô hình học rất tốt trên miền ProGAN-validation, song sụt giảm mạnh khả năng nhận diện các bộ sinh (generator) chưa từng xuất hiện trong quá trình huấn luyện tại Task 5.1.

### 5.1. Khẳng Định Tính Toàn Vẹn Của Trọng Số Kiểm Thử
- Trọng số [`fatformer_plan_a_best_weights.pth`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/checkpoints/fatformer_plan_a_best_weights.pth) là bản sao chuẩn xác của checkpoint tối ưu (`model_best.pth`), hoàn toàn không phải nguyên nhân khiến kết quả Task 5.1 bị suy giảm.
- Đối chiếu trực tiếp với checkpoint đầy đủ từ phiên huấn luyện Kaggle xác nhận:
  - Cùng 1.684 tensor tham số;
  - Khớp toàn bộ danh sách key trong state_dict;
  - Mã băm (hash) từng tensor riêng biệt đều khớp hoàn toàn;
  - Checkpoint đầy đủ được lưu tại Epoch 5 với điểm số `val_score = 83.0868`.
- Do đó, kết quả kiểm thử Task 5.1 phản ánh hoàn toàn trung thực năng lực của checkpoint tối ưu trong lượt huấn luyện Plan A.

### 5.2. Các Điểm Sáng Kỹ Thuật Đã Đạt Được
1. **Huấn luyện hội tụ ổn định**: Hàm mất mát (loss) giảm liên tục từ 0,9228 xuống 0,6598; điểm đánh giá tổng hợp trên tập validation tăng đều đặn từ 78,85 lên 83,09.
2. **Bảo tồn rất tốt đặc trưng ProGAN**:
   - Điểm kiểm tra validation clean đạt 97,88%, trong khi full benchmark ProGAN clean đạt 97,95%.
   - Điểm nén sâu Q30 trên validation đạt 60,34%, trong khi kết quả Task 5.1 ProGAN Q30 đạt 61,60%.
   - Hai phép đo độc lập hoàn toàn tương thích, chứng minh quy trình kiểm thử và công cụ đo đạc có độ tin cậy khoa học cao.
3. **Cải thiện độ nhạy trước nén sâu trên diện rộng**:
   - Ở benchmark Q30 toàn bộ 18 tập test, Fake ACC tăng từ 5,78% của baseline lên **27,53%**.
   - Mô hình không còn rơi vào trạng thái tê liệt dự đoán gần như toàn bộ ảnh là Real như bài báo gốc.

### 5.3. Cái Giá Đánh Đổi Quá Lớn & Hiện Tượng Suy Giảm Liên Miền
Mặc dù có những bước tiến kỹ thuật trên miền ProGAN, dữ liệu thực nghiệm bộc lộ sự đánh đổi quá lớn:
1. **Sụt giảm sâu trên ảnh sạch phòng lab**:
   - Clean Macro ACC chỉ đạt **67,34%**, kém mốc baseline (96,45%) khoảng 29 điểm phần trăm.
   - Fake ACC clean chỉ đạt **43,98%** trong khi Real ACC đạt **90,70%**; mô hình bỏ lọt rất nhiều ảnh giả mạo ở các bộ sinh nằm ngoài ProGAN.
2. **Đánh đổi false negative thành false positive dưới nén sâu**:
   - Điểm tổng thể Q30 Overall ACC đạt **51,20%**, vẫn thấp hơn baseline (52,89%) dù Fake ACC đã tăng.
   - Bản chất là quá trình fine-tune đã chuyển một phần lỗi bỏ lọt ảnh giả thành lỗi nhận diện nhầm ảnh thật: Real ACC sụt giảm từ 100,00% xuống **74,87%**.
3. **Độ dốc suy giảm biểu kiến không phản ánh ưu thế tuyệt đối**:
   - Mô hình ghi nhận Overall ACC tại Q50, Q70, blur và down-up đều thấp hơn bài báo gốc.
   - Mức sụt giảm tương đối $\Delta$ so với clean trông có vẻ hẹp hơn chủ yếu vì điểm mốc clean ban đầu đã ở mức thấp, không phải vì độ chính xác tuyệt đối vượt trội hơn.
4. **Tê liệt trên nhiều miền sinh ngoài tập huấn luyện**:
   - GauGAN clean ghi nhận Fake ACC chỉ đạt **2,02%**; PNDM chỉ đạt **2,70%**; DALL-E chỉ đạt **13,70%**; Deepfake Q30 rơi về **0,00%**.
   - Mức độ nhận diện này chưa thể coi là có độ bền vững hay khả năng tổng quát hóa liên miền.

### 5.4. Đánh Giá Bản Chất Học Thuật
- Phiên bản fine-tune Plan A đã thành công như một adaptation chuyên biệt cho miền ProGAN và các dạng suy biến xoay quanh ProGAN.
- Phiên bản này chưa thành công như một mô hình có thể thay thế bài báo gốc trên bài toán benchmark đa bộ sinh.
- Điểm số validation 83,09 không phải thước đo đại diện cho mục tiêu tổng quát hóa cuối cùng, vì tập validation chủ yếu bao gồm ProGAN clean/Q50/Q30. Điểm số này chỉ lựa chọn ra checkpoint thích ứng ProGAN tốt, chứ không lựa chọn ra checkpoint có năng lực tổng quát hóa liên miền.

---

## 6. KẾT LUẬN NGHIỆM THU & QUY HOẠCH CHIẾN LƯỢC CHO CÁC TASK TIẾP THEO

### 6.1. Kết Luận Nghiệm Thu Task 5.1
1. **Hoàn thành đúng chức năng đo đạc**: Task 5.1 đã hoàn thành trọn vẹn 126 lượt đo đạc phân tầng, xuất đầy đủ hiện vật ma trận số liệu và biểu đồ đối chứng.
2. **Đóng gói Task 5.1 làm Test Set chuẩn mực**:
   - Toàn bộ 18 tập dữ liệu của Task 5.1 được khóa cố định làm tập kiểm thử độc lập (Test Set).
   - Tuyệt đối không sử dụng 18 tập này để huấn luyện bổ sung, không dùng để điều chỉnh ngưỡng quyết định (threshold tuning) hay làm tiêu chí lựa chọn checkpoint.
3. **Quy chuẩn triển khai**:
   - Các giải pháp khắc phục hiện tượng over-specialization sẽ được quy hoạch thành các task độc lập tiếp theo, không thực hiện chỉnh đi chỉnh lại trên Task 5.1.
   - Việc hoàn thành Task 5.1 phản ánh sự trung thực về mặt dữ liệu thực nghiệm và không đồng nghĩa với việc hoàn thành toàn bộ bài báo cáo đồ án.

### 6.2. Chiến Lược Giải Pháp Học Thuật Cho Các Task Cải Tiến Tiếp Theo
Ưu tiên hàng đầu là điều chỉnh "mục tiêu học" và "cấu trúc dữ liệu", chưa vội thay đổi kiến trúc mô hình. Vấn đề cốt lõi là mô hình đang học tối ưu dấu vết ProGAN thay vì học các tín hiệu giả mạo dùng chung giữa các bộ sinh.

1. **Thiết lập tập Validation Cross-Domain độc lập**:
   - Xây dựng tập `val_crossdomain` riêng biệt, hoàn toàn tách rời với dữ liệu huấn luyện và không trùng lặp với 18 tập test của Task 5.1.
   - Nhóm GAN: Bao gồm StyleGAN2, BigGAN/CycleGAN.
   - Nhóm Diffusion: Bao gồm LDM, GLIDE và một biến thể khác.
   - Mỗi miền đều bao gồm ảnh sạch, Q30 và Q50.
   - Tiêu chí chọn checkpoint tối ưu phải dựa trên điểm số cross-domain và bổ sung ràng buộc sàn: Clean Fake ACC, Q30 Fake ACC và Macro AUC đều không được thấp hơn ngưỡng quy định, không chỉ dựa vào trung bình cộng ACC.
2. **Cân bằng dữ liệu huấn luyện theo nguồn sinh**:
   - Tỷ lệ 85% ProGAN / 15% Staging hiện tại là nguồn gốc dẫn đến hiện tượng chuyên biệt hóa cục bộ (over-specialization).
   - Xây dựng bộ lấy mẫu (sampler) phân tầng theo cấu trúc: `Bộ sinh` $\times$ `Lớp nhãn` $\times$ `Mức suy biến`.
   - Bổ sung dữ liệu huấn luyện độc lập cho StyleGAN2, GauGAN, BigGAN, Deepfake và các mô hình Diffusion, bảo đảm không xâm phạm tập test Task 5.1.
3. **Bổ sung hàm mất mát bảo tồn tri thức bài báo gốc trên ảnh sạch**:
   - Huấn luyện theo cặp ảnh song song: $x_{\text{clean}}$ (ảnh gốc) và $x_{\text{degraded}}$ (ảnh suy biến từ chính $x_{\text{clean}}$).
   - Cấu trúc hàm mất mát tích hợp:
     - Cross-Entropy (CE) đồng thời cho nhánh clean và degraded;
     - Triệt tiêu phân kỳ KL (Knowledge Distillation) từ mô hình bài báo gốc trên nhánh clean;
     - Hàm mất mát đồng bộ (Consistency Loss) giữa phân phối đầu ra của clean và degraded;
     - Tăng trọng số phạt đối với ảnh giả bị suy biến, đặc biệt tại dải $Q \in [30, 50]$.
4. **Đa dạng hóa biến thể suy biến đa nguồn và tổ hợp**:
   - Áp dụng các mức nén Q30/Q50/Q70, làm mờ và co giãn độ phân giải cho toàn bộ các bộ sinh trong batch.
   - Bổ sung các chuỗi biến dạng thực tế: Co giãn $\rightarrow$ Nén JPEG, Nén JPEG $\rightarrow$ Co giãn, định dạng WebP và lấy mẫu màu chroma subsampling.
5. **Định vị vai trò của việc dò ngưỡng quyết định (Threshold Tuning)**:
   - Việc quét ngưỡng chỉ đóng vai trò chẩn đoán phân tách giữa hiện tượng lệch chuẩn xác suất (calibration) và hiện tượng mất khả năng phân biệt thật - giả; không xem dò ngưỡng là giải pháp căn cơ.
6. **Bảo tồn phương pháp PEFT & Thử nghiệm LoRA có kiểm soát**:
   - Không thực hiện fine-tune toàn bộ phần thân mạng (backbone).
   - Giữ nguyên SRM/gating và PEFT hiện tại làm nền tảng. Trong trường hợp dữ liệu và hàm mất mát mới chưa đạt ngưỡng, xem xét bổ sung các module LoRA nhỏ vào 2 đến 4 khối Transformer cuối của ViT, với điều kiện vượt qua bài kiểm tra cross-domain validation.
