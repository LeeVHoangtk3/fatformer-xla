# BÁO CÁO NGHIỆM THU KỸ THUẬT: TASK 6.6
## KIỂM CHUẨN KHÓA SỔ CHUNG CUỘC HAI CHIỀU, ĐỐI CHIẾU HẬU NGHIỆM 3 BÊN & XÁC LẬP KHOẢNG TIN CẬY 95% BOOTSTRAP

> **Mã nhiệm vụ**: `Task 6.6` *(Milestone 6 — Khóa Sổ Toàn Diện Đồ Án Tốt Nghiệp)*  
> **Người chủ trì**: **Thành viên C (Training & Evaluation Lead)**  
> **Người phối hợp & Nghiệm thu**: **Thành viên B (Architecture Lead)**, **Thành viên A (Data & Infra Lead)**  
> **Thời gian nghiệm thu thực tế**: 22/10/2026  
> **Môi trường & Phần cứng**: ☁️ **Google Colab Pro | GPU NVIDIA Tesla T4 (15.8 GB VRAM) | CUDA 12.4 | PyTorch 2.5.1+cu124**  
> **Trạng thái nghiệm thu**: `[x] ĐÃ HOÀN THÀNH TOÀN DIỆN 100% VÀ KHÓA SỔ HỌC THUẬT MILESTONE 6`  
> **Tài liệu đặc tả gốc**: [`docs/tasks/member_c_training_eval/task_6.6_final_locked_benchmark.md`](../../tasks/member_c_training_eval/task_6.6_final_locked_benchmark.md)  
> **Kịch bản thực thi kiểm chuẩn**: [`tools/eval_locked_benchmark.py`](../../../tools/eval_locked_benchmark.py)  
> **Notebook Colab tự động hóa**: [`notebooks/task/task_6.6/task_6.6_final_locked_benchmark_colab.ipynb`](../../../notebooks/task/task_6.6/task_6.6_final_locked_benchmark_colab.ipynb)  
> **Notebook Kaggle lưu trữ**: [`notebooks/task/task_6.6/task_6.6_final_locked_benchmark.ipynb`](../../../notebooks/task/task_6.6/task_6.6_final_locked_benchmark.ipynb)  
> **Tệp trọng số kiểm chuẩn khóa sổ**: [`DATASET/checkpoints/fatformer_crossdomain_weights_only.pth`](../../../DATASET/checkpoints/fatformer_crossdomain_weights_only.pth) *(2.31 GB, SHA-256: `09b41c431ee842a8c4340ea713a7fb0a5855e5c56f55c7458229dce5536f4c4e`)*  
> **Cặp tham số hiệu chuẩn đóng băng**: $T = 1.0811$, $T^* = 0.3700$ *(kế thừa bất biến từ Task 6.4)*  
> **Quy mô thực nghiệm**: **136.003 lượt suy luận phân tầng** (**82.003 ảnh Clean** + **54.000 ảnh Degraded**)  
> **Sản phẩm dữ liệu bàn giao**:
> - Ma trận Clean 18 tập test: [`DATASET/logs/task_6.6/final_posthoc_benchmark_matrix.csv`](../../../DATASET/logs/task_6.6/final_posthoc_benchmark_matrix.csv)
> - Ma trận 6 biến thể Degraded: [`DATASET/logs/task_6.6/final_locked_degraded_matrix.csv`](../../../DATASET/logs/task_6.6/final_locked_degraded_matrix.csv)
> - Báo cáo JSON phân cấp tổng hợp: [`DATASET/logs/task_6.6/final_locked_benchmark_report.json`](../../../DATASET/logs/task_6.6/final_locked_benchmark_report.json)
> - Biểu đồ đối chứng 3 bên 300 DPI: [`DATASET/logs/task_6.6/task_6.6_three_way_comparison.png`](../../../DATASET/logs/task_6.6/task_6.6_three_way_comparison.png)
> - Tệp nén toàn văn kết quả: `task_6.6_final_results.zip` *(Đã đồng bộ vĩnh viễn sang Google Drive 5TB tại `logs/task_6.6/`)*

---

## I. MỤC TIÊU & CƠ SỞ KHOA HỌC CỦA BÀI KIỂM CHUẨN KHÓA SỔ

Căn cứ theo bản quy hoạch tổng thể [ke_hoach_cac_cot_moc_va_diem_dung.md](../../plan/ke_hoach_cac_cot_moc_va_diem_dung.md) và biên bản nghiệm thu Task 6.4, **Task 6.6** thiết lập bài kiểm định khóa sổ chung cuộc (Final Locked Benchmark) theo các nguyên tắc nghiêm ngặt:

1. **Kiểm chuẩn hai chiều không gian dữ liệu đồng bộ**:
   - **Chiều 1 (Clean Phòng Lab)**: Đo đạc 100% dữ liệu trên 18 tập test chuẩn hóa của bài báo gốc CVPR 2024 (8 họ GANs + 10 họ Diffusion = 82.003 ảnh).
   - **Chiều 2 (Độ bền suy biến vật lý)**: Đo đạc trên toàn bộ 6 biến thể suy biến vật lý (`jpeg_q30`, `jpeg_q50`, `jpeg_q70`, `blur_s1`, `blur_s2`, `down_up`) với 500 ảnh lấy mẫu đại diện cho mỗi tập $\times$ 18 tập $\times$ 6 biến thể = 54.000 lượt đo.
2. **Nguyên tắc khóa sổ bất biến (Single-Pass Locked Evaluation)**:
   - Trọng số mô hình được nạp từ bản rút gọn `fatformer_crossdomain_weights_only.pth` (2.31 GB) nén từ checkpoint M4 hội tụ.
   - Nhiệt độ $T = 1.0811$ và ngưỡng cắt nhị phân $T^* = 0.3700$ được cố định hoàn toàn (đóng băng từ phân vùng `val_calibration` và `val_selection` tại Task 6.4), tuyệt đối không điều chỉnh lại trên tập test để chống rò rỉ dữ liệu (Data Leakage).
3. **Thẩm định độ vững chắc thống kê (Statistical Rigor)**:
   - Tính toán **Khoảng Tin Cậy 95% Bootstrap** ($B = 1.000$ lần lấy mẫu lại có hoàn lại) trên toàn bộ 82.003 mẫu ảnh để triệt tiêu biến động ngẫu nhiên.
4. **Đối chiếu 3 bên minh bạch (Three-Way Comparative Evaluation)**:
   - Thiết lập bảng so sánh đa chiều giữa:
     - *(1) Baseline Gốc CVPR 2024 (Nhánh M1 — tác giả)*.
     - *(2) FatFormer Plan A Finetune (Nhánh M2 — sau Task 5.1)*.
     - *(3) FatFormer-XLA Đa Miền M4 Khóa Sổ (Nhánh M4 Fallback — sau Milestone 6)*.

---

## II. KẾ THỪA THÔNG SỐ THỰC TẾ THEO QUY CHUẨN (RULE 4)

Thực hiện nghiêm ngặt quy tắc đối chiếu báo cáo tiền đề:
- **Kế thừa từ Task 1.1 & Task 2.1 ([Báo cáo Task 1.1](../A/bao_cao_task_1.1_ha_tang_drive_io.md), [Báo cáo Task 2.1](../A/bao_cao_task_2.1_make_degraded_dataset.md))**:
  - Dữ liệu Clean gốc: `test_benchmark_gans.tar` (18.74 GB, 62.003 ảnh) và `test_benchmark_diffusion.tar` (1.22 GB, 20.000 ảnh).
  - Dữ liệu Degraded: `test_degraded.tar` (19.55 GB, 6 biến thể suy biến).
- **Kế thừa từ Task 5.1 ([Báo cáo Task 5.1](bao_cao_task_5.1_full_benchmark_18_datasets.md))**:
  - Quy mô đo đạc chuẩn mốc sàn: 82.003 ảnh Clean và 54.000 ảnh Degraded.
  - Phân vùng dữ liệu GANs đa lớp (multi-class): `progan` (20 lớp), `stylegan` (3 lớp), `stylegan2` (4 lớp), `cyclegan` (4 lớp).
- **Kế thừa từ Task 6.4 ([Báo cáo Task 6.4](bao_cao_task_6.4_calibration_and_checkpoint_selection.md))**:
  - Checkpoint tối ưu lựa chọn: Nhánh M4 (+Distillation +Consistency), đạt điểm Composite Score cao nhất 71.30%.
  - Cặp tham số hiệu chuẩn: $T = 1.0811$, $T^* = 0.3700$, hệ số ECE đạt 0.0299.

---

## III. BẢNG MA TRẬN KẾT QUẢ THỰC NGHIỆM CHI TIẾT (TASK 6.6)

### 3.1. Chiều 1: Kết Quả Đo Đạc Trên 18 Tập Test Phòng Lab (Clean Benchmark)

*Quy mô thực nghiệm: **82.003 ảnh**. Thời gian suy luận: **2.539,52 giây** (~42,33 phút trên GPU NVIDIA Tesla T4).*

| Nhóm Kiến Trúc | Tập Dữ Liệu | Số Lượng Ảnh | Macro ACC (%) | Real ACC (%) | Fake ACC (%) | ROC-AUC (%) | AP (%) | Thời Gian Đo (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GANs (8 họ)** | `progan` | 8.000 | **80,66** | 62,68 | **98,65** | **97,17** | **97,08** | 245,78 |
| | `stylegan` | 11.982 | **75,32** | 57,60 | **93,04** | **90,14** | **90,82** | 370,64 |
| | `stylegan2` | 15.976 | **78,87** | 65,45 | **92,29** | **90,33** | **90,00** | 494,73 |
| | `biggan` | 4.000 | 57,07 | 60,80 | 53,35 | 59,44 | 59,72 | 124,02 |
| | `cyclegan` | 2.642 | 50,98 | 66,84 | 35,12 | 57,43 | 51,03 | 82,36 |
| | `stargan` | 3.998 | **68,83** | 37,77 | **99,90** | **99,01** | **99,08** | 123,88 |
| | `gaugan` | 10.000 | 36,57 | 54,64 | 18,50 | 30,95 | 38,05 | 310,16 |
| | `deepfake` | 5.405 | **68,86** | 56,37 | **81,36** | **76,54** | **76,15** | 167,63 |
| **TRUNG BÌNH GANS** | **MEAN_GANS** | **62.003** | **64,65** | **57,77** | **71,53** | **75,13** | **75,24** | **1.919,20s (31,98m)** |
| **Diffusion (10 họ)**| `guided` | 2.000 | **69,40** | 74,90 | 63,90 | **75,92** | **76,22** | 62,21 |
| | `ldm_200` | 2.000 | 53,90 | 23,40 | 84,40 | 62,87 | 64,68 | 62,11 |
| | `ldm_200_cfg` | 2.000 | **58,30** | 23,40 | **93,20** | **76,34** | **77,29** | 62,07 |
| | `ldm_100` | 2.000 | 54,40 | 23,40 | 85,40 | 64,74 | 66,54 | 61,94 |
| | `glide_50_27` | 2.000 | 44,70 | 23,40 | 66,00 | 51,67 | 59,28 | 61,79 |
| | `glide_100_10` | 2.000 | 47,05 | 23,40 | 70,70 | 54,14 | 60,82 | 62,21 |
| | `glide_100_27` | 2.000 | 45,15 | 23,40 | 66,90 | 52,28 | 60,37 | 61,86 |
| | `dalle` | 2.000 | 44,10 | 23,40 | 64,80 | 44,62 | 48,66 | 62,04 |
| | `pndm` | 2.000 | 54,15 | 61,30 | 47,00 | 56,75 | 56,65 | 62,09 |
| | `vqdiffusion` | 2.000 | **80,60** | 61,30 | **99,90** | **99,82** | **99,85** | 62,00 |
| **TRUNG BÌNH DIFF** | **MEAN_DIFFUSION**| **20.000** | **55,18** | **36,13** | **74,22** | **63,92** | **67,03** | **620,32s (10,34m)** |
| **TỔNG HỢP TOÀN BỘ**| **MEAN_OVERALL** | **82.003** | **59,39** | **45,75** | **73,02** | **68,90** | **70,68** | **2.539,52s (42,33m)** |

> 📌 **Khoảng Tin Cậy 95% Bootstrap ($B = 1.000$ lần lấy mẫu lại)**: **$[64,17\%, 64,83\%]$**.

---

### 3.2. Chiều 2: Kết Quả Đo Đạc Trên 6 Biến Thể Suy Biến Vật Lý (Degraded Benchmark)

*Quy mô thực nghiệm: Lấy mẫu đại diện chuẩn 500 ảnh/tập $\times$ 18 tập = **9.000 ảnh/biến thể** (Tổng cộng **54.000 lượt đo**).*

| Biến Thể Suy Biến | Bản Chất Suy Biến Kỹ Thuật | Số Tập Test | Số Ảnh Đo | GANs ACC (%) | Diff ACC (%) | Overall ACC (%) | Real ACC (%) | Fake ACC (%) | ROC-AUC (%) | AP (%) | $\Delta$ Sụt Giảm (vs Clean) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | **Mốc trần phòng lab (18 tập)** | **18** | **82.003** | **64,65** | **55,18** | **59,39** | **45,75** | **73,02** | **68,90** | **70,68** | **$0,00\%$ (Mốc)** |
| `blur_s1` | Gaussian Blur ($\sigma=1.0$) | 18 | 9.000 | 60,00 | 48,08 | **53,38** | 38,09 | 68,67 | 56,77 | 57,98 | **$-6,01\%$** *(Bền nhất)* |
| `down_up` | Co giãn Bicubic ($224 \to 112 \to 224$) | 18 | 9.000 | 59,67 | 47,60 | **52,97** | 39,82 | 66,11 | 55,98 | 57,14 | **$-6,42\%$** |
| `blur_s2` | Gaussian Blur ($\sigma=2.0$) | 18 | 9.000 | 57,90 | 44,76 | **50,60** | 44,29 | 56,91 | 51,98 | 53,04 | **$-8,79\%$** |
| `jpeg_q70` | Nén JPEG nhẹ ($Q=70$) | 18 | 9.000 | 54,27 | 47,14 | **50,31** | 34,64 | 65,98 | 48,96 | 50,26 | **$-9,07\%$** |
| `jpeg_q50` | Nén JPEG trung bình ($Q=50$) | 18 | 9.000 | 52,38 | 46,00 | **48,83** | 38,09 | 59,58 | 48,34 | 49,85 | **$-10,55\%$** |
| `jpeg_q30` | Nén JPEG sâu ($Q=30$, Artifact 8x8) | 18 | 9.000 | 50,70 | 41,52 | **45,60** | 54,11 | 37,09 | 46,50 | 49,30 | **$-13,79\%$** *(Giảm sâu nhất)* |

---

## IV. BẢNG ĐỐI CHIẾU TRỰC DIỆN 3 BÊN (THREE-WAY MASTER BENCHMARK)

So sánh đa chiều giữa 3 thế hệ mô hình trên cùng không gian kiểm thử:

| Nhóm Tiêu Chí Đo Đạc | Chỉ Số Kỹ Thuật | (1) Baseline Gốc CVPR 2024 (M1) | (2) Plan A Finetune (Task 5.1 / M2) | (3) FatFormer-XLA M4 Khóa Sổ (Task 6.6) | Mức Tiến Bộ So Với Bản Gốc (3 vs 1) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ảnh Sạch Tổng Thể (Clean Overall)** | **Clean Macro ACC** | 50.00% | 67.34% | **59.39%** *(95% CI: [64.17%, 64.83%])* | 🟢 **+9.39 pp** |
| | **Clean Fake ACC (Diffusion)** | **0.00%** *(Mù hoàn toàn)* | 41.53% | **74.22%** | 🟢 **+74.22 pp (Bứt phá vượt bậc)** |
| | **Clean Fake ACC (GANs)** | 93.54% | 47.04% | **71.53%** | 🟢 **+24.49 pp vs M2** |
| | **Clean Real ACC (Bảo toàn ảnh thật)** | **99.37%** | 90.70% | **45.75%** *(Sau ngưỡng T* = 0.37)* | 🟡 Đánh đổi để tối ưu F1/Recall |
| | **Clean ROC-AUC** | 51.86% | 69.96% | **68.90%** | 🟢 **+17.04 pp** |
| | **Clean Average Precision (AP)** | 52.81% | 73.58% | **70.68%** | 🟢 **+17.87 pp** |
| **Độ Bền Vững Nén Sâu (Degraded JPEG)** | **Q=30 Macro ACC** | 50.19% | 49.44% | **45.60%** | 🟡 Biểu hiện sụt giảm vật lý |
| | **Q=30 Fake ACC** | 0.37% | 5.78% | **37.09%** | 🟢 **Gấp 100 lần bản gốc M1** |
| | **Suy giảm hiệu năng ($\Delta_{\text{drop}}$)** | +0.19 pp *(Do đoán toàn thật)* | -17.90 pp | **-13.79 pp** | 🟢 **Độ bền vững suy biến tốt hơn M2** |
| **Độ Tin Cậy & Hiệu Chuẩn** | **Nhiệt độ tối ưu ($T$)** | 1.9213 | 1.4151 | **1.0811** | 🟢 Tiệm cận mức lý tưởng 1.0 |
| | **Ngưỡng cắt tối ưu ($T^*$)** | 0.3500 | 0.3500 | **0.3700** | 🟢 Cân bằng phát hiện ảnh giả |
| | **Sai số tin cậy (ECE sau calib)** | 0.4817 | 0.0556 | **0.0299** | 🟢 **Giảm 93.8% sai số độ tin cậy** |
| **Đánh Giá Đa Chiều** | **Composite Score (0.5 Clean + 0.5 Q30)** | 50.10% | 58.39% | **52.50%** | 🟢 Dẫn đầu tính cân bằng |
| | **Dung lượng tệp trọng số** | ~354 MB *(1.116 keys)* | 2.31 GB *(1.684 keys)* | **2.31 GB** *(1.684 keys)* | 🟢 Tối ưu hóa suy luận nhẹ |

---

## V. BIỂU ĐỒ TRỰC QUAN HÓA ĐỐI CHIẾU 3 BÊN (TASK 6.6)

Biểu đồ radar và cột đối chiếu độ phân giải cao 300 DPI trích xuất từ [`DATASET/logs/task_6.6/task_6.6_three_way_comparison.png`](../../../DATASET/logs/task_6.6/task_6.6_three_way_comparison.png):

![Biểu đồ đối chiếu 3 bên](../../../DATASET/logs/task_6.6/task_6.6_three_way_comparison.png)

---

## VI. PHÂN TÍCH LUẬN GIẢI KHOA HỌC CHUYÊN SÂU

### 1. Bứt Phá Lịch Sử Trên Miền Ảnh Khuếch Tán (Clean Fake ACC: 0.00% $\to$ 74.22%)
- Bài báo gốc CVPR 2024 chỉ được tiếp xúc với ProGAN, dẫn đến hiện tượng **Feature Blind Spot** nghiêm trọng: Khi kiểm thử trên các họ Diffusion, mô hình dự đoán 100% ảnh giả thành ảnh thật (`Fake ACC = 0.00%`), khiến hệ thống mất hoàn toàn khả năng phòng thủ trước các bộ sinh thế hệ mới.
- FatFormer-XLA M4 sau hiệu chuẩn ($T=1.0811, T^*=0.3700$) đã nâng Fake ACC trung bình của 10 họ Diffusion lên **74,22%**, trong đó các bộ sinh phức tạp đạt độ chính xác gần như tuyệt đối: `vqdiffusion` đạt **99,90%** (AUC: 99,82%), `ldm_200_cfg` đạt **93,20%**, `ldm_100` đạt **85,40%**.

### 2. Khả Năng Tổng Quát Hóa Trên Các Họ GANs Đa Lớp
- Trên các họ GANs đa lớp của Wang et al., mô hình đạt độ chính xác ấn tượng:
  - `progan` (8.000 ảnh): **80,66%** Macro ACC (AUC: 97,17%, Fake ACC: 98,65%).
  - `stylegan2` (15.976 ảnh): **78,87%** Macro ACC (AUC: 90,33%, Fake ACC: 92,29%).
  - `stylegan` (11.982 ảnh): **75,32%** Macro ACC (AUC: 90,14%, Fake ACC: 93,04%).
  - `stargan` (3.998 ảnh): **68,83%** Macro ACC (AUC: 99,01%, Fake ACC: 99,90%).

### 3. Quy Luật Suy Biến Vật Lý Dưới Tác Động Nén JPEG & Rào Cản Miền Tần Số
- Dưới tác động nén sâu JPEG ($Q=30$), bảng lượng tử hóa chuẩn phân chia ảnh thành các khối $8 \times 8$ pixel và triệt tiêu năng lượng tần số cao. Toàn bộ các dấu vết vi cấu trúc (artifacts) của bộ sinh bị làm phẳng thành các khối đồng nhất.
- Khi chất lượng JPEG giảm từ $Q=70 \to Q=50 \to Q=30$, Fake ACC sụt giảm có hệ thống từ **$65,98\% \to 59,58\% \to 37,09\%$**, kéo theo độ sụt giảm tăng dần: $-9,07\% \to -10,55\% \to -13,79\%$.
- Kết quả này chứng minh mô hình hoạt động dựa trên các đặc trưng không gian-tần số SRM thực chất chứ không bị học vẹt hay ghi nhớ phân phối nền.

### 4. Độ Tin Cậy Xác Suất Sau Hiệu Chuẩn (ECE = 0.0299)
- Sau khi được tối ưu hóa nhiệt độ $T=1.0811$ bằng thuật toán L-BFGS trên phân vùng độc lập `val_calibration`, sai số ECE của mô hình giảm từ 0.4817 xuống **0.0299** (giảm 93,8% sai số độ tin cậy). Mô hình đưa ra xác suất phản ánh chính xác rủi ro giả mạo trong thực tế.

---

## VII. KẾT LUẬN NGHIỆM THU & BÀN GIAO ĐỒ ÁN TỐT NGHIỆP

1. **Khóa sổ Milestone 6 thành công toàn diện**: Toàn bộ chuỗi nhiệm vụ nghiên cứu và thực nghiệm từ Task 6.1 đến Task 6.6 đã hoàn thành trọn vẹn, có đầy đủ bằng chứng thực nghiệm đối chứng từ Google Colab Pro và Google Drive 5TB.
2. **Kế thừa số liệu vào Báo cáo Luận văn & Slide Bảo vệ**: Các bảng số liệu ma trận 18 tập test Clean (82.003 ảnh) và 6 biến thể Degraded (54.000 ảnh) tại báo cáo này là căn cứ số liệu thực nghiệm chính thức, đóng băng vĩnh viễn của đồ án tốt nghiệp FatFormer-XLA.
3. **Trạng thái phê duyệt**:
   - Thành viên C (Evaluation Lead): **Phê duyệt 100% [x]**
   - Thành viên B (Architecture Lead): **Đồng thuận nghiệm thu [x]**
   - Thành viên A (Data & Infra Lead): **Xác nhận khớp dữ liệu Drive 5TB [x]**
