# BÁO CÁO KỸ THUẬT: TASK 7.14
## KIỂM CHUẨN KHÓA SỔ TOÀN DIỆN PHASE 7 (FINAL LOCKED BENCHMARK)

> **Người thực hiện**: Thành viên C (Training & Evaluation Lead)  
> **Người phối hợp**: Thành viên A (Data Infra & Integrity Audit), Thành viên B (Architecture & Router Review)  
> **Phiên bản báo cáo**: `report_schema: v3` (Evidence Level: `measured`)  
> **Commit hash**: `dfd3c4afc802e910670e7b79dc8df6eaa10ce9bc`  
> **Thời điểm khóa sổ**: `2026-11-08T09:11:42 UTC`  
> **Trạng thái**: `[~] ĐÃ CÓ SỐ ĐO THẬT (Chờ hậu kiểm calibration v2, ROC & logic gate)` — Áp dụng Điều khoản kết quả âm §II.3 của Kế hoạch tổng thể

---

## 1. TỔNG QUAN VÀ PHÁN QUYẾT HAI CỔNG CHẶN (GATES VERDICT)

Task 7.14 đại diện cho đợt kiểm chuẩn khóa sổ (Locked Benchmark) tối hậu của Phase 7, được tiến hành theo nguyên tắc đo đạc đơn nhất (Single Execution Run), không chọn lọc kết quả, không tinh chỉnh ngưỡng hậu nghiệm.

Toàn bộ các con số trong báo cáo này được trích xuất trực tiếp từ các file kết quả thực nghiệm GPU lưu tại [`DATASET/logs/phase7/locked_benchmark_phase7/`](../../../DATASET/logs/phase7/locked_benchmark_phase7/) (`final_phase7_report.json`, `indist_phase7.json`, `per_generator_phase7.csv`).

### 1.1. Bảng Tổng Hợp Tiêu Chuẩn Cổng G7.2 & G7.3 (Master Plan §II.3)

| Mã Cổng | Chỉ Tiêu Kỹ Thuật | Ngưỡng Mục Tiêu | Kết Quả Đo Đạc Thực Tế | Trạng Thái | Cơ Chế Nghiệm Thu |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **G7.2** | CFE Clean: Fake @ Real = 90% | $\ge 65.0\%$ | **64.12%** (CI95: $[59.34, 69.77]\%$) | ❌ FAIL | Thiếu 0.88 pp; Liêm chính học thuật |
| **G7.2** | CFE Clean: ROC-AUC | $\ge 78.0\%$ | **88.67%** (CI95: $[87.12, 90.16]\%$) | ✅ **PASS** | Vượt +10.67 pp |
| **G7.2** | CFE Q30: ROC-AUC | $\ge 60.0\%$ | **79.87%** (CI95: $[77.64, 82.06]\%$) | ✅ **PASS** | Vượt +19.87 pp |
| **G7.2** | CFE Q30: Fake @ Real = 90% | $\ge 40.0\%$ | **42.62%** (CI95: $[37.80, 47.75]\%$) | ✅ **PASS** | Vượt +2.62 pp |
| **G7.2** | In-Distribution: Macro ACC (18 tập) | $\ge 90.0\%$ | **90.46%** (90,46035%) | ✅ **PASS** | Vượt +0.46 pp |
| **G7.2** | In-Distribution: Real ACC (Ràng buộc cứng) | $\ge 90.0\%$ | **90.67%** | ✅ **PASS** | Bảo vệ mốc sàn chống báo oan |
| **G7.2** | Worst-Subset Fake ACC (CFE) | $\ge 45.0\%$ | **28.40%** | ❌ FAIL | Miền autoregressive/pixel khó |
| **G7.2** | ECE sau hiệu chuẩn (Q30) | $\le 0.08$ | **0.1349** (13.49%) | ❌ FAIL | Lệch phân phối nén sâu |
| **G7.3** | Single Checkpoint Forward Pass | Bắt buộc | **ĐẠT (1 Checkpoint, 1 `forward()`)** | ✅ **PASS** | Không rẽ nhánh runtime |
| **G7.3** | GPU Latency overhead per image | $< 5.0$ ms | **44.79 ms (Backbone forward B=1)** | ⚠️ Chờ 7.15 | Chưa đo overhead wrapper $\Delta T$ |
| **G7.3** | Delta vs Best Condition Model | $\le 1.5$ pp | **Chưa đánh giá** | ⚠️ Chờ 7.15 | Cần đối đầu trực tiếp trên cùng tập |

### 1.2. Phán Quyết Chính Thức

1. **Cổng G7.2 (Năng Lực Thật & Độ Bền Ngoại Vi)**:
   - Kết quả: Đạt **5/8** chỉ tiêu.
   - Phán quyết: **FAIL (Ghi nhận trung thực theo Điều khoản kết quả âm §II.3 của Kế hoạch tổng thể Master Plan)**.
   - Căn cứ pháp lý đồ án: Khoản II.3 Kế hoạch tổng thể quy định: *"Một kết quả không đạt mục tiêu nhưng được chứng minh bằng thực nghiệm trung thực, chặt chẽ và có giá trị khoa học sẽ được nghiệm thu đầy đủ dưới dạng negative result, không được làm tròn khống hay gọt giũa dữ liệu"*.
2. **Cổng G7.3 (Hợp Nhất Kiến Trúc & Vận Hành)**:
   - Đạt tiêu chuẩn cấu trúc: Mô hình vận hành từ một checkpoint duy nhất, một lượt `forward()` không cần metadata ngoại vi.
   - Chưa đạt tiêu chuẩn độ trễ và chênh lệch: Con số 44.79 ms hiện tại là thời gian forward của toàn bộ backbone CLIP tại batch 1, chưa đo phần overhead do wrapper thêm vào. Tiêu chí chênh lệch $\le 1.5$ pp chưa được đối đầu trực tiếp. Cả hai tiêu chí này được chuyển giao hoàn tất trong Task 7.15.

---

## 2. THÔNG SỐ KẾ THỪA & ĐỐI TƯỢNG KIỂM CHUẨN

* **Checkpoint mục tiêu**: `fatformer_anchored_v2_best.pth` (Epoch 4 từ Task 7.11)
  - Vị trí nguồn Drive: `/content/drive/MyDrive/Fatformer/checkpoint/phase7/anchored_v2_run1/fatformer_anchored_v2_best.pth`
  - Mã băm kiểm tra: `SHA-256 = 5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c`
  - Kiểm tra Canonical Load: `0 missing keys / 0 unexpected keys` (Khớp hoàn hảo 1.200 tensors).
* **Kiến trúc mạng**: `FatFormer-XLA`
  - Backbone: `CLIP:ViT-L/14` (OpenAI pretrained).
  - Tần số: `SRM 3-Kernels` (Horizontal, Vertical, Diagonal High-pass).
  - Điều phối: `Dynamic Frequency Gating` + `3 Bottleneck Adapters` tại các block $\{7, 15, 23\}$ ($n=3$).
* **Lưu ý Quan Trọng Về Hiệu Chuẩn Xác Suất (Calibration)**:
  - Lần chạy này sử dụng cặp thông số tham chiếu của Baseline: $T = 1.6402, T^* = 0.9127$ (kế thừa từ `baseline.json` có SHA-256: `3b9044...`).
  - **Minh bạch học thuật**: Đây **chưa phải** là calibration tối ưu được fit riêng cho checkpoint v2 (`5cd492...`). Ở Task 7.15, checkpoint v2 sẽ được chạy fit calibration độc lập trên `val_calib` để tạo file `anchored_v2.json` chính danh, không giả mạo hash hay áp đặt số liệu cũ.

---

## 3. KẾT QUẢ KIỂM CHUẨN 18 TẬP IN-DISTRIBUTION (82.003 ẢNH)

Phép đo quét qua toàn bộ 82.003 ảnh phòng lab tiêu chuẩn của bài báo gốc chứng minh mô hình giữ vững năng lực trên miền dữ liệu huấn luyện, không bị quên thảm khốc (Catastrophic Forgetting).

### 3.1. Tổng Quan Chỉ Số In-Distribution
* **Tổng số lượng mẫu**: 82.003 ảnh.
* **Macro Accuracy**: **90.46%** (Tính toán lại chính xác: `90.46035%`).
* **Real Accuracy (Mốc sàn bảo vệ)**: **90.67%** (Đạt mục tiêu $\ge 90.0\%$, khống chế nguy cơ báo oan).
* **Micro Accuracy**: **93.43%**.
* **Fake Accuracy**: **96.19%**.
* **ROC-AUC Toàn Cục**: **98.28%** (Đo từ dự đoán tổng hợp).  
  *(Lưu ý: Không dựng đường cong ROC mô phỏng bằng công thức hàm mũ; biểu đồ ROC chỉ biểu diễn các tập có mảng logits từng mẫu thực tế).*

### 3.2. Bảng Phân Tách Chi Tiết 18 Tập Dữ Liệu

| Nhóm | Tập Dữ Liệu (Subset) | Số Mẫu | Macro ACC (%) | Real ACC (%) | Fake ACC (%) | ROC-AUC (%) | AP (%) | Thời Gian Đo (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GANs** | `progan` | 8.000 | 95.56 | 91.12 | 100.00 | 99.96 | 99.94 | 88.89 |
| **GANs** | `stylegan` | 11.982 | 97.21 | 94.89 | 99.53 | 99.78 | 99.75 | 130.49 |
| **GANs** | `stylegan2` | 15.976 | 95.72 | 95.99 | 95.46 | 98.77 | 98.85 | 173.65 |
| **GANs** | `biggan` | 4.000 | 97.00 | 94.40 | 99.60 | 99.56 | 99.51 | 43.90 |
| **GANs** | `cyclegan` | 2.642 | 96.33 | 92.66 | 100.00 | 99.19 | 99.07 | 29.04 |
| **GANs** | `stargan` | 3.998 | 90.25 | 80.49 | 100.00 | 99.75 | 99.74 | 43.76 |
| **GANs** | `gaugan` | 10.000 | 99.02 | 98.04 | 100.00 | 99.89 | 99.80 | 108.89 |
| **GANs** | `deepfake` | 5.405 | 86.74 | 81.75 | 91.73 | 94.32 | 94.36 | 58.87 |
| **Diffusion** | `guided` | 2.000 | 73.30 | 94.30 | 52.30 | 82.93 | 85.04 | 22.13 |
| **Diffusion** | `ldm_200` | 2.000 | 88.60 | 78.30 | 98.90 | 99.04 | 99.26 | 21.94 |
| **Diffusion** | `ldm_200_cfg` | 2.000 | 87.25 | 78.30 | 96.20 | 97.12 | 97.81 | 21.97 |
| **Diffusion** | `ldm_100` | 2.000 | 88.60 | 78.30 | 98.90 | 98.92 | 99.18 | 21.96 |
| **Diffusion** | `glide_50_27` | 2.000 | 82.65 | 78.30 | 87.00 | 90.97 | 93.48 | 21.91 |
| **Diffusion** | `glide_100_10` | 2.000 | 82.60 | 78.30 | 86.90 | 90.89 | 93.46 | 21.95 |
| **Diffusion** | `glide_100_27` | 2.000 | 83.10 | 78.30 | 87.90 | 92.16 | 94.22 | 21.95 |
| **Diffusion** | `dalle` | 2.000 | 88.50 | 78.30 | 98.70 | 98.50 | 98.78 | 21.97 |
| **Diffusion** | `pndm` | 2.000 | 97.95 | 96.20 | 99.70 | 99.91 | 99.91 | 21.99 |
| **Diffusion** | `vqdiffusion` | 2.000 | 97.90 | 96.20 | 99.60 | 99.85 | 99.85 | 21.98 |
| **TỔNG HỢP** | **MEAN GANs** | **62.003** | **94.73** | **91.17** | **98.29** | **98.90** | **98.88** | **677.49 s** |
| **TỔNG HỢP** | **MEAN DIFFUSION** | **20.000** | **87.05** | **83.48** | **90.61** | **95.03** | **96.10** | **219.75 s** |

---

## 4. KẾT QUẢ NGOẠI MIỀN CFE HELD-OUT & 6 BIẾN THỂ SUY BIẾN VẬT LÝ

Kiểm chuẩn ngoại miền được thực hiện trên phân vùng độc lập 1.600 ảnh CFE Held-Out (5 Subsets) qua 7 điều kiện vật lý (Clean + 6 biến thể suy biến `post_resize`). Cả 7 file NPZ đều khớp mã băm và kích thước mảng.

### 4.1. Ma Trận Đa Điều Kiện Suy Biến Vật Lý

| Điều Kiện Suy Biến | Tham Số Kỹ Thuật | ROC-AUC (%) | 95% CI AUC | Fake@Real90 (%) | 95% CI Fake@Real90 | Macro ACC (%) | ECE (%) | NLL |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Clean` | Mốc sàn chuẩn phòng lab | **88.67** | $[87.12, 90.16]$ | **64.12** | $[59.34, 69.77]$ | 73.75 | 9.56 | 0.4816 |
| `downup05` | Thu phóng tỷ lệ 0.5 | **87.00** | $[85.35, 88.76]$ | **64.75** | $[58.68, 68.89]$ | 77.38 | 11.47 | 0.5573 |
| `blur15` | Gaussian blur $\sigma = 1.5$ | **84.22** | $[82.35, 86.02]$ | **50.62** | $[45.43, 56.21]$ | 71.00 | 10.09 | 0.5623 |
| `q75` | Nén JPEG Quality 75 | **83.31** | $[81.34, 85.22]$ | **48.75** | $[43.11, 55.30]$ | 66.19 | 12.80 | 0.5853 |
| `q50` | Nén JPEG Quality 50 | **81.55** | $[79.41, 83.50]$ | **47.62** | $[42.10, 52.32]$ | 66.31 | 13.04 | 0.6147 |
| `q30` | Nén JPEG Quality 30 (Hard) | **79.87** | $[77.64, 82.06]$ | **42.62** | $[37.80, 47.75]$ | 65.31 | 13.49 | 0.6368 |
| `webp40` | Nén WebP Quality 40 | **79.00** | $[76.70, 81.25]$ | **41.88** | $[36.48, 47.84]$ | 65.94 | 14.94 | 0.6894 |

### 4.2. Nhận Xét Quy Luật Suy Biến (Robustness Insights)
1. **Phân cấp suy biến thực tế**: Hiện tượng suy giảm tín hiệu giả mạo diễn ra nghiêm trọng nhất ở các phép nén chuyển đổi khối (Block-transform compression như JPEG Q30 và WebP Q40). Biến dạng không gian liên tục (DownUp0.5, Gaussian Blur 1.5) giữ lại được đặc trưng tần số tốt hơn đáng kể (AUC vẫn đạt $84\% - 87\%$).
2. **Khoảng tin cậy Bootstrap 95%**: Khoảng tin cậy đạt mức $\pm 1.5$ pp trên tập Clean ($[87.12, 90.16]\%$) và dao động đến $\pm 2.21$ pp trên tập nén sâu Q30 ($[77.64, 82.06]\%$), phản ánh sự phân tán lớn hơn khi ảnh chịu suy biến nặng.

---

## 5. PHÂN TÍCH PER-GENERATOR (5 SUBSETS × 4 NHÓM KIẾN TRÚC)

Được trích xuất từ [`per_generator_phase7.csv`](../../../DATASET/logs/phase7/locked_benchmark_phase7/per_generator_phase7.csv):

| Phân Vùng CFE | Họ Kiến Trúc Generator | Số Real | Số Fake | AUC Clean (%) | Fake@Real90 Clean (%) | AUC Q30 (%) | Fake@Real90 Q30 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_commercial` | Diffusion (Midjourney, DALL-E 3) | 237 | 237 | 88.17 | 65.40 | 85.21 | 56.96 |
| `2_latent_diffusion` | Latent Diffusion (SD v1.5, SDXL) | 251 | 241 | 87.45 | 60.17 | 75.70 | 34.02 |
| `3_pixel_diffusion` | Pixel Diffusion (Imagen, DeepFloyd) | 160 | 158 | 88.28 | 53.80 | 78.95 | 30.38 |
| `4_modern_gans` | Modern GANs (StyleGAN-XL, ProGAN) | 78 | 83 | **99.94** | **100.00** | **92.17** | **71.08** |
| `5_other_autoregressive`| Autoregressive (Parti, Muse) | 74 | 81 | **75.76** | **55.56** | **61.19** | **20.99** |

* Nhóm Modern GANs đạt kết quả vượt trội (AUC 99.94% Clean, 92.17% Q30).
* Nhóm Autoregressive và Pixel Diffusion là vùng suy giảm mạnh nhất dưới nén JPEG sâu, dẫn đến chỉ số `Worst-subset Fake ACC` dừng ở 28.40%.

---

## 6. LUẬN GIẢI HỌC THUẬT & TRÁCH NHIỆM PHƯƠNG PHÁP LUẬN

1. **Liêm chính số liệu (Scientific Integrity)**:
   - Chỉ số Clean Fake@Real90 thực tế đạt **64.12%**, thiếu hụt **0.88 pp** so với mục tiêu đặt ra ($\ge 65.0\%$).
   - Báo cáo không làm tròn khống lên 65%, không cherry-pick mẫu, không chỉnh sửa ngưỡng để ép kết quả PASS.
2. **Giá trị của Điều khoản kết quả âm (§II.3)**:
   - Điều khoản kết quả âm ghi nhận tính trung thực của kết quả thực nghiệm, nhưng **không thay thế cho việc kiểm chứng phương pháp luận**.
   - Việc tách biệt calibration của baseline và calibration riêng của v2 là bắt buộc để đảm bảo tính khách quan khoa học.
3. **Ứng viên cho bước đóng gói**: Checkpoint `fatformer_anchored_v2_best.pth` là ứng viên triển vọng nhất hiện có của Phase 7 nhờ giữ vững Real ACC >90% trên 82k ảnh. Tuy nhiên, việc khẳng định tính tối ưu Pareto và mức độ tổng quát toàn diện đòi hỏi phải đối chiếu trực tiếp với các biến thể khác trên cùng tập kiểm thử trong các nghiên cứu tiếp theo.

---

## 7. KẾ HOẠCH HÀNH ĐỘNG CHUYỂN TIẾP (LỘ TRÌNH 6 BƯỚC)

1. **Vá lỗi runner & báo cáo**: Sửa lỗi falsy Python tại `run_locked_benchmark_phase7.py`, bỏ pass cứng G7.3, bỏ ROC mô phỏng. *(Đã hoàn thành)*
2. **Xây dựng công cụ đóng gói & CLI**: Viết `tools/package_unified_fatformer.py` và `tools/infer_unified.py`, kiểm thử local bằng unit test fixture.
3. **Thực thi trên Colab GPU**: Chạy checkpoint v2 trên `val_calib` độc lập để tạo `anchored_v2.json`, đóng gói artifact `unified_fatformer_v2.pth`.
4. **Đối chiếu hành vi**: So sánh dự đoán giữa checkpoint gốc và package trên cùng mẫu thử để chứng minh đóng gói không làm thay đổi hành vi suy luận.
5. **Kiểm thử môi trường độc lập**: Đảm bảo package chạy không cần nạp thêm file rời.
6. **Đo đạc Latency overhead**: Đo thời gian backbone gốc, thời gian qua wrapper, xác định overhead $\Delta T$ (< 5 ms) trên các batch size $1, 8, 32$.
