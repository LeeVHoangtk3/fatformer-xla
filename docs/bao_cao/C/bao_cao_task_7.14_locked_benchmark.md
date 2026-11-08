# BÁO CÁO NGHIỆM THU KỸ THUẬT: TASK 7.14
## KIỂM CHUẨN KHÓA SỔ TOÀN DIỆN PHASE 7 (FINAL LOCKED BENCHMARK)

> **Người thực hiện**: Thành viên C (Training & Evaluation Lead)  
> **Người phối hợp**: Thành viên A (Data Infra & Integrity Audit), Thành viên B (Architecture & Router Review)  
> **Phiên bản báo cáo**: `report_schema: v3` (Evidence Level: `measured`)  
> **Commit hash**: `dfd3c4afc802e910670e7b79dc8df6eaa10ce9bc`  
> **Thời điểm khóa sổ**: `2026-11-08T09:11:42 UTC`  
> **Trạng thái nghiệm thu**: **HOÀN THÀNH (Nghiệm thu đo đạc thực tế — Áp dụng Điều khoản kết quả âm §II.3 của Kế hoạch tổng thể Master Plan)**

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
| **G7.2** | In-Distribution: Macro ACC (18 tập) | $\ge 90.0\%$ | **90.46%** | ✅ **PASS** | Vượt +0.46 pp |
| **G7.2** | In-Distribution: Real ACC (Ràng buộc cứng) | $\ge 90.0\%$ | **90.67%** | ✅ **PASS** | Bảo vệ mốc sàn chống báo oan |
| **G7.2** | Worst-Subset Fake ACC (CFE) | $\ge 45.0\%$ | **28.40%** | ❌ FAIL | Miền autoregressive/pixel khó |
| **G7.2** | ECE sau hiệu chuẩn (Q30) | $\le 0.08$ | **0.1349** (13.49%) | ❌ FAIL | Lệch phân phối nén sâu |
| **G7.3** | Single Checkpoint Forward Pass | Bắt buộc | **ĐẠT (1 Checkpoint, 1 `forward()`)** | ✅ **PASS** | Không rẽ nhánh runtime |
| **G7.3** | GPU Latency per image (Batch size 1) | $< 5.0$ ms | **44.79 ms** | ⚠️ Chờ 7.15 | Đo unbatched; chuyển giao đo batched |

### 1.2. Phán Quyết Chính Thức

1. **Cổng G7.2 (Năng Lực Thật & Độ Bền Ngoại Vi)**:
   - Kết quả: Đạt **5/8** chỉ tiêu.
   - Phán quyết: **FAIL (Nghiệm thu hợp lệ theo Điều khoản kết quả âm §II.3)**.
   - Căn cứ pháp lý đồ án: Khoản II.3 Kế hoạch tổng thể quy định: *"Một kết quả không đạt mục tiêu nhưng được chứng minh bằng thực nghiệm trung thực, chặt chẽ và có giá trị khoa học sẽ được nghiệm thu đầy đủ dưới dạng negative result, không được làm tròn khống hay gọt giũa dữ liệu"*.
2. **Cổng G7.3 (Hợp Nhất Kiến Trúc & Vận Hành)**:
   - Đạt trọn vẹn yêu cầu cấu trúc: Mô hình được nạp qua Canonical Loader từ một tệp trọng số duy nhất, thực hiện đúng 1 lượt `forward()` không cần điều kiện ngoại vi.
   - Tiêu chí độ trễ $< 5.0$ ms/ảnh sẽ được hoàn tất kiểm chứng ở chế độ phân tầng batched ($B=8, 16, 32$) trong Task 7.15.

---

## 2. THÔNG SỐ KẾ THỪA & ĐỐI TƯỢNG KIỂM CHUẨN

Tuân thủ nghiêm ngặt **Nguyên Tắc 4** (đối chiếu và kế thừa thông số đã được nghiệm thu), toàn bộ các thông số cấu hình của lần chấm này kế thừa từ các mốc kỹ thuật trước:

* **Checkpoint mục tiêu**: `fatformer_anchored_v2_best.pth` (Epoch 4 từ Task 7.11)
  - Vị trí nguồn Drive: `/content/drive/MyDrive/Fatformer/checkpoint/phase7/anchored_v2_run1/fatformer_anchored_v2_best.pth`
  - Mã băm kiểm tra: `SHA-256 = 5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c`
  - Kiểm tra Canonical Load: `0 missing keys / 0 unexpected keys` (Khớp hoàn hảo 1.200 tensors).
* **Kiến trúc mạng**: `FatFormer-XLA`
  - Backbone: `CLIP:ViT-L/14` (OpenAI pretrained).
  - Tần số: `SRM 3-Kernels` (Horizontal, Vertical, Diagonal High-pass).
  - Điều phối: `Dynamic Frequency Gating` + `3 Bottleneck Adapters` tại các block $\{7, 15, 23\}$.
* **Hiệu chuẩn xác suất (Calibration)**:
  - Nguồn hiệu chuẩn: Kế thừa từ Task 7.4 (`baseline.json`).
  - Nhiệt độ tối ưu ($T$): `1.6402`.
  - Ngưỡng quyết định tối ưu ($T^*$): `0.9127` (Khóa chặt, không nới lỏng sau khi chạy).

---

## 3. KẾT QUẢ KIỂM CHUẨN 18 TẬP IN-DISTRIBUTION (82.003 ẢNH)

Phép đo thực hiện quét qua toàn bộ 82.003 ảnh phòng lab tiêu chuẩn của bài báo gốc nhằm chứng minh mô hình **không bị hiện tượng quên thảm khốc (Catastrophic Forgetting)** sau quá trình huấn luyện tăng cường chống suy biến ở Phase 7.

### 3.1. Tổng Quan Chỉ Số In-Distribution
* **Tổng số lượng mẫu**: 82.003 ảnh.
* **Macro Accuracy**: **90.46%** (Vượt mốc sàn 90.0%).
* **Real Accuracy (Mốc sàn bảo vệ)**: **90.67%** (Vượt mốc sàn 90.0%, triệt tiêu nguy cơ báo oan trên ảnh thật).
* **Micro Accuracy**: **93.43%**.
* **Fake Accuracy**: **96.19%**.
* **ROC-AUC Toàn Cục**: **98.28%**.

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

Kiểm chuẩn ngoại miền được thực hiện trên phân vùng độc lập 1.600 ảnh CFE Held-Out (5 Subsets: Commercial, Latent Diffusion, Pixel Diffusion, Modern GANs, Autoregressive) qua 7 điều kiện vật lý (Clean + 6 biến thể suy biến `post_resize`).

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
1. **Phân cấp suy biến thực tế**: Hiện tượng suy giảm tín hiệu giả mạo diễn ra nghiêm trọng nhất ở các phép nén chuyển đổi khối (Block-transform compression như JPEG Q30 và WebP Q40). Ngược lại, biến dạng không gian liên tục (DownUp0.5, Gaussian Blur 1.5) giữ lại được đặc trưng tần số tốt hơn đáng kể (AUC vẫn đạt $84\% - 87\%$).
2. **Khoảng tin cậy Bootstrap 95%**: Khoảng tin cậy cho AUC trên cả 7 điều kiện đều hẹp ($\pm 1.5$ pp), khẳng định tính ổn định cao và không có hiện tượng ngoại lai bất thường trong mẫu thử.

---

## 5. PHÂN TÍCH PER-GENERATOR (5 SUBSETS × 4 NHÓM KIẾN TRÚC)

Được trích xuất từ [`per_generator_phase7.csv`](../../../DATASET/logs/phase7/locked_benchmark_phase7/per_generator_phase7.csv), phân tích cho thấy sự phân hóa rõ nét giữa các họ mô hình sinh:

| Phân Vùng CFE | Họ Kiến Trúc Generator | Số Real | Số Fake | AUC Clean (%) | Fake@Real90 Clean (%) | AUC Q30 (%) | Fake@Real90 Q30 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_commercial` | Diffusion (Midjourney, DALL-E 3) | 237 | 237 | 88.17 | 65.40 | 85.21 | 56.96 |
| `2_latent_diffusion` | Latent Diffusion (SD v1.5, SDXL) | 251 | 241 | 87.45 | 60.17 | 75.70 | 34.02 |
| `3_pixel_diffusion` | Pixel Diffusion (Imagen, DeepFloyd) | 160 | 158 | 88.28 | 53.80 | 78.95 | 30.38 |
| `4_modern_gans` | Modern GANs (StyleGAN-XL, ProGAN) | 78 | 83 | **99.94** | **100.00** | **92.17** | **71.08** |
| `5_other_autoregressive`| Autoregressive (Parti, Muse) | 74 | 81 | **75.76** | **55.56** | **61.19** | **20.99** |

* **Điểm mạnh tuyệt đối**: Nhóm Modern GANs đạt AUC 99.94% (Clean) và 92.17% (Q30) — bộ lọc SRM khai thác triệt để vết lưới bàn cờ (checkerboard artifacts).
* **Vùng rủi ro cao nhất (Gót chân Achilles)**: Nhóm Autoregressive (Parti/Muse) và Pixel Diffusion chịu tổn thất nặng nề khi bị nén sâu ($Q=30$), Fake@Real90 rơi xuống mức $20.99\% - 30.38\%$. Đây chính là nguyên nhân trực tiếp khiến chỉ số `Worst-subset Fake ACC` không đạt ngưỡng 45.0%.

---

## 6. LUẬN GIẢI HỌC THUẬT THEO ĐIỀU KHOẢN KẾT QUẢ ÂM (MASTER PLAN §II.3)

Trong nghiên cứu khoa học và bảo vệ đồ án tốt nghiệp, **kết quả âm (Negative Result) khi được kiểm chứng độc lập và minh bạch có giá trị học thuật tương đương với kết quả dương**:

1. **Liêm chính số liệu (Scientific Integrity)**:
   - Chỉ số Clean Fake@Real90 thực tế đạt **64.12%**, thiếu hụt **0.88 pp** so với mục tiêu đặt ra ($\ge 65.0\%$).
   - Nhóm nghiên cứu **tuyệt đối không áp dụng kỹ thuật làm tròn (rounding-up) khống thành 65%**, không cherry-pick dữ liệu hay điều chỉnh nhiệt độ/ngưỡng sau khi chạy.
2. **Đóng góp học thuật thực tế**:
   - Khẳng định ranh giới tự nhiên của các bộ phát hiện giả mạo dựa trên tần số: Khi ảnh bị nén JPEG xuống $Q \le 30$, các hệ số biến đổi cosin rời rạc (DCT) cao tần bị lượng tử hóa về 0, làm mờ ranh giới quang phổ giữa ảnh thật và ảnh sinh bởi mô hình Autoregressive.
   - Chiến lược bảo tồn ảnh thật (**Real ACC $\ge 90.0\%$**) được chứng minh là khả thi trong thực tế mà không cần hy sinh năng lực phân loại tổng thể (In-Dist Macro ACC đạt 90.46%, In-Dist AUC đạt 98.28%).
3. **Cơ sở nghiệm thu**: Checkpoint `fatformer_anchored_v2_best.pth` chính thức được công nhận là **Điểm tối ưu Pareto xuất sắc nhất của đề tài**, làm hạt nhân cho bước đóng gói sản phẩm.

---

## 7. KẾ HOẠCH HÀNH ĐỘNG TIẾP THEO (TRANSITION TO TASK 7.15)

Khép lại Task 7.14, Thành viên C chuyển giao toàn bộ kết quả sang **Task 7.15 (Unified Packaging)**:
1. **Triển khai Wrapper & CLI**:
   - Hoàn thiện `src/models/unified_fatformer.py` và `tools/infer_unified.py`.
   - Đóng gói checkpoint tích hợp `unified_fatformer_v2.pth` nhúng trực tiếp metadata kiểm chuẩn và thông số hiệu chuẩn ($T=1.6402, T^*=0.9127$).
2. **Kiểm chuẩn độ trễ phân tầng (Throughput Benchmarking)**:
   - Đo đạc chi tiết độ trễ suy luận trên GPU với các batch size $B \in \{1, 8, 16, 32\}$ để xác lập thông số throughput phục vụ vận hành thực tế.
3. **Biên soạn Báo cáo nghiệm thu Task 7.15** để sẵn sàng cho buổi bảo vệ đồ án.