# BÁO CÁO NGHIỆM THU NHIỆM VỤ 2.2 (TASK 2.2)
## BENCHMARK BASELINE GỐC TRÊN TOÀN BỘ 18 TẬP CLEAN (CVPR 2024 PAPER REPRODUCTION)

> **Người thực hiện**: Thành viên C *(Training & Evaluation Lead)*  
> **Người nghiệm thu**: Cả nhóm *(Lead A, Lead B, Lead C)*  
> **Thời gian hoàn thành**: 2026-09-28  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC (PASS 100% TIÊU CHUẨN DoD)`  
> **Môi trường & Phần cứng**: Google Colab Pro | GPU NVIDIA Tesla T4 16GB VRAM | Batch size 32 | Workers 4  
> **Tài nguyên lưu trữ**: Google Drive 5TB (`/content/drive/MyDrive/Fatformer/`)  

---

## I. MỤC TIÊU VÀ BỐI CẢNH KỸ THUẬT

1. **Tái lập tính hợp lệ khoa học của bài báo CVPR 2024**:
   - Kiểm chứng mô hình FatFormer gốc (`fatformer_4class_ckpt.pth`) trên toàn bộ 18 tập test ảnh sạch (Clean) nhằm xác nhận độ chính xác khớp chuẩn mực với công bố của tác giả:
     - Trung bình 8 kiến trúc GANs: $\mathbf{98.4\%}$ ACC / $\mathbf{99.5\%}$ AP.
     - Trung bình 10 kiến trúc Diffusion: $\mathbf{95.0\%}$ ACC / $\mathbf{98.0\%}$ AP.
     - Tập kiểm thử khó nhất (Guided Diffusion): $\mathbf{76.1\%}$ ACC.
2. **Thiết lập mốc trần hiệu năng (*Upper Bound Performance*)**:
   - Kết quả này đóng vai trò mốc chuẩn tham chiếu bất biến. Trong các giai đoạn tiếp theo (khi tích hợp SRM 3-Kernels, Dynamic Frequency Gating và Curriculum Degradation), hiệu năng trên ảnh sạch không được phép suy giảm quá $\pm 0.5\%$.
3. **Lưu trữ và bàn giao số liệu**:
   - Toàn bộ kết quả thực nghiệm chi tiết được lưu trực tiếp vào tệp `baseline_clean_results.csv` và `baseline_clean_results.md` trên Google Drive 5TB để phục vụ viết Chương 1 & 2 của Báo cáo đồ án.

---

## II. KẾ THỪA THÔNG SỐ TỪ CÁC TASK TIỀN ĐỀ (RULE 4)

1. **Kế thừa từ Task 1.1** ([`docs/bao_cao/A/bao_cao_task_1.1_ha_tang_drive_io.md`](../A/bao_cao_task_1.1_ha_tang_drive_io.md) - Lead A):
   - Đường dẫn Drive 5TB: `/content/drive/MyDrive/Fatformer`.
   - Các file dữ liệu nén `.tar` nguyên bản:
     - `test_benchmark_gans.tar` (**18.74 GB**): 8 tập GANs (62,003 ảnh).
     - `test_benchmark_diffusion.tar` (**1.22 GB**): 10 tập Diffusion (20,000 ảnh).
   - Giải nén sang SSD NVMe máy ảo Colab tại `/content/dataset_local/test_clean` (tốc độ đọc ~1.2 GB/s, loại bỏ hoàn toàn hiện tượng nghẽn cổ chai I/O).
2. **Kế thừa từ Task 1.2** ([`docs/bao_cao/B/bao_cao_task_1.2_fatformer_model_and_weights_loader.md`](../B/bao_cao_task_1.2_fatformer_model_and_weights_loader.md) - Lead B):
   - Nạp checkpoint với cờ `strict=True`, bảo đảm khớp chính xác tuyệt đối **1.116/1.116 tensors** (~933.16M tham số) của FatFormer gốc.
3. **Kế thừa từ Task 1.4** ([`docs/bao_cao/C/bao_cao_task_1.4_fast_eval_pipeline.md`](bao_cao_task_1.4_fast_eval_pipeline.md) - Lead C):
   - Đo đạc đầy đủ bộ 5 chỉ số chuẩn hóa: **ACC (%)**, **AP (%)**, **ROC-AUC (%)**, **Real ACC (%)**, **Fake ACC (%)**.

---

## III. KẾT QUẢ THỰC NGHIỆM CHI TIẾT TRÊN GOOGLE COLAB (GPU TESLA T4)

### 1. Bảng Dữ Liệu Thực Nghiệm Chi Tiết Toàn Bộ 18 Tập (82,003 Ảnh)

| Nhóm | # | Tên Tập Dữ Liệu | Số Lượng Ảnh | Thời Gian (s) | ACC (%) | AP (%) | AUC (%) | Real ACC (%) | Fake ACC (%) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GANs** | 0 | `progan` | 8,000 | 628.68 | 99.89 | 100.00 | 100.00 | 100.00 | 99.78 |
| **GANs** | 1 | `stylegan` | 11,982 | 939.67 | 97.13 | 99.75 | 99.69 | 100.00 | 94.26 |
| **GANs** | 2 | `stylegan2` | 15,976 | 1252.40 | 98.80 | 99.92 | 99.90 | 99.97 | 97.63 |
| **GANs** | 3 | `biggan` | 4,000 | 311.85 | 99.50 | 99.98 | 99.97 | 99.10 | 99.90 |
| **GANs** | 4 | `cyclegan` | 2,642 | 207.15 | 99.36 | 100.00 | 100.00 | 98.71 | 100.00 |
| **GANs** | 5 | `stargan` | 3,998 | 312.94 | 99.75 | 100.00 | 100.00 | 99.50 | 100.00 |
| **GANs** | 6 | `gaugan` | 10,000 | 779.52 | 99.43 | 100.00 | 100.00 | 98.88 | 99.98 |
| **GANs** | 7 | `deepfake` | 5,405 | 419.21 | 93.27 | 97.99 | 97.01 | 99.45 | 87.06 |
| 📊 | - | **MEAN GANS (8 tập)** | **62,003** | **4851.42** | **98.39** | **99.70** | **99.57** | **99.45** | **97.33** |
| **Diff** | 8 | `guided` | 2,000 | 156.99 | 76.05 | 91.92 | 91.13 | 97.90 | 54.20 |
| **Diff** | 9 | `ldm_200` | 2,000 | 156.55 | 98.55 | 99.83 | 99.78 | 99.30 | 97.80 |
| **Diff** | 10 | `ldm_200_cfg` | 2,000 | 157.61 | 94.85 | 99.22 | 99.02 | 99.30 | 90.40 |
| **Diff** | 11 | `ldm_100` | 2,000 | 156.20 | 98.60 | 99.89 | 99.87 | 99.30 | 97.90 |
| **Diff** | 12 | `glide_50_27` | 2,000 | 157.23 | 94.60 | 99.50 | 99.40 | 99.30 | 89.90 |
| **Diff** | 13 | `glide_100_10` | 2,000 | 157.46 | 94.15 | 99.33 | 99.18 | 99.30 | 89.00 |
| **Diff** | 14 | `glide_100_27` | 2,000 | 157.59 | 94.30 | 99.27 | 99.11 | 99.30 | 89.30 |
| **Diff** | 15 | `dalle` | 2,000 | 157.24 | 98.70 | 99.84 | 99.80 | 99.30 | 98.10 |
| **Diff** | 16 | `pndm` | 2,000 | 156.35 | 99.25 | 99.99 | 99.98 | 100.00 | 98.50 |
| **Diff** | 17 | `vqdiffusion` | 2,000 | 156.86 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 📊 | - | **MEAN DIFFUSION (10 tập)**| **20,000** | **1570.08** | **94.91** | **98.88** | **98.73** | **99.30** | **90.51** |
| 🏆 | - | **MEAN OVERALL (18 TẬP)** | **82,003** | **6421.54 s (107.03 phút)** | **96.45** | **99.25** | **99.10** | **99.37** | **93.54** |

---

## IV. ĐỐI CHIẾU TIÊU CHUẨN NGHIỆM THU (DoD VERIFICATION GATE)

| Tiêu chuẩn nghiệm thu (DoD) | Kỳ Vọng Paper CVPR 2024 | Thực Nghiệm Colab T4 | Sai số $\Delta$ | Ngưỡng cho phép | Đánh giá nghiệm thu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Trung bình 8 họ GANs** | $\mathbf{98.40\%}$ | $\mathbf{98.39\%}$ | $\mathbf{0.01\%}$ | $\le \pm 0.50\%$ | ✅ **PASS XUẤT SẮC** |
| **Trung bình 10 họ Diffusion** | $\mathbf{95.00\%}$ | $\mathbf{94.91\%}$ | $\mathbf{0.09\%}$ | $\le \pm 0.50\%$ | ✅ **PASS XUẤT SẮC** |
| **Tập khó nhất (Guided Diff)** | $\mathbf{76.10\%}$ | $\mathbf{76.05\%}$ | $\mathbf{0.05\%}$ | $\le \pm 0.50\%$ | ✅ **PASS XUẤT SẮC** |
| **Độ phủ dữ liệu** | 18 tập chuẩn | **18 tập (82.003 ảnh)** | $0$ ảnh thiếu | $100\%$ full | ✅ **PASS 100%** |
| **Lưu trữ Drive 5TB** | File CSV và Markdown an toàn | Đã ghi nhận nguyên vẹn | - | Đầy đủ dữ liệu | ✅ **ĐÃ LƯU TRỮ** |

---

## V. CÁC QUAN SÁT KHOA HỌC THEN CHỐT CHO BÁO CÁO (CHƯƠNG 1–2)

1. **Tính Bất Đối Xứng Giữa Real ACC và Fake ACC**:
   - Trên ảnh sạch, mô hình gốc nhận diện ảnh thật gần như hoàn hảo (**Real ACC = 99.37%**).
   - Tuy nhiên, độ chính xác nhận diện ảnh giả (**Fake ACC = 93.54%**) có độ phân tán lớn giữa các kiến trúc sinh:
     - Nhóm GANs đạt Fake ACC **97.33%** (riêng `deepfake` bị sụt xuống **87.06%**).
     - Nhóm Diffusion đạt Fake ACC **90.51%**, trong đó `guided` sụt giảm mạnh nhất chỉ còn **54.20%** (xấp xỉ đoán ngẫu nhiên).
2. **Nguyên Nhân Tử Huyệt Trên Guided Diffusion**:
   - Mô hình Guided Diffusion sử dụng classifier guidance với hệ số lớn để khử nhiễu, làm mịn các đặc trưng vi mô miền tần số cao mà CLIP phụ thuộc. Kết quả thực nghiệm này khẳng định luận điểm khoa học: *Đặc trưng giả mạo vi mô của Diffusion hiện đại rất dễ bị che giấu nếu chỉ dựa vào nhánh ngữ nghĩa Visual CLIP gốc*.
3. **Mốc Trần Tham Chiếu (Upper Bound)**:
   - Chỉ số **96.45% ACC** và **99.25% AP** là mốc trần mốc sàn. Mọi cải tiến tiếp theo của FatFormer-XLA (SRM 3-Kernels, Dynamic Gating, Focal Loss) phải bảo toàn mức hiệu năng này trên ảnh sạch.

---

## VI. BÀN GIAO THÀNH PHẨM

| Tệp Bàn Giao | Đường Dẫn Lưu Trữ | Mô Tả |
| :--- | :--- | :--- |
| **Tệp số liệu CSV** | `/content/drive/MyDrive/Fatformer/log/baseline_clean_results.csv` | 21 dòng chi tiết, đầy đủ 5 chỉ số và thời gian đo |
| **Báo cáo Markdown** | `/content/drive/MyDrive/Fatformer/log/baseline_clean_results.md` | Bảng tổng kết phân nhóm GANs / Diffusion chuẩn |
| **Mã nguồn Full-Eval** | [`tools/full_eval.py`](../../../tools/full_eval.py) | Hỗ trợ `--subsets all`, auto-merge và cổng kiểm tra DoD |
| **Notebook thực thi** | [`notebooks/benchmark_baseline.ipynb`](../../../notebooks/benchmark_baseline.ipynb) | Pipeline Colab khép kín từ giải nén SSD đến xuất kết quả |
