# BÁO CÁO NGHIỆM THU NHIỆM VỤ 2.1 (TASK 2.1)
## XÂY DỰNG BỘ KIỂM THỬ SUY BIẾN VẬT LÝ (DEGRADED TEST SET)

> **Người thực hiện**: Thành viên A *(Data & Infra Lead)*  
> **Người nghiệm thu**: Cả nhóm *(Lead A, Lead B, Lead C)*  
> **Thời gian hoàn thành**: 2026-09-29  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC (PASS 100% TIÊU CHUẨN DoD)`  
> **Môi trường & Phần cứng**: Google Colab CPU đa nhân (2 vCPU) | Đa tiến trình `ProcessPoolExecutor` (Workers = 4)  
> **Tài nguyên lưu trữ**: Google Drive 5TB (`/content/drive/MyDrive/Fatformer/datasets/test_degraded.tar`)  

---

## I. MỤC TIÊU VÀ BỐI CẢNH KỸ THUẬT

1. **Chuẩn hóa bộ dữ liệu kiểm thử suy biến mạng xã hội**:
   - Hiện thực hóa pipeline tự động áp dụng 6 biến thể suy biến vật lý phổ biến nhất trên Internet lên toàn bộ 18 tập test Clean nguyên bản của bài báo CVPR 2024:
     - **Nén JPEG sâu $Q=30$**: Mô phỏng thuật toán nén nặng của các nền tảng chat/mạng xã hội (WeChat, Facebook Messenger), tạo artifact khối $8 \times 8$ cường độ cao - là "tử huyệt" làm sụp đổ các mô hình dựa trên Visual CLIP.
     - **Nén JPEG trung bình $Q=50$** và **nhẹ $Q=70$**: Đo đạc độ dốc sụt giảm hiệu năng theo chất lượng nén.
     - **Làm mờ Gauss (Gaussian Blur)**: $\sigma = 1.0$ và $\sigma = 2.0$, triệt tiêu các đặc trưng vi mô tần số cao.
     - **Biến dạng Downsample-Upsample (Down-Up)**: Thu nhỏ $224 \rightarrow 112$ rồi phóng to lại $224$ (Bicubic).
2. **Bảo toàn cấu trúc phân loại nhị phân**:
   - Giữ nguyên $100\%$ cấu trúc thư mục con `0_real/` và `1_fake/`, đổi đuôi ảnh đồng nhất sang `.jpg` để tương thích hoàn toàn với DataLoader chuẩn của FatFormer.
3. **Mở khóa điểm nghẽn cho Task 2.3**:
   - Đóng gói toàn bộ tập dữ liệu thành `test_degraded.tar` đẩy lên Google Drive 5TB, trực tiếp cung cấp đầu vào để Thành viên C thực thi Task 2.3 (Benchmark mốc sàn suy giảm).

---

## II. KẾ THỪA THÔNG SỐ TỪ CÁC TASK TIỀN ĐỀ (RULE 4)

1. **Kế thừa từ Task 1.1** ([`docs/bao_cao/A/bao_cao_task_1.1_ha_tang_drive_io.md`](bao_cao_task_1.1_ha_tang_drive_io.md) - Lead A):
   - Đường dẫn Drive 5TB: `/content/drive/MyDrive/Fatformer/`.
   - Giải nén từ `test_benchmark_gans.tar` (18.74 GB) và `test_benchmark_diffusion.tar` (1.22 GB) sang ổ SSD NVMe máy ảo Colab tại `/content/dataset_local/test_clean` (tốc độ đọc ~1.2 GB/s).
2. **Kế thừa từ Task 2.2** ([`docs/bao_cao/C/bao_cao_task_2.2_baseline_clean_benchmark.md`](../C/bao_cao_task_2.2_baseline_clean_benchmark.md) - Lead C):
   - Tập Clean nguyên bản đã được xác nhận độ chính xác mốc trần (**96.45% ACC** / **99.25% AP**).

---

## III. KẾT QUẢ THỰC NGHIỆM CHI TIẾT TRÊN GOOGLE COLAB

### 1. Số Liệu Sinh Ảnh Chi Tiết Theo Từng Biến Thể

Script đa tiến trình [`tools/make_degraded.py`](../../../tools/make_degraded.py) đã thực thi qua notebook chuyên dụng [`notebooks/task/task_2.1/task_2.1_make_degraded.ipynb`](../../../notebooks/task/task_2.1/task_2.1_make_degraded.ipynb):

| # | Tên Biến Thể | Quy Chuẩn Vật Lý | Số Lượng Ảnh Tạo Mới | Thời Gian (s) | Tốc Độ Xử Lý | Số Lỗi |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `jpeg_q30` | Nén lượng tử JPEG $Q=30$ | 110,329 | 691.66 s | 160.04 ảnh/s | 0 |
| 2 | `jpeg_q50` | Nén lượng tử JPEG $Q=50$ | 110,329 | 698.70 s | 158.40 ảnh/s | 0 |
| 3 | `jpeg_q70` | Nén lượng tử JPEG $Q=70$ | 110,329 | 719.87 s | 153.71 ảnh/s | 0 |
| 4 | `blur_s1` | Gaussian Blur $\sigma=1.0$ | 110,329 | 1357.02 s | 81.43 ảnh/s | 0 |
| 5 | `blur_s2` | Gaussian Blur $\sigma=2.0$ | 110,329 | 1356.26 s | 81.48 ảnh/s | 0 |
| 6 | `down_up` | Down-Up Bicubic ($224 \rightarrow 112 \rightarrow 224$) | 110,329 | 892.47 s | 123.92 ảnh/s | 0 |
| 🏆 | **TỔNG CỘNG** | **6 Biến Thể Suy Biến** | **661,974 ảnh** | **5716.40 s (95.27 phút)** | **~115.8 ảnh/s** | **0 (PASS)** |

### 2. Kiểm Toán Tính Toàn Vẹn Cấu Trúc (Audit Verification)
- Tất cả 6 thư mục biến thể con trong `/content/dataset_local/test_degraded/` đều đạt chính xác tuyệt đối **110.329 ảnh (.jpg)**.
- Toàn bộ các cặp thư mục `0_real/` và `1_fake/` của 18 subsets được nhân bản nguyên vẹn, không có hiện tượng mất mát nhãn hay lỗi tệp 0 byte.

### 3. Đóng Gói Lưu Trữ Trên Google Drive 5TB (Giao Thức SSD NVMe + Flush Cache)
- **Quy trình thực thi**: Đóng gói nội bộ trên SSD NVMe máy ảo (`/content/test_degraded.tar`, 550.0s) -> Sao chép nguyên khối sang Drive (237.8s) -> Ép đồng bộ bộ nhớ đệm `drive.flush_and_unmount()`.
- **Đường dẫn tệp**: `/content/drive/MyDrive/Fatformer/datasets/test_degraded.tar`
- **Dung lượng thực tế**: **19.55 GB** (xác nhận tồn tại vĩnh viễn trên Drive 5TB).

---

## IV. ĐỐI CHIẾU TIÊU CHUẨN NGHIỆM THU (DoD VERIFICATION GATE)

| Tiêu chuẩn nghiệm thu (DoD) | Yêu cầu thiết kế | Thực nghiệm Colab | Đánh giá nghiệm thu |
| :--- | :--- | :--- | :---: |
| **Độ ổn định & Bộ nhớ** | Chạy đa tiến trình không lỗi, không rò rỉ RAM | 661.974 lượt chuyển đổi ảnh thành công với 0 lỗi | ✅ **PASS XUẤT SẮC** |
| **Bảo toàn nhãn nhị phân** | Đầy đủ `0_real/` và `1_fake/` trên toàn bộ tập | Đạt chính xác 110.329 ảnh / biến thể | ✅ **PASS 100%** |
| **Lưu trữ Drive 5TB** | Tệp `test_degraded.tar` sẵn sàng trên Drive | Đạt 19.55 GB tại `datasets/test_degraded.tar` (Flush verified) | ✅ **ĐÃ LƯU TRỮ VĨNH VIỄN** |

---

## V. Ý NGHĨA KHOA HỌC VÀ BÀN GIAO CHO TASK 2.3

1. **Giải phóng điểm nghẽn toàn diện**:
   - Thành viên C đã có thể giải nén `test_degraded.tar` và chạy ngay `tools/full_eval.py` / `tools/fast_eval.py` trên các tập nén để xác lập mốc sàn sụt giảm (Task 2.3).
2. **Cơ sở định lượng cho Báo cáo Chương 1–2**:
   - Bộ dữ liệu này chính là công cụ thực nghiệm để chứng minh tử huyệt của FatFormer gốc dưới nén sâu $Q=30$.
