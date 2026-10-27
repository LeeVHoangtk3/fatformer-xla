# TASK 6.6: KIỂM CHUẨN CUỐI CÙNG & ĐỐI CHIẾU HẬU NGHIỆM MINH BẠCH

- **Mã nhiệm vụ**: `Task 6.6` *(Milestone 6 - Tuần 6)*
- **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**
- **Người nghiệm thu**: **Cả nhóm (Lead A, Lead B, Lead C)**
- **Môi trường thực thi**: ☁️ **Colab GPU (T4 / A100)**
- **Trạng thái**: `[~] HOÀN THÀNH BENCHMARK CÓ ĐIỀU KIỆN (M4 FALLBACK, CHƯA ĐẠT GATE ROBUSTNESS Q=30)`

---

## 1. Mục Tiêu & Cơ Sở Khoa Học

Để bảo đảm tính liêm chính học thuật và phản ánh khách quan năng lực của mô hình sau chiến dịch Milestone 6, bài kiểm chuẩn cuối cùng được thiết lập theo cấu trúc 2 tầng minh bạch:

1. **Tầng 1 - Benchmark Đối Chiếu Hậu Nghiệm (Post-hoc Comparative Benchmark)**:
   - Đo đạc lại trên 18 tập test của bài báo gốc CVPR 2024 (ở cả 2 trạng thái Clean và Degraded $Q \in \{30, 50, 70\}$).
   - Khai báo rõ ràng trong báo cáo khoa học: Đây là phép đo đối chiếu hậu nghiệm nhằm so sánh trực tiếp sự cải thiện trước và sau khi can thiệp Milestone 6 (Baseline Gốc vs Plan A vs Mô hình Đa Miền).
2. **Tầng 2 - Đánh Giá Trên Phân Vùng Held-out Selection (Held-out Selection-Split Evaluation)**:
   - Đánh giá trên phân vùng dữ liệu `val_selection` (270 ảnh Midjourney v5 & SD v1.5) tách biệt với pha hiệu chuẩn nhiệt độ `val_calibration` nhằm xác nhận hiệu năng nội bộ sau hiệu chuẩn.
   - *Lưu ý học thuật*: Phân vùng này đã tham gia quá trình chấm điểm chọn checkpoint M4 tại Task 6.4, do đó không thay thế cho bài kiểm thử ngoại vi độc lập (*external unseen test* như `test_final_v2` với manifest và kiểm tra overlap = 0).

---

## 2. Quy Tắc Thực Nghiệm Khóa Bất Biến

1. **Thực thi kiểm chuẩn đúng 1 lượt**: Toàn bộ quá trình kiểm chuẩn chỉ được chạy một lượt, không lặp lại để thử nghiệm tham số.
2. **Đóng băng hoàn toàn mô hình và ngưỡng**: Trọng số mô hình và cặp tham số hiệu chuẩn $(T, T^*)$ được cố định nguyên vẹn từ kết quả của Task 6.4, tuyệt đối không tinh chỉnh trên dữ liệu kiểm thử.
3. **Khoảng tin cậy thống kê**: Mọi chỉ số trung bình Macro ACC, AP, AUC phải đi kèm khoảng tin cậy 95% tính toán bằng phương pháp Bootstrap ($B = 1.000$ lần lấy mẫu lại).

---

## 3. Nội Dung Thực Hiện Chi Tiết

### Bước 1: Nạp Checkpoint Tuyển Chọn & Tham Số Hiệu Chuẩn
- Nạp tệp trọng số rút gọn `fatformer_crossdomain_weights_only.pth` từ Task 6.4.
- Thiết lập nhiệt độ $T$ và ngưỡng phân loại $T^*$ đã đóng băng.

### Bước 2: Thực thi vòng đo đạc đối chiếu hậu nghiệm trên 18 tập test
- Chạy script kiểm chuẩn:
  ```bash
  python tools/eval_locked_benchmark.py \
      --checkpoint /content/checkpoint/fatformer_crossdomain_weights_only.pth \
      --temperature_calibrated \
      --threshold_calibrated \
      --test_dir /content/test_benchmark \
      --output_csv DATASET/logs/final_posthoc_benchmark_matrix.csv
  ```

### Bước 3: Đánh giá trên tập kiểm chuẩn độc lập mở rộng (`test_final_v2`)
- Chạy đánh giá trên tập kiểm chuẩn độc lập để kiểm chứng khả năng phát hiện trên các kiến trúc bộ sinh hoàn toàn mới.
- Xuất tệp `DATASET/logs/final_independent_test_matrix.csv`.

### Bước 4: Lập bảng đối chiếu 3 bên & Báo cáo chung cuộc
- Lập bảng tổng hợp so sánh trực diện:
  1. Mô hình Baseline gốc CVPR 2024.
  2. Mô hình sau đợt finetune Plan A (Task 5.1).
  3. Mô hình Đa Miền sau Milestone 6.
- Đánh giá sự cải thiện về độ bền vững trước nhiễu nén mạng xã hội và khả năng bảo toàn trên ảnh sạch.

---

## 4. Tiêu Chuẩn Nghiệm Thu (Definition of Done)

- [x] Hoàn thành vòng đo đạc đối chiếu hậu nghiệm trên 18 tập test hai chiều (Clean + Degraded) đúng 1 lượt ([`tools/eval_locked_benchmark.py`](../../../tools/eval_locked_benchmark.py)).
- [~] External unseen test: Chưa thực hiện do chưa có bộ dữ liệu `test_final_v2` được kiểm chứng overlap; sử dụng phân vùng `val_selection` làm held-out evaluation sau calibration, không thay thế external test.
- [x] Xuất đầy đủ các bảng dữ liệu CSV chi tiết và bảng tổng hợp 3 bên kèm khoảng tin cậy 95% Bootstrap phân định rõ ràng giữa Estimator Cấp Subset (Cluster CI) và Cấp Ảnh Gộp (Sample CI) ([`task_6.6_three_way_comparison.png`](../../../DATASET/logs/task_6.6_three_way_comparison.png)).
- [~] Trạng thái nghiệm thu: Hoàn thành benchmark khóa sổ; checkpoint M4 báo cáo ở trạng thái fallback có điều kiện từ Task 6.4 (chưa đạt gate robustness Q=30).
- [x] Biên soạn báo cáo nghiệm thu tại [`docs/bao_cao/C/bao_cao_task_6.6_final_locked_benchmark.md`](../../bao_cao/C/bao_cao_task_6.6_final_locked_benchmark.md).
