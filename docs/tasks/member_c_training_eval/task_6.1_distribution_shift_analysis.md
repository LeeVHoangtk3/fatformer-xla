# ĐẶC TẢ NHIỆM VỤ 6.1: PHÂN TÍCH ĐỊNH LƯỢNG HIỆN TƯỢNG LỆCH PHÂN PHỐI
## DIAGNOSING CROSS-GENERATOR DISTRIBUTION SHIFT & ERROR PROFILES

> **Mã nhiệm vụ**: `Task 6.1` *(Milestone 6)*  
> **Người thực hiện chính**: **Thành viên C (Training & Evaluation Lead)**  
> **Người phối hợp phân tích**: **Thành viên B (Architecture Lead)**  
> **Môi trường thực thi**: Môi trường tính toán cục bộ / Phân tích dữ liệu nhật ký Colab  
> **Dữ liệu đầu vào**: Ma trận kiểm thử chi tiết từ Task 5.1 ([`final_full_benchmark_matrix.csv`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv)) và tệp nhật ký dự đoán thô per-sample (nếu có sẵn).  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH NGHIỆM THU`  
> **Nguyên tắc kỹ thuật**: **Hoàn toàn không sửa đổi trọng số mô hình**. Tuyệt đối **không chạy lại Task 5.1 để quét ngưỡng (tune threshold)** nhằm tránh biến tập test thành tập điều chỉnh tham số.

---

## 1. MỤC TIÊU VÀ BỐI CẢNH KỸ THUẬT

1. **Bóc Tách Hiện Tượng Lệch Phân Phối Của Từng Bộ Sinh**:
   - Từ kết quả đo đạc thực tế của Task 5.1, mô hình Plan A biểu hiện sự phân hóa rõ rệt giữa các bộ sinh (ProGAN đạt 97,95% ACC trong khi GauGAN rơi xuống 45,02%, PNDM 49,60%, DALL-E 48,80%).
   - Cần tính toán và phân tích phân tầng cho từng bộ sinh các chỉ số: **Fake Recall**, **Real Recall**, **ROC-AUC / AP**, và sự phân hóa giữa nhánh GANs vs Diffusion.
2. **Quy Chuẩn Kiểm Tra Đầu Vào Thực Tế (DoD Tiền Đề)**:
   - Rà soát xác minh sự tồn tại của tệp nhật ký dự đoán thô từng ảnh (logits / raw probabilities).
   - **Kịch bản A (Có tệp log thô)**: Tính toán phân vị xác suất, đường cong ECE và biểu đồ phân bổ độ tin cậy.
   - **Kịch bản B (Chỉ có ma trận tổng hợp CSV)**: Phân tích dựa trên các chỉ số phân tầng có sẵn (Macro ACC, Real ACC, Fake ACC, AP, độ sụt giảm $\Delta_{\text{drop}}$). Toàn bộ việc quét ngưỡng chi tiết và Temperature Scaling sẽ được thực thi trên tập kiểm định độc lập `val_calibration` tại Task 6.4.

---

## 2. KẾ HOẠCH NỘI DUNG VÀ CÁC BƯỚC THỰC HIỆN

### Bước 1: Rà Soát & Xác Minh Đầu Vào Dữ Liệu
- Kiểm tra kho lưu trữ xem có lưu tệp log xác suất per-sample của đợt chạy Task 5.1 hay không.
- Nếu không có: Khóa phạm vi phân tích ở mức số liệu tổng hợp phân tầng, tuyệt đối không chạy lại mã nguồn trên 18 tập test Task 5.1 chỉ để lấy logit phục vụ dò ngưỡng.

### Bước 2: Phân Tích Phân Tầng Theo Nhóm Kiến Trúc Bộ Sinh
- Tính toán độ lệch Fake ACC và Real ACC trên 8 kiến trúc GANs và 10 kiến trúc Diffusion ở cả 2 trạng thái Clean và Degraded $Q=30$.
- Phân nhóm các cụm thách thức:
  - Cụm 1 - Suy giảm nhận diện ảnh giả trên ảnh sạch: GauGAN, PNDM, DALL-E.
  - Cụm 2 - Suy giảm nhận diện ảnh giả dưới nén sâu: Deepfake, StyleGAN2, Glide.
  - Cụm 3 - Nhóm bảo tồn tốt: ProGAN, StarGAN, VQDiffusion.

### Bước 3: Phân Định Nguyên Nhân Gây Lỗi (Feature vs Metric)
- So sánh chỉ số AP (Average Precision) và Accuracy:
  - Nếu AP vẫn duy trì mức cao (trên 70%) nhưng ACC thấp quanh mức 50% $\rightarrow$ Dấu hiệu của việc lệch điểm cắt phân loại do hàm mất mát gây ra.
  - Nếu cả AP và AUC đều sụt giảm sâu (dưới 50%) $\rightarrow$ Bắt nguồn từ việc biểu diễn đặc trưng của bộ sinh bị hòa lẫn với ảnh thật.

### Bước 4: Lập Báo Cáo Chẩn Đoán Học Thuật
- Tổng hợp toàn bộ phát hiện thành báo cáo kỹ thuật chi tiết: `docs/bao_cao/C/bao_cao_task_6.1_distribution_shift_analysis.md`.
- Đưa ra khuyến nghị định lượng về danh mục bộ sinh cần bổ sung cho Task 6.2 và định hướng hàm mất mát cho Task 6.3-B.

---

## 3. TIÊU CHUẨN HOÀN THÀNH (Definition of Done)

- [x] Xác minh trạng thái của log dự đoán thô (xác nhận kịch bản A hoặc B minh bạch - đã kích hoạt Kịch bản B).
- [x] Lập bảng số liệu chẩn đoán phân tầng per-generator trên 18 tập test (Fake Recall, Real Recall, AP, $\Delta_{\text{drop}}$).
- [x] Phân định rõ ràng các nhóm bộ sinh bị mất đặc trưng thực sự so với nhóm bị ảnh hưởng bởi độ lệch điểm cắt.
- [x] Hoàn tất báo cáo nghiệm thu Task 6.1 làm tiền đề trực tiếp cho việc thu thập dữ liệu tại Task 6.2 ([báo cáo Task 6.1](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/C/bao_cao_task_6.1_distribution_shift_analysis.md)).
