# TASK 3.7: SOẠN THẢO BÁO CÁO ĐỒ ÁN - CHƯƠNG 3 (PHƯƠNG PHÁP LUẬN)

- **Mã nhiệm vụ**: `Task 3.7`
- **Người phụ trách**: **Thành viên B** *(Architecture & Loss Lead)*
- **Môi trường thực hiện**: 🖥️ **Local** (LaTeX / Word / Overleaf)
- **Thời hạn hoàn thành**: Tuần 3 (Ngày 21)
- **Độ ưu tiên**: 🟢 Bình thường (Thực thi chiến lược viết báo cáo cuốn chiếu)
- **Đầu vào (Input)**:
  - Sơ đồ kiến trúc 4 tầng của pipeline FatFormer-XLA.
  - Công thức Haar Wavelet 2D DWT, ma trận lọc SRM 3-Kernels, công thức Sigmoid gating $\lambda(x)$, giáo trình Curriculum và hàm mục tiêu Dual-Stream Focal Loss.
- **Đầu ra (Output)**:
  - Bản thảo **Chương 3 (Phương pháp luận & Kiến trúc đề xuất FatFormer-XLA)** dạng PDF/LaTeX kèm sơ đồ khối vector độ nét cao.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Soạn thảo chương quan trọng nhất của báo cáo đồ án (chương chiếm trọng số điểm khoa học cao nhất trước hội đồng).
2. Chuẩn mực hóa các công thức toán học và biểu đồ luồng dữ liệu, chứng minh sự cộng hưởng hữu cơ giữa 3 chiến lược:
   - Chiến lược 1 giải quyết biểu diễn đặc trưng bất biến không gian - tần số.
   - Chiến lược 2 giải quyết rào cản sốc gradient khi tiếp xúc nhiễu nén nặng.
   - Chiến lược 3 phá vỡ thiên lệch GANs và tập trung gradient vào các mẫu khó.

---

## 2. NỘI DUNG TRỌNG TÂM CẦN TRÌNH BÀY

### 1. Tổng Quan Kiến Trúc FatFormer-XLA
- Sơ đồ tổng thể dòng chảy đặc trưng từ ảnh đầu vào $\rightarrow$ Nhánh Visual (CLIP ViT + FAA + SRM + Gating) $\rightarrow$ Nhánh Alignment (LGA) $\rightarrow$ Dự đoán nhị phân.

### 2. Tầng Trích Xuất Vết Dư Không Gian SRM (`SpatialResidualBlock`)
- Chứng minh toán học: Phép vi sai bậc 1 triệt tiêu nền phẳng, phép vi sai bậc 2 Laplacian bắt độ cong biên, ma trận $5 \times 5$ bắt tạo tác nội suy pixel.
- Lý do tại sao đặt `requires_grad = False` và tác dụng triệt tiêu hoàn toàn thông tin ngữ nghĩa.

### 3. Tầng Cổng Thích Ứng Tần Số Động (`DynamicFrequencyGating` $\lambda(x)$)
- Phân tích cơ chế suy diễn: Khi ảnh sạch $\rightarrow \lambda(x) \approx 1.0 \sim 1.5$; khi ảnh nén sâu $\rightarrow \lambda(x) \rightarrow 0.05 \sim 0.1$.
- So sánh với hệ số tĩnh của paper gốc để làm bật tính đột phá của đề tài.

### 4. Tầng Lập Lịch Suy Thoái Curriculum & Hàm Mục Tiêu Focal Loss
- Định nghĩa hàm bước nhảy tham số theo 3 giai đoạn epoch.
- Phân tích toán học hàm `DualStreamFocalLoss`: cơ chế hệ số điều chế $(1 - p_t)^\gamma$ triệt tiêu gradient của mẫu dễ, kéo mô hình hội tụ trên vùng biên phân tách khó.

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Bản thảo Chương 3 hoàn chỉnh (khoảng 10–12 trang), định dạng chuẩn LaTeX/Word.
- [ ] Đầy đủ sơ đồ kiến trúc vector sắc nét, công thức toán đánh số thứ tự chuẩn mực.
- [ ] Cả 3 thành viên họp thống nhất ký duyệt nội dung phương pháp luận trước khi bước vào tuần huấn luyện chính thức.
