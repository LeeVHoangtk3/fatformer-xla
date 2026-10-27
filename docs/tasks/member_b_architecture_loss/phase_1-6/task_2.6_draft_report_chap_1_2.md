# TASK 2.6: SOẠN THẢO BÁO CÁO ĐỒ ÁN - CHƯƠNG 1 & CHƯƠNG 2

- **Mã nhiệm vụ**: `Task 2.6`
- **Người phụ trách**: **Thành viên B** *(Chủ trì)*, **Thành viên C** *(Review)*
- **Môi trường thực hiện**: 🖥️ **Local** (Overleaf / LaTeX / Word / Markdown)
- **Thời hạn hoàn thành**: Tuần 2 (Ngày 14)
- **Độ ưu tiên**: 🟢 Bình thường (Thực thi chiến lược viết báo cáo cuốn chiếu)
- **Đầu vào (Input)**:
  - Tài liệu nghiên cứu tổng quan dự án, paper FatFormer CVPR 2024.
  - Bảng số liệu Baseline Clean (Task 2.2) và Baseline Degraded (Task 2.3).
  - Bộ ảnh Grad-CAM mốc sàn (Task 2.4).
- **Đầu ra (Output)**:
  - Bản thảo **Chương 1 (Giới thiệu bài toán & Động lực nghiên cứu)** và **Chương 2 (Các công trình liên quan - Related Work)** dạng PDF/LaTeX.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Triển khai mô hình **Viết báo cáo cuốn chiếu (Progressive Paper Writing)**: Hoàn tất 40% khối lượng văn bản học thuật ngay trong Tuần 2, giải tỏa triệt để áp lực dồn việc vào tuần cuối.
2. Trình bày sắc bén động lực nghiên cứu: Phân tích thực trạng các mô hình phát hiện ảnh giả mạo hiện đại (như FatFormer, UnivFD, RINE) đạt trên 98% trên ảnh sạch, nhưng sụt giảm thảm hại (còn dưới 60%) khi ảnh bị nén bởi các nền tảng mạng xã hội.
3. Tổng hợp bức tranh tổng thể các hướng tiếp cận hiện tại: Phát hiện dựa trên không gian màu, dựa trên phổ tần số Fourier/Wavelet, dựa trên trích xuất đặc trưng vi sai pháp y (SRM, PRNU).

---

## 2. NỘI DUNG TRỌNG TÂM CẦN TRÌNH BÀY

### Chương 1: Giới Thiệu & Động Lực Nghiên Cứu
- Bối cảnh bùng nổ của các mô hình sinh ảnh Diffusion (Midjourney v5/v6, Stable Diffusion XL, FLUX).
- Vấn đề thực tiễn: Ảnh lan truyền trên mạng xã hội luôn bị biến dạng nén lượng tử JPEG mất mát (lossy compression) và làm mờ.
- Lỗ hổng kỹ thuật của FatFormer gốc: Quá phụ thuộc vào dải tần số cao DWT cố định ($\lambda = 0.5 \sim 1.0$), khi bị nén JPEG thì dải tần số này bị phá hủy thành nhiễu khối, đánh lừa mô hình.
- Mục tiêu và đóng góp của đồ án FatFormer-XLA: Hợp nhất SRM 3-Kernels, Cổng tần số thích ứng $\lambda(x)$, Lập lịch Curriculum và Hàm Focal Loss.

### Chương 2: Các Công Trình Liên Quan (Related Work)
- **Mô hình nền tảng đa phương thức (Vision-Language Models - CLIP)** trong giám định ảnh AI.
- **Biến đổi sóng con (Wavelet Transforms & DWT)** trong nhận diện tần số tạo tác.
- **Kỹ thuật trích xuất vết dư không gian (Spatial Rich Models - SRM)** trong pháp y hình ảnh cổ điển.
- **Học thích ứng theo giáo trình (Curriculum Learning)** trong việc nâng cao độ bền vững.

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Bản thảo Chương 1 & Chương 2 hoàn chỉnh (độ dài tối thiểu 8–10 trang chuẩn format đồ án).
- [ ] Có đầy đủ trích dẫn tài liệu tham khảo (BibTeX/IEEE style).
- [ ] Chèn bảng số liệu thực tế đo đạc từ Baseline Task 2.2 và 2.3.
- [ ] Thành viên C review và ký duyệt nội dung kỹ thuật.
