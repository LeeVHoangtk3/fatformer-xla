# TASK 5.4: LẮP RÁP BÁO CÁO ĐỒ ÁN HOÀN CHỈNH & SLIDE BẢO VỆ

- **Mã nhiệm vụ**: `Task 5.4`
- **Người phụ trách**: **Thành viên C** *(Chủ trì)*, **Cả 3 thành viên**
- **Môi trường thực hiện**: 🖥️ **Local** (LaTeX / Google Docs / Overleaf / PowerPoint)
- **Thời hạn hoàn thành**: Tuần 5 (Ngày 35 - Kết thúc đồ án)
- **Độ ưu tiên**: 🔴 Cực kỳ khẩn cấp (Sản phẩm cuối cùng bảo vệ trước Hội đồng)
- **Đầu vào (Input)**:
  - Bản thảo Chương 1–2 (Task 2.6) và Chương 3 (Task 3.7) từ Thành viên B.
  - Bản thảo Chương 4 (Task 4.6) từ Thành viên B & A.
  - Bảng số liệu Full-Benchmark (Task 5.1) và Bảng Ablation Study (Task 5.2).
  - Bộ 10 ảnh Grad-CAM đối đầu (Task 5.3) từ Thành viên B.
- **Đầu ra (Output)**:
  - **Báo cáo đồ án hoàn chỉnh** dạng PDF (độ dài ~35–45 trang theo đúng quy chuẩn trường/khoa).
  - **Bộ slide thuyết trình bảo vệ đồ án** (khoảng 20–25 slide).

---

## 1. MỤC TIÊU KỸ THUẬT
1. Lắp ráp và hoàn thiện toàn bộ báo cáo đồ án theo đúng tiến độ cuốn chiếu:
   - Viết **Chương 5 (Kết quả thực nghiệm & Thảo luận)**: Phân tích sâu sắc các bảng số liệu Benchmark, Ablation Study và hình ảnh Grad-CAM.
   - Viết **Chương 6 (Kết luận & Hướng phát triển tương lai)**: Đánh giá mức độ hoàn thành so với mục tiêu ban đầu, đề xuất mở rộng sang video deepfake hoặc các chuẩn nén mạng xã hội WebP/HEIC mới.
2. Thiết kế bộ Slide thuyết trình ấn tượng, logic, nêu bật được tính mới của phương pháp FatFormer-XLA và sự bền vững của mô hình.

---

## 2. KẾ HOẠCH PHÂN BỔ THUYẾT TRÌNH BẢO VỆ (SLIDE BREAKDOWN)

| Phần trình bày | Nội dung chính | Thành viên phụ trách nói |
| :--- | :--- | :---: |
| **Phần 1: Mở đầu** (Slide 1–5) | Lý do chọn đề tài, tử huyệt nén của các mô hình hiện nay, mục tiêu đồ án | **Thành viên A** |
| **Phần 2: Phương pháp** (Slide 6–12) | Kiến trúc FatFormer-XLA, SRM 3-Kernels, Cổng $\lambda(x)$, Curriculum, Focal Loss | **Thành viên B** |
| **Phần 3: Thực nghiệm** (Slide 13–18) | Môi trường A100/T4, Bảng số liệu Benchmark, Ablation Study, Ảnh XAI Grad-CAM | **Thành viên C** |
| **Phần 4: Demo & Kết luận** (Slide 19–22) | Trực tiếp chạy Live Demo kéo thả ảnh, Kết luận và Hướng phát triển | **Thành viên A (+ Cả 3 Q&A)** |

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Báo cáo đồ án hoàn chỉnh (PDF) xuất từ LaTeX/Word không có lỗi chính tả, mục lục và trích dẫn chuẩn mực.
- [ ] File Slide thuyết trình (.pptx / PDF) hoàn tất, hình ảnh minh họa sắc nét.
- [ ] Cả nhóm tổng duyệt diễn tập thuyết trình thử (Rehearsal) đạt thời lượng 15–20 phút chuẩn quy định.
