# TASK 4.6: SOẠN THẢO BÁO CÁO ĐỒ ÁN - CHƯƠNG 4 (THIẾT LẬP THỰC NGHIỆM)

- **Mã nhiệm vụ**: `Task 4.6`
- **Người phụ trách**: **Thành viên B** *(Chủ trì)*, **Thành viên A** *(Đóng góp phần Data)*
- **Môi trường thực hiện**: 🖥️ **Local** (LaTeX / Word / Overleaf)
- **Thời hạn hoàn thành**: Tuần 4 (Ngày 28)
- **Độ ưu tiên**: 🟢 Bình thường (Thực thi chiến lược viết báo cáo cuốn chiếu)
- **Đầu vào (Input)**:
  - Cấu hình hạ tầng thực tế tài khoản Google AI Pro (GPU A100 SXM4 40GB, L4 24GB, T4 16GB, Google Drive 5TB).
  - Toàn bộ siêu tham số huấn luyện (learning rate, batch accumulation, optimizer AdamW, scheduler, loss weights).
  - Chi tiết tập dữ liệu Staging 90/10 từ Thành viên A và 18 tập benchmark kiểm thử.
- **Đầu ra (Output)**:
  - Bản thảo **Chương 4 (Thiết lập thực nghiệm & Môi trường huấn luyện)** dạng PDF/LaTeX.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Hoàn tất toàn bộ bối cảnh thực nghiệm trước khi bước sang Tuần 5 đánh giá benchmark.
2. Trình bày chi tiết, minh bạch và chính xác các tham số kỹ thuật để bất kỳ nhà nghiên cứu nào cũng có thể tái lập (*reproducibility*) lại toàn bộ kết quả của nhóm.
3. Giải thích rõ ràng phương pháp chia tách tập dữ liệu, các phép biến dạng vật lý và cách tính các thang đo định lượng (ACC, Average Precision - AP, AUC).

---

## 2. NỘI DUNG TRỌNG TÂM CẦN TRÌNH BÀY

### 1. Môi Trường Phần Cứng & Thư Viện Phần Mềm
- Mô tả chi tiết hạ tầng Google AI Pro: GPU NVIDIA A100 SXM4 40GB VRAM, hệ thống lưu trữ NVMe SSD và Google Drive 5TB.
- Phiên bản hệ thống: PyTorch 2.x, CUDA 12.x, torchvision, timm, streamlit.

### 2. Thiết Kế Tập Dữ Liệu Huấn Luyện & Kiểm Thử
- Bảng thống kê tập Staging 90/10: Phân bổ số lượng ảnh giữa ProGAN 4-class (`car, cat, chair, horse`) và 3.600 ảnh GenImage xúc tác (SD v1.5 + Midjourney v5).
- Bảng kê 18 tập test paper chuẩn: 8 kiến trúc GANs (ProGAN, StyleGAN, BigGAN, CycleGAN, StarGAN, GauGAN, DeepFake, FaceForensics++) và 10 kiến trúc Diffusion (LDM, GLIDE, DALL-E, Guided Diffusion...).
- Chi tiết thông số các tập nén suy biến: JPEG ($Q=30, 50, 70$), Gaussian Blur ($\sigma=1.0, 2.0$), Down-Up ($224 \rightarrow 112 \rightarrow 224$).

### 3. Siêu Tham Số Huấn Luyện & Thiết Kế Ablation Study
- Learning rate phân tầng: $10^{-4}$ cho adapter, $10^{-3}$ cho gating network; giảm một nửa ở Pha 3.
- Bảng kê 4 phiên bản kiểm thử triệt tiêu: Baseline gốc, Aug-only, SRM-only, và FatFormer-XLA tối ưu.

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Bản thảo Chương 4 hoàn chỉnh (khoảng 6–8 trang).
- [ ] Bảng biểu tham số rõ ràng, số liệu ăn khớp 100% với code thực thi trong `configs/train_config.yaml`.
- [ ] Hoàn thành đúng hạn vào cuối Tuần 4, sẵn sàng để Tuần 5 chỉ cần điền kết quả vào Chương 5.
