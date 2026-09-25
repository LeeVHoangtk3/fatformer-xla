# FatFormer-XLA: Master Architectural & Project Blueprint

Tài liệu thiết kế kiến trúc và cấu trúc tổng thể toàn diện của đồ án **FatFormer-XLA** được đặt tại:

👉 **[Bản Thiết Kế Kiến Trúc & Tổng Quan Hệ Thống (tong_quan_du_an.md)](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/overall/tong_quan_du_an.md)**

---

## Tóm Tắt Nhanh Nội Dung Tài Liệu:
1. **Bối Cảnh & Đột Phá Kỹ Thuật**: Nâng cao độ bền pháp y trước suy thoái mạng xã hội (JPEG, Gaussian Blur, Resize down-up) bằng cách tích hợp SRM 3-Kernels + Dynamic Frequency Gating $\lambda(x)$ + Dual-Stream Augmentation (Phương Án 2).
2. **Bản Đồ Cấu Trúc Toàn Bộ Dự Án**: Phân loại chi tiết toàn bộ các thư mục (`src/`, `FatFormer-main/`, `tools/`, `notebooks/`, `docs/`, `configs/`).
3. **Sơ Đồ Luồng Dữ Liệu (Mermaid)**: Trực quan hóa đường đi của ảnh từ đầu vào qua FAA Wavelet, SRM, Language-Guided Alignment đến hàm mất mát Eq. (10).
4. **Kiến Trúc Hạ Tầng (Colab A100 & Drive 5TB)**: Cơ chế kết nối Shortcut 0 byte và quy tắc giải nén SSD NVMe chống sập I/O.
5. **Ma Trận Phân Vai 3 Thành Viên**: Phân định ranh giới trách nhiệm giữa Member A (Data/Drive), Member B (Model/Loss), Member C (Train/Eval).
6. **Sổ Tay Kỹ Thuật & Cheat-Sheet Lệnh**: Cú pháp chạy Smoke Test, Fast-Eval và Huấn luyện chuẩn hóa.
