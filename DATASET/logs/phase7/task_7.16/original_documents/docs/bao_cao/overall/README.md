# FatFormer-XLA: Master Architectural & Project Blueprint

Tài liệu thiết kế kiến trúc và cấu trúc tổng thể toàn diện của đồ án **FatFormer-XLA** được đặt tại:

👉 **[Bản Thiết Kế Kiến Trúc & Tổng Quan Hệ Thống (tong_quan_du_an.md)](tong_quan_du_an.md)**  
👉 **[Báo Cáo Nghiệm Thu Kiểm Định Ngoại Vi CommunityForensics-Eval (CVPR 2025)](bao_cao_kiem_dinh_communityforensics_eval.md)**  

---

## 🚨 Đính Chính Quan Trọng & Kế Hoạch Hiện Hành (27/10/2026)

> **Mọi tài liệu trong thư mục này phải được đọc kèm hai văn kiện dưới đây.**

| Văn kiện | Vai trò | Ghi chú |
| :--- | :--- | :--- |
| **[Kế hoạch Phase 7 — Tái lập đo lường & Hợp nhất mô hình](../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md)** | **Source of truth** cho toàn bộ công việc tiếp theo | `v3.0`; 16 task, 4 cổng G7.0–G7.3; bản `v2.0` lưu tại [`archive/`](../../plan/archive/) |
| **[Phần IX — Đính chính kỹ thuật của báo cáo CFE](bao_cao_kiem_dinh_communityforensics_eval.md#ix-đính-chính-kỹ-thuật-ngày-27102026)** | Vô hiệu hóa số liệu WiSE-FT, giữ hiệu lực Baseline/Anchored | Bắt buộc đọc trước khi trích dẫn bất kỳ con số nào |
| **[Điểm dừng 6 — Phase 7](../../plan/ke_hoach_cac_cot_moc_va_diem_dung.md)** | Cổng nghiệm thu thay thế Gate 4 | G7.0–G7.3 + điều khoản kết quả âm |
| **[Khế ước đo lường v3 (Phụ lục A)](../../plan/ke_hoach_test_communityforensics_eval.md)** | Quy tắc báo cáo metric | Cấm báo cáo "overall accuracy" đơn lẻ |

**Ba điều phải nhớ:**

1. **WiSE-FT Gen 1 bị cách ly** (`status: invalid_layout`): checkpoint chứa 10 khối adapter `{2,5,7,8,11,14,15,17,20,23}`, trong khi kiến trúc chỉ sinh được vị trí `(i+1)·24/n − 1` với `n ∈ {3,4,6,8,12,24}` ⇒ không nạp đầy đủ được. Mọi con số của nó (`52,05%`, `4,10%`, `0,20%`, `96,58%`) **không còn hiệu lực**.
2. **Baseline và Anchored Recovery vẫn hợp lệ** — đây là hai mốc được phép trích dẫn.
3. **Thiết kế đích là một checkpoint duy nhất** (`UnifiedFatFormerXLA`, G7.3), không phải hệ thống hai chế độ. Kiến trúc kép chỉ là phương án dự phòng nếu G7.3 thất bại.

---

## Tóm Tắt Nhanh Nội Dung Tài Liệu:
1. **Bối Cảnh & Đột Phá Kỹ Thuật**: Nâng cao độ bền pháp y trước suy thoái mạng xã hội (JPEG, Gaussian Blur, Resize down-up) bằng cách tích hợp SRM 3-Kernels + Dynamic Frequency Gating $\lambda(x)$ + Dual-Stream Augmentation (Phương Án 2).
2. **Bản Đồ Cấu Trúc Toàn Bộ Dự Án**: Phân loại chi tiết toàn bộ các thư mục (`src/`, `FatFormer-main/`, `tools/`, `notebooks/`, `docs/`, `configs/`).
3. **Sơ Đồ Luồng Dữ Liệu (Mermaid)**: Trực quan hóa đường đi của ảnh từ đầu vào qua FAA Wavelet, SRM, Language-Guided Alignment đến hàm mất mát Eq. (10).
4. **Kiến Trúc Hạ Tầng (Colab A100 & Drive 5TB)**: Cơ chế kết nối Shortcut 0 byte và quy tắc giải nén SSD NVMe chống sập I/O.
5. **Ma Trận Phân Vai 3 Thành Viên**: Phân định ranh giới trách nhiệm giữa Member A (Data/Drive), Member B (Model/Loss), Member C (Train/Eval).
6. **Sổ Tay Kỹ Thuật & Cheat-Sheet Lệnh**: Cú pháp chạy Smoke Test, Fast-Eval và Huấn luyện chuẩn hóa.
