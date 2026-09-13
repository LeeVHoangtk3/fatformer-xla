# BẢNG ĐIỀU KHIỂN NHIỆM VỤ: THÀNH VIÊN B (ARCHITECTURE & LOSS LEAD)

> **Họ và tên**: [Điền tên Thành viên B]  
> **Vai trò**: Trưởng nhóm Kiến trúc Mô hình & Hàm Mất Mát (Architecture & Loss Lead)  
> **Trách nhiệm cốt lõi**: Xây dựng mã nguồn mô hình tại `models/`, quản lý tính toàn vẹn của 1,116 tensors, hiện thực hóa khối **SRM 3-Kernels** và mạng **Dynamic Frequency Gating $\lambda(x)$**, đóng băng 94.1% CLIP ViT-L/14, xuất ảnh Grad-CAM và xây dựng bảng Ablation Study.

---

## 📌 BẢNG CHECKLIST TIẾN ĐỘ TỔNG THỂ (SPRINT PROGRESS)

| Tuần | Mã Task | Tên Nhiệm Vụ | Hạn Chót | Trạng Thái | Tệp Chi Tiết |
| :---: | :---: | :--- | :---: | :---: | :--- |
| **Tuần 1** | **Task 1.2** | Xây dựng mã nguồn FatFormer & Trình nạp khớp 1,116 tensors | Ngày 7 | `[ ] Chưa xong` | [`week_1/2026-09-13_task_1.2_fatformer_model_and_weights_loader.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_b_architecture_loss/week_1/2026-09-13_task_1.2_fatformer_model_and_weights_loader.md) |
| **Tuần 2** | **Task 2.4** | Trích xuất ảnh nhiệt Grad-CAM đối chứng trên Baseline gốc | Ngày 14 | `[ ] Chưa xong` | [`week_2/2026-09-13_task_2.4_gradcam_baseline_extraction.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_b_architecture_loss/week_2/2026-09-13_task_2.4_gradcam_baseline_extraction.md) |
| **Tuần 3** | **Task 3.3** | Hiện thực hóa khối SRM 3-Kernels & Dynamic Frequency Gating | Ngày 21 | `[ ] Chưa xong` | [`week_3/2026-09-13_task_3.3_srm_and_frequency_gating_integration.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_b_architecture_loss/week_3/2026-09-13_task_3.3_srm_and_frequency_gating_integration.md) |
| **Tuần 3** | **Task 3.4** | Cấu hình đóng băng 94.1% CLIP ViT & Thiết lập File cấu hình Train | Ngày 21 | `[ ] Chưa xong` | [`week_3/2026-09-13_task_3.4_freeze_clip_and_train_config.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_b_architecture_loss/week_3/2026-09-13_task_3.4_freeze_clip_and_train_config.md) |
| **Tuần 4** | **Task 4.3** | Giám sát & Phân tích cơ chế đóng mở cổng tần số $\lambda(x)$ | Ngày 28 | `[ ] Chưa xong` | [`week_4/2026-09-13_task_4.3_frequency_gating_analysis.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_b_architecture_loss/week_4/2026-09-13_task_4.3_frequency_gating_analysis.md) |
| **Tuần 5** | **Task 5.2** | Tổng hợp bảng phân tích triệt tiêu (Ablation Study 4 phiên bản) | Ngày 35 | `[ ] Chưa xong` | [`week_5/2026-09-13_task_5.2_ablation_study_matrix.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_b_architecture_loss/week_5/2026-09-13_task_5.2_ablation_study_matrix.md) |

---

## 🧭 4 QUY TẮC KỸ THUẬT BẮT BUỘC CỦA LEAD B

1. **Quy tắc Typo Bắt Buộc**: Trong class `LanguageGuidedAlignment`, module enhancer bắt buộc phải đặt tên thuộc tính là `self.patch_basaed_enhancer` (có chữ `a` thừa) để khớp 100% với tên key trong checkpoint gốc `fatformer_4class_ckpt.pth`.
2. **Quy tắc Nạp Weights**: Tệp [`ViT-L-14.pt`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/ViT-L-14.pt) là TorchScript JIT archive, nạp bằng `torch.jit.load()` hoặc `torch.load(..., weights_only=False)`. Checkpoint bọc trong khóa `ckpt['model']`.
3. **Quy tắc SRM 3-Kernels**: Luôn đặt `requires_grad = False` cho 3 ma trận lọc vi sai SRM để biến nó thành bộ lọc cố định, không tốn VRAM hay trọng số huấn luyện.
4. **Quy tắc Đóng Băng**: Chỉ 5.9% tham số (FAA, SRM projection, Gating, LGA, soft prompts) được phép có `requires_grad = True`. 94.1% CLIP ViT-L/14 phải đóng băng tuyệt đối.
