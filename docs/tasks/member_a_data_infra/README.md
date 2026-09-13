# BẢNG ĐIỀU KHIỂN NHIỆM VỤ: THÀNH VIÊN A (DATA & INFRA LEAD)

> **Họ và tên**: [Điền tên Thành viên A]  
> **Vai trò**: Trưởng nhóm Dữ liệu & Hạ tầng (Data & Infrastructure Lead)  
> **Trách nhiệm cốt lõi**: Quản lý Google Drive 5TB nhóm, chống nghẽn I/O trên Colab Pro, chuẩn hóa các tập dữ liệu huấn luyện/kiểm thử (ProGAN, GenImage, Degraded) và xây dựng module Data Augmentation 2 luồng.

---

## 📌 BẢNG CHECKLIST TIẾN ĐỘ TỔNG THỂ (SPRINT PROGRESS)

| Tuần | Mã Task | Tên Nhiệm Vụ | Hạn Chót | Trạng Thái | Tệp Chi Tiết |
| :---: | :---: | :--- | :---: | :---: | :--- |
| **Tuần 1** | **Task 1.1** | Thiết lập kết nối Drive 5TB, Colab Pro & Dummy Dataset | Ngày 7 | `[ ] Chưa xong` | [`week_1/2026-09-13_task_1.1_setup_drive_colab_io.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_a_data_infra/week_1/2026-09-13_task_1.1_setup_drive_colab_io.md) |
| **Tuần 2** | **Task 2.1** | Xây dựng script tạo bộ kiểm thử suy biến vật lý (Degraded test set) | Ngày 14 | `[ ] Chưa xong` | [`week_2/2026-09-13_task_2.1_make_degraded_dataset.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_a_data_infra/week_2/2026-09-13_task_2.1_make_degraded_dataset.md) |
| **Tuần 3** | **Task 3.1** | Đóng gói tập con GenImage (Midjourney + SD v1.5) hỗn hợp | Ngày 21 | `[ ] Chưa xong` | [`week_3/2026-09-13_task_3.1_prepare_genimage_subset.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_a_data_infra/week_3/2026-09-13_task_3.1_prepare_genimage_subset.md) |
| **Tuần 3** | **Task 3.2** | Xây dựng module Data Augmentation 2 luồng chống học vẹt nén | Ngày 21 | `[ ] Chưa xong` | [`week_3/2026-09-13_task_3.2_dual_stream_augmentations.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_a_data_infra/week_3/2026-09-13_task_3.2_dual_stream_augmentations.md) |
| **Tuần 5** | **Task 5.5** | Xây dựng pipeline suy luận Live Demo trên ảnh mạng xã hội | Ngày 35 | `[ ] Chưa xong` | [`week_5/2026-09-13_task_5.5_live_demo_pipeline.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_a_data_infra/week_5/2026-09-13_task_5.5_live_demo_pipeline.md) |

---

## 🧭 NGUYÊN TẮC HOẠT ĐỘNG CỦA LEAD A

1. **Quy tắc Vàng về I/O**: Tuyệt đối không đọc từng ảnh trực tiếp từ Google Drive mount. Mọi dataset đưa lên Drive bắt buộc phải nén thành `.tar`, sau đó viết code để Colab tự copy file `.tar` vào `/content/` và giải nén ra SSD máy ảo trước khi DataLoader bắt đầu.
2. **Quy tắc Cấu trúc Dữ liệu**: Mọi thư mục con của tập train/val/test đều phải tuân thủ nghiêm ngặt 2 nhãn nhị phân:
   ```
   dataset_name/
   ├── 0_real/   # Nhãn 0 (Ảnh thật)
   └── 1_fake/   # Nhãn 1 (Ảnh sinh bởi AI)
   ```
3. **Cơ chế 2 Tài khoản (Phương Án A)**: Bạn chịu trách nhiệm hướng dẫn thành viên chạy Colab thực hiện thao tác "Add shortcut to Drive" từ thư mục `FatFormer_Hub` của tài khoản 5TB để không tiêu hao 15GB của tài khoản Colab.
