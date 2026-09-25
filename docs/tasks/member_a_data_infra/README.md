# BẢNG ĐIỀU KHIỂN NHIỆM VỤ: THÀNH VIÊN A (DATA & INFRA LEAD)

> **Vai trò**: Trưởng nhóm Dữ liệu & Hạ tầng (Data & Infrastructure Lead)  
> **Trách nhiệm cốt lõi**: Quản trị kho dữ liệu Google Drive 5TB (tài khoản Google AI Pro), thiết lập đường truyền I/O chống nghẽn SSD NVMe trên Colab, chuẩn hóa tập dữ liệu ProGAN & GenImage Catalyst (tỷ lệ Staging 90/10), xây dựng module `CurriculumDegradationScheduler` 3 giai đoạn theo epoch, đảm nhận huấn luyện Checkpoint Ablation 2 và phát triển ứng dụng Live Demo Web.

---

## 📌 BẢNG LỘ TRÌNH VÀ CHECKLIST NHIỆM VỤ (TUẦN 1 - TUẦN 5)

| Tuần | Mã Task | Tên nhiệm vụ | Môi trường | Đầu vào (Input) | Đầu ra (Output) | Tệp Chi Tiết | Trạng thái |
| :---: | :---: | :--- | :---: | :--- | :--- | :---: | :---: |
| **Tuần 1** | **Task 1.1** | Thiết lập hạ tầng Drive 5TB, I/O Colab & Dataset Dummy | 🖥️ **Local** +<br>☁️ **Colab** | • 20 ảnh mẫu (10 real, 10 fake)<br>• Tài khoản Google AI Pro 5TB | • Thư mục `Fatformer/` trên Drive 5TB<br>• Các tệp `.tar` sẵn sàng<br>• Notebook `00_setup_env.ipynb` | [`task_1.1_setup_drive_colab_io.md`](task_1.1_setup_drive_colab_io.md) | `[x] Đã hoàn thành` |
| **Tuần 2** | **Task 2.1** | Xây dựng bộ kiểm thử suy biến vật lý (`test_degraded.tar`) | 🖥️ **Local** +<br>☁️ **Colab CPU/T4** | • 18 tập test Clean gốc paper CVPR 2024<br>• Tham số JPEG, Blur, Down-Up | • Script `tools/make_degraded.py`<br>• Tệp nén `test_degraded.tar` đẩy lên Drive 5TB | [`task_2.1_make_degraded_dataset.md`](task_2.1_make_degraded_dataset.md) | `[ ] Chưa xong` |
| **Tuần 3** | **Task 3.1** | Chuẩn hóa dữ liệu Staging 90/10 (`diffusion_staging.tar`) | ☁️ **Colab** | • ProGAN 4-class (`car, cat, chair, horse`)<br>• Tập GenImage (SD v1.5 + Midjourney) | • Tệp `diffusion_staging.tar` trên Drive 5TB (tỷ lệ 90/10) | [`task_3.1_prepare_genimage_staging.md`](task_3.1_prepare_genimage_staging.md) | `[ ] Chưa xong` |
| **Tuần 3** | **Task 3.2** | Xây dựng module `CurriculumDegradationScheduler` | 🖥️ **Local** | • Quy chuẩn giáo trình suy biến 3 giai đoạn theo epoch | • Module `src/datasets/transforms.py` tích hợp scheduler hoàn chỉnh | [`task_3.2_curriculum_degradation_scheduler.md`](task_3.2_curriculum_degradation_scheduler.md) | `[ ] Chưa xong` |
| **Tuần 4** | **Task 4.3** | Huấn luyện Checkpoint Ablation 2 (Chỉ Augmentation tĩnh) | ☁️ **Colab T4/L4** | • Tập ProGAN 4-class gốc<br>• Augmentation tĩnh ngẫu nhiên ($Q \in [40, 95]$)<br>• Hàm Cross-Entropy | • Checkpoint `fatformer_aug_only.pth` lưu trên Drive 5TB<br>• Log hội tụ 5 epoch | [`task_4.3_train_ablation_aug_only.md`](task_4.3_train_ablation_aug_only.md) | `[ ] Chưa xong` |
| **Tuần 5** | **Task 5.5** | Xây dựng ứng dụng Live Demo Web kéo thả ảnh | 🖥️ **Local** | • Checkpoint cuối `fatformer_srm_robust_final.pth`<br>• Thư viện Streamlit / Gradio | • Ứng dụng `tools/inference_demo.py` hoạt động mượt mà | [`task_5.5_live_demo_web_app.md`](task_5.5_live_demo_web_app.md) | `[ ] Chưa xong` |

---

## 🧭 NGUYÊN TẮC KỸ THUẬT CỐT LÕI CỦA LEAD A

1. **Hạ Tầng Siêu Nút Google AI Pro 5TB**: Toàn bộ tài nguyên tính toán và lưu trữ nằm chung trên một tài khoản Google AI Pro. Đường dẫn làm việc thống nhất trên Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   DRIVE_HUB = "/content/drive/MyDrive/Fatformer"
   ```
2. **Quy Tắc I/O Chống Nghẽn SSD NVMe (Bắt buộc)**:
   - Tuyệt đối không để DataLoader đọc trực tiếp từng tệp ảnh nhỏ từ Google Drive FUSE.
   - Luôn nén dataset thành file `.tar` trên Drive, copy về `/content/` của máy ảo và giải nén sang SSD NVMe trước khi huấn luyện:
     ```bash
     cp /content/drive/MyDrive/FatFormer_Hub/datasets/progan_train.tar /content/
     tar -xf /content/progan_train.tar -C /content/dataset_local/
     ```
3. **Cấu Trúc Thư Mục Dữ Liệu Chuẩn**:
   Mọi tập dataset (train, val, test) luôn phải tuân thủ nghiêm ngặt 2 nhãn nhị phân:
   ```
   dataset_name/
   ├── 0_real/   # Nhãn 0 (Ảnh thật)
   └── 1_fake/   # Nhãn 1 (Ảnh sinh bởi AI)
   ```
4. **Tính Tự Chủ Triển Khai**: Chi tiết thuật toán, tổ chức mã nguồn trong `src/datasets/` và giao diện demo do Thành viên A tự chủ thiết kế và tối ưu, bảo đảm đúng chuẩn giao tiếp đầu vào/đầu ra với các thành viên khác.
