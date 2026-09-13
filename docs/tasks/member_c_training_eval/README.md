# BẢNG ĐIỀU KHIỂN NHIỆM VỤ: THÀNH VIÊN C (TRAINING & EVALUATION LEAD)

> **Họ và tên**: [Điền tên Thành viên C]  
> **Vai trò**: Trưởng nhóm Huấn Luyện & Đánh Giá Benchmark (Training & Evaluation Lead)  
> **Trách nhiệm cốt lõi**: Quản trị tài nguyên **300 Compute Units** trên Colab Pro, vận hành GPU A100 với AMP FP16 & Gradient Accumulation, thiết lập pipeline đánh giá nhanh `fast_eval.py` (500 ảnh/tập) và `full_eval.py` (18 tập test), thực thi 2 pha huấn luyện và chủ trì soạn thảo Báo cáo kỹ thuật đồ án.

---

## 📌 BẢNG CHECKLIST TIẾN ĐỘ TỔNG THỂ (SPRINT PROGRESS)

| Tuần | Mã Task | Tên Nhiệm Vụ | Hạn Chót | Trạng Thái | Tệp Chi Tiết |
| :---: | :---: | :--- | :---: | :---: | :--- |
| **Tuần 1** | **Task 1.3** | Thiết lập Notebook Huấn Luyện GPU A100 trên Colab Pro | Ngày 7 | `[ ] Chưa xong` | [`week_1/2026-09-13_task_1.3_setup_colab_a100_training_notebook.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_c_training_eval/week_1/2026-09-13_task_1.3_setup_colab_a100_training_notebook.md) |
| **Tuần 1** | **Task 1.4** | Xây dựng module chỉ số Benchmark & Pipeline Fast-Eval | Ngày 7 | `[ ] Chưa xong` | [`week_1/2026-09-13_task_1.4_fast_eval_metrics_pipeline.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_c_training_eval/week_1/2026-09-13_task_1.4_fast_eval_metrics_pipeline.md) |
| **Tuần 2** | **Task 2.2** | Benchmark Baseline gốc trên toàn bộ tập Clean (8 GANs + 6 Diffusion) | Ngày 14 | `[ ] Chưa xong` | [`week_2/2026-09-13_task_2.2_baseline_clean_benchmark.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_c_training_eval/week_2/2026-09-13_task_2.2_baseline_clean_benchmark.md) |
| **Tuần 2** | **Task 2.3** | Benchmark Baseline gốc trên tập Degraded ($Q=30, 50, 70$, Blur) | Ngày 14 | `[ ] Chưa xong` | [`week_2/2026-09-13_task_2.3_baseline_degraded_benchmark.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_c_training_eval/week_2/2026-09-13_task_2.3_baseline_degraded_benchmark.md) |
| **Tuần 3** | **Task 3.5** | Cài đặt cơ chế tự động sao lưu Checkpoint và Resume phiên Colab | Ngày 21 | `[ ] Chưa xong` | [`week_3/2026-09-13_task_3.5_checkpoint_autosave_and_resume.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_c_training_eval/week_3/2026-09-13_task_3.5_checkpoint_autosave_and_resume.md) |
| **Tuần 4** | **Task 4.1** | Thực thi Huấn luyện Pha 1 (Adaptive Generalization) trên A100 | Ngày 25 | `[ ] Chưa xong` | [`week_4/2026-09-13_task_4.1_train_phase_1_adaptive.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_c_training_eval/week_4/2026-09-13_task_4.1_train_phase_1_adaptive.md) |
| **Tuần 4** | **Task 4.2** | Thực thi Huấn luyện Pha 2 (Hard Degradation Hardening) trên A100 | Ngày 28 | `[ ] Chưa xong` | [`week_4/2026-09-13_task_4.2_train_phase_2_hardening.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_c_training_eval/week_4/2026-09-13_task_4.2_train_phase_2_hardening.md) |
| **Tuần 4** | **Task 4.4** | Giám sát hội tụ Loss và Validation qua Fast-Eval sau mỗi Epoch | Ngày 28 | `[ ] Chưa xong` | [`week_4/2026-09-13_task_4.4_fast_validation_tracking.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_c_training_eval/week_4/2026-09-13_task_4.4_fast_validation_tracking.md) |
| **Tuần 5** | **Task 5.1** | Chạy Full-Benchmark 18 tập test hai chiều (Clean vs Degraded) | Ngày 32 | `[ ] Chưa xong` | [`week_5/2026-09-13_task_5.1_full_benchmark_18_datasets.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_c_training_eval/week_5/2026-09-13_task_5.1_full_benchmark_18_datasets.md) |
| **Tuần 5** | **Task 5.3** | Xuất bộ ảnh trực quan hóa Grad-CAM phân giải cao minh chứng XAI | Ngày 33 | `[ ] Chưa xong` | [`week_5/2026-09-13_task_5.3_xai_gradcam_comparison.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_c_training_eval/week_5/2026-09-13_task_5.3_xai_gradcam_comparison.md) |
| **Tuần 5** | **Task 5.4** | Chủ trì soạn thảo Báo cáo kỹ thuật đồ án (PDF) và Slide thuyết trình | Ngày 35 | `[ ] Chưa xong` | [`week_5/2026-09-13_task_5.4_technical_report_and_slides.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/tasks/member_c_training_eval/week_5/2026-09-13_task_5.4_technical_report_and_slides.md) |

---

## 🧭 NGUYÊN TẮC QUẢN LÝ TÀI NGUYÊN CỦA LEAD C

1. **Quản trị Compute Units (CU)**:
   - Các tác vụ nhẹ (kiểm tra code, nạp weights, chạy thử dummy): Ưu tiên dùng GPU **T4** (tiêu tốn chỉ ~1.5–2 CU/giờ).
   - Các tác vụ nặng (Train Pha 1, Train Pha 2, Full-Benchmark): Bật GPU **A100 (40GB VRAM)** (tiêu tốn ~12–14 CU/giờ). Tổng mức tiêu thụ toàn dự án phải được kiểm soát dưới **100 CU** (giữ lại 200 CU dự phòng).
2. **Chiến lược Đánh giá Phân tầng**: Luôn chạy `fast_eval.py` (500 ảnh/tập, mất ~8 phút) để kiểm tra mô hình sau mỗi epoch; chỉ chạy `full_eval.py` (mất hàng giờ) khi đã có checkpoint cuối cùng ưng ý.
3. **An toàn Checkpoint**: Tuyệt đối không để checkpoint nằm duy nhất trên SSD Colab (vì sẽ mất sạch khi disconnect session). Sau mỗi epoch bắt buộc phải sao lưu file `.pth` sang thư mục Drive 5TB `FatFormer_Hub/checkpoints/`.
