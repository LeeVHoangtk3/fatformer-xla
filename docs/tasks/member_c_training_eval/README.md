# BẢNG ĐIỀU KHIỂN NHIỆM VỤ: THÀNH VIÊN C (TRAINING & EVALUATION LEAD)

> **Vai trò**: Trưởng nhóm Huấn Luyện & Đánh Giá Benchmark (Training & Evaluation Lead)  
> **Trách nhiệm cốt lõi**: Quản trị tài nguyên Compute Units trên Colab Pro (tài khoản Google AI Pro 5TB), vận hành GPU A100 SXM4 và T4/L4 với AMP FP16 & Gradient Accumulation; xây dựng notebook huấn luyện `notebooks/train.ipynb`; thiết lập pipeline đánh giá nhanh `tools/fast_eval.py` và `tools/full_eval.py` trên 18 tập test; thiết lập mốc sàn Baseline (Clean & Degraded); chủ trì thực thi chiến dịch huấn luyện 8 epoch mô hình chính FatFormer-XLA theo giáo trình Curriculum; chạy Full-Benchmark 4 phiên bản Ablation Study; chủ trì tổng hợp Báo cáo kỹ thuật hoàn chỉnh (Chương 5, Kết luận) và Slide bảo vệ đồ án.

---

## 📌 BẢNG LỘ TRÌNH VÀ CHECKLIST NHIỆM VỤ (TUẦN 1 - TUẦN 5)

| Tuần | Mã Task | Tên nhiệm vụ | Môi trường | Đầu vào (Input) | Đầu ra (Output) | Tệp Chi Tiết | Trạng thái |
| :---: | :---: | :--- | :---: | :--- | :--- | :---: | :---: |
| **Tuần 1** | **Task 1.3** | Thiết lập Notebook Huấn Luyện Colab với AMP FP16 & Grad Accumulation | ☁️ **Colab T4** | • Mô hình từ Task 1.2<br>• Dữ liệu dummy từ Task 1.1 | • Notebook `notebooks/train.ipynb`<br>• Đo đạc thời gian 1 step forward-backward | [`task_1.3_setup_colab_training_notebook.md`](task_1.3_setup_colab_training_notebook.md) | `[ ] Chưa xong` |
| **Tuần 1** | **Task 1.4** | Xây dựng pipeline đánh giá nhanh Fast-Eval (< 8 phút) | 🖥️ **Local** +<br>☁️ **Colab T4** | • Module mô hình hoàn chỉnh<br>• Công thức tính ACC, AP, AUC | • Script `tools/fast_eval.py`<br>• Pipeline test thử trên dummy dataset | [`task_1.4_fast_eval_metrics_pipeline.md`](task_1.4_fast_eval_metrics_pipeline.md) | `[ ] Chưa xong` |
| **Tuần 2** | **Task 2.2** | Đánh giá Baseline gốc trên toàn bộ tập Clean (8 GANs + 10 Diffusion) | ☁️ **Colab T4/A100** | • Checkpoint `fatformer_4class_ckpt.pth`<br>• 18 tập test Clean trên SSD Colab | • Bảng số liệu Baseline Clean chi tiết | [`task_2.2_baseline_clean_benchmark.md`](task_2.2_baseline_clean_benchmark.md) | `[ ] Chưa xong` |
| **Tuần 2** | **Task 2.3** | Đánh giá Baseline gốc trên tập Degraded ($Q=30, 50, 70$, Blur) | ☁️ **Colab T4** | • Checkpoint gốc<br>• Tập test `test_degraded.tar` trên SSD Colab | • Bảng số liệu Baseline Degraded mốc sàn | [`task_2.3_baseline_degraded_benchmark.md`](task_2.3_baseline_degraded_benchmark.md) | `[ ] Chưa xong` |
| **Tuần 2** | **Task 2.5** | Chủ trì Họp nhóm Milestone 1 chốt mốc sàn & chỉ tiêu cải thiện | 🖥️ **Họp nhóm** | • Kết quả Task 2.2, Task 2.3 và ảnh Grad-CAM Task 2.4 | • Biên bản kỹ thuật Milestone 1 | [`task_2.5_milestone_1_baseline_freeze.md`](task_2.5_milestone_1_baseline_freeze.md) | `[ ] Chưa xong` |
| **Tuần 3** | **Task 3.5** | Cài đặt cơ chế tự động backup Checkpoint & Resume phiên Colab | ☁️ **Colab** | • Thư mục Drive 5TB `FatFormer_Hub/checkpoints/`<br>• PyTorch save/load `state_dict` | • Cơ chế auto-saving và tham số `--resume` trong `train.ipynb` | [`task_3.5_checkpoint_autosave_and_resume.md`](task_3.5_checkpoint_autosave_and_resume.md) | `[ ] Chưa xong` |
| **Tuần 3** | **Task 3.6** | Chủ trì Cổng kiểm thử tích hợp (Smoke Test Gate) trên GPU T4 | ☁️ **Colab T4** | • Mã nguồn tích hợp từ Lead A (transforms) và Lead B (SRM, Gating, Focal Loss) | • Biên bản xác nhận Smoke Test Pass | [`task_3.6_integration_smoke_test_gate.md`](task_3.6_integration_smoke_test_gate.md) | `[ ] Chưa xong` |
| **Tuần 4** | **Task 4.1** | Huấn luyện Mô hình Chính - Giai đoạn 1 & 2 (Epoch 1-5) trên GPU A100 | ☁️ **Colab A100** | • Pipeline pass Task 3.6<br>• Dữ liệu `diffusion_staging.tar` (90/10) | • Checkpoint `fatformer_srm_phase2.pth` trên Drive 5TB<br>• Log hội tụ loss & validation | [`task_4.1_train_phase_1_adaptive.md`](task_4.1_train_phase_1_adaptive.md) | `[ ] Chưa xong` |
| **Tuần 4** | **Task 4.2** | Huấn luyện Mô hình Chính - Giai đoạn 3 (Epoch 6-8) trên GPU A100 | ☁️ **Colab A100** | • Checkpoint `fatformer_srm_phase2.pth`<br>• Giáo trình nén sâu $Q \in [30, 50]$, Down-Up $224 \rightarrow 112$ | • Checkpoint hoàn thiện `fatformer_srm_robust_final.pth`<br>• Toàn bộ log huấn luyện 8 epoch | [`task_4.2_train_phase_2_hardening.md`](task_4.2_train_phase_2_hardening.md) | `[ ] Chưa xong` |
| **Tuần 4** | **Task 4.5** | Quản lý Khoảng đệm dự phòng 2 ngày & Kiểm soát hội tụ 4 Checkpoint | ☁️ **Colab** /<br>🖥️ **Họp nhóm** | • Log huấn luyện và trạng thái của 4 checkpoint | • Cả 4 checkpoint đều ở trạng thái hội tụ hoàn hảo | [`task_4.5_buffer_and_convergence_monitoring.md`](task_4.5_buffer_and_convergence_monitoring.md) | `[ ] Chưa xong` |
| **Tuần 5** | **Task 5.1** | Thực thi Full-Benchmark trên 18 tập test hai chiều (Clean vs Degraded) | ☁️ **Colab T4** | • Checkpoint cuối `fatformer_srm_robust_final.pth`<br>• 18 tập test paper chuẩn | • Ma trận số liệu Benchmark toàn diện (ACC, AP) | [`task_5.1_full_benchmark_18_datasets.md`](task_5.1_full_benchmark_18_datasets.md) | `[ ] Chưa xong` |
| **Tuần 5** | **Task 5.2** | Lập bảng Ablation Study 4 phiên bản & Khai báo giới hạn học thuật | 🖥️ **Local** /<br>☁️ **Colab T4** | • 4 checkpoint đối chứng từ Tuần 4 | • Bảng Ablation Study định lượng giá trị từng cải tiến | [`task_5.2_ablation_study_matrix.md`](task_5.2_ablation_study_matrix.md) | `[ ] Chưa xong` |
| **Tuần 5** | **Task 5.4** | Chủ trì Lắp ráp Báo cáo đồ án hoàn chỉnh & Hoàn tất Slide bảo vệ | 🖥️ **Local** | • Bản thảo Chương 1–4 từ Lead B & A<br>• Toàn bộ số liệu Task 5.1, 5.2 và ảnh Grad-CAM Task 5.3 | • Báo cáo đồ án hoàn chỉnh (PDF)<br>• Bộ slide bảo vệ đồ án chuẩn mực | [`task_5.4_technical_report_and_slides.md`](task_5.4_technical_report_and_slides.md) | `[ ] Chưa xong` |

---

## 🧭 NGUYÊN TẮC QUẢN LÝ TÀI NGUYÊN & KỸ THUẬT CỦA LEAD C

1. **Quản Trị Ngân Sách Compute Units (CU)**:
   - Các tác vụ nhẹ (kiểm tra code, chạy thử dummy, test smoke gate, fast-eval, full-benchmark): **Bắt buộc dùng GPU T4** (chỉ tốn ~1.5–2 CU/giờ).
   - Chỉ kích hoạt GPU cao cấp **A100 SXM4 (40GB)** cho tác vụ huấn luyện chính 8 epoch (Task 4.1 & 4.2). Tổng ngân sách tiêu thụ cho huấn luyện chính kiểm soát nghiêm ngặt ở mức **~25–30 CU**, giữ lại hơn 200 CU dự phòng.
2. **Chiến Lược Đánh Giá Phân Tầng (Tiered Evaluation)**:
   - Sau mỗi epoch huấn luyện, chỉ chạy `fast_eval.py` (500 ảnh/tập, mất ~8 phút) để theo dõi xu hướng hội tụ mà không làm gián đoạn tài nguyên.
   - Chỉ chạy `full_eval.py` (toàn bộ ảnh của 18 tập test) một lần duy nhất ở Tuần 5 khi đã có checkpoint cuối cùng ưng ý.
3. **An Toàn Tuyệt Đối Cho Checkpoint**:
   - Tuyệt đối không lưu checkpoint duy nhất trên ổ SSD tạm thời của máy ảo Colab (`/content/`). Sau mỗi epoch huấn luyện, mã nguồn phải tự động sao lưu file `.pth` trực tiếp sang Google Drive 5TB: `/content/drive/MyDrive/FatFormer_Hub/checkpoints/`.
4. **Tính Tự Chủ Triển Khai**: Thành viên C tự chủ tối ưu các siêu tham số huấn luyện (learning rate schedule, gradient clipping, batch accumulation), thiết kế cấu trúc notebook và kịch bản phân bổ tài nguyên, bảo đảm tiến độ và chất lượng thực nghiệm cao nhất.
