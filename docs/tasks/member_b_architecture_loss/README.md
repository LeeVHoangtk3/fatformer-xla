# BẢNG ĐIỀU KHIỂN NHIỆM VỤ: THÀNH VIÊN B (ARCHITECTURE & LOSS LEAD)

> **Vai trò**: Trưởng nhóm Kiến trúc Mô hình & Hàm Mất Mát (Architecture & Loss Lead)  
> **Trách nhiệm cốt lõi**: Quản lý tính toàn vẹn của mã nguồn mô hình tại `src/models/`, bảo đảm nạp khớp 100% (1.116 tensors) checkpoint gốc; hiện thực hóa tầng kiến trúc **Spatial Residual Block (SRM 3-Kernels)** và mạng cổng thích ứng tần số động **$\lambda(x) \in [0.0, 2.0]$**; cấu hình đóng băng 94.1% CLIP ViT-L/14; xây dựng hàm mục tiêu thích ứng **`DualStreamFocalLoss`**; huấn luyện Checkpoint Ablation 3; trích xuất ảnh giải thích mô hình Grad-CAM (XAI) và chủ trì soạn thảo bản thảo Báo cáo đồ án (Chương 1–4).

---

## 📌 BẢNG LỘ TRÌNH VÀ CHECKLIST NHIỆM VỤ (TUẦN 1 - TUẦN 5)

| Tuần | Mã Task | Tên nhiệm vụ | Môi trường | Đầu vào (Input) | Đầu ra (Output) | Tệp Chi Tiết | Trạng thái |
| :---: | :---: | :--- | :---: | :--- | :--- | :---: | :---: |
| **Tuần 1** | **Task 1.2** | Kiểm chứng mã nguồn mô hình & Trình nạp khớp 1.116 tensors | 🖥️ **Local** /<br>☁️ **Colab T4** | • Mã nguồn `src/models/`<br>• Checkpoint `fatformer_4class_ckpt.pth`<br>• Backbone `ViT-L-14.pt` | • Báo cáo pass `strict=True` 1.116 tensor<br>• Script `tools/test_src_load.py` | [`task_1.2_fatformer_model_and_weights_loader.md`](task_1.2_fatformer_model_and_weights_loader.md) | `[ ] Chưa xong` |
| **Tuần 2** | **Task 2.4** | Trích xuất Grad-CAM ban đầu đối chứng trên mô hình gốc | 🖥️ **Local** /<br>☁️ **Colab T4** | • Checkpoint gốc<br>• 5 cặp ảnh mẫu (Real vs Fake) ở trạng thái Clean và Degraded ($Q=30$) | • Script `tools/visualize_cam.py`<br>• Bộ 10 ảnh Grad-CAM đối chứng mốc sàn | [`task_2.4_gradcam_baseline_extraction.md`](task_2.4_gradcam_baseline_extraction.md) | `[ ] Chưa xong` |
| **Tuần 2** | **Task 2.6** | Soạn thảo Chương 1 (Giới thiệu) & Chương 2 (Công trình liên quan) | 🖥️ **Local** | • Tài liệu nghiên cứu, paper CVPR 2024, phân tích mốc sàn Baseline | • Bản thảo Chương 1 & Chương 2 (PDF/LaTeX) | [`task_2.6_draft_report_chap_1_2.md`](task_2.6_draft_report_chap_1_2.md) | `[ ] Chưa xong` |
| **Tuần 3** | **Task 3.3** | Hiện thực hóa SRM 3-Kernels, Dynamic Gating & Dual-Stream Focal Loss | 🖥️ **Local** +<br>☁️ **Colab T4** | • Công thức toán học SRM 3-Kernels, gating Sigmoid $\lambda(x)$, Focal Loss ($\gamma=2.0, \alpha=0.25$) | • Code hoàn chỉnh: `src/models/srm.py`, `src/models/gating.py`, `src/training/loss.py` | [`task_3.3_srm_and_frequency_gating_integration.md`](task_3.3_srm_and_frequency_gating_integration.md) | `[ ] Chưa xong` |
| **Tuần 3** | **Task 3.4** | Cấu hình đóng băng 94.1% CLIP ViT & File cấu hình huấn luyện | 🖥️ **Local** | • Kiến trúc mô hình hoàn chỉnh từ Task 3.3 | • File `configs/train_config.yaml`<br>• Logic set `requires_grad` trong code | [`task_3.4_freeze_clip_and_train_config.md`](task_3.4_freeze_clip_and_train_config.md) | `[ ] Chưa xong` |
| **Tuần 3** | **Task 3.7** | Soạn thảo Chương 3: Phương pháp luận & Kiến trúc FatFormer-XLA | 🖥️ **Local** | • Công thức toán SRM, Haar DWT, Gating $\lambda(x)$, Curriculum và Dual-Stream Focal Loss | • Bản thảo Chương 3 (PDF/LaTeX kèm sơ đồ kiến trúc vector) | [`task_3.7_draft_report_chap_3.md`](task_3.7_draft_report_chap_3.md) | `[ ] Chưa xong` |
| **Tuần 4** | **Task 4.4** | Huấn luyện Checkpoint Ablation 3 (Thêm SRM, không Gating/Focal) | ☁️ **Colab L4/A100** | • Mô hình có nhánh SRM 3-Kernels<br>• Dữ liệu ProGAN gốc, hàm Cross-Entropy Loss | • Checkpoint `fatformer_srm_only.pth` trên Drive 5TB<br>• Log hội tụ 5 epoch | [`task_4.4_train_ablation_srm_only.md`](task_4.4_train_ablation_srm_only.md) | `[ ] Chưa xong` |
| **Tuần 4** | **Task 4.6** | Chủ trì soạn thảo Chương 4: Thiết lập thực nghiệm & Môi trường | 🖥️ **Local** | • Siêu tham số (LR, batch size, optimizer), phần cứng A100/T4, giáo trình Curriculum | • Bản thảo Chương 4 hoàn chỉnh | [`task_4.6_draft_report_chap_4.md`](task_4.6_draft_report_chap_4.md) | `[ ] Chưa xong` |
| **Tuần 5** | **Task 5.3** | Xuất bộ ảnh Grad-CAM đối đầu chứng minh tính giải thích (XAI) | 🖥️ **Local** /<br>☁️ **Colab T4** | • Checkpoint gốc vs Checkpoint cuối `fatformer_srm_robust_final.pth`<br>• Tập ảnh test nén sâu $Q=30$ | • Bộ 10 ảnh Grad-CAM phân giải cao so sánh song song | [`task_5.3_xai_gradcam_comparison.md`](task_5.3_xai_gradcam_comparison.md) | `[ ] Chưa xong` |

---

## 🧭 4 QUY TẮC KỸ THUẬT BẮT BUỘC CỦA LEAD B

1. **Quy Tắc Typo Bắt Buộc (Khớp Checkpoint Gốc)**:
   - Trong class `LanguageGuidedAlignment`, module enhancer bắt buộc phải đặt tên thuộc tính chính xác là `self.patch_basaed_enhancer` (có chữ `a` thừa) để khớp 100% với tên key trong file checkpoint gốc `fatformer_4class_ckpt.pth`.
2. **Quy Tắc Nạp Weights Backbone**:
   - Tệp `ViT-L-14.pt` là TorchScript JIT archive, nạp thông qua `torch.jit.load()` hoặc `torch.load(..., weights_only=False)`. Checkpoint huấn luyện được bọc trong khóa `ckpt['model']`.
3. **Quy Tắc Bộ Lọc Vi Sai SRM 3-Kernels**:
   - 3 ma trận lọc vi sai SRM ($1^{st}$ order, $2^{nd}$ order Laplacian, $5 \times 5$ square) luôn phải đặt `requires_grad = False` để trở thành bộ lọc cố định, tuyệt đối không tính gradient hay tiêu tốn VRAM tối ưu.
4. **Quy Tắc Đóng Băng Tuyệt Đối (Freeze CLIP)**:
   - Nghiêm cấm mở khóa gradient của các tầng ViT gốc. Chỉ 5.9% tham số được phép có `requires_grad = True` (FAA adapters, SRM projection layer, Gating MLP network, LGA và soft prompts).
5. **Tính Tự Chủ Triển Khai**: Chi tiết cài đặt hàm toán học, căn chỉnh shape tensor trong `src/models/` do Thành viên B tự chủ xử lý, miễn bảo đảm đúng tiêu chuẩn đầu ra và vượt qua cổng Smoke Test Gate.
