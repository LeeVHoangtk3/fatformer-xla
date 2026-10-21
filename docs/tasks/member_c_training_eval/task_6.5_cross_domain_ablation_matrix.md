# TASK 6.5: LẬP MA TRẬN ABLATION STUDY ĐA MIỀN & PHÂN TÍCH ĐÓNG GÓP THÀNH PHẦN

- **Mã nhiệm vụ**: `Task 6.5` *(Milestone 6 - Tuần 6)*
- **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**
- **Người phối hợp**: **Thành viên B (Architecture Lead)**
- **Vị trí trong luồng quy trình**: **Thực thi ngay sau Task 6.3-C và trước Task 6.4**
- **Môi trường thực thi**: ☁️ **Colab GPU** / 🖥️ **Local**
- **Độ ưu tiên**: 🟠 Cao (Cung cấp luận cứ khoa học định lượng từng biến số)
- **Trạng thái**: `[x] ĐÃ HOÀN THÀNH`

---

## 1. Mục Tiêu & Cơ Sở Khoa Học

Để bảo đảm tính khoa học trước hội đồng, mọi ứng viên mô hình (từ M1 đến M5) phải được kiểm thử song song trên cùng một tập dữ liệu và điều kiện đánh giá trước khi bước vào khâu hiệu chuẩn xác suất và tuyển chọn tại Task 6.4.

Mục tiêu chính:
1. Thu thập và đo đạc đồng bộ 5 cấu hình thử nghiệm: Baseline gốc (M1), Plan A (M2), + Dữ liệu đa miền (M3), + Distillation & Consistency Loss (M4), + LoRA ViT Adapter (M5).
2. Định lượng đóng góp cô lập của từng thành phần kỹ thuật trên cả 2 chiều: Bảo toàn ảnh sạch (Clean) và Bền vững nén sâu (Degraded $Q=30$).
3. Bàn giao toàn bộ kết quả đo đạc thô cho Task 6.4 thực hiện hiệu chuẩn nhiệt độ và sàng lọc theo Gate 4 điều kiện.

---

## 2. Ma Trận Cấu Hình Thử Nghiệm Đối Đầu (M1 – M5)

| STT | Cấu Hình Mô Hình | Dữ Liệu Huấn Luyện | Hàm Mất Mát | Kiến Trúc Can Thiệp | Mục Tiêu Khoa Học |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **M1** | **Baseline Gốc** | ProGAN 4-class (Tác giả) | Cross-Entropy chuẩn | CLIP ViT-L/14 + FAA gốc | Xác lập mốc chuẩn tham chiếu |
| **M2** | **Plan A Finetune** | 144k ProGAN + 3.6k Staging | CE + Curriculum Scheduler | SRM 3-Kernels + Gating $\lambda(x)$ | Đo đạc hiệu quả của Chiến lược S1 + S2 |
| **M3** | **+ Dữ Liệu Đa Miền** | `train_crossdomain` (Task 6.2) | CE + Curriculum Scheduler | SRM 3-Kernels + Gating $\lambda(x)$ | Chứng minh giá trị của kho bộ sinh mở rộng |
| **M4** | **+ Distillation & Consistency** | `train_crossdomain` (Task 6.2) | CE + KD Loss + Consistency Loss | SRM 3-Kernels + Gating $\lambda(x)$ | Chứng minh vai trò giữ mỏ neo ảnh sạch |
| **M5** | **+ LoRA Adapter (Mở rộng)** | `train_crossdomain` (Task 6.2) | CE + KD Loss + Consistency Loss | SRM + Gating + LoRA ViT ($r=8$) | Đánh giá khả năng mở rộng tham số ViT |

---

## 3. Các Chỉ Số Đo Lường Phân Tầng

Trên tập kiểm định `val_selection` (Task 6.2), toàn bộ 5 cấu hình được đo đạc qua 5 chỉ số khách quan:
1. **Clean Macro ACC**: Đánh giá mức độ bảo toàn tri thức ảnh sạch.
2. **Clean Fake ACC**: Khả năng phân biệt ảnh giả trên điều kiện lý tưởng.
3. **Q30 Overall ACC**: Độ bền vững tổng thể khi ảnh bị nén sâu.
4. **Q30 Fake ACC**: Khả năng bóc tách tạo tác giả mạo dưới nhiễu nén.
5. **Macro AUC / AP**: Năng lực phân định ranh giới không phụ thuộc vào ngưỡng phân loại.

---

## 4. Tiêu Chuẩn Nghiệm Thu (Definition of Done)

- [x] Hoàn thành script đánh giá tự động `tools/eval_crossdomain_ablation.py`.
- [x] Xuất tệp dữ liệu chi tiết `DATASET/logs/crossdomain_ablation_matrix.csv` gồm đầy đủ 5 cấu hình.
- [x] Lập biểu đồ radar chart hoặc bar chart trực quan hóa sự dịch chuyển hiệu năng giữa M1 $\rightarrow$ M5 ([`tools/plot_crossdomain_ablation.py`](../../../tools/plot_crossdomain_ablation.py)).
- [x] Bàn giao toàn bộ danh sách checkpoint và logits thô cho Task 6.4 thực hiện hiệu chuẩn và xếp hạng ([`val_split_indices.json`](../../../DATASET/logs/val_split_indices.json)).
- [x] Biên soạn báo cáo nghiệm thu tại [`docs/bao_cao/C/bao_cao_task_6.5_cross_domain_ablation_matrix.md`](../../bao_cao/C/bao_cao_task_6.5_cross_domain_ablation_matrix.md).
