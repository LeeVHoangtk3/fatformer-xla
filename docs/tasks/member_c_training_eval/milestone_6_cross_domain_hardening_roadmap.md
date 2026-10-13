# KẾ HOẠCH LỘ TRÌNH CHIẾN LƯỢC: MILESTONE 6
## KHẮC PHỤC LỆCH PHÂN PHỐI & HUẤN LUYỆN ĐA MIỀN TỔNG QUÁT (CROSS-DOMAIN HARDENING)

> **Mã giai đoạn**: `Milestone 6`  
> **Chủ trì kỹ thuật**: **Thành viên C (Training & Evaluation Lead)**  
> **Phối hợp chuyên môn**: **Thành viên B (Architecture Lead)**, **Thành viên A (Data & Infra Lead)**  
> **Bối cảnh khởi tạo**: Kế thừa kết quả thực chứng từ Task 5.1 ([Báo cáo Task 5.1](../../bao_cao/C/bao_cao_task_5.1_full_benchmark_18_datasets.md)). Lượt fine-tune Plan A đã bảo toàn tốt miền ProGAN và tăng Fake ACC tại mức nén sâu Q=30 gấp 4,76 lần, nhưng gặp hiện tượng chuyên biệt hóa cục bộ (over-specialization), làm suy giảm độ chính xác trên các bộ sinh lạ.  
> **Nguyên tắc học thuật cốt lõi**: Khóa cố định 18 tập test của Task 5.1 làm bài đối chiếu hậu nghiệm (Post-hoc Comparative Benchmark). Tuyệt đối không huấn luyện ép trên tập test để tránh rò rỉ dữ liệu (Data Leakage). Giải quyết tận gốc hiện tượng lệch phân phối (Distribution Shift) bằng dữ liệu huấn luyện và kiểm định độc lập.

---

## 1. TIÊU CHUẨN NGHIỆM THU GATE 4 ĐIỀU KIỆN RÀNG BUỘC (TRÊN CÙNG TẬP VAL_CROSSDOMAIN)

Để bảo đảm tính công bằng khoa học và triệt tiêu sai lệch do độ khó khác nhau giữa các tập dữ liệu, mốc sàn so sánh bắt buộc phải được đo đạc trực tiếp từ mô hình baseline gốc trên tập kiểm định đa miền độc lập `val_crossdomain` ($\text{Base}_{\text{clean}}$, $\text{Base}_{\text{fake\_clean}}$, $\text{Base}_{\text{q30}}$, $\text{Base}_{\text{fake\_q30}}$, $\text{Base}_{\text{real\_q30}}$).

Mô hình ứng viên chỉ vượt qua Gate khi thỏa mãn đồng thời cả 4 điều kiện chênh lệch tương đối (percentage points - pp) sau đây:

1. **Điều kiện 1 (Bảo tồn ảnh sạch)**: Clean Macro ACC $\ge \text{Base}_{\text{clean}} - 1.5\text{ pp}$ (giới hạn độ sụt giảm tối đa 1.5 điểm phần trăm).
2. **Điều kiện 2 (Ổn định độ nhạy ảnh giả sạch)**: Clean Fake ACC $\ge \text{Base}_{\text{fake\_clean}} - 2.0\text{ pp}$.
3. **Điều kiện 3 (Bảo toàn độ chính xác tổng thể nén sâu)**: Q30 Overall ACC $\ge \text{Base}_{\text{q30}}$ (tối thiểu ngang bằng mốc sàn baseline trên cùng tập kiểm định).
4. **Điều kiện 4 (Bứt phá khả năng bắt ảnh giả nén sâu)**: Q30 Fake ACC $\ge \text{Base}_{\text{fake\_q30}} + 10.0\text{ pp}$, đồng thời Q30 Real ACC $\ge 60.0\%$ (không bị sụp đổ phân loại ảnh thật).

Nếu không thỏa mãn đồng thời cả 4 tiêu chuẩn trên, mô hình chưa được phép đưa vào nghiệm thu kiểm chuẩn cuối cùng.

---

## 2. LUỒNG THỰC THI CHUẨN HÓA & DANH MỤC NHIỆM VỤ (TASK 6.1 – TASK 6.6)

Quy trình triển khai tuân thủ thứ tự khoa học: **Thực hiện toàn bộ các nhánh thử nghiệm cắt bỏ (Ablation) trước, sau đó mới hiệu chuẩn và xếp hạng để tuyển chọn mô hình đại diện**:

`Task 6.1` $\rightarrow$ (`Task 6.2` song song `Task 6.3-B`) $\rightarrow$ `Smoke Gate M6` $\rightarrow$ `Task 6.3-C` $\rightarrow$ `Task 6.5` $\rightarrow$ `Task 6.4` $\rightarrow$ `Task 6.6`

| Mã Task | Tên Nhiệm Vụ | Mục Tiêu Kỹ Thuật | Phương Pháp Triển Khai | Đầu Ra Bàn Giao |
| :---: | :--- | :--- | :--- | :--- |
| **Task 6.1** | **Phân tích định lượng hiện tượng lệch phân phối** | Xác minh tệp log thô hoặc phân tích metric tổng hợp per-generator | Bóc tách Fake Recall, Real Recall, AUC giữa các họ GAN vs Diffusion; không chạy lại Task 5.1 để tune ngưỡng | Báo cáo chẩn đoán lỗi chuyên sâu & Biểu đồ phân tích độ nhạy |
| **Task 6.2** | **Xây dựng kho train/val đa bộ sinh độc lập & Đo mốc sàn** | Chuẩn bị dữ liệu độc lập ngoài 18 tập test; đo baseline trên val | Kiểm toán pHash near-duplicate; tách domain ảnh Real; lập `data_manifest.json`; đo mốc sàn baseline gốc trên val | Kho dữ liệu cân bằng, Manifest chuẩn & Bảng mốc sàn baseline trên val |
| **Task 6.3-B**| **Thiết kế hàm loss Distillation & Paired Augmentation** | Xây dựng cơ chế giữ mỏ neo tri thức và đồng bộ logit | Module `CrossDomainDistillationLoss` KL trên ảnh clean; `PairedConsistencyLoss`; cấu hình LoRA ViT Adapter | Module loss hoàn thiện & Pipeline ghép cặp Clean/Degraded |
| **Smoke Gate**| **Cổng kiểm thử tích hợp Milestone 6** | Xác nhận an toàn kỹ thuật trước khi cấp phát GPU lớn | Kiểm tra forward/backward 1 batch: paired transform đúng cặp, teacher không có gradient, loss hữu hạn, sampler chuẩn | Biên bản xác nhận Smoke Gate Pass 100% |
| **Task 6.3-C**| **Huấn luyện thích ứng chống lệch miền trên GPU** | Thực thi huấn luyện các nhánh mô hình với điều kiện cố định | Cố định random seed (42), data manifest và schedule; giới hạn đĩa chỉ lưu checkpoint latest và best | Checkpoints ứng viên M3, M4, M5 & Log hội tụ |
| **Task 6.5** | **Nghiên cứu cắt bỏ thành phần (Ablation Study đa miền)** | Thu thập kết quả đo lường thô của toàn bộ các cấu hình M3, M4, M5 | Đánh giá đối chứng Plan A, thêm dữ liệu đa miền (M3), thêm distillation (M4), thêm LoRA ViT (M5) | Ma trận số liệu Ablation Study đa chiều hoàn chỉnh |
| **Task 6.4** | **Hiệu chuẩn xác suất & Tuyển chọn Checkpoint** | Hiệu chuẩn nhiệt độ, quét ngưỡng và chọn mô hình tối ưu | Tách `val_calibration` (fit Temperature, quét ngưỡng) và `val_selection` (xếp hạng độc lập theo Gate 4 điều kiện) | Checkpoint tuyển chọn đại diện & Bảng hiệu chuẩn xác suất |
| **Task 6.6** | **Kiểm chuẩn khóa sau cùng (Final Benchmark)** | Đánh giá độc lập lần cuối khẳng định năng lực tổng quát | Chạy 1 lượt kiểm chuẩn; đối chiếu hậu nghiệm trên 18 tập test Task 5.1 và kiểm thử trên tập độc lập mở rộng | Báo cáo kiểm chuẩn chung cuộc & Hiện vật nghiệm thu |

---

## 3. NGUYÊN TẮC QUẢN TRỊ KỸ THUẬT

1. **Minh Bạch Về Tập Kiểm Chuẩn**: Tập dữ liệu Task 5.1 đóng vai trò bài kiểm tra đối chiếu hậu nghiệm. Mọi kết luận tổng quát hóa độc lập phải được kiểm chứng thêm qua phân chia validation độc lập.
2. **Quy Trình Triển Khai Cuốn Chiếu**: Bắt đầu bằng việc hiểu rõ bản chất lỗi tại Task 6.1 trước khi bắt tay vào chuẩn bị dữ liệu ở Task 6.2 và huấn luyện ở Task 6.3.
3. **Phân Tách Hiệu Chuẩn và Tuyển Chọn**: Tuyệt đối không dùng chung một tập dữ liệu để vừa tối ưu hóa tham số hiệu chuẩn vừa tính điểm xếp hạng mô hình.
