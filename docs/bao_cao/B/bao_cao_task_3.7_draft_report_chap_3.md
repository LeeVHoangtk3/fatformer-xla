# BÁO CÁO NGHIỆM THU NHIỆM VỤ 3.7 (TASK 3.7)
## SOẠN THẢO BÁO CÁO ĐỒ ÁN - CHƯƠNG 3 (PHƯƠNG PHÁP LUẬN VÀ KIẾN TRÚC ĐỀ XUẤT FATFORMER-XLA)

> **Người thực hiện**: **Thành viên B** *(Architecture & Loss Lead - Chủ trì)*  
> **Người nghiệm thu / Review kỹ thuật**: **Thành viên C** *(Training & Evaluation Lead - Trưởng Dự Án)*  
> **Ngày hoàn thành**: 2026-10-05  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC (PASS 100% TIÊU CHUẨN DoD)`  
> **Văn bản hoàn thiện**: [`docs/report/chuong_3_ban_thao.md`](../../report/chuong_3_ban_thao.md)  
> **Quy mô văn bản**: 15+ trang tài liệu học thuật chuyên sâu | 5 sơ đồ cấu trúc Mermaid & Bảng toán học | Chuẩn mực ký hiệu IEEE  

---

## I. MỤC TIÊU VÀ Ý NGHĨA KHOA HỌC

1. **Hoàn thiện "trái tim học thuật" của đồ án tốt nghiệp**:
   - Chương 3 là chương chiếm trọng số điểm khoa học cao nhất trước Hội đồng chấm đồ án, chịu trách nhiệm lý giải toàn bộ cơ chế hoạt động, chứng minh toán học và làm rõ tính đột phá của hệ thống đề xuất.
2. **Kế thừa và chuẩn mực hóa toàn bộ thành quả Mốc 2**:
   - Chuyển hóa toàn bộ mã nguồn PyTorch đã được kiểm thử tại Tasks 3.1, 3.2, 3.3, 3.4, 3.5 và Cổng kiểm thử 3.6 thành hệ thống công thức giải tích hình thức, sơ đồ khối vector và bảng thông số chuẩn mực.
3. **Thực hiện đúng chiến lược viết báo cáo cuốn chiếu (*Progressive Paper Writing*)**:
   - Đảm bảo khi kết thúc Mốc 2, nhóm đã hoàn thành **60% tổng khối lượng văn bản báo cáo** (Chương 1, 2 và 3), triệt tiêu hoàn toàn áp lực viết tài liệu dồn ứ ở Tuần 5.

---

## II. KẾ THỪA THÔNG SỐ TỪ CÁC TASK TIỀN ĐỀ (RULE 4)

Bản thảo Chương 3 kế thừa chính xác 100% các giá trị kỹ thuật thực tế đã nghiệm thu:
1. **Kế thừa từ Task 2.3 & Task 2.5** ([Báo cáo Task 2.3](../C/bao_cao_task_2.3_baseline_degraded_benchmark.md) & [Biên bản Mốc 1](../../milestones/milestone_1_freeze.md)):
   - Thực nghiệm One-Sided Collapse: Clean ACC **96.45%**, $Q=30$ ACC **52.89%** và Fake ACC sụp đổ xuống **5.78%** làm động lực bắt buộc cho việc thiết kế hàm `DualStreamFocalLoss`.
2. **Kế thừa từ Task 3.1** ([Báo cáo Task 3.1](../A/bao_cao_task_3.1_prepare_genimage_staging.md)):
   - Tỷ lệ dữ liệu Staging 90/10 (32.400 ProGAN + 3.600 GenImage catalyst SD v1.5 & Midjourney v5).
3. **Kế thừa từ Task 3.2** ([Báo cáo Task 3.2](../A/bao_cao_task_3.2_curriculum_degradation_scheduler.md)):
   - Thông số 3 giai đoạn Curriculum: Pha 1 ($Q \in [70, 90]$, Down-Up 192), Pha 2 ($Q \in [45, 70]$, Down-Up 160), Pha 3 ($Q \in [30, 50]$, Down-Up 112).
4. **Kế thừa từ Task 3.3** ([Báo cáo Task 3.3](bao_cao_task_3.3_srm_gating_focal_loss.md)):
   - Định nghĩa toán học của 3 bộ lọc vi sai SRM ($k_1, k_2, k_3$), mạng Bottleneck MLP của $\lambda(x) \in [0.0, 2.0]$ và Focal Loss ($\gamma=2.0, \alpha=0.25$).
5. **Kế thừa từ Task 3.4 & Task 3.6** ([Báo cáo Task 3.4](bao_cao_task_3.4_freeze_clip_train_config.md) & [Báo cáo Task 3.6](../C/bao_cao_task_3.6_integration_smoke_test_gate.md)):
   - Tỷ lệ tham số PEFT chuẩn: Đóng băng **94.1%** Backbone CLIP (~304M tham số), chỉ mở khóa **~5.9%** tham số thích ứng (~55M tham số), chốt chặn an toàn `assert_srm_kernels_frozen()` với 0 tham số rò rỉ gradient.

---

## III. NỘI DUNG CHI TIẾT BẢN THẢO HOÀN THIỆN

Văn bản đầy đủ lưu trữ tại [`docs/report/chuong_3_ban_thao.md`](../../report/chuong_3_ban_thao.md) bao gồm 7 mục trọng tâm:

- **Mục 3.1: Tổng quan thiết kế kiến trúc hệ sinh thái FatFormer-XLA**: Sơ đồ tổng thể dòng chảy dữ liệu 4 tầng qua Mermaid biểu diễn liên kết từ ảnh đầu vào đến hàm mất mát hai luồng.
- **Mục 3.2: Tầng trích xuất vết dư không gian SRM**: Chứng minh giải tích toán học của 3 ma trận lọc vi sai bậc 1, bậc 2 Laplacian và ma trận $5 \times 5$ square filter; luận chứng nguyên tắc khóa gradient (`requires_grad = False`).
- **Mục 3.3: Tầng cổng thích ứng tần số động $\lambda(x)$**: Phân tích nghịch lý hệ số tĩnh của CVPR 2024; mô hình hóa toán học cấu trúc Bottleneck MLP và hàm kích hoạt Scaled Sigmoid $\in [0.0, 2.0]$.
- **Mục 3.4: Chiến lược đóng băng tham số PEFT**: Bảng phân vùng chi tiết 8 nhóm module tham số của CLIP ViT-L/14, bảo toàn tri thức nền tảng và tối ưu hóa tài nguyên VRAM.
- **Mục 3.5: Hàm mục tiêu thích ứng hai luồng (Dual-Stream Focal Loss)**: Công thức giải tích $(1 - p_t)^\gamma$ triệt tiêu gradient của mẫu dễ, ép mô hình tập trung gradient vào các mẫu nén sâu $Q=30$.
- **Mục 3.6: Cơ chế lập lịch suy thoái giáo trình 3 giai đoạn**: Biểu đồ Gantt và bảng siêu tham số của lộ trình 8 epoch thích ứng tăng tiến chống sốc gradient.
- **Mục 3.7: Tổng kết 5 đóng góp học thuật của chương**.

---

## IV. ĐỐI SOÁT TIÊU CHUẨN HOÀN THÀNH (DoD CHECKLIST)

- [x] Bản thảo Chương 3 hoàn chỉnh, định dạng chuẩn Markdown/LaTeX học thuật chuyên sâu.
- [x] Đầy đủ sơ đồ kiến trúc hệ thống vector sắc nét (Mermaid Diagram) và bảng phân tích toán học.
- [x] Đầy đủ các công thức toán học đánh số chuẩn xác, giải thích rõ các ký hiệu và miền giá trị.
- [x] Nhất quán $100\%$ với mã nguồn thực tế tại `src/models/`, `src/training/` và `configs/train_config.yaml`.
- [x] Trưởng dự án (Lead C) rà soát và ký duyệt nghiệm thu chính thức.
