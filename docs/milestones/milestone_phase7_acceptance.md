# Biên bản trạng thái nghiệm thu Phase 7

Ngày lập hồ sơ:09/11/2026. Người lập bản tổng hợp: Codex, từ dữ liệu đã hậu kiểm. **Đây là biên bản kỹ thuật chờ con người xác nhận, không phải chữ ký hoặc quyết định nghiệm thu của A/B/C.**

## 1. Phạm vi và kết luận

Run `corrected_20261109_102954_3d28ad`, package `248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`. [Phán quyết gốc](../../DATASET/logs/phase7/task_7.15/current/raw/acceptance.json) và [view tổng hợp có nguồn](../../DATASET/logs/phase7/task_7.16/final_phase7_report.json). Model ở Drive, ZIP không có weights; local không băm lại actual package. Không lấy thí nghiệm/đo/tài liệu hoàn tất thay cho PASS tất cả gate.

| Cổng | Kết quả hiện hành | Bằng chứng / phần còn thiếu |
| :--- | :--- | :--- |
| G7.0 | **PENDING** theo DoD toàn repo | Canonical core: 3 checkpoint nạp 0/0, Gen 1 bị từ chối; còn loader legacy và CI chẩn đoán n3/n8 chưa đủ. |
| G7.1 | **FAIL** | Gen 2: Clean Fake@Real90 48,88% <55%; Q30 AUC 43,67% <55%. |
| G7.2 | **FAIL — 6/8** | Clean Fake@Real90 64,12% <65%; worst-subset Fake ACC 30,86% <45%. |
| G7.3 | **PENDING** | 20 ảnh và overhead L4 0,188055 ms/ảnh đạt; thiếu reference tốt nhất theo điều kiện. |

G7.2 negative result có thể bàn giao để đánh giá khoa học với hạn chế rõ; chưa có chứng minh Pareto/best-reference và chữ ký, nên **không tuyên bố nghiệm thu toàn Phase 7**.

## 2. Trạng thái 16 task

| Task | Chủ trì | Trạng thái | Kết quả và giới hạn |
| :--- | :--- | :--- | :--- |
| 7.1 | C | [~] | Có audit 4 checkpoint; Gen 1 invalid_layout; n3/n8 verdict inconclusive vì thiếu CI phù hợp. |
| 7.2 | B | [~] | Canonical core và từ chối layout đã xác minh; repo-wide legacy/exception review chưa đóng. |
| 7.3 | C | [~] | Metric/CI đã đo; raw corrected schema v1 và view tài liệu v3 tách rõ; thiếu phân tầng generator chuẩn. |
| 7.4 | C | [x] | Calibration runtime 400 ảnh đã hội tụ; T=2,78324914; FPR calibration 5%, không bảo đảm mọi miền. |
| 7.5 | C | [x] | 12 ứng viên measured; chọn α=0,55, β=0,35; metadata nhúng và actual hash khớp matrix. |
| 7.6 | C | [x] | Đã đo đủ 4 cấu hình ×2 điều kiện; kết quả G7.1 FAIL, không phải lỗi chưa chạy. |
| 7.7 | A | [x] | 14.400 bản ghi; train/val_calib/val_select=12.240/1.080/1.080; audit 0 vi phạm trong reference đã ghi. |
| 7.8 | A | [x] | Pipeline v2 và benchmark CPU synthetic: overhead 14,08% <15%; không suy thành latency CLIP/GPU. |
| 7.9 | B | [x] | Loss v2 có code/smoke; kiểm thử thành phần được ghi riêng. Không đồng nghĩa đã chứng minh tăng khả năng tổng quát. |
| 7.10 | B | [x] | Selection worst-group đã dùng trong training; CPU toy overhead 1,046×, không phải CLIP benchmark. |
| 7.11 | C | [~] | Có 5 epoch, chọn epoch 4; val_select worst 46,67%; log grad_norm NaN còn ở epoch 1,4. |
| 7.12 | C+B | [~] | Có ablation 4 cấu hình/seed 42; chưa chứng minh contribution tổng quát của hai loss; không đổi gate từ report ablation. |
| 7.13 | B | [~] | Router code có regression; experiment deferred, deployment tắt; số synthetic v1 thu hồi; fixed fallback dùng được. |
| 7.14 | C | [~] | Đã chấm package thực, đủ dự đoán và CI; G7.2 FAIL 6/8; taxonomy/schema còn ghi rõ giới hạn. |
| 7.15 | C | [~] | 20 ảnh và overhead L4 đạt; G7.3 chờ reference; weights mới trên Drive, không trong ZIP local. |
| 7.16 | A+B+C | [~] | Biên soạn/sync tài liệu và kiểm tra tự động hoàn tất; chờ cross-review và chữ ký thật A/B/C. |

## 3. Checklist 11 tài liệu bắt buộc

- [x] [ke_hoach.md](../plan/ke_hoach.md) — đã cập nhật số thật, nguồn và phạm vi hiệu lực.
- [x] [ke_hoach_cac_cot_moc_va_diem_dung.md](../plan/ke_hoach_cac_cot_moc_va_diem_dung.md) — đã cập nhật số thật, nguồn và phạm vi hiệu lực.
- [x] [tong_hop_checkpoints_va_lich_su_thuc_nghiem.md](../plan/tong_hop_checkpoints_va_lich_su_thuc_nghiem.md) — đã cập nhật số thật, nguồn và phạm vi hiệu lực.
- [x] [bao_cao_kiem_dinh_communityforensics_eval.md](../bao_cao/overall/bao_cao_kiem_dinh_communityforensics_eval.md) — đã cập nhật số thật, nguồn và phạm vi hiệu lực.
- [x] [ke_hoach_test_communityforensics_eval.md](../plan/ke_hoach_test_communityforensics_eval.md) — đã cập nhật số thật, nguồn và phạm vi hiệu lực.
- [x] [README_A.md](../tasks/member_a_data_infra/README_A.md) — đã cập nhật số thật, nguồn và phạm vi hiệu lực.
- [x] [README_B.md](../tasks/member_b_architecture_loss/README_B.md) — đã cập nhật số thật, nguồn và phạm vi hiệu lực.
- [x] [README_C.md](../tasks/member_c_training_eval/README_C.md) — đã cập nhật số thật, nguồn và phạm vi hiệu lực.
- [x] [README.md](../bao_cao/overall/README.md) — đã cập nhật số thật, nguồn và phạm vi hiệu lực.
- [x] [tong_quan_du_an.md](../bao_cao/overall/tong_quan_du_an.md) — đã cập nhật số thật, nguồn và phạm vi hiệu lực.
- [x] [bao_cao_task_6.6_r_locked_recovery.md](../bao_cao/C/bao_cao_task_6.6_r_locked_recovery.md) — đã cập nhật số thật, nguồn và phạm vi hiệu lực.

## 4. Hồ sơ bàn giao và kiểm tra máy

- [x] [bao_cao_ra_soat_so_lieu_phase7.md](../bao_cao/overall/bao_cao_ra_soat_so_lieu_phase7.md) — số cũ→mới→JSON pointer.
- [x] [chuong_5_ket_luan_va_han_che.md](../bao_cao/overall/chuong_5_ket_luan_va_han_che.md) — đủ 5.1–5.4, số pp thiếu và CI/giới hạn.
- [x] [slide_phase7_5phut.md](../bao_cao/overall/slide_phase7_5phut.md) —6slide/300 giây, số sinh từ cùng JSON.
- [x] [evidence_catalog.json](../../DATASET/logs/phase7/task_7.16/evidence_catalog.json) và source hash; snapshot các tài liệu trước sửa trong `original_documents/`.
- [x]103test mục tiêu và 53test thành phần pass: **156 case trên 9file riêng biệt**, không tuyên bố toàn repository pytest xanh.
- [x] Phân biệt current report path với các vị trí loader legacy/khởi tạo/diagnostic; không xóa grep để làm giả repo-wide sạch.
- [ ] Dọn hoặc phê duyệt rõ phạm vi legacy của G7.0; hiện còn pending, không thuộc việc ký khống để đóng7.16.
- [ ] Best-reference theo điều kiện cho G7.3; không tạo phép đo hoặc CI thay thế.

## 5. Cross-review và chữ ký thật

| Vai trò | Họ tên | Nội dung phải đọc chéo | Xác nhận/ngày ký |
| :--- | :--- | :--- | :--- |
| A | Chưa cung cấp | Kiến trúc/đóng gói của B+C; kiểm tra counts và phạm vi reference | **Chưa ký** |
| B | Chưa cung cấp | Báo cáo CFE/Chương5 của C và hồ sơ dữ liệu của A | **Chưa ký** |
| C | Chưa cung cấp | README_A/README_B, đối chiếu nguồn, metric, gate và slide | **Chưa ký** |

Không ai được đánh dấu đã đọc hoặc đã ký khi chưa có xác nhận thực. Chữ ký/đánh giá này không được tạo bởi script. Trạng thái7.16: **biên soạn và kiểm tra tự động hoàn tất; nghiệm thu chính thức chờ cross-review A/B/C**. Dòng này không thay trạng thái FAIL/pending của các gate.
