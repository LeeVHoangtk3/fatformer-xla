# Hồ sơ kết quả Task 7.15

**Bản sửa mới:** [temperature_converged_20261109_local](runs/temperature_converged_20261109_local/README.md). Đã sửa hội tụ bộ fit T, fit lại ngưỡng từ cùng 400 logits, đóng gói mới và kiểm chứng local; 116 tests pass. NLL calibration 0,5013 → 0,4230; ECE 11,70% → 6,06%. Chưa nghiệm thu GPU/20 ảnh thật/benchmark evaluation cho package mới. Có bundle Colab chứa đúng code sửa; xem hướng dẫn trong bản sửa.

Run mới nhất nhập từ `wait/task7-15/`: **unified_packaging_20261109_083821_8ddcb401**.

- [Phân tích và đánh giá chi tiết](runs/unified_packaging_20261109_083821_8ddcb401/analysis/README.md)
- [6 artifact gốc, gồm package và logits](runs/unified_packaging_20261109_083821_8ddcb401/raw/)
- [Manifest SHA256](runs/unified_packaging_20261109_083821_8ddcb401/import_manifest.json)
- [Kiểm chứng dữ liệu và trạng thái nghiệm thu](runs/unified_packaging_20261109_083821_8ddcb401/analysis/verification.json)
- [Chẩn đoán hội tụ calibration](runs/unified_packaging_20261109_083821_8ddcb401/analysis/optimizer_convergence_diagnostic.json)
- [Tóm tắt latency L4](runs/unified_packaging_20261109_083821_8ddcb401/analysis/latency_summary.csv)

**Kết luận về run nộp gốc:** calibration đúng nguồn/400 ảnh, package đủ trọng số và overhead L4 đạt. Bộ tối ưu T của run này chưa hội tụ; đã xử lý trong bản sửa riêng phía trên. Chưa nghiệm thu hoàn toàn Task 7.15/G7.3 và chưa chọn package nào ở đây làm bản cuối cho báo cáo.

Giữ nguyên notebook tại `notebooks/task/task_7.15/` và nguồn `wait`. Không thay thế package/calibration dùng chung, không gộp số benchmark 7.14 cũ thành số đo package mới. Các hồ sơ `task_7.15_original_68cd0a8` và `task_7.15_recovery` được bảo toàn như lịch sử riêng.
