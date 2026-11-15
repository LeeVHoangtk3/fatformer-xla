# Master full-data audit

Hồ sơ hiện hành: [master_audit_20261115_040751_7beca5](runs/master_audit_20261115_040751_7beca5/README.md), chạy ngày **15/11/2026**, source commit `1f39ca8fdc11890fa1daed8cf885506c4b79eba8`.

**Đã hoàn tất 8 archive, 37/37 job, 1.618.061 lượt dự đoán; G7.2 FAIL 6/8, G7.3 PENDING.** Đây là kiểm thử mở rộng của package Task 7.15 đang được chọn, không phải model mới hoặc nghiệm thu đạt tất cả mục tiêu.

- [Danh mục run](index.json)
- [Báo cáo hậu kiểm và giới hạn](runs/master_audit_20261115_040751_7beca5/analysis/README.md)
- [Metric theo điều kiện và subset](runs/master_audit_20261115_040751_7beca5/raw/measured_condition_and_subset_summary.csv)
- [Phán quyết gốc](runs/master_audit_20261115_040751_7beca5/raw/gates.json)
- [Kiểm chứng local](runs/master_audit_20261115_040751_7beca5/analysis/verification.json)
- [Hash và vị trí bằng chứng](runs/master_audit_20261115_040751_7beca5/import_manifest.json)

Raw là bản sao nguyên byte từ `wait/master_audit_20261115_040751_7beca5`, nguồn `wait` được giữ nguyên. SQLite và NPZ được giữ trong hồ sơ local nhưng bị Git bỏ qua để tránh đưa dữ liệu nhị phân lớn vào repository; các file JSON/CSV/XML/TXT và manifest được phép theo dõi. Clone Git riêng sẽ không có SQLite/NPZ: cần phục hồi chúng từ bộ kết quả gốc, đối chiếu manifest rồi mới tái tính metric từng ảnh. Không có weights trong hồ sơ này.

Không thay calibration, threshold, package hoặc số đo lịch sử tại [Task 7.15](../task_7.15/README.md). G7.0/G7.1 không được đánh giá lại trong run này; pHash, generator-disjoint audit toàn kho và đối chứng best-by-condition chưa hoàn tất.
