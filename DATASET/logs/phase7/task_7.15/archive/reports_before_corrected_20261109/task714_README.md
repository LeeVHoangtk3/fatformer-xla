# Hồ sơ Task 7.14

Run vừa nhập: [locked_benchmark_phase7_20261109_074134](runs/locked_benchmark_phase7_20261109_074134/analysis/README.md).

- Phân loại: Anchored v2 tại operating point baseline cố định (T=1,6402; threshold=0,9127).
- Dữ liệu: 31 tệp gốc, 7 điều kiện CFE × 1.600 ảnh; 18 tập in-dist, 82.003 ảnh.
- Hậu kiểm: metric/CI CFE và tổng hợp in-dist đã kiểm tra; xem verification.json để biết giới hạn.
- G7.2: 5/8 tiêu chí, tổng thể FAIL tại operating point này. G7.3 còn pending.
- Chưa phải kết quả package cuối 7.15 dùng calibration v2.

[index.json](index.json) phân biệt latest_imported_run với latest_verified_final_package_run. Không thay run cũ tại `../locked_benchmark_phase7/`. Notebook trong `notebooks/task/task_7.14/` và nguồn trong `wait` được giữ nguyên byte.
