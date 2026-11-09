# Lịch sử Task 7.15 — không dùng làm bản hiện hành

Nguồn duy nhất để chạy tiếp/lấy số liệu hiện hành là [../current/](../current/README.md).

- `submission_20261109/`: sáu artifact và hậu kiểm run nộp ngày 09/11, package `8ddcb401…`. Calibration đúng 400 ảnh nhưng fit T chưa hội tụ; latency L4 chỉ thuộc package này.
- `original_68cd0a8/`: bản gốc có calibration 2.000 ảnh, không dùng để nghiệm thu calibration độc lập.
- `recovery_legacy/`: hồ sơ khôi phục trước khi nhận logits mới và trước khi sửa hội tụ.
- `shared_paths_before_consolidation/`: các tệp Task 7.15 trước đây nằm rải ở `phase7/` và `phase7/calibration/`.
- `navigation_before_consolidation/`: README, index và báo cáo trước khi thống nhất thư mục.
- `export_before_consolidation/`: ZIP và kiểm chứng export trước khi sửa các đường dẫn mặc định.

Nội dung bằng chứng cũ được giữ nguyên byte. Các lệnh/đường dẫn trong chúng là ghi nhận lịch sử, không phải hướng dẫn chạy hiện tại. Dùng [PATH_MIGRATION.json](../PATH_MIGRATION.json) để tra vị trí mới. Không cộng hoặc ghép các kết quả giữa các package khác hash.
