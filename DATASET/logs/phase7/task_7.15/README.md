# Task 7.15 — thư mục log duy nhất

**Chỉ dùng `current/` để chạy tiếp và lấy số liệu hiện hành.** `archive/` giữ lịch sử, không dùng thay kết quả hiện hành.

- [Hướng dẫn chạy và bản sửa hiện tại](current/README.md)
- [ZIP code dùng cho Colab](current/colab/phase7_corrected_colab_bundle.zip) — [SHA256](current/colab/phase7_corrected_colab_bundle.zip.sha256)
- [Calibration hiện tại](current/calibration/anchored_v2.json)
- [Package hiện tại](current/artifacts/unified_fatformer_v2.pth) — [metadata](current/artifacts/metadata.json)
- [Kiểm chứng local](current/verification/repair_validation.json)
- [Trạng thái latency của package hiện tại](current/verification/latency_phase7.json)
- [Lịch sử đã gom](archive/README.md)

Trạng thái: **đã sửa hội tụ, hiệu chuẩn và kiểm chứng local; chưa nghiệm thu toàn bộ G7.3**. T≈2,78636, ngưỡng≈0,79532, NLL calibration=0,4230, ECE=6,06%; FPR calibration=5%. Không lấy số liệu calibration làm accuracy evaluation. Package hiện tại có hash `16606e16c89901f39ab78c29951c1f83d9412f98f27ab55853c2e2dd13ee5f74`.

Chạy tiếp bằng notebook tổng hợp hiện có và ZIP trong `current/colab/`; không sửa notebook. Khi có kết quả GPU, chỉ cập nhật bản hiện hành sau khi đối chiếu cùng hash package và lưu bản trước vào `archive/`. Không ghép latency/metric từ package cũ vào báo cáo package mới.

`index.json` là danh mục hồ sơ. `PATH_MIGRATION.json` lưu đường dẫn cũ → mới và hash trước khi di chuyển để đọc các bản ghi lịch sử. Các đường dẫn ghi trong metadata cũ phản ánh thời điểm chạy; không sửa chúng để giả thành một run mới. Mặc định của công cụ xuất bundle đã trỏ về `current/`; đầu ra extraction chưa kiểm chứng mặc định nằm trong `work/` bên dưới cùng thư mục Task 7.15.
