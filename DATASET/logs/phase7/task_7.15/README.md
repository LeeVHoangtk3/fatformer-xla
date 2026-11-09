# Task 7.15 — thư mục log duy nhất

**Chỉ dùng `current/` để chạy tiếp và lấy số liệu hiện hành.** `archive/` giữ lịch sử, không dùng thay kết quả hiện hành.

- [Kết quả hiện hành và đánh giá](current/README.md) — [phân tích chi tiết](current/analysis/README.md)
- [ZIP code dùng cho Colab](current/colab/phase7_corrected_colab_bundle.zip) — [SHA256](current/colab/phase7_corrected_colab_bundle.zip.sha256)
- [Calibration hiện tại](current/calibration/anchored_v2.json)
- [Vị trí package hiện tại trên Drive](current/artifacts/package_reference.json) — [metadata](current/artifacts/metadata.json)
- [Hậu kiểm từ dự đoán từng ảnh](current/analysis/verification.json)
- [Trạng thái latency của package hiện tại](current/verification/latency_phase7.json)
- [Lịch sử đã gom](archive/README.md)

Run hiện hành: **corrected_20261109_102954_3d28ad**, đã chạy trên L4/FP16/batch16 và hậu kiểm dữ liệu. **G7.2: 6/8, tổng thể FAIL. G7.3: pending đối chiếu reference tốt nhất theo điều kiện**; 20 ảnh verification và overhead L4 đã có. Clean AUC 88,66%, Fake@Real90 64,12%; Q30 AUC 79,88%, ECE 5,08%; in-dist macro balanced ACC 90,101847%. Package có hash `248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`, weights nằm trên Drive vì ZIP kết quả không kèm `.pth`.

Không cần chạy lại cùng notebook vì lỗi kỹ thuật từ run này. Dùng số liệu và giới hạn trong báo cáo hiện hành để tổng hợp; muốn hoàn tất G7.3 cần reference phù hợp, muốn nâng hai chỉ tiêu G7.2 cần thí nghiệm có kiểm soát. Bản local FP32 cũ đã chuyển vào `archive/temperature_converged_20261109_local/`; không dùng weights `16606e16…` thay cho model GPU mới. Không ghép latency/metric giữa các package khác hash.

`index.json` là danh mục hồ sơ. `PATH_MIGRATION.json` lưu đường dẫn cũ → mới và hash trước khi di chuyển để đọc các bản ghi lịch sử. Các đường dẫn ghi trong metadata cũ phản ánh thời điểm chạy; không sửa chúng để giả thành một run mới. Mặc định của công cụ xuất bundle đã trỏ về `current/`; đầu ra extraction chưa kiểm chứng mặc định nằm trong `work/` bên dưới cùng thư mục Task 7.15.
