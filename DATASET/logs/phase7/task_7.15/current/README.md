# Kết quả hiện hành — corrected_20261109_102954_3d28ad

**Đã hoàn tất chạy và hậu kiểm. G7.2: 6/8 tiêu chí, tổng thể FAIL. G7.3: pending vì chưa có đối chiếu với reference tốt nhất theo từng điều kiện.** Đây là kết quả đo package thực, không còn chỉ là bản sửa calibration local.

- [Phân tích và đánh giá công tâm](analysis/README.md)
- [Phán quyết nghiệm thu của run](raw/acceptance.json)
- [Báo cáo benchmark gốc](raw/benchmark/final_report.json)
- [Metric/CI tính lại từ dự đoán từng ảnh](analysis/recomputed_metrics.json) — [CI95](analysis/ci_verification.json)
- [Tổng hợp 7 điều kiện CFE](analysis/condition_summary.csv) — [18 tập in-dist](analysis/per_subset_indist.csv)
- [Calibration GPU hiện hành](calibration/anchored_v2.json)
- [Latency L4 cùng package](verification/latency_phase7.json) — [bảng tái tính](analysis/latency_summary.csv)
- [Thông tin vị trí model trên Drive](artifacts/package_reference.json)
- [Kiểm chứng toàn bộ hồ sơ](analysis/verification.json) — [manifest nhập nguồn](import_manifest.json)
- [ZIP code đã dùng trong run](colab/phase7_corrected_colab_bundle.zip) — [SHA256](colab/phase7_corrected_colab_bundle.zip.sha256)

Điểm chính: Clean AUC **88,66%**, Fake@Real90 **64,12%**; Q30 AUC **79,88%**, Fake@Real90 **42,62%**, ECE **5,08%**. In-dist macro balanced ACC **90,101847%**, pooled Real ACC **90,406282%**. Điểm yếu lớn là Fake ACC nhóm `5_other_autoregressive` chỉ **30,86%**. Không diễn giải FPR calibration 5% thành bảo đảm FPR ≤5% trên mọi miền.

GPU **NVIDIA L4**, FP16/batch16; calibration T **2,7832491397857666**, ngưỡng **0,7918508648872375**. Kiểm chứng 20 ảnh dữ liệu thực báo sai số xác suất 0. Overhead median batch1 **0,188055 ms/ảnh**, đạt <5 ms trên GPU này; package latency tổng **44,98 ms/ảnh**.

Model của run có hash **`248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`**, nằm tại:

```text
MyDrive/Fatformer/checkpoint/phase7/corrected_20261109_102954_3d28ad/unified_fatformer_v2.pth
```

ZIP kết quả không kèm weights, nên không có `.pth` tương ứng ở `current/`. Không dùng package cũ `16606e16…` thay cho model này. JSON/log và dự đoán nhất quán cùng hash; chưa băm lại bytes của model mới tại local.

Giữ nguyên notebook có output và nguồn `wait/phase7_corrected_rerun/`. Bản local FP32 trước đó được chuyển vào `../archive/temperature_converged_20261109_local/`. Không cần chạy lại cùng notebook chỉ để xử lý lỗi kỹ thuật từ run này; phần còn thiếu cho G7.3 là reference phù hợp, còn cải thiện hai tiêu chí G7.2 cần thí nghiệm model/dữ liệu mới có kiểm soát.
