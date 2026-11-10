# Vòng 2 — chuẩn bị phép đối chứng calibration transfer

Cập nhật 10/11/2026. **Đã hậu kiểm run `calib_transfer_20261110_155948`; điểm calibration clean-only mới không đạt guard Real Q30≥90%.** Fake recall tăng, nhưng FPR Q30 lên 13,05% P0/12,71% incumbent. Không chọn điểm mới, chưa train hoặc thay model; primary package `24869052…`, G7.2 FAIL/đạt 6/8, G7.3 pending giữ nguyên. Xem [kết quả và quyết định](runs/calib_transfer_20261110_155948/README.md), [index hiện hành](index.json).

Code: [`tools/phase7_calibration_transfer.py`](../../../../../tools/phase7_calibration_transfer.py). Protocol và controls: [`configs/phase7_calibration_transfer/plan.json`](../../../../../configs/phase7_calibration_transfer/plan.json). Notebook/hướng dẫn: [`notebooks/task/phase7_calibration_transfer/README.md`](../../../../../notebooks/task/phase7_calibration_transfer/README.md).

Run đã hoàn tất từ source `253942a5262432fbed7ce7ca8b12b4e455bfc2ce`; không cần chạy lại y nguyên. [Notebook trong repo](../../../../../notebooks/task/phase7_calibration_transfer/phase7_calibration_transfer_colab.ipynb) vẫn là template không outputs; chưa có notebook thực thi để đối chiếu cell. Hậu kiểm dựa trên source, metadata và arrays đã nộp. Các mục thiết kế/preparation bên dưới là hồ sơ trước run, không thay kết quả hiện hành.

## Thiết kế đã khóa trước phép đo

P0 và incumbent nguồn giữ nguyên SHA. Với mỗi checkpoint, đối chứng là calibration CFE cũ đúng model, can thiệp là T/ngưỡng fit **chỉ trên `val_calib` v2 clean**, 1.080 ảnh/590 real/490 fake. Không lấy labels `val_select` để fit, không augmentation hoặc đổi weights. Một ngưỡng mới được dùng cho cả clean/Q30. Empirical FPR5 cho phép tối đa 29/590 false positives, comparator `>=` theo float32.

Chấm cả hai operating point trên cùng raw validation logits. Chỉ reuse cache khi inference source, GPU/Torch/CUDA/batch khớp; nếu khác thì đo lại validation hai model cùng protocol và batch. Calibration không tạo ranking gain; báo raw-margin AUC riêng và lưu thay đổi do float32 score ties. New calibration là research schema, `deployable=false`; mọi guard chỉ là mô tả development split, không đổi gate hoặc package.

So sánh này gộp thay đổi domain/cỡ mẫu/tỷ lệ lớp calibration; không suy nguyên nhân riêng từng yếu tố. CI chỉ conditional on fit/seed hiện tại, không bao quát fit uncertainty/generator dependence. Chưa có bằng chứng mới về CFE hoặc khả năng tổng quát. Kết quả được đọc trước khi thiết kế một can thiệp robustness Q30 riêng.

## Đã kiểm gì ở local?

- Frozen manifest, split calibration/validation 1.080 mỗi tập, không giao SHA; controls đúng checkpoint và source hashes.
- Toàn bộ 16 file reference staging từ folder **và** ZIP đã được thử với dữ liệu diagnostics thật đã nộp, giữ đúng hashes; không forward lại model thật ở local.
- **30 test CPU/code đạt**: 22 test transfer +8 test Git workflow. Có mock CUDA để thử luồng success/failure, cache reuse/remeasure và ZIP không chứa weights; không coi đó là đo GPU thật. Xem [`preparation/validation/local_tests.xml`](preparation/validation/local_tests.xml).
- [`preparation/readiness.json`](preparation/readiness.json) ghi `prepared_not_gpu_run`. [`preparation/preservation.json`](preparation/preservation.json) bảo vệ 365 file bằng chứng/notebook cũ, gồm notebook diagnostics người dùng vừa cập nhật. Không sửa hoặc xóa outputs của notebook cũ.

Kết quả nhận từ `wait/calibration_transfer_without_weights/`; nguồn Drive là `MyDrive/Fatformer/phase7/improvement_round2/calibration_transfer/calib_transfer_20261110_155948/`. Log run mới nằm tại `runs/calib_transfer_20261110_155948/`, có T/ngưỡng, scores và hậu kiểm riêng. `preparation/` giữ trạng thái trước khi chạy, không được sửa thành kết quả mới.
