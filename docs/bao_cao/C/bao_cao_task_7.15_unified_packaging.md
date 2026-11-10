# Task 7.15 — đóng gói và kiểm chứng package đo trên GPU

Run hiện hành **corrected_20261109_102954_3d28ad**, [hồ sơ duy nhất](../../../DATASET/logs/phase7/task_7.15/README.md). Bản sửa local FP32 trước đó được giữ trong `archive/temperature_converged_20261109_local/`; chỉ dùng `current/` cho số liệu hiện hành.

## Phần đã thực hiện và hậu kiểm

- Notebook hoàn tất 11 cell, không có output lỗi, source khớp notebook mẫu trong ZIP. Dùng đúng ZIP đã chuẩn bị, kiểm tra đủ hash 140 tệp. Notebook và nguồn kết quả được giữ nguyên.
- Đo lại đúng **400 ảnh calibration, 200 real + 200 fake**, không trùng ID với 1.600 evaluation; FP16/CUDA/batch 16 thống nhất với benchmark. Bộ fit nhiệt độ đã hội tụ và được xác nhận bằng tối ưu độc lập.
- [Calibration hiện hành](../../../DATASET/logs/phase7/task_7.15/current/calibration/anchored_v2.json): **T=2,7832491397857666**, ngưỡng 5% **0,7918508648872375**, FPR calibration **5%**, Real/Fake ACC **95%/58,5%**. ECE **14,38% → 5,39%**, NLL **0,6444 → 0,4227**. Không dùng metric trên tập fit thay cho chất lượng evaluation.
- [JSON kiểm chứng GPU](../../../DATASET/logs/phase7/task_7.15/current/raw/real_image_verification.json) báo đủ **1.200 tensor bằng source**, standalone khi CLIP path không tồn tại, **20 ảnh dữ liệu thực (10 real + 10 fake)**, sai số xác suất và nhãn reference cùng calibration bằng 0. Hậu kiểm local đọc/verify đủ 20 PNG, khớp ID calibration và 20 file có hash nội dung khác nhau.
- [Benchmark](../../../DATASET/logs/phase7/task_7.15/current/raw/benchmark/final_report.json) lưu dự đoán từng ảnh. [Hậu kiểm](../../../DATASET/logs/phase7/task_7.15/current/analysis/verification.json) tính lại metric và CI95, xác nhận báo cáo cùng hash package. Router tắt, dùng fallback clean cố định; không công bố đã nghiệm thu degradation router.

## Package và phạm vi bằng chứng

Package SHA256 **`248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`**, source checkpoint ghi nhận `5cd49252…`. File model nằm tại:

```text
MyDrive/Fatformer/checkpoint/phase7/corrected_20261109_102954_3d28ad/unified_fatformer_v2.pth
```

[Package reference](../../../DATASET/logs/phase7/task_7.15/current/artifacts/package_reference.json) ghi rõ vị trí, hash và giới hạn xác minh. ZIP kết quả không kèm weights, nên local chưa băm lại tệp `.pth` hoặc chạy forward độc lập bằng package mới. Không dùng file local cũ `16606e16…` thay thế. Kiểm chứng tensor/forward dựa trên JSON/log GPU được ràng buộc cùng hash và source code đã đối chiếu; metric được tái tính từ dự đoán đã nộp.

## Latency cùng package

NVIDIA L4 **22,03 GiB**, FP16, TF32 tắt; **20 warmup, 200 cặp đo/batch**, luân phiên thứ tự backbone/package. Đã tái tính median, p95, throughput và delta từ mảng thô:

- Batch 1: package **44,98 ms/ảnh**, overhead median **0,188055 ms/ảnh**, p95 overhead **2,481857 ms/ảnh**, khoảng **22,2 FPS**.
- Batch 8: package **9,60 ms/ảnh**, khoảng **104,2 FPS**.
- Batch 16: package **9,61 ms/ảnh**, khoảng **104,1 FPS**.
- Batch 32: package **10,38 ms/ảnh**, khoảng **96,4 FPS**.

**Overhead <5 ms tại batch 1 đạt trên L4.** Median chênh lệch từng cặp không nhất thiết bằng hiệu hai median. Delta âm nhỏ tại batch 16 được giữ nguyên là dao động đo, không chứng minh wrapper tăng tốc backbone. Tensor tổng hợp không gồm I/O/preprocessing; không suy diễn thời gian đầu-cuối hoặc số đo sang GPU khác. [Latency gốc](../../../DATASET/logs/phase7/task_7.15/current/raw/latency_phase7.json), [bảng tái tính](../../../DATASET/logs/phase7/task_7.15/current/analysis/latency_summary.csv).

## Phán quyết và bước còn thiếu

**G7.2 đạt 6/8, tổng thể FAIL. G7.3 pending**, vì chưa có đối chiếu reference tốt nhất theo từng điều kiện với delta AUC/Fake@Real90 ≤1,5 pp trên cùng partition/protocol. Tương đương source/package và 20 ảnh không thay thế so sánh với best model. Trạng thái sau cùng nằm trong [acceptance.json](../../../DATASET/logs/phase7/task_7.15/current/raw/acceptance.json); metadata và benchmark giữ trạng thái tại thời điểm được tạo trước latency.

Không cần chạy lại y nguyên toàn bộ notebook vì lỗi kỹ thuật từ run này. Muốn hoàn tất G7.3 cần thêm reference phù hợp; muốn tăng hai tiêu chí G7.2 còn thiếu cần thí nghiệm model/dữ liệu có kiểm soát trên train/val_select, không chỉnh ngưỡng theo evaluation.

Chất lượng hiện được xác minh ở FP16/CUDA/batch 16. CLI ảnh thông thường trong code nộp mặc định FP32; nếu triển khai kênh đó cần kiểm chứng protocol tương ứng. Có 1 mẫu Q50 và 5 mẫu in-dist đổi nhãn khi tính softmax trên CPU do sát ngưỡng, chênh xác suất ≤3×10⁻⁷. Giữ dự đoán GPU làm nguồn cho báo cáo CUDA, không thay bằng nhãn CPU.

Xem [phân tích công tâm đầy đủ](../../../DATASET/logs/phase7/task_7.15/current/analysis/README.md) và [báo cáo 7.14](bao_cao_task_7.14_locked_benchmark.md). Đây là post-hoc confirmation, không phải kiểm thử mù mới và không chứng minh FPR ≤5% cho mọi miền.

## Bổ sung 10/11/2026 — không đổi package sau vòng cải thiện 1

Matched evaluation mới tạo package P0 `e26a1627…` và incumbent refit `de36061f…`. P0 chỉ đạt 4/8 tiêu chí G7.2, không được chọn thay deployment. Incumbent refit có dự đoán trùng kết quả current nhưng vẫn là binary package khác. Vì vậy package báo cáo tiếp tục là **`24869052…`**, giữ nguyên T/ngưỡng và latency của run corrected nêu trên; không thay overhead 0,188055 bằng 0,151093 của P0 hoặc 0,205143 của refit. G7.3 vẫn thiếu verified best-by-condition reference. Xem [báo cáo vòng cải thiện và trạng thái chẩn đoán mới](bao_cao_phase7_improvement_round1_va_chan_doan_vong2.md).
