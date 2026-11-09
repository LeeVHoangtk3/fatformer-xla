# Báo cáo Task 7.15 — đóng gói và hậu kiểm

Cập nhật ngày 09/11/2026. Hồ sơ log duy nhất: [DATASET/logs/phase7/task_7.15](../../../DATASET/logs/phase7/task_7.15/README.md). **Chỉ dùng `current/` cho bản hiện hành; `archive/` là lịch sử.** Notebook và dữ liệu nộp gốc được giữ nguyên.

## Kết quả hiện hành

Đã sửa bộ fit nhiệt độ dừng sớm bằng bộ giải NLL lồi theo nghịch đảo nhiệt độ, có kiểm tra điều kiện tối ưu. Fit lại T và các ngưỡng từ đúng **400 logits thực nghiệm FP32**, gồm 200 real và 200 fake, không trùng ID với 1.600 evaluation. Không huấn luyện lại backbone và không dùng evaluation để chọn T/ngưỡng.

Nguồn số liệu: [calibration](../../../DATASET/logs/phase7/task_7.15/current/calibration/anchored_v2.json), [NPZ 400 ảnh](../../../DATASET/logs/phase7/task_7.15/current/inputs/calibration_400.npz), [kiểm chứng](../../../DATASET/logs/phase7/task_7.15/current/verification/repair_validation.json).

- T: **2,7863595485687256**; ngưỡng FPR 5%: **0,7953211069107056**.
- FPR calibration **5%**, Real ACC **95%**, Fake ACC **59%**.
- NLL calibration sau sửa: **0,4230**, so với **0,5013** của run nộp mới trước sửa hội tụ.
- ECE calibration sau sửa: **6,06%**, so với **11,70%** trước sửa hội tụ.
- Bộ giải hội tụ sau 35 vòng; nghiệm được xác nhận bằng tối ưu số độc lập trên cùng calibration logits.

Đây là cải thiện hiệu chuẩn trên tập fit, không phải bằng chứng AUC/Fake@Real90 hay accuracy evaluation tăng. Độc lập phân vùng được kiểm tra theo ID, chưa thay thế kiểm tra trùng nội dung ảnh dưới tên khác.

## Package và kiểm chứng

[Package hiện hành](../../../DATASET/logs/phase7/task_7.15/current/artifacts/unified_fatformer_v2.pth) có SHA256 `16606e16c89901f39ab78c29951c1f83d9412f98f27ab55853c2e2dd13ee5f74`. Toàn bộ 1.200 entry tensor của state_dict khớp package nguồn; layout `[7,15,23]`, không partial. Router tắt đúng chính sách chưa có held-out win; dùng fallback cố định đã hiệu chuẩn.

[Kiểm chứng package](../../../DATASET/logs/phase7/task_7.15/current/verification/package_verification.json) xác nhận nạp standalone không cần CLIP path ngoài, smoke tổng hợp CPU khớp xác suất/nhãn reference cùng T/ngưỡng, sai số tối đa 0. Có **116 tests pass** cho bản sửa; lần xuất ZIP trước khi gom thư mục đã qua 38 tests từ chính code giải nén. Kiểm chứng export hiện hành được lưu riêng cạnh package để tránh nhầm các phiên bản ZIP.

Hash checkpoint huấn luyện `5cd49252…` được kế thừa đúng nguồn NPZ/package; không tuyên bố đã băm lại file huấn luyện trên Drive ở local. Metadata giữ thông tin repackage và hash container nguồn.

## Trạng thái nghiệm thu

**G7.3 vẫn pending.** Chưa có đối chiếu ≥20 ảnh thật, benchmark `package.predict()` trên evaluation/reference tốt nhất theo từng điều kiện, và phép đo latency mới cùng hash package hiện hành.

Run nộp trước sửa có overhead batch 1 khoảng **0,119 ms/ảnh trên L4**, đạt ngưỡng <5 ms, nhưng số này thuộc package `8ddcb401…`; không gán sang package hiện hành. [Trạng thái latency hiện tại](../../../DATASET/logs/phase7/task_7.15/current/verification/latency_phase7.json) ghi pending, không tạo số đo thay thế. Chấp nhận GPU CUDA tương thích đủ VRAM; không bắt buộc T4 và không suy diễn số đo giữa các GPU.

Run Task 7.14 trước đó dùng operating point baseline, không phải phép chấm trực tiếp của package mới. Giữ nguyên kết luận G7.2 5/8 của run đó; cần chấm lại package đúng protocol trước khi dùng cho báo cáo cuối.

## Chạy tiếp và tra cứu

Dùng [hướng dẫn hiện hành](../../../DATASET/logs/phase7/task_7.15/current/README.md) và [ZIP trong current](../../../DATASET/logs/phase7/task_7.15/current/colab/phase7_corrected_colab_bundle.zip). Notebook tổng hợp giữ nguyên, nạp ZIP thay cho clone GitHub; đồng bộ calibration/batch/precision với benchmark GPU, kiểm chứng ảnh thật rồi đánh giá 7.14 và latency cùng package. Đây là xác nhận sau hậu kiểm, không gọi evaluation đã xem là phép thử mù mới.

[Archive](../../../DATASET/logs/phase7/task_7.15/archive/README.md) giữ bản gốc 2.000 ảnh, recovery cũ và run nộp ngày 09/11. Không dùng các bản đó làm kết quả hiện hành. [PATH_MIGRATION.json](../../../DATASET/logs/phase7/task_7.15/PATH_MIGRATION.json) ghi vị trí mới và hash trước khi gom; các đường dẫn trong bằng chứng lịch sử giữ nguyên như thời điểm chạy.
