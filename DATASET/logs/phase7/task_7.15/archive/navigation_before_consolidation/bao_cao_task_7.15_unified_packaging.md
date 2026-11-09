# BÁO CÁO TASK 7.15 — ĐÓNG GÓI VÀ HẬU KIỂM

> Cập nhật ngày 09/11/2026. Trạng thái: [~] đã sửa phân vùng calibration và đóng gói lại; chưa nghiệm thu toàn bộ G7.3.
> Notebook 7.14 và 7.15, cùng dữ liệu chạy gốc, được giữ nguyên. Đây là sửa sau hậu kiểm, không phải một lần kiểm thử mù mới.
> Điều chỉnh phần cứng ngày 09/11/2026 theo yêu cầu người dùng: chấp nhận GPU CUDA tương thích có đủ dung lượng. Không bắt buộc T4; mọi số đo latency chỉ áp dụng cho GPU thực tế được ghi trong báo cáo.

## 1. Kết luận

Bản chạy commit `68cd0a8` tạo được package và có log standalone/GPU. Tuy nhiên, calibration 2.000 ảnh chứa đủ 400 ảnh calibration và toàn bộ 1.600 ảnh evaluation của CFE. Vì vậy kết luận “val_calib độc lập” và “G7.3 PASS 100%” của báo cáo trước được thu hồi.

Đã khôi phục đúng 400 dòng từ logits lưu sẵn, kiểm tra khớp nhãn và không trùng ID với evaluation, fit lại calibration, rồi đóng gói lại bằng đúng tensor của package cũ. Không huấn luyện lại backbone. Hash checkpoint huấn luyện được kế thừa từ metadata/NPZ đã đo; tệp huấn luyện gốc trên Drive chưa được băm lại trong lần sửa local này.

Kết quả cũ được lưu tại [task_7.15_original_68cd0a8](../../../DATASET/logs/phase7/task_7.15_original_68cd0a8/). [archive_manifest.json](../../../DATASET/logs/phase7/task_7.15_original_68cd0a8/archive_manifest.json) ghi SHA-256 từng tệp và hai notebook.

## 2. Nguyên nhân và biện pháp sửa

Split dùng ID `subset/0/file.png`, nhưng ảnh nằm trong `subset/0_real/file.png`. Bộ trích xuất cũ tìm bằng glob không khớp, rồi tự quét toàn thư mục. Số mẫu tăng từ 400 lên 2.000 mà không báo lỗi.

Bộ đọc mới chuẩn hóa lớp 0/0_real và 1/1_fake, giữ tên subset, từ chối ID trùng/nhãn sai, yêu cầu đủ từng ảnh trong split. Thiếu split hoặc ảnh phải báo lỗi. Trích xuất measured yêu cầu cả calibration/evaluation partition, ảnh lỗi đọc làm dừng thay vì âm thầm bỏ mẫu. Logits dummy được gắn synthetic và không được fit thành calibration measured.

Công cụ [recover_calibration_split.py](../../../tools/recover_calibration_split.py) chọn đúng ID calibration từ NPZ cũ. [partition_audit.json](../../../DATASET/logs/phase7/task_7.15_recovery/partition_audit.json) ghi nhận:
- Pool gốc: 2.000 ảnh; trùng evaluation: 1.600.
- Calibration khôi phục: 400 ảnh, gồm 200 real và 200 fake.
- Evaluation tham chiếu: 1.600 ảnh; overlap ID sau sửa: 0.
- Đơn vị kiểm tra: subset/class/filename. Chưa kiểm tra trùng nội dung ảnh dưới tên khác.
- Evaluation đã được xem trước; lần sửa này phải ghi là post-hoc recovery.

## 3. Calibration hiện hành

Nguồn: [anchored_v2.json](../../../DATASET/logs/phase7/calibration/anchored_v2.json) và [logits khôi phục](../../../DATASET/logs/phase7/task_7.15_recovery/logits_anchored_v2_calibration_400.npz).

- Checkpoint huấn luyện tham chiếu: `5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c`.
- Temperature: **1.4670088291168213**.
- Ngưỡng FPR 5%: **0.9273644089698792**.
- FPR thực đo trên calibration: 1%, 5%, 10% ở các ngưỡng tương ứng.
- Real ACC: 95,00%; Fake ACC: 58,50%, chỉ trên 400 ảnh dùng fit.
- ECE: 14,29% → 11,56%; NLL: 0,6436 → 0,5003 (không có đơn vị phần trăm).
- T và ngưỡng giữ đủ độ chính xác khi serialize; xử lý điểm bằng nhau tại biên để FPR calibration không vượt mục tiêu.

Các giá trị này không bảo đảm FPR trên miền khác. Không dùng 1.600 ảnh evaluation để chỉnh T/ngưỡng cho đạt gate.

Baseline trong `baseline.json` có T=1,6402, ngưỡng FPR 1%=0,999, 5%=0,9127 và 10%=0,0258; ECE 34,00% → 33,54%, NLL 3,4835 → 2,1392, Fake ACC calibration 34,00%. Những con số baseline khác trong báo cáo trước không có căn cứ từ artifact này. Không suy luận v2 tốt hơn có ý nghĩa thống kê chỉ từ bảng calibration.

## 4. Package hiện hành và kiểm chứng

[unified_fatformer_v2.pth](../../../DATASET/checkpoints/phase7/unified_fatformer_v2.pth) chứa tensor và metadata calibration đã sửa. Metadata ghi `repackaged_from`, hash package nguồn, commit nền và trạng thái worktree có sửa đổi. Không gán hash tệp package thành hash checkpoint huấn luyện.

Tool đối chiếu toàn bộ 1.200 tensor bằng `torch.equal` khi nạp lại. 1.200 là số entry của state_dict, bao gồm các alias của visual/text encoder; không phải 1.200 tensor tham số độc lập. Model không có `head.weight/head.bias` như báo cáo cũ mô tả.

Router vẫn tắt vì chưa có bằng chứng thắng held-out hợp lệ. Fallback dùng calibration đã sửa. [package_verification.json](../../../DATASET/logs/phase7/task_7.15_recovery/package_verification.json) ghi tách biệt kiểm tra tensor, smoke test và nghiệm thu 20 ảnh thật. Không suy diễn smoke test thành kiểm chuẩn CFE hay chứng nhận triển khai tuyệt đối.

SHA-256 được sinh từ các tệp hiện có trong [artifact_manifest.json](../../../DATASET/logs/phase7/task_7.15_recovery/artifact_manifest.json). Các hash JSON chép sai trong báo cáo trước được thay bằng manifest có thể kiểm tra lại.

## 5. Độ trễ và G7.3

Log gốc đo trên **NVIDIA A100-SXM4-40GB**, FP16 autocast, warmup **15**, đo **40** bước. Batch 1: backbone 47,56 ms, package 47,83 ms; overhead ghi nhận 0,2743 ms, throughput 20,9 ảnh/s. Batch 32: 80,43 ms/batch và 397,8 ảnh/s.

Đây là phép đo của package cũ. Không chuyển số đo đó sang package vừa sửa hay suy ra chắc chắn đạt trên T4. Overhead batch 8 bằng 0 trong log cũ là kết quả bị kẹp âm về 0; không chứng minh chi phí wrapper triệt tiêu.

Benchmark mới đo các cặp backbone/package xen kẽ, đồng bộ CUDA trước/sau, lưu mẫu thô và delta có dấu. Mục tiêu <5 ms overhead được chấm trên GPU CUDA thực tế theo yêu cầu cập nhật; T4 chỉ là tham chiếu lịch sử. Batch quá lớn được ghi out-of-memory, không xóa kết quả batch nhỏ và không tạo số đo giả.

[latency_phase7.json](../../../DATASET/logs/phase7/latency_phase7.json) hiện ghi **pending** cho package mới, có đường dẫn tới số đo A100 lịch sử. Chưa có phép đo GPU mới trong phiên local.

G7.3 còn yêu cầu trong 1,5 pp so với kết quả tốt nhất theo từng điều kiện clean và Q30. Đây không phải chênh lệch clean trừ Q30. Chưa có phép so sánh đủ căn cứ để chấm toàn bộ gate PASS.

## 6. Quan hệ với Task 7.14

Giữ nguyên kết quả 7.14 tại operating point baseline tham chiếu: Clean AUC 88,67%, Fake@Real90 64,12%, Macro ACC 73,75%; Q30 AUC 79,87%, Fake@Real90 42,62%, Macro ACC 65,31%. G7.2 đạt 5/8 tiêu chí theo báo cáo cũ, chưa đạt toàn bộ.

Hai số 75,81% và 70,81% trong báo cáo trước không được hỗ trợ bởi bộ locked benchmark này. In-dist Macro ACC 90,46% là một tiêu chí đạt, không tự chứng minh toàn bộ G7.2 PASS hoặc không có quên so với baseline.

Không ghi đè kết quả frozen 7.14 bằng số chấm lại sau thay calibration.

## 7. Các bước còn lại

1. Chạy CLI và đối chiếu reference/package trên ít nhất 20 ảnh thật, cùng precision/preprocessing; ghi rõ tiêu chí kiểm chứng.
2. Chấm package mới qua predict trên đúng 1.600 ảnh evaluation, clean/Q30; ghi là xác nhận sau thay calibration.
3. So với cấu hình tốt nhất có bằng chứng cùng partition theo điều kiện; lưu metric và delta, không tự đặt PASS.
4. Đo overhead trên GPU CUDA được cấp có đủ VRAM; lưu median/p95/mẫu thô, hash package, GPU, VRAM và protocol.
5. Cập nhật tiến độ từ kết quả thực tế. Có thể bàn giao sản phẩm best-effort với hạn chế rõ ràng; đóng gói thành công không làm gate chất lượng tự PASS.

Lệnh chạy và quy trình tái lập được ghi trong [task_7.15_recovery/README.md](../../../DATASET/logs/phase7/task_7.15_recovery/README.md).
