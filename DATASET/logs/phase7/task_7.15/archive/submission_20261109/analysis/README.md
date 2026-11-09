# Hậu kiểm Task 7.15 — run 09/11/2026, package 8ddcb401

## Kết luận

**Đã sửa được lỗi cell 4–5 và tạo được package có calibration đúng nguồn, nhưng chưa đủ điều kiện nghiệm thu hoàn toàn Task 7.15/G7.3.** Điểm mới cần xử lý là bộ tối ưu nhiệt độ chưa hội tụ: số liệu khớp code hiện tại, nhưng không phải nghiệm cực tiểu NLL như mục tiêu hiệu chuẩn. Phần đo overhead trên L4 đạt yêu cầu; không quy lỗi cho việc dùng GPU khác T4.

Đã nhập nguyên vẹn **6 tệp gốc** vào [raw](../raw/), gồm package 1.972.728.623 byte, NPZ, calibration JSON, metadata, latency và reliability PNG. Notebook trong `notebooks/task/task_7.15/` và nguồn trong `wait/task7-15/` được giữ nguyên. Không ghi đè package/calibration đang có tại các đường dẫn dùng chung hoặc các run cũ. Manifest SHA256 nằm ở [import_manifest.json](../import_manifest.json).

## Phần thực hiện đúng và có thể xác minh

- Notebook lưu 8 code cell đã thực thi, execution count `1,2,3,8,9,10,11,12`, không còn output lỗi. Cell 4–5 khớp nguyên nội dung hai cell hỗ trợ đã cấp. Đây là bằng chứng đã chạy tiếp thành công sau sửa lỗi, không phải yêu cầu execution count phải liên tiếp.
- NPZ chứa **400 logits float32, 200 real + 200 fake**, hữu hạn, ID/nhãn/đường dẫn thống nhất. Khớp đúng frozen calibration; không trùng ID với **1.600 ảnh evaluation**. Cả hai file split và hash được lưu riêng trong [evidence](../evidence/). Kiểm tra ID không thay thế kiểm tra trùng nội dung ảnh dưới tên khác.
- Checkpoint nguồn ghi nhận SHA256 `5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c`. NPZ, calibration và metadata đều thống nhất nguồn này. File checkpoint huấn luyện trên Drive không được cung cấp local để băm lại; không diễn giải chuỗi hash tự khai thành phép kiểm tra byte độc lập.
- Calibration JSON trỏ đúng NPZ thực nộp, SHA256 `3b0f66dfdc7c4655df6aa7fcab7346f0315d2663927559b66005bafa2c657942`. Không dùng logits baseline để fit; file baseline split chỉ cung cấp ID/nhãn phân vùng đã chốt.
- Package SHA256 **`8ddcb401f1e160776c653a3d073d2512cbafa1e08e5d39510201be83ab097dd8`** khớp notebook và log latency. Có đủ **1.200 tensor**, layout `n=3, indices=[7,15,23], partial=false`. Metadata trong `.pth` bằng JSON ngoài.
- Đối chiếu trực tiếp toàn bộ key, dtype và giá trị tensor với package gốc lưu ở `task_7.15_original_68cd0a8`: không có tensor thay đổi. Đây là so với bản package lưu trước, không thay cho đối chiếu trực tiếp file huấn luyện trên Drive. Bằng chứng ở [tensor_comparison.json](tensor_comparison.json).
- Kiểm tra local nạp package khi CLIP path không tồn tại và chạy 1 tensor 0 trên CPU: xác suất wrapper khớp backbone chia nhiệt độ, sai số tối đa **0**. Đây là smoke tổng hợp, không tính thành kiểm thử 20 ảnh thật. Xem [standalone_cpu_smoke.json](standalone_cpu_smoke.json).
- Router tắt đúng chủ đích vì chưa có bằng chứng thắng trên held-out; package dùng nhánh cố định đã hiệu chuẩn. Không có cơ sở công bố router đã được nghiệm thu.

## Calibration: số liệu đúng, tối ưu chưa hội tụ

Tính lại từ 400 logits cho kết quả khớp tệp nộp:

- `T = 1.4675993919372559`; `T* = 0.9293636083602905` tại FPR mục tiêu 5%.
- FPR thực trên 200 ảnh real calibration: **10/200 = 5%**; Real ACC **95%**, Fake ACC **59%**.
- ECE: **14,72% → 11,70%**. Đây là ECE 10 bins của confidence tại quyết định argmax, khác việc tính ACC tại ngưỡng vận hành T*.
- NLL: **0,6453 → 0,5013**. Các ngưỡng 1%/5%/10% và FPR tương ứng đều tái lập được.

Chạy lại đúng `ModelWithTemperature.fit` cho T giống hệt, nên **không có dấu hiệu số liệu này được ghi sai so với code**. Tuy nhiên, kiểm tra hội tụ độc lập trên chính 400 calibration logits cho:

- Tối ưu một biến trong miền `[0.05,10]`: `T ≈ 2.78635945`, NLL ổn định **0,42296390**.
- Giải nghiệm đạo hàm theo nghịch đảo nhiệt độ cho `T ≈ 2.78635949`, thống nhất với phép tối ưu trên.
- Gradient NLL theo T tại T đang đóng gói: khoảng **−0,18212**, chưa gần điểm dừng. NLL còn giảm được khoảng **0,07830** trên tập fit.

Nguyên nhân phù hợp với code: L-BFGS dùng `lr=0.01, max_iter=100`, rồi nhận kết quả mà không kiểm tra hội tụ. Không nên gọi T hiện tại là nhiệt độ tối ưu. Đây là vấn đề thuật toán/tối ưu; không phải lỗi truyền `evaluation_split` vừa sửa.

**Giữ nguyên calibration/package nộp.** T chẩn đoán chỉ ghi trong [optimizer_convergence_diagnostic.json](optimizer_convergence_diagnostic.json), không phát hành thành calibration thay thế. Không dùng evaluation để chọn T. Sửa T thì phải fit lại ngưỡng trên calibration, đóng gói mới và kiểm tra lại; không đổi riêng T bằng tay. NLL tốt hơn trên tập fit cũng không bảo đảm kết quả held-out tốt hơn. Temperature scaling dương của mô hình hai lớp giữ thứ tự điểm về mặt toán học, nên không tự khắc phục AUC hay Fake@Real90 còn thiếu.

## Latency: đạt overhead trên L4, giới hạn diễn giải rõ ràng

Đã tính lại median, p95, throughput và mọi delta từ **200 cặp mẫu thô/batch**, sau **20 warmup**, khớp JSON. GPU **NVIDIA L4, 22,03 GiB**, FP16 autocast, TF32 tắt. Xem [latency_summary.csv](latency_summary.csv).

- Batch 1: package **44,98 ms/ảnh**, overhead median **0,118957 ms/ảnh**, p95 overhead **1,071266 ms/ảnh**; khoảng **22,2 FPS**.
- Batch 8: package **9,30 ms/ảnh**, overhead median **−0,031251 ms/ảnh**, khoảng **107,5 FPS**.
- Batch 32: package **10,51 ms/ảnh**, overhead median **0,002716 ms/ảnh**, khoảng **95,2 FPS**.

**Tiêu chí overhead batch 1 <5 ms/ảnh đạt trên L4.** Không so thời gian toàn package 44,98 ms với ngưỡng overhead 5 ms. Delta âm nhỏ ở batch 8 được giữ nguyên như dao động đo, không chứng minh wrapper làm backbone nhanh hơn. Batch lớn nhất cũng không mặc định có throughput tốt nhất.

Input đo là tensor tổng hợp; không gồm I/O và preprocessing. Không suy diễn thành thời gian đầu-cuối của ứng dụng hoặc số đo T4. Calibration/extraction dùng FP32 và softmax CPU, còn phép đo tốc độ dùng FP16 trên CUDA: **chưa có kiểm chứng chất lượng package ở đúng precision/backend của phép đo tốc độ**. Sự khác biệt không tự chứng minh model sai, nhưng cần đo trước khi gắn accuracy FP32 vào throughput FP16.

## Standalone và G7.3: còn thiếu bằng chứng nào

Cell 8 báo đã chạy standalone trên **4 ảnh**, chưa đủ yêu cầu **20 ảnh thật** và không đối chiếu xác suất/nhãn với reference trên cùng ảnh. Tệp `demo_inference_output.json` không nằm trong 6 artifact xuất ra; notebook chỉ giữ 4 dòng kết quả đã làm tròn. Hai ảnh có tên bắt đầu `fake_` được dự đoán REAL, nhưng không được lấy 4 ảnh tiện chọn làm ước lượng chất lượng toàn bộ mô hình. Thời gian ảnh đầu khoảng 2.168,7 ms có khởi động lạnh, không so trực tiếp với median sau warmup.

G7.3 hiện có:

- Một package đủ trọng số và wrapper một lần gọi backbone: cấu trúc đã kiểm tra, CPU smoke đạt.
- Overhead trên GPU thực tế: **đạt trên L4/FP16**.
- Kiểm tra tương đương xác suất/nhãn trên ít nhất 20 ảnh thật, cùng calibration và precision: **thiếu**.
- Chấm `package.predict()` trên frozen evaluation, so AUC/Fake@Real90 với reference tốt nhất theo từng điều kiện trong giới hạn 1,5 pp: **thiếu**.
- Hội tụ nhiệt độ hiệu chuẩn: **cần sửa trước khi chốt artifact triển khai**.

Vì vậy trạng thái tổng thể là **pending**, không gán FAIL cho phần latency đã đạt và không gán PASS toàn bộ G7.3. `metadata.validation_status=pending_real_prediction_equivalence_and_G7.3` đang phản ánh đúng phần thiếu. `build.dirty=true` không tự chứng minh code sai, nhưng thiếu runtime source diff nên chưa tái dựng được hoàn toàn môi trường nguồn. Dòng mô tả notebook cũ còn nhắc bắt buộc T4 là nội dung lỗi thời; theo yêu cầu người dùng, GPU CUDA đủ VRAM được chấp nhận.

## Liên hệ Task 7.14 và chẩn đoán hậu kiểm

Run 7.14 `locked_benchmark_phase7_20261109_074134` đã nhập trước đó dùng cùng checkpoint v2 nhưng operating point baseline `T=1.6402, threshold=0.9127`. **Không ghép báo cáo đó thành benchmark trực tiếp của package mới**. Kết luận G7.2 trước đó vẫn 5/8, chưa đạt; run 7.15 không tự thay đổi phán quyết đó.

Đã tính lại điểm từ logits 7.14 với T/ngưỡng của package mới và lưu [historical_logits_rescore.csv](historical_logits_rescore.csv). Đây là **post-hoc rescore trên cache backbone**, không phải chạy mới package, không dùng để chọn T hoặc ngưỡng:

- Clean: fixed FPR **4,25%**, fixed Fake ACC **52,75%**.
- Q30: fixed FPR **8,25%**, fixed Fake ACC **39,375%**.
- Các điều kiện suy biến còn lại cũng có FPR >5% trong cache này. Vì vậy FPR 5% trên calibration không phải bảo đảm FPR ≤5% trên mọi điều kiện độc lập.

Các file cache, nguồn partition và hash được ghi trong [verification.json](verification.json); không chạy lại forward GPU ở local để tạo những điểm này.

## Trình tự giải quyết hợp lý

1. Sửa/kiểm tra hội tụ của bộ fit T trên **400 ảnh calibration**. NPZ mới đã đúng nguồn và phân vùng, nên **không cần trích xuất lại cell 4 chỉ để sửa tối ưu** nếu giữ FP32/protocol hiện tại. Fit lại T và toàn bộ ngưỡng, không chỉnh hash hoặc số liệu cũ.
2. Chạy lại phần hiệu chuẩn → đóng gói → kiểm tra ≥20 ảnh thật → latency với hash package mới. Tương đương reference/package cần cùng T/ngưỡng, preprocessing và precision. Nếu chọn FP16 để báo cáo hiệu năng, chuẩn hóa protocol đó trước calibration/benchmark; lúc đó cần đo lại logits calibration theo protocol đã chọn.
3. Chấm lại Task 7.14 trên package đã chốt, đúng 1.600 evaluation và các bộ in-dist theo protocol thống nhất. So reference theo từng điều kiện; công bố FAIL/pending nếu thực tế chưa đạt. Đây là xác nhận sau sửa calibration, không gọi evaluation đã xem là phép thử mù mới.
4. Chỉ sau đó dùng đúng hash package, báo cáo và gate có bằng chứng để cập nhật Task 7.16/báo cáo cuối.

Phiên hậu kiểm này **chỉ nhập và phân tích kết quả**, không chỉnh notebook, code huấn luyện/calibrator, calibration hay weights gốc. Không có lần đo GPU mới trong phiên local.
