# Chương 5 — Kết luận, đóng góp, hạn chế và hướng phát triển

Nguồn thực nghiệm: run `corrected_20261109_102954_3d28ad`, package `248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`. [View số liệu cho tài liệu](../../../DATASET/logs/phase7/task_7.16/final_phase7_report.json) được tổng hợp từ báo cáo GPU đã hậu kiểm; không phải một lần đo mới. Toàn bộ chương sử dụng metric có điều kiện vận hành, không thay AUC bằng accuracy hoặc pooled bằng macro. Evaluation đã được xem trước; kết quả là **post-hoc confirmation**, không phải kiểm thử mù mới.

## 5.1. Kết luận và trả lời câu hỏi nghiên cứu

### Câu hỏi 1: Độ tin cậy của kết luận có phụ thuộc vào cách nạp checkpoint và đo lường không?

Có. Audit cấu trúc cho thấy Gen 1 chứa hợp 10 khối adapter, không biểu diễn đầy đủ được bằng số khối hỗ trợ của constructor. Nạp partial n3 hoặc n8 tạo ra hai phép đo khác nhau, không phải cùng một model hợp lệ. Diagnostic n3/n8 chỉ để tham khảo lịch sử, `invalid_layout`; verdict hiện vẫn **inconclusive** vì không có CI đối chiếu phù hợp. Canonical loader xử lý đúng ba checkpoint hợp lệ và từ chối Gen 1 có tên lỗi. Đó là bằng chứng xác định về cấu trúc, không cần gắn CI thống kê cho số lượng khối. Tuy nhiên G7.0 rộng vẫn pending do các đường legacy và phạm vi kiểm tra chưa đóng toàn repo.

WiSE-FT Gen 2 sửa điều kiện tương thích bằng cách trộn Baseline và Anchored cùng n3. Sweep 12 ứng viên chọn α=0,55, β_srm=0,35, actual hash `eb170fd62700b3613124b8c1519bc796c659ba1e4733be9abb1fb9f5801e429c`. Gen 2 không tạo quick win: Clean Fake@Real90 48,88% và Q30 AUC 43,67% không đạt mốc 55%. Số Gen 1 cũ không được dùng làm bằng chứng ưu thế hoặc định lý rằng phải có hai model.

### Câu hỏi 2: Mô hình sau huấn luyện và hiệu chuẩn có đáp ứng mục tiêu độ bền không?

Mới đáp ứng một phần. Package Anchored v2 được chấm trực tiếp trên 1.600 CFE evaluation tách khỏi 400 calibration, với FP16/CUDA/batch 16 và canonical CLIP preprocessing:

| Điều kiện | AUC % [CI 95%] | AP % [CI 95%] | Fake@Real90 % [CI 95%] | ECE % |
| :--- | :--- | :--- | :--- | ---: |
| clean | 88,66 [87,11; 90,16] | 89,08 [87,35; 90,85] | 64,12 [59,25; 69,77] | 4,39 |
| q30 | 79,88 [77,65; 82,07] | 78,47 [75,33; 81,49] | 42,62 [37,87; 47,91] | 5,08 |

Nguồn: [final_report.json](../../../DATASET/logs/phase7/task_7.15/current/raw/benchmark/final_report.json) `/cfe`; CI95 bootstrap 1.000 lần/seed 42 đã được tính lại từ dự đoán. Clean Fake@Real90 64,12% có CI [59,25; 69,77] bao gồm 65%. Vì vậy chỉ kết luận điểm ước lượng không đạt cổng, không gọi đó là bằng chứng thống kê rằng năng lực tổng thể thấp hơn 65%.

Trên18 tập in-dist, macro là trung bình balanced accuracy của từng tập: **90,10%**, pooled Real ACC **90,41%**; pooled accuracy 93,46% là đại lượng khác. Hai mốc 90% đạt theo điểm ước lượng, nhưng không chứng minh mọi tập đều được bảo vệ hoặc không có quên mô hình. G7.2 chỉ đạt **6/8**; thiếu Clean Fake@Real90 **0,88 pp** và worst-subset Fake ACC **14,14 pp**.

Nhiệt độ đã hội tụ T=2,78324914, ngưỡng=0,79185086. NLL calibration sau fit 0,4227, ECE 5,39%, nhưng chỉ là kết quả trên tập fit. FPR calibration 5% không bảo đảm ngoài miền; tại ngưỡng cố định Q30 có FPR 8,625%. Temperature scaling sửa confidence, không tự tăng thứ tự phân biệt/AUC về mặt toán học.

### Câu hỏi 3: Có thể vận hành bằng một artifact thay vì chọn model thủ công không?

Về cấu trúc và đường chạy đã đo, có thể đóng gói một checkpoint với calibration/metadata, một backbone forward và fixed fallback. Log GPU đối chiếu1.200 tensor và 20 ảnh dữ liệu thực (10 real+10 fake), sai số xác suất tối đa0. Overhead median batch 1 trên L4 là **0,188055 ms/ảnh**, p95 **2,481857 ms/ảnh**, dưới ngưỡng 5 ms; tổng package **44,98 ms/ảnh** không phải overhead. Nguồn [real_image_verification.json](../../../DATASET/logs/phase7/task_7.15/current/raw/real_image_verification.json) và [latency_phase7.json](../../../DATASET/logs/phase7/task_7.15/current/raw/latency_phase7.json).

Nhưng chưa thể trả lời rằng artifact này đạt chất lượng tốt nhất trong mọi điều kiện: G7.3 vẫn **pending** phép so sánh với reference tốt nhất cùng partition/protocol trong 1,5 pp. DegradationRouter tắt vì thí nghiệm held-out chưa có; frequency gating trong backbone không đồng nghĩa router đã hoạt động. Không suy từ pending thành bắt buộc hai chế độ hay router đã thua.

## 5.2. Đóng góp được bằng chứng hỗ trợ

1. **Đo lường có truy vết:** chuyển trọng tâm sang AUC/AP/TPR@FPR, CI, source hashes và phân biệt ngưỡng vận hành với metric ROC. Lưu dự đoán từng ảnh cho 7 điều kiện và 18 tập, kiểm tra nhãn/ID/đếm, không tạo số liệu thay thế.
2. **Phát hiện và cô lập lỗi kiến trúc:** chứng minh layout Gen 1 không thể nạp đầy đủ; canonical core từ chối thay vì im lặng bỏ khối. Gen 2 cùng layout được đo riêng, chấp nhận G7.1 FAIL thay vì dùng số Gen 1 làm kết luận.
3. **Quy trình hiệu chuẩn và bàn giao có kiểm soát:** tách 400/1.600, sửa hội tụ nhiệt độ, giữ cùng precision/batch/backend cho calibration và benchmark, kiểm chứng package và latency cùng hash. Code loss/selection và ablation được cung cấp như hạ tầng nghiên cứu; không khẳng định mỗi module đều tạo cải thiện tổng quát.

## 5.3. Hạn chế và phạm vi diễn giải

**Mục tiêu chưa đạt.** G7.2 thiếu hai tiêu chí: Clean Fake@Real90 thiếu 0,88 pp, worst-subset Fake ACC thiếu 14,14 pp. Nhóm `5_other_autoregressive` chỉ đạt 30,86% Fake ACC. Không hạ ngưỡng theo evaluation hoặc đổi định nghĩa gate để che kết quả này. Negative result có giá trị mô tả nhưng không tự trở thành nghiệm thu đầy đủ hoặc chứng minh checkpoint Pareto tối ưu.

**Quy mô và tính độc lập.** Tổng pool CFE 2.000 được tách 400 calibration/1.600 evaluation, chỉ 5 subset. CI rộng và không suy được cho generator ngoài phạm vi đã đo. Kho train v2 14.400 ảnh với val_calib 1.080 là một nguồn khác, không được gọi nhầm là tập400 CFE. In-dist82.003 là số lượt/bản ghi; có candidate dùng chung real reference giữa các tập, chưa có đủ ảnh gốc để chứng minh nội dung độc lập. CI theo ảnh chưa mô tả hết phụ thuộc giữa dataset/generator.

**Điểm gộp che chênh lệch miền.** Macro in-dist chỉ vượt 90% khoảng 0,10 pp. Mean-subset Real ACC khoảng 85,79%, trong khi pooled Real ACC 90,41%;9/18 tập Real ACC dưới 90%. CI chẩn đoán resample theo18 subset khoảng[86,76%;93,32%], không phải quy tắc gate mới. Không biến điểm vượt mốc thành bảo đảm ổn định giữa các miền.

**Giới hạn ablation.** Bốn cấu hình cùng seed 42; report không có CI độc lập cho selection worst-group và `primary_driver=null`. Không thể tách riêng đóng góp dữ liệu khi cả bốn đều dùng v2; không thể tuyên bố RealGuard/FeatureConsistency chắc chắn nâng chất lượng ngoài miền. Log training còn `grad_norm=NaN` ở epoch 1,4; điều này chưa được giải thích, nhưng cũng không tự chứng minh weights cuối chứa NaN vì các dự đoán hiện hành hữu hạn.

**Protocol và khả năng triển khai.** In-dist cũ stretch 224×224 khác preprocessing mới, nên so sánh trước/sau không quy hết cho calibration. Chất lượng hiện tại là FP16/CUDA/batch 16; CLI ảnh thông thường mặc định FP32. Tự tính lại CPU softmax có 1 mẫuQ50 và 5 mẫuin-dist đổi nhãn sát ngưỡng dù sai khác xác suất≤3×10⁻⁷. Không chuyển quality claim sang mọi backend/batch mà chưa xác minh.

**Giới hạn bàn giao và quản trị.** Model mới ở Drive; ZIP không kèm weights nên hậu kiểm local không băm lại file `.pth` 24869052… hoặc source checkpoint. G7.3 thiếu best-reference; G7.0 còn legacy. Raw benchmark dùng schema corrected/v1; view v3 trong hồ sơ này chỉ là bản tổng hợp, không thay schema gốc. Không có taxonomy generator chuẩn để tự tạo bảng per-generator. Chữ ký và cross-review A/B/C chưa có, không được giả lập.

## 5.4. Hướng phát triển và điều kiện bàn giao tiếp

1. **Thí nghiệm chất lượng có kiểm soát:** audit nhóm khó và nhãn nguồn, chọn thay đổi trên train/val_select; nhiều seed, paired CI và tập test mới chưa xem. Đối chiếu reference cùng partition/precision/preprocessing trước khi chốt G7.3. Không tái dùng evaluation để chọn T/ngưỡng.
2. **Độ tin cậy vận hành:** đồng bộ CLI với protocol đã đánh giá, kiểm tra batch 1/16 và CPU/CUDA quanh biên ngưỡng; đo I/O và preprocessing nếu cần latency đầu-cuối. Đánh giá FPR theo miền và calibration chuyên điều kiện mà không làm rò rỉ test.
3. **Tái lập và nghiệm thu:** đưa đúng weights cùng hash vào bàn giao, xử lý hoặc phân loại loader legacy, giải thích gradient telemetry và reconcile metadata; thu thập cross-review/chữ ký thật. Router chỉ được bật sau fit/evaluation held-out hợp lệ, không từ số synthetic hoặc unit test.

Hồ sơ này cho phép báo cáo một kết quả thực nghiệm có kiểm chứng và các mục chưa đạt, không tuyên bố toàn bộ Phase 7 đã PASS. Các nguồn chi tiết, số cũ→mới và kiểm tra tự động nằm ở [bao_cao_ra_soat_so_lieu_phase7.md](bao_cao_ra_soat_so_lieu_phase7.md).

## 5.5. Bổ sung 10/11/2026 — kiểm chứng một hướng cải thiện và kết quả âm

Pilot preprocessing vòng 1 train mới P0 legacy và P1 CLIP, cùng validation CLIP, dữ liệu/loss/batch/seed 42 và 5 epoch. P0 worst-group validation 42% so với 35,33% của P1, được chọn trước khi chấm CFE. Matched evaluation sau đó cho P0 Clean Fake@Real90 61%, Q30 38,875%, macro in-dist 89,6794%, thấp hơn điểm tương ứng 64,125%/42,625%/90,1018% của incumbent refit. P0 chỉ đạt 4/8 tiêu chí G7.2; incumbent giữ 6/8. Model báo cáo `24869052…` không thay đổi.

Kết quả này chưa hỗ trợ hướng đổi preprocessing train trong phạm vi pilot một seed, không chứng minh mọi CLIP preprocessing hoặc mọi thí nghiệm train mới đều thất bại. Paired CI của delta recall clean/Q30 còn chứa 0; không coi tất cả chênh lệch là có ý nghĩa thống kê. P0 có lợi thế ở một số chỉ số phụ như Q30 TPR tại FPR thấp và in-dist ECE/NLL; tiêu chí chọn model đã định vẫn chưa đạt. [Báo cáo đầy đủ](../C/bao_cao_phase7_improvement_round1_va_chan_doan_vong2.md) cung cấp hashes, CI và các giới hạn.

Kiểm kê tiếp cho thấy ba họ fake chỉ có một generator và tất cả generator validation đã có trong train. [Chẩn đoán GPU đã hậu kiểm](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/README.md) cung cấp 4.320 lượt dự đoán clean/Q30 trên 1.080 ảnh: AUC điểm gần nhau nhưng CI paired chưa xác lập ưu thế/tương đương. Có suy giảm Q30 rõ trên một số mẫu như CIPS, nhưng DiT recall đã thấp ở clean; recall lịch sử khớp không chứng minh bit-exact logits. Ngưỡng 0,9127 là điểm chẩn đoán, còn CFE đã dùng calibration riêng; không dùng validation để xác lập nguyên nhân CFE hoặc loại trừ hạn chế biểu diễn. Ưu tiên phép so sánh calibration độc lập trước, robustness augmentation là giả thuyết riêng; chưa train hoặc thay model báo cáo.

## 5.6. Đối chứng calibration transfer và giới hạn operating point

[Calibration transfer đã hậu kiểm](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/README.md) fit đúng clean `val_calib` độc lập, đạt29/590 FP ở mỗi model. Trên `val_select`, P0 Fake clean tăng62,04→84,69%, Q30 49,18→80%; incumbent63,06→84,49% và49,18→78,78%. Tuy nhiên Real Q30 giảm còn86,95%/87,29%, dưới guard90 khai báo trước. NLL/ECE điểm tốt hơn không bù được việc tăng báo oan; hai point mới không được chọn để thay package.

Đây là bằng chứng tradeoff operating point trên development split, không phải cải thiện ranking hoặc CFE mới. Raw logits dùng chung không đổi; calibration không thể khép khoảng thiếu CFE Fake@Real90 chỉ bằng đổi T/ngưỡng. Recipe đổi đồng thời domain, cỡ mẫu và prior calibration nên chưa tách nguyên nhân riêng. Muốn tăng ranking cần thí nghiệm weights/data/robustness có control và bằng chứng độc lập; muốn một operating point robust có thể thử clean/Q30 calibration một ngưỡng chung như một thí nghiệm khác, không fit theo validation/test. Giữ model báo cáo `24869052…`, G7.2 FAIL6/8, G7.3 pending; chưa train thêm.
