# Task 7.14 — kết quả đánh giá package hiện hành

Run **corrected_20261109_102954_3d28ad**, hoàn tất ngày 09/11/2026. [Hồ sơ và phân tích chi tiết](../../../DATASET/logs/phase7/task_7.15/current/analysis/README.md), [benchmark gốc](../../../DATASET/logs/phase7/task_7.15/current/raw/benchmark/final_report.json), [acceptance](../../../DATASET/logs/phase7/task_7.15/current/raw/acceptance.json).

## Protocol và tính nhất quán

Chấm trực tiếp `package.predict()` trên **1.600 evaluation CFE ×7 điều kiện** và **82.003 lượt ảnh/18 tập in-dist**, tách khỏi 400 calibration. GPU L4, FP16 backbone autocast, logits/softmax float32, batch 16; cùng calibration, preprocessing và package hash `248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`.

Đã tính lại metric pooled, 35 nhóm CFE và 18 tập in-dist từ dự đoán, kiểm tra nhãn/ID/thứ tự, hash artifact và CI95 bootstrap 1.000 lần, seed42. Kết quả khớp báo cáo nộp. Đây là post-hoc confirmation trên evaluation đã xem trước, không gọi là phép thử mù mới.

## Kết quả chính

- Clean AUC **88,66%**, CI95 **[87,11;90,16]**; Fake@Real90 **64,12%**, CI95 **[59,25;69,77]**; accuracy **74,375%**; ECE **4,39%**.
- Q30 AUC **79,88%**, CI95 **[77,65;82,07]**; Fake@Real90 **42,62%**, CI95 **[37,87;47,91]**; accuracy **65,8125%**; ECE **5,08%**.
- In-dist macro, trung bình 18 balanced accuracies: **90,101847%**. Pooled accuracy **93,464873%**, AUC **98,41%**. Pooled Real ACC **90,406282%**, Fake ACC **96,524136%**.
- Nhóm CFE yếu nhất `5_other_autoregressive`: Fake ACC **25/81=30,8642%**, Real ACC **86,49%**.

Số liệu đầy đủ và các điều kiện Q50/Q75/WebP/blur/down-up ở [condition_summary.csv](../../../DATASET/logs/phase7/task_7.15/current/analysis/condition_summary.csv), [CFE subsets](../../../DATASET/logs/phase7/task_7.15/current/analysis/per_subset_cfe.csv), [in-dist subsets](../../../DATASET/logs/phase7/task_7.15/current/analysis/per_subset_indist.csv).

## G7.2: 6/8, tổng thể FAIL

Đạt: Clean AUC; Q30 AUC; Q30 Fake@Real90; in-dist macro ACC; pooled Real ACC; ECE Q30.

Chưa đạt: Clean Fake@Real90 **64,12% <65%** và worst-subset Fake ACC **30,86% <45%**. Mức thiếu lần lượt khoảng **0,88 pp** và **14,14 pp**. Đây là phán quyết theo điểm ước lượng của cổng đã định. CI Clean Fake@Real90 chứa 65%, nên không diễn giải điểm FAIL thành chứng minh có ý nghĩa thống kê rằng năng lực tổng thể thấp hơn 65%.

ECE Q30 đã chuyển từ không đạt ở run operating-point baseline cũ sang đạt ở run này. AUC/Fake@Real90 gần như giữ nguyên, phù hợp với việc sửa hiệu chuẩn mà không train lại. In-dist cũ dùng stretch224×224 còn run này dùng canonical CLIP preprocessing; không gán toàn bộ thay đổi in-dist cho calibration.

## Giới hạn cần công bố

FPR 5% chỉ được fit trên calibration: FPR evaluation tại ngưỡng cố định là **4,375% Clean**, **8,625% Q30**, còn các suy biến khác tới **12,125%**; pooled in-dist **9,593718%**. Fake@Real90 khác Fake ACC tại ngưỡng vận hành, không trộn hai chỉ số.

Mean-subset Real ACC chỉ khoảng **85,79%**; 9/18 tập có Real ACC dưới90%, dù cổng pooled vẫn đạt. Macro chỉ vượt90% khoảng0,10pp; bootstrap chẩn đoán theo 18 subset cho CI khoảng **[86,76%;93,32%]**. Không tự thay quy tắc gate bằng CI mới, cũng không gọi mức vượt này là bảo đảm cho mọi miền.

Một số real reference có tên tương đối và logits trùng giữa các tập; chưa đủ ảnh gốc để khẳng định trùng nội dung toàn bộ. 82.003 là số lượt/bản ghi, không chứng minh 82.003 nội dung ảnh độc lập; CI theo ảnh chưa mô tả hết phụ thuộc giữa các tập.

G7.3 còn pending phép so sánh với reference tốt nhất theo từng điều kiện. Kiểm chứng20ảnh và overhead L4 đã có trong hồ sơ7.15. Model `.pth` nằm trên Drive, ZIP kết quả không kèm weights; hậu kiểm local không băm lại file model hoặc source checkpoint.

Run baseline-operating-point trước đó vẫn được giữ ở [hồ sơ lịch sử7.14](../../../DATASET/logs/phase7/task_7.14/runs/locked_benchmark_phase7_20261109_074134/analysis/README.md); không ghi đè dữ liệu gốc và không ghép metric giữa các hash package.

## Bổ sung 10/11/2026 — phép đối chứng P0 sau vòng train mới

Matched run `p0_eval_20261110_065901` chấm P0 và incumbent weights hiệu chuẩn lại: P0 G7.2 FAIL/đạt 4/8, incumbent refit FAIL/đạt 6/8. Clean Fake@Real90 61% so với 64,125%; Q30 38,875% so với 42,625%; macro 89,6794% so với 90,1018%. Arrays của incumbent refit trùng logits/xác suất/nhãn với kết quả current trên cả 8 quần thể, nhưng package refit hash khác. Không thay raw hay nguồn số liệu chính của báo cáo này bằng gói refit/P0. Xem [báo cáo vòng 1 và chẩn đoán vòng 2](bao_cao_phase7_improvement_round1_va_chan_doan_vong2.md). Chẩn đoán validation sau đó đã hoàn tất GPU và hậu kiểm; đây không phải lần chấm CFE tiếp theo, không thay G7.2 bằng số validation.
