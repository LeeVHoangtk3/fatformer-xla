# Phân tích Task 7.14 — run locked_benchmark_phase7_20261109_074134

Ngày hậu kiểm: 09/11/2026. Run nguồn commit `bb43fd90dc65df67707de2eee0c41b5a63eec335`; báo cáo kết thúc lúc `2026-11-09T08:11:35.040240+00:00`. GPU ghi trong notebook: **NVIDIA L4 (22.03 GiB VRAM)**. Phân tích local không chạy lại backbone/GPU.

## Kết luận sử dụng

Đã nhập **31 tệp gốc**, giữ nguyên byte, tên và dữ liệu vào [raw](../raw/). Đây là kết quả đánh giá **Anchored v2 tại operating point baseline cố định**, không phải benchmark package 7.15 dùng calibration v2 đã sửa. Việc dùng GPU L4 hợp lệ; vấn đề phân loại kết quả nằm ở model/calibration/protocol, không ở tên GPU.

Checkpoint v2 ghi nhận `5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c`. Calibration dùng `baseline.json`, checkpoint nguồn `3b904401ad4ce7d0bc33661e83742ec4b659850c076b73cd544e87e26d8cb3db`, **T=1.6402, threshold=0.9127**. Trường `mode=frozen_reference_operating_point` trong báo cáo lần này đã mô tả đúng quan hệ đó. Không đổi tên thành calibration chính chủ của v2.

## Kiểm tra đã thực hiện

- 207 phép kiểm tra đã đạt tại thời điểm tính kết quả: hash 7 NPZ, shape/giá trị hữu hạn, nhãn, ID, thứ tự cặp ảnh, metadata checkpoint/calibration/partition, metric và CI95.
- Mỗi điều kiện có đúng 1.600 ảnh: 800 real + 800 fake, cùng ID/nhãn. Khớp toàn bộ frozen evaluation và không trùng ID với 400 calibration. Kiểm tra ID không thay cho kiểm tra trùng nội dung ảnh.
- Tính lại toàn bộ metric pooled và từng subset; CI95 bootstrap 1.000 lần, seed 42, khớp báo cáo trong dung sai 0,02 pp do làm tròn. Không fit T/ngưỡng trên evaluation.
- In-dist có 18 tập, tổng 82.003 ảnh; mean của 18 balanced accuracies bằng **90.46035007%**. CI bootstrap theo subset cũng khớp.
- Hash notebook trước/sau được giữ trong [verification.json](verification.json) và import manifest. Không chỉnh, reset output, format hoặc sao chép đè notebook.

## Kết quả CFE

- `clean`: AUC **88.67%**, Fake@Real90 **64.12%**, accuracy 73.75%, FPR tại ngưỡng cố định 4.250%.
- `q30`: AUC **79.87%**, Fake@Real90 **42.62%**, accuracy 65.31%, FPR tại ngưỡng cố định 8.000%.
- `q50`: AUC **81.55%**, Fake@Real90 **47.62%**, accuracy 66.31%, FPR tại ngưỡng cố định 6.500%.
- `q75`: AUC **83.31%**, Fake@Real90 **48.75%**, accuracy 66.19%, FPR tại ngưỡng cố định 5.375%.
- `webp40`: AUC **79.00%**, Fake@Real90 **41.88%**, accuracy 65.94%, FPR tại ngưỡng cố định 10.250%.
- `blur15`: AUC **84.22%**, Fake@Real90 **50.62%**, accuracy 71.00%, FPR tại ngưỡng cố định 11.000%.
- `downup05`: AUC **87.00%**, Fake@Real90 **64.75%**, accuracy 77.38%, FPR tại ngưỡng cố định 11.500%.

Số chi tiết 7 điều kiện ở [condition_summary.csv](condition_summary.csv); 35 điểm theo subset ở [per_subset_summary.csv](per_subset_summary.csv). Cột `macro_acc` của evaluator CFE là accuracy theo ảnh; toàn tập cân bằng 800/800 nên bằng balanced accuracy, nhưng không mặc định như vậy cho từng subset.

Clean AUC CI95 **[87.12, 90.16]**, Fake@Real90 CI95 **[59.34, 69.77]**. Q30 tương ứng **[77.64, 82.06]** và **[37.8, 47.75]**. Không mô tả tất cả CI là ±1,5 pp.

JPEG Q30 làm AUC giảm 8.80 pp và Fake@Real90 giảm 21.50 pp. Đây là mô tả suy biến, không phải tiêu chí delta ≤1,5 pp của G7.3. Nhóm autoregressive yếu nhất ở clean: Fake ACC 28,40% tại ngưỡng triển khai tham chiếu; không được thay bằng Fake@Real90 dùng ngưỡng khác.

## In-distribution và gate

Macro ACC mean 18 subset **90.46035%**, Micro ACC ghi nhận **93.43317%**. Real ACC pooled **90.67454%** khác Real ACC mean 18 subset **86.89720%**. Báo cáo dùng pooled Real ACC cho gate; phải ghi rõ cách tổng hợp, không trộn hai con số.

G7.2 tại operating point này: **5/8 đạt, tổng thể FAIL**. Ba tiêu chí chưa đạt:
- Clean Fake@Real90: 64,12% so với 65%.
- Worst-subset Fake ACC: 28,40% so với 45%.
- ECE Q30: 13,49% so với 8%.

G7.3: **pending**, chưa có số đo overhead wrapper, verification package hoặc delta với best model theo điều kiện trong bộ tệp này. **44.40 ms** chỉ là tổng thời gian backbone batch 1 trên L4; không đem so trực tiếp với mục tiêu overhead <5 ms.

Không có logits từng ảnh của 82k in-dist trong gói nộp. Vì vậy AUC pooled **98.28088%** chỉ được đối chiếu giữa các báo cáo, chưa tính lại độc lập từ dự đoán. ROC gốc chỉ thể hiện CFE clean/Q30 có logits, không suy ra đường ROC in-dist từ một số AUC.

## Lưu ý về bảng generator và config

`per_generator_phase7.csv` có 20 dòng khuôn, chỉ 5 dòng đủ real/fake. Các điểm số số học tính lại khớp, nhưng nhóm generator được suy từ đường dẫn/tên subset, chưa được xác nhận bằng metadata generator độc lập. Ngưỡng Real90 của bảng lấy trên pooled real; không bảo đảm Real ACC bằng 90% riêng trong mỗi nhóm. Không coi 20 dòng là 20 generator độc lập.

`LOCKED_CONFIG.json` gốc chứa ngưỡng FPR 1%=0,95 và 10%=0,88, không khớp baseline thực tế 0,999 và 0,0258; ngưỡng 5%=0,9127 đang dùng khớp. Chuỗi router “71 tests” cũng là metadata cũ, không phải chứng cứ kiểm thử của run. Giữ nguyên raw config để truy vết; [effective_config.json](effective_config.json) là cấu hình quan sát sau chạy, không giả nhận đã được khóa trước thí nghiệm.

Chưa băm lại tệp checkpoint gốc trên Drive hay kiểm tra pixel ảnh tại local. Legacy dataset reader có khả năng thay ảnh lỗi bằng ảnh xám/zero; log CFE không thấy cảnh báo đọc lỗi, nhưng không đủ để chứng minh mọi ảnh in-dist được giải mã đúng. Một run tốt không tự chứng minh ưu thế thống kê hoặc không quên miền cũ.

## Bố trí hồ sơ

- [raw](../raw/): 31 tệp gốc của run, không sửa.
- [evidence](../evidence/): baseline calibration và cặp frozen split đối chiếu, không phải calibration v2 mới.
- [verification.json](verification.json): kiểm tra máy đọc được, metric/CI/gate và giới hạn.
- [notebook_stdout.txt](notebook_stdout.txt): trích stream output của notebook, không sửa file notebook.

Giữ run lịch sử `DATASET/logs/phase7/locked_benchmark_phase7/` nguyên trạng. Run vừa nhập dùng cho báo cáo dưới nhãn frozen reference operating point; không ghi đè hoặc tự nhận là kết quả package cuối 7.15.
