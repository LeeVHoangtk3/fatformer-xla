# Báo cáo tổng hợp công việc sau Task 7.15

**Ngày chốt số liệu: 10/11/2026.** Các số liệu chi tiết và nguồn tương ứng nằm trong [bảng kết quả](02_bang_so_lieu.md) và [danh mục nguồn](05_danh_muc_nguon.md).

## 1. Mục tiêu và mốc đối chứng

Sau khi hoàn tất hiệu chuẩn và đóng gói ở Task 7.15, công việc tiếp theo nhằm xác định liệu huấn luyện bổ sung hoặc thay đổi calibration có tạo được phương án tốt hơn model hiện hành hay không. Các hoạt động gồm tổng hợp hồ sơ Task 7.16, pilot huấn luyện hai nhánh, đánh giá đối chứng P0 với incumbent, chẩn đoán lỗi validation và thử calibration trên miền v2.

Mốc đối chứng là run `corrected_20261109_102954_3d28ad`, package SHA-256 `248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`. Model này dùng source Anchored v2, fixed fallback; router chưa có đủ bằng chứng held-out để được kích hoạt. Calibration hiện hành có T = **2,7832491398**, ngưỡng = **0,7918508649**. Hồ sơ gốc nằm tại [Task 7.15/current](../../../DATASET/logs/phase7/task_7.15/current/README.md).

Model đạt **6/8 tiêu chí G7.2**, nhưng tổng thể G7.2 vẫn FAIL. Hai khoảng thiếu là Clean Fake@Real90 **64,125% <65%** và worst-subset Fake accuracy **30,8642% <45%**. G7.3 còn thiếu đối chiếu với reference tốt nhất từng điều kiện trên cùng partition và protocol. Những giới hạn này là lý do phải kiểm chứng cải thiện bằng đối chứng thay vì chỉ dựa vào accuracy tổng hợp.

## 2. Hoàn thiện tài liệu sau đóng gói

Task 7.16 đã tổng hợp nguồn số liệu, kết luận, hạn chế và hồ sơ nghiệm thu. Công việc này không huấn luyện model hoặc tạo phép đo chất lượng mới. Cross-review/chữ ký A/B/C còn chờ xác nhận người thật; không xem việc biên soạn xong là toàn Phase 7 PASS. Trạng thái chi tiết lấy từ [hồ sơ Task 7.16](../../../DATASET/logs/phase7/task_7.16/README.md).

## 3. Pilot cải thiện vòng 1

Run `pilot_20261109_182359` thử hai nhánh huấn luyện: P0 dùng preprocessing legacy ImageNet khi train, P1 dùng CLIP. Cả hai được validation bằng CLIP, cùng seed 42, cùng 5 epoch, batch hiệu dụng 32 và các thiết lập dữ liệu/khởi tạo/loss chung. Thiết kế này khảo sát preprocessing train; không thay thế ablation bốn nhánh của Task 7.12.

Luật chọn checkpoint dựa vào worst-group của bốn recall clean/Q30 × Real/Fake, với điều kiện bảo vệ Real đã định. P0 đạt **42%**, P1 đạt **35,3333%**, cả hai chọn epoch 5. P0 được chọn làm ứng viên cho bước đánh giá độc lập với quyết định chọn checkpoint; không chọn từ kết quả CFE. Tuy nhiên, đây chỉ là một seed và Q30 pilot dùng 300 ảnh biến đổi từ validation, nên chưa đủ kết luận P0 tốt hơn mọi phương án hoặc preprocessing CLIP luôn kém. [Nguồn pilot](../../../DATASET/logs/phase7/improvement_round1/runs/pilot_20261109_182359/README.md).

## 4. Matched evaluation: P0 chưa thay được incumbent

Run `p0_eval_20261110_065901` đánh giá P0 và weights incumbent hiệu chuẩn lại trên cùng 400 ảnh calibration, 1.600 ảnh CFE ×7 điều kiện, cùng 82.003 lượt ảnh in-dist thuộc 18 tập. Hai model dùng chung protocol L4/FP16/batch16, calibration được fit riêng. Gói incumbent refit có hash khác package hiện hành; nó là đối chứng, không tự động thay package chuẩn.

P0 đạt Clean Fake@Real90 **61%**, Q30 Fake@Real90 **38,875%**, macro in-dist **89,6794%**; đối chứng lần lượt đạt **64,125%**, **42,625%**, **90,1018%**. Worst-subset Fake accuracy của P0 là **25,9259%**, thấp hơn **30,8642%** của đối chứng. P0 đạt 4/8 tiêu chí, đối chứng đạt 6/8; cả hai FAIL. Vòng 1 vì vậy chưa tạo ứng viên thay thế được chấp nhận.

P0 vẫn có điểm tốt ở một số vùng vận hành, như Q30 TPR tại FPR thấp và in-dist NLL/ECE. Tuy nhiên, các điểm đó không bù được tiêu chí đã chốt còn thiếu. Một số khoảng tin cậy chênh lệch recall chứa 0; không kết luận mọi suy giảm đều có ý nghĩa thống kê. Kết quả là bằng chứng cho quyết định giữ model, chưa phải kết luận mọi cách huấn luyện mới đều thất bại. [Nguồn matched evaluation và CI](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/README.md).

## 5. Chẩn đoán validation vòng 2

Run `val_diag_20261110_142828` đo cả hai model trên cùng 1.080 ảnh `val_select` gồm 590 Real và 490 Fake, ở clean và JPEG Q30 sau crop. T = **1,6402**, ngưỡng = **0,9127** là điểm lịch sử dùng chung để chẩn đoán, không phải calibration triển khai của mỗi model.

P0 có score AUC khoảng **96,37% clean**, **92,13% Q30**, nhưng Fake recall tại điểm lịch sử chỉ **60,41%**, **46,94%**. Điều này hỗ trợ khảo sát operating point, chưa chứng minh ngưỡng là nguyên nhân duy nhất của lỗi CFE. Trước đó P0 đã được calibration riêng khi chấm CFE; không thể quy kết kết quả test kém là do còn dùng ngưỡng lịch sử.

Hậu kiểm cũng sửa các nhận xét quá mức: CIPS giảm recall **10/12 →1/12**, nhưng mẫu nhỏ; SD v1.5 Q30 của P0 là **30/90 =33,33%**; DiT là **3/13 ở cả clean và Q30**, không có sụt recall giữa hai điều kiện. Khớp bốn recall lịch sử không tương đương kiểm chứng bit-exact logits. [Nguồn diagnostics và đính chính](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/README.md).

## 6. Calibration transfer: lợi ích và đánh đổi

Run `calib_transfer_20261110_155948` giữ weights cố định, fit T/ngưỡng chỉ trên 1.080 ảnh clean `val_calib` v2 không trùng SHA với `val_select`. Mỗi model so hai điểm vận hành: calibration CFE sẵn có và calibration clean-v2 mới. Cả hai điểm dùng cùng validation logits; run thực tế tái dùng arrays đã đóng băng, không forward validation mới.

Cả hai fit đạt **29/590 false positives =4,9153%** trên clean calibration, đúng ngân sách 5%. Khi áp dụng sang validation Q30, P0 tăng Fake recall **49,18% →80%**, nhưng Real recall giảm **98,14% →86,95%**. Incumbent tăng Fake recall **49,18% →78,78%**, nhưng Real recall giảm **97,29% →87,29%**. FPR Q30 tương ứng tăng lên **13,05%** và **12,71%**.

Hai điểm mới đều không đạt guard Real recall ≥90% ở cả clean/Q30 đã khai báo trước. NLL/ECE điểm có cải thiện nhưng raw ranking không đổi. Đây là thay đổi điểm vận hành, không tạo model có biểu diễn mới và không chứng minh cải thiện G7.2 trên CFE. Quyết định là **không chấp nhận calibration clean-only mới**. [Nguồn calibration transfer](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/README.md).

## 7. Kết luận và giá trị của giai đoạn bổ sung

Các thí nghiệm sau Task 7.15 đã làm rõ rằng đổi preprocessing train trong pilot hiện tại chưa cải thiện đủ các mục tiêu, còn calibration clean-only tăng Fake recall bằng đánh đổi mức bảo vệ ảnh thật. Giá trị của giai đoạn này là đối chứng được ứng viên, tái lập số liệu, nhận diện giới hạn và loại bỏ diễn giải chưa có căn cứ; chưa tạo model tốt hơn được chấp nhận.

Giữ package `24869052…` cùng calibration Task 7.15 làm bản chuẩn cho báo cáo. Kết quả chính vẫn **G7.2 FAIL — 6/8 tiêu chí**, **G7.3 PENDING**. Để cải thiện tiếp, ưu tiên bổ sung reference phù hợp cho G7.3 và thiết kế thí nghiệm dữ liệu/weights/robustness có đối chứng, lựa chọn bằng train/validation. Không điều chỉnh model hoặc ngưỡng dựa trên CFE đã xem, không hạ tiêu chí sau khi có kết quả. [Quyết định và hướng tiếp theo](04_ket_luan_va_huong_tiep_theo.md).
