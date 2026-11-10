# Báo cáo Phase 7 — pilot vòng 1, matched evaluation và chuẩn bị chẩn đoán vòng 2

Ngày cập nhật **10/11/2026**. Kết luận hiện hành: **chưa chọn được model thay thế incumbent; vòng cải thiện 1 là kết quả âm.** Model báo cáo vẫn là package `24869052…`, G7.2 FAIL 6/8, G7.3 pending. Notebook chẩn đoán vòng 2 đã tạo, chưa chạy GPU và chưa train thêm.

## 1. Lịch sử và phạm vi bằng chứng

Task 7.13 cung cấp hạ tầng router/fixed fallback; router hiện tắt do thiếu thí nghiệm held-out đủ bằng chứng. Các số synthetic cũ bị thu hồi, không dùng số test để khẳng định ưu thế router. Task 7.14 đo benchmark; Task 7.15 hiệu chuẩn/đóng gói và chạy corrected GPU ngày 09/11. Task 7.16 tổng hợp tài liệu, không tạo model hoặc phép đo chất lượng mới; chữ ký/cross-review còn chờ.

Run corrected `corrected_20261109_102954_3d28ad` là nguồn chính cho kết quả incumbent, không bị thay bằng pilot mới. Run benchmark 7.14 trước corrected được giữ như lịch sử. Hai vòng tiếp theo là thí nghiệm cải thiện bổ sung, không thay tên hoặc sửa raw của các task cũ.

Nguồn local: [hồ sơ 7.15 hiện hành](../../../DATASET/logs/phase7/task_7.15/current/analysis/README.md), [pilot train](../../../DATASET/logs/phase7/improvement_round1/runs/pilot_20261109_182359/README.md), [matched evaluation](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/README.md), [kiểm kê vòng 2](../../../DATASET/logs/phase7/improvement_round2/diagnostics/README.md). Arrays/weights không được mặc định là đã có trong Git; gói kết quả local và Drive là nơi lưu bằng chứng tương ứng.

## 2. Pilot train vòng 1 — lựa chọn hợp lệ trong phạm vi validation

Run `pilot_20261109_182359`, source `2665014`, NVIDIA L4, seed 42, 5 epoch cho mỗi nhánh mới. P0 dùng legacy ImageNet preprocessing khi train, P1 dùng CLIP; cả hai validation bằng CLIP. Dữ liệu, khởi tạo/teacher, loss, số epoch và effective batch 32 được giữ chung. Đây là pilot hai preprocessing, không thay ablation bốn nhánh Guard/Feature của Task 7.12.

P0 worst-group **42%**, P1 **35,3333%**, cả hai chọn epoch 5. Worst-group là min recall clean real/fake và Q30 real/fake trên `val_select`, không phải worst-generator CFE. P0 được chọn trước khi đo evaluation, không chọn checkpoint từ CFE. Source P0 SHA-256 `9d97613c5f271801a30ff514326557f262a372abeb4d52ce864c78f20fc1069b`.

Đã nhập 27 file nguyên byte và notebook thực thi, tái tính 3.830 record optimizer trace, đối chiếu 146 source hashes. P0 áp dụng 1.914/1.915 update, P1 1.913/1.915; bỏ qua lần lượt 1/2 update do norm không hữu hạn, không áp dụng update có norm không hữu hạn. Có 28.832 phép kiểm tra dữ liệu đạt; đây không phải 28.832 unit test. Instrumentation mới không suy ngược số update bị bỏ qua trong log Anchored v2 cũ.

## 3. Matched evaluation vòng 1 — không đủ cơ sở thay incumbent

Run `p0_eval_20261110_065901`, source `977b282`, notebook ba code cell hoàn tất, không có error output; thời gian cell đánh giá khoảng 81,82 phút. Hai model được fit calibration riêng trên cùng 400 ảnh cân bằng, chấm cùng 1.600 ảnh CFE cân bằng ×7 điều kiện và 82.003 lượt ảnh in-dist/18 tập. Cùng L4, FP16, batch 16, preprocessing và TF32 tắt. Calibration/evaluation tách ID.

Đối chứng là **weights incumbent hiệu chuẩn lại**, không phải nạp lại đúng file package deployment `24869052…`. Các số sau lấy từ arrays/comparison tương ứng; đơn vị %, trừ khi ghi khác:

- Clean AUC: P0 **87,96**, đối chứng **88,66**. Fake@Real90: **61,000** so với **64,125**, mục tiêu ≥65.
- Q30 AUC: **79,02** so với **79,88**. Fake@Real90: **38,875** so với **42,625**, mục tiêu ≥40.
- Macro in-dist (trung bình balanced accuracy 18 tập): **89,6794** so với **90,1018**, mục tiêu ≥90.
- Pooled Real accuracy: **90,36** so với **90,41**, cả hai đạt ≥90.
- Worst-subset Fake accuracy CFE clean `5_other_autoregressive`: **21/81=25,93** so với **25/81=30,86**, mục tiêu ≥45.
- ECE Q30: **6,17** so với **5,08**, cả hai đạt ≤8. Không lấy ECE trên tập calibration thay ECE evaluation này.

P0 đạt **4/8** tiêu chí, trượt clean Fake@Real90, Q30 Fake@Real90, macro và worst-subset Fake. Đối chứng đạt **6/8**, trượt clean Fake@Real90 và worst-subset Fake. Cả hai FAIL; không có gate mới được tự hạ hoặc thay định nghĩa.

P0 không kém mọi chỉ số: Q30 TPR tại FPR 1%/5% đạt **15,12%/29%**, cao hơn **12,38%/28,25%** của đối chứng. In-dist ECE/NLL **6,76%/0,3121**, tốt hơn **6,99%/0,3386**. Các điểm tốt này chưa đủ bù các tiêu chí đã chốt còn thiếu. Calibration FPR 5% không bảo đảm ngoài tập fit; P0 có FPR clean **5,375%**, Q30 **8,5%** ở ngưỡng vận hành của nó.

## 4. Độ chắc chắn và kiểm chứng local

Đã nhập 90 file, 65.926.966 byte, và bản sao nguyên byte notebook thực thi. Đối chiếu 155 source files của commit đã chạy, 17 lệnh/step markers và hash output; tái tính pooled/per-subset metrics từ **186.406 lượt dự đoán**. Nhiệt độ được kiểm bằng tối ưu NLL độc lập, ngưỡng CPU/GPU trong tolerance dưới `1e-6`. Có 2.446 phép kiểm tra dữ liệu và 25 unit test hậu kiểm/workflow liên quan đạt; không gọi các phép kiểm tra dữ liệu là unit test.

Paired bootstrap được chạy lại: 1.000 resample phân tầng nhãn cho mỗi điều kiện, cùng indices hai model, seed 42. Delta **P0−đối chứng**, đơn vị điểm phần trăm:

- Clean AUC −0,7034, CI95 [−1,1033; −0,3035]; Q30 AUC −0,8575, CI95 [−1,4258; −0,2916].
- Clean Fake@Real90 −3,125, CI95 [−6,1281; +0,5000]; Q30 −3,750, CI95 [−6,500; +1,375]. Hai CI recall chứa 0, không kết luận mọi suy giảm đều có ý nghĩa thống kê.
- Macro delta −0,4225, CI chẩn đoán 10.000 resample theo 18 tập [−0,8464; +0,0325]. CI này không thay luật gate.

AUC điểm P0 thấp hơn cả 7 điều kiện; 6/7 CI AUC chưa điều chỉnh nằm dưới 0. Đây là một seed train, chưa điều chỉnh nhiều so sánh, CI ảnh chưa bao quát phụ thuộc generator/dataset. CFE đã được xem trước, nên đây là hậu kiểm, không phải blind test mới. Không suy từ kết quả này rằng mọi cách train lại hoặc mọi CLIP preprocessing đều thất bại.

Weights và archive ảnh đầy đủ không có trong gói nộp. Local không băm lại `.pth` hoặc forward lại model thật. Log GPU báo tensor/20 ảnh tương đương; local kiểm chuỗi nguồn/hash, giải mã 20 PNG và tái tính arrays. Khoảng tin cậy paired đã được chạy lại; không chạy lại toàn bộ marginal bootstrap CI của từng model tại local trong lần hậu kiểm này.

Nguồn: [verification](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/verification.json), [paired verification](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/paired_verification.json), [decision](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/decision.json).

## 5. Danh tính model và latency không được trộn

- Source incumbent `5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c`; deployment **`248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`**. Đây vẫn là model báo cáo.
- P0 source `9d97613c…`; package vòng 1 `e26a16273d3da6f851a4fe41c994307b585339e9f5f7f9ac58c58a3621c863e2`; T=3,0949893, ngưỡng=0,76518965. Không được chấp nhận thay deployment.
- Incumbent refit package `de36061f4cc10e8b5a7b599842c160bfc5109e2a29bef2d659cf77388aaf1a5e`; T=2,7832491, ngưỡng=0,79185086. Hash khác package deployment; không đổi tham chiếu chính sang refit.

Dự đoán refit giống hoàn toàn arrays deployment cũ trên 7 điều kiện CFE và in-dist: IDs/logits/xác suất/nhãn trùng, sai số xác suất tối đa 0, không đảo nhãn. Đây là kiểm chứng tái lập dự đoán, không phải so sánh lại hai file `.pth` tại local.

Primary deployment đo L4 batch1 tổng **44,98 ms/ảnh**, overhead **0,188055 ms/ảnh**. Vòng matched mới: P0 tổng **44,53 ms/ảnh**, overhead **0,151093**; refit tổng **45,82**, overhead **0,205143**. Không lấy số mới thay latency của package cũ hoặc suy ra kiến trúc nhanh hơn. Overhead và tổng inference khác nhau. G7.3 vẫn thiếu verified best-by-condition reference; một incumbent không mặc nhiên là tốt nhất từng điều kiện.

## 6. Chẩn đoán vòng 2 — đã chuẩn bị, chưa có kết quả GPU

Manifest 14.400 ảnh có 17 generator fake: autoregressive/commercial/pixel diffusion mỗi họ 1, GAN 6, latent diffusion 8. Tất cả generator `val_select` đã có trong train. Metadata labels/family nhất quán và SHA/path không trùng giữa split; ngữ nghĩa nhãn cần xem ảnh. Local chưa có 14.400 ảnh v2 hoặc logits từng ảnh validation; chưa biết lỗi cụ thể của từng generator.

P0 tại operating point chọn model lịch sử T=1,6402, ngưỡng=0,9127 giữ 586/590 real nhưng chỉ nhận ra 296/490 fake clean; Q30 lịch sử 147/150 real và 63/150 fake. Đây là lý do cần đo ranking/phân bố điểm trước khi train tiếp, không phải bằng chứng chỉ cần hạ ngưỡng. Các ngưỡng này khác calibration deployment nêu ở mục 5.

Đã chuẩn bị runner tại commit **`0dabd72`**, 41 test CPU/code đạt; tạo notebook mới [`phase7_validation_diagnostics_colab.ipynb`](../../../notebooks/task/phase7_validation_diagnostics/phase7_validation_diagnostics_colab.ipynb). Notebook có ba code cell, chưa thực thi, không có outputs giả. Đo hai model trên cùng 1.080 `val_select` clean/Q30; giữ riêng Q30 300 ảnh lịch sử, kiểm hash ảnh, không train và không chấm test.

Sau khi có dữ liệu mới mới quyết định thử calibration transfer, độ phủ dữ liệu/nhãn hoặc robustness Q30; chưa khóa nhánh train. Kế hoạch và vị trí file/Drive: [kế hoạch sau vòng 1](../../plan/ke_hoach_cai_thien_phase7_sau_vong1.md). Giữ package hiện hành và các gate chưa đạt khi viết báo cáo; không coi việc tạo notebook hoặc hoàn thành tài liệu là model đã tốt hơn.
