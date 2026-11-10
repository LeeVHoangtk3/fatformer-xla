# Phase 7 — cập nhật sau vòng cải thiện 1 và kế hoạch chẩn đoán vòng 2

Cập nhật **10/11/2026**. Model báo cáo hiện hành vẫn là package `248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`, nguồn Anchored v2 `5cd49252…`. **G7.2 FAIL, đạt 6/8 tiêu chí; G7.3 pending verified best-by-condition reference.** Không thay model bằng P0.

## 1. Những giai đoạn đã thực hiện

1. **Task 7.13 — router và fallback:** code hạ tầng có kiểm thử, nhưng thí nghiệm router held-out chưa đủ; deployment dùng fixed fallback. Không lấy số synthetic đã thu hồi hoặc số unit test làm bằng chứng router thắng.
2. **Task 7.14–7.15 — hiệu chuẩn và chấm package thực:** run `corrected_20261109_102954_3d28ad` ngày 09/11 dùng 400 ảnh CFE calibration, 1.600 ảnh evaluation ×7 điều kiện và 82.003 lượt ảnh in-dist/18 tập. Nhiệt độ hội tụ, package/20 ảnh/latency L4 có log cùng hash. Kết quả không đạt toàn bộ gate; giữ raw và bộ log 7.14 trước đó để truy vết.
3. **Task 7.16 — cập nhật hồ sơ:** tài liệu, Chương 5 và bảng nguồn đã được biên soạn; cross-review/chữ ký A/B/C còn chờ người thật. Cập nhật tài liệu không tự đóng G7.0/G7.3 hoặc đổi kết quả âm thành PASS.
4. **Pilot train vòng 1:** run `pilot_20261109_182359`, code `2665014`, hai nhánh train mới P0 legacy và P1 CLIP, 5 epoch/nhánh, seed 42, L4. Cùng validation CLIP, dữ liệu, loss và batch hiệu dụng 32. Worst-group selection của P0 42%, P1 35,33%; cả hai chọn epoch 5. Chọn P0 bằng validation trước khi chấm CFE. Đây không thay ablation bốn nhánh Task 7.12.
5. **Matched evaluation vòng 1:** run `p0_eval_20261110_065901`, code `977b282`. P0 và weights incumbent được hiệu chuẩn riêng, chấm cùng protocol. P0 chỉ đạt 4/8 tiêu chí G7.2, incumbent refit đạt 6/8; cả hai FAIL. Chốt vòng 1 là kết quả âm, giữ package hiện hành.
6. **Chẩn đoán vòng 2:** run `val_diag_20261110_142828` đã hoàn tất L4, source `0dabd725e5b7a351514a9b7ee34f4bd6014b584f`; hậu kiểm tái lập đủ 4.320 lượt dự đoán và báo cáo nhóm/lỗi. **Chưa train hoặc fit calibration mới.** Có căn cứ thử giả thuyết calibration transfer và robustness Q30, chưa xác lập nguyên nhân P0 thua CFE. Xem [hậu kiểm và đính chính](../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/README.md).

Báo cáo diễn giải, số đo và giới hạn: [báo cáo vòng 1 và chẩn đoán vòng 2](../bao_cao/C/bao_cao_phase7_improvement_round1_va_chan_doan_vong2.md). Kết quả primary 7.14–7.15 vẫn lấy từ [`task_7.15/current`](../../DATASET/logs/phase7/task_7.15/current/analysis/README.md), không trộn với gói refit của vòng 1.

## 2. Vì sao chưa train lại ngay?

P0 trên evaluation có Clean Fake@Real90 61% so với 64,125% của incumbent refit; Q30 38,875% so với 42,625%; macro in-dist 89,6794% so với 90,1018%. Đổi preprocessing train trong pilot này chưa tạo model thay thế tốt hơn. Chưa có bằng chứng để chọn tăng epoch, đổi loss hay bổ sung dữ liệu làm phương án chính thức.

Trên validation ở ngưỡng chọn model lịch sử, P0 giữ đúng 586/590 real nhưng chỉ nhận ra 296/490 fake clean; Q30 lịch sử đúng 147/150 real, 63/150 fake. Cần phân biệt khả năng xếp hạng với ngưỡng bảo thủ; không tự hạ ngưỡng để tăng recall rồi bỏ qua FPR.

Manifest có đủ họ autoregressive, nhưng chỉ một generator; commercial và pixel diffusion cũng mỗi họ một generator. Tất cả generator validation đã có ở train. Đây là giới hạn độ phủ, chưa chứng minh quan hệ nhân quả với subset test yếu. Logits `val_select` mới đã được nộp và hậu kiểm; local vẫn chưa có full ảnh v2 để xác nhận nhãn ngữ nghĩa. Không suy ngược lỗi từng ảnh từ CSV tổng hợp cũ.

## 3. Phép chẩn đoán validation đã hoàn tất — quy trình tái lập

Run và notebook đã được hậu kiểm; không cần chạy lại y nguyên. Phần hướng dẫn dưới đây giữ để tái lập. Bước cần thiết kế tiếp hiện nằm ở mục 5.

Mở notebook mới [`phase7_validation_diagnostics_colab.ipynb`](../../notebooks/task/phase7_validation_diagnostics/phase7_validation_diagnostics_colab.ipynb). Chạy ba code cell lần lượt: kiểm tra GPU, clone nhánh `hoang` và checkout commit đã khóa, mount Drive và forward.

Nguồn code là Git, không cần ZIP code. GPU có thể là bất kỳ CUDA GPU tương thích đủ bộ nhớ. Runner kiểm hash archive/checkpoint, stage lên SSD Colab, giải mã/hash đủ ảnh validation, probe cả hai model rồi chọn batch chung trong [1,16], FP16, TF32 tắt. Nếu đổi GPU/commit hoặc thất bại, giữ run cũ và tạo run mới; không thay batch giữa phiên đo.

Input Drive:

```text
MyDrive/Fatformer/phase7/train_crossdomain_v2.tar
MyDrive/Fatformer/phase7/improvement_round1/pilot_20261109_182359/p0_legacy_full/best.pth
MyDrive/Fatformer/checkpoint/phase7/anchored_v2_run1/fatformer_anchored_v2_best.pth
```

Phạm vi đo:

- Hai checkpoint đã đóng băng, cùng 1.080 ảnh `val_select` v2: 590 real/490 fake. Đây là tập khác với CFE calibration/evaluation 400/1.600.
- Clean và JPEG Q30 **sau resize/crop**; CLIP resize cạnh ngắn 224 và crop 224. Không đồng nhất mọi protocol JPEG trước/sau resize.
- Q30 chạy đủ 1.080 ảnh; giữ riêng 300 ảnh lịch sử, 150 real/150 fake, seed 42. Phần lịch sử có autoregressive 33, GAN 30, commercial 26, pixel diffusion 19, latent diffusion 42 ảnh fake.
- Dùng chung operating point lịch sử T=1,6402, ngưỡng=0,9127; không dùng nó thay calibration deployment của mỗi model. Không fit T/ngưỡng mới ở bước này.
- Xuất logits, xác suất, nhãn, ID/hash và path từng ảnh, metric theo family/generator và danh sách lỗi. AUC nhóm fake dùng cùng 590 real reference; nhóm nhỏ và dùng chung real có phụ thuộc thống kê.

Kế hoạch máy đọc được: [`configs/phase7_validation_diagnostics/plan.json`](../../configs/phase7_validation_diagnostics/plan.json). Hướng dẫn đầy đủ: [`notebooks/task/phase7_validation_diagnostics/README.md`](../../notebooks/task/phase7_validation_diagnostics/README.md). 41 test CPU/code đã đạt; đây không phải bằng chứng forward GPU thật hoặc cải thiện chất lượng.

## 4. Hậu kiểm và quyết định sau phép đo

Lưu notebook đã chạy vào `notebooks/task/phase7_validation_diagnostics/phase7_validation_diagnostics_executed.ipynb`; kết quả vào `wait/phase7_validation_diagnostics/<DIAG_RUN_ID>/`. Drive lưu tại `MyDrive/Fatformer/phase7/improvement_round2/diagnostics/<DIAG_RUN_ID>/`. ZIP là ZIP kết quả, không chứa weights; có thể chép trực tiếp folder thay ZIP.

Dữ liệu đã được đối chiếu hash/IDs/split/source, ranking/recall và báo cáo lỗi; nhãn ngữ nghĩa vẫn cần xem ảnh thật vì local không có full archive. Các giả thuyết sau được xem xét theo bằng chứng, chưa coi là kết luận nhân quả:

1. Ranking tốt nhưng recall tại ngưỡng thấp: thử giả thuyết chuyển miền calibration; fit trên `val_calib` độc lập, giữ ràng buộc Real/FPR, không xuất ngưỡng ước lượng từ `val_select` làm deployment.
2. Lỗi tập trung ở nhóm độ phủ hẹp: kiểm nhãn/ảnh và nguồn trước; dữ liệu bổ sung phải qua denylist/isolation, không lấy ảnh test vào train.
3. Ranking giảm rõ ở Q30: thiết kế một thay đổi về robustness/degradation và control tương ứng; không đồng thời đổi loss, data, preprocessing và epoch.

`val_select` đã được dùng chọn P0, nên kết quả là chẩn đoán, không phải test mù. Không chọn epoch/seed/ngưỡng từ CFE đã xem. Muốn kết luận tổng quát mạnh hơn cần holdout mới và nhiều seed. G7.3 vẫn là công việc riêng: thu thập reference tốt nhất từng điều kiện có cùng partition/protocol/provenance; chỉ đối chiếu một incumbent chưa đủ.

Hồ sơ và trạng thái hiện tại: [`improvement_round2/diagnostics`](../../DATASET/logs/phase7/improvement_round2/diagnostics/README.md). Chưa có nhánh train vòng 2 được khóa hoặc khởi chạy.

## 5. Sau chẩn đoán: thử calibration có đối chứng trước khi train

Các số gộp validation đã khớp, nhưng không gọi bốn recall khớp là bit-exact logits. SD v1.5 P0 Q30 recall đúng là **33,33%**; DiT **23,08% ở cả clean và Q30**. CIPS có suy giảm mạnh trên 12 ảnh, chưa chứng minh kiến trúc sụp đổ. Median score fake gần ngưỡng nhưng phân bố rộng; real không phải đều dưới 0,02.

Ngưỡng 0,9127/T1,6402 là điểm lịch sử của chẩn đoán; P0 đã được fit T3,0949893/ngưỡng0,76518965 khi chấm CFE. Vì vậy không quy kết CFE kém là do chưa hiệu chuẩn. AUC cao trên generator đã thấy cũng không loại trừ hạn chế biểu diễn ở miền khác.

Thí nghiệm đầu tiên được đề xuất: giữ nguyên từng checkpoint, so calibration CFE sẵn có với T/ngưỡng FPR5 fit trên **val_calib v2 clean, 1.080 ảnh độc lập**. Dùng một operating point cố định cho cả clean và Q30 khi chẩn đoán trên `val_select`. Giữ precision/batch/preprocessing/IDs, báo Real/Fake recall, FPR, AUC/AP và calibration metrics theo nhóm. Không fit trên `val_select`, không tăng Fake bằng cách bỏ qua Real guard, không chọn ngưỡng/model theo CFE evaluation đã xem. Chưa khởi chạy thí nghiệm này.

Nếu ranking/robustness Q30 vẫn là giới hạn thì làm một thí nghiệm augmentation riêng, có control phù hợp; không đồng thời đổi calibration và train rồi gán nguyên nhân. [Kế hoạch có cấu trúc](../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/analysis/post_diagnosis_plan.json) và [đính chính chi tiết](../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/README.md) ghi phạm vi chấp nhận, giới hạn và trạng thái chưa train.

## 6. Giai đoạn chuẩn bị code calibration transfer — hồ sơ trước run

[Kế hoạch calibration transfer vòng 2](ke_hoach_calibration_transfer_phase7_round2.md) đã khóa hai point trên mỗi checkpoint: calibration CFE có sẵn và calibration fit chỉ trên clean `val_calib` v2. [Notebook/hướng dẫn](../../notebooks/task/phase7_calibration_transfer/README.md) lấy source Git `hoang`; runner chỉ reuse validation logits khi inference source/GPU/Torch/CUDA/batch khớp, còn lại đo lại cả hai model theo cùng protocol. Kết quả nghiên cứu chưa được dùng thay model/gate hiện hành.

[Preparation](../../DATASET/logs/phase7/improvement_round2/calibration_transfer/README.md) có 30 test CPU/code đạt và đã thử staging đủ 16 reference files từ folder/ZIP thật. Đây là trạng thái trước run; kết quả đã được cập nhật ở mục 7. Thí nghiệm thay domain/cỡ mẫu/tỷ lệ lớp calibration đồng thời, không tách nguyên nhân riêng từng yếu tố.

## 7. Calibration transfer đã hoàn tất nhưng point mới không đạt guard

[Run `calib_transfer_20261110_155948`](../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/README.md) được hậu kiểm đủ 56 file/170 source hashes, 8.640 scored decisions; metrics/CSV/paired CI khớp. Fit chỉ trên `val_calib`, đúng 29 FP/590 real mỗi model. P0 Fake recall clean62,04→84,69%, Q30 49,18→80%; incumbent clean63,06→84,49%, Q30 49,18→78,78%. Đổi lại Real Q30 chỉ86,95%/87,29%, dưới guard90 đã khai báo. Không chấp nhận point clean-only mới; current package/G7.2/G7.3 giữ nguyên.

Phân biệt mục tiêu tiếp: calibration robust clean/Q30 một ngưỡng chung là thử nghiệm operating point riêng; tăng ranking/Fake@Real90 của G7.2 cần can thiệp weights/data/robustness có control. Không dùng `val_select` hoặc CFE evaluation để chọn ngưỡng sau khi thấy kết quả. Chưa khởi chạy thí nghiệm sau run này. Notebook calibration trong repo vẫn là template, chưa có bản thực thi; hậu kiểm hiện dựa vào arrays/metadata/source, không tự xác nhận các cell đã chạy từ notebook chưa nộp.
