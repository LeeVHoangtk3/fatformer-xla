# Rà soát số liệu Phase 7 — nguồn cho Task 7.16

Run hiện hành `corrected_20261109_102954_3d28ad`; package `248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`. Tài liệu này được lập trước khi cập nhật 11 tệp mục tiêu. Chỉ các số mới có file và JSON pointer bên dưới được dùng cho báo cáo hiện hành.

## Số cũ → số mới → nguồn

Các số cũ trong bảng chỉ để tham khảo lịch sử; Gen 1 là `invalid_layout`. Không hiểu các cặp số khác metric/protocol là mức cải thiện trực tiếp.

| Số/mô tả cũ | Ngữ cảnh cũ | Hiện hành | File + trường JSON | Diễn giải |
| :--- | :--- | :--- | :--- | :--- |
| 52.05% | Gen 1 n=8, ACC diagnostic, invalid_layout | CFE clean accuracy: **74,38%** | [final_report.json](../../../DATASET/logs/phase7/task_7.15/current/raw/benchmark/final_report.json) `/cfe/clean/metrics/accuracy` | Không phải mức tăng cùng protocol; model và partition khác nhau. |
| 4.10% | Gen 1 n=8, Fake ACC diagnostic, invalid_layout | CFE clean Fake ACC: **53,12%** | [final_report.json](../../../DATASET/logs/phase7/task_7.15/current/raw/benchmark/final_report.json) `/cfe/clean/metrics/fake_acc` | Chỉ ở ngưỡng vận hành hiện hành; không thay thế Fake@Real90. |
| 96.58% | Overall Gen 1 bị invalid_layout, từng gọi sai là macro | In-dist mean 18 balanced ACC: **90,10%** | [final_report.json](../../../DATASET/logs/phase7/task_7.15/current/raw/benchmark/final_report.json) `/indist/macro_acc` | Không so trực tiếp overall cũ với macro mới. |
| 91.5% | Mốc Phase 6-R, chỉ để tham khảo lịch sử | CFE clean AUC: **88,66%** | [final_report.json](../../../DATASET/logs/phase7/task_7.15/current/raw/benchmark/final_report.json) `/cfe/clean/metrics/auc` | Không đổi ACC thành AUC; chỉ thay nguồn trích dẫn hiện hành. |
| 5/8 G7.2 | Run operating point baseline trước sửa | G7.2 hiện hành: **6/8; FAIL** | [acceptance.json](../../../DATASET/logs/phase7/task_7.15/current/raw/acceptance.json) `/G72` | Điểm ECE Q30 nay đạt; hai mục vẫn FAIL. |
| 13.49% ECE Q30 | Run baseline operating point | ECE Q30: **5,08%** | [final_report.json](../../../DATASET/logs/phase7/task_7.15/current/raw/benchmark/final_report.json) `/cfe/q30/metrics/ece` | Khác calibration/precision; không gán là tăng năng lực phân biệt. |
| T=1.4676 / 2.78636 | Bản trước hội tụ / local FP32 | T FP16/CUDA batch16: **2.7832491397857666** | [anchored_v2.json](../../../DATASET/logs/phase7/task_7.15/current/calibration/anchored_v2.json) `/T` | Không ép T bằng cache local; ngưỡng cũng fit lại. |
| 0.9127 / 0.795321 | Ngưỡng baseline / local FP32 | Ngưỡng hiện hành: **0.7918508648872375** | [anchored_v2.json](../../../DATASET/logs/phase7/task_7.15/current/calibration/anchored_v2.json) `/T_star_default` | Fit trên 400 calibration, không dùng evaluation. |
| 44.79 ms hoặc 44.98 ms | Backbone/package tổng, từng nhầm với overhead | Overhead batch1: **0,188055 ms** | [latency_phase7.json](../../../DATASET/logs/phase7/task_7.15/current/raw/latency_phase7.json) `/batches/1/overhead_delta_ms` | Median từng cặp đo; không lấy hiệu hai median đã làm tròn. |
| Chờ 20 ảnh / chờ T4 | Trạng thái trước run GPU | Verification: **20 ảnh; L4 đã đo** | [real_image_verification.json](../../../DATASET/logs/phase7/task_7.15/current/raw/real_image_verification.json) `/n_images` | T4 không bắt buộc theo yêu cầu người dùng. |
| Router thắng / 71 tests = nghiệm thu | Số synthetic v1 đã thu hồi | Router: **Tắt; experiment deferred** | [router_params.json](../../../DATASET/logs/phase7/router_params.json) `/deployment_enabled` | Unit test không thay cho thí nghiệm held-out. |
| Gen 2 alpha=.50 metadata rời | Ứng viên cũ, unselected | Gen 2 đã chọn: **alpha=.55; beta=.35** | [gen2_verified_metadata.json](../../../DATASET/logs/phase7/task_7.16/gen2_verified_metadata.json) `/metadata` | Đã băm actual .pth và đối chiếu metadata nhúng + CSV + matrix. |
| 12–16k dự kiến | Kho dữ liệu v2 | Số bản ghi: **14.400** | [data_isolation_report.json](../../../DATASET/logs/phase7/data_isolation_report.json) `/integrity/summary/n_records` | Train 12.240; val_calib 1.080; val_select 1.080, khác CFE 400/1.600. |

Toàn bộ giá trị chi tiết, kể cả AUC/AP/TPR/CI, count, calibration và latency: [evidence_catalog.json](../../../DATASET/logs/phase7/task_7.16/evidence_catalog.json) và [source_values.csv](../../../DATASET/logs/phase7/task_7.16/source_values.csv). Hash nguồn được giữ trong catalog; không gõ số slide độc lập.

## Những điều không được suy ra

G7.0 còn DoD legacy trên toàn repo; G7.1 FAIL; G7.2 FAIL 6/8; G7.3 pending reference tốt nhất theo điều kiện. Không biến việc hoàn tất thí nghiệm hoặc viết tài liệu thành PASS tất cả cổng.

Pooled accuracy, macro balanced accuracy và mean-subset Real ACC là ba đại lượng khác nhau. FPR calibration 5% không bảo đảm FPR của miền khác. CI theo ảnh không bao quát phụ thuộc giữa generator/dataset. Full checkpoint GPU không có trong ZIP; không dùng file local cũ khác hash để kiểm chứng thay thế.

## Kiểm tra source và phạm vi công cụ

AST ghi nhận 20 vị trí `strict` không hằng True; không phải tất cả đều là lỗi checkpoint trong đường báo cáo. [strict_loading_review.json](../../../DATASET/logs/phase7/task_7.16/strict_loading_review.json) phân biệt initialization, diagnostic và legacy. Không xóa chuỗi grep hoặc đổi source loader để làm đẹp DoD. E1–E6 trong pipeline hiện hành được đối chiếu cụ thể ở phụ lục A.9 của kế hoạch CFE.

Kiểm thử mục tiêu hiện tại: **103 test pipeline + 53 test thành phần = 156 passed, 0 failed/error/skipped**, trên 9 file riêng biệt; phạm vi ghi trong [targeted_tests.json](../../../DATASET/logs/phase7/task_7.16/validation/targeted_tests.json). Không tuyên bố đã chạy xanh toàn bộ repository. Cross-review và chữ ký A/B/C vẫn chờ người thật.

## Bổ sung ngày 10/11/2026 — nguồn số vòng cải thiện, không thay current

Catalog Task 7.16 phía trên tiếp tục mô tả run corrected/primary package `24869052…`. Các số sau thuộc thí nghiệm bổ sung và không ghi đè catalog hoặc raw hiện hành:

- Pilot train `pilot_20261109_182359`: P0/P1 worst-group validation 42%/35,3333%, đều epoch 5, source P0 `9d97613c…`. Nguồn [selection.json](../../../DATASET/logs/phase7/improvement_round1/runs/pilot_20261109_182359/raw/selection.json) `/arms`. Đây không phải Fake@Real90 hoặc worst-subset CFE.
- Matched evaluation `p0_eval_20261110_065901`: P0/incumbent refit Clean Fake@Real90 61%/64,125%, Q30 38,875%/42,625%; macro 89,67938199%/90,10184712%. Nguồn [paired verification](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/paired_verification.json) `/comparison/CFE`, `/comparison/indist`. Bản nguồn raw có thể làm tròn 64,12 và 42,62; không coi đó là số của một run khác.
- Phán quyết P0 FAIL 4/8, incumbent refit FAIL 6/8; deployment không đổi. Nguồn [decision.json](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/decision.json). Refit package `de36061f…` khác current `24869052…`; không đổi latency hoặc danh tính model báo cáo dù arrays đối chứng trùng kết quả cũ.
- Chẩn đoán vòng 2 đã hoàn tất run `val_diag_20261110_142828`: [verification](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/analysis/verification.json) xác nhận 159 source hashes, 4.320 lượt dự đoán, 112 dòng group metrics và 933 dòng lỗi; [interpretation](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/analysis/interpretation.json) ghi CI, số đếm và giới hạn. Source [plan](../../../configs/phase7_validation_diagnostics/plan.json) không đổi. Có 25 test CPU hậu kiểm/workflow liên quan đạt lần này; 41 test chuẩn bị và 156 test trước đó là các phạm vi khác có trùng lặp, không cộng như full-suite mới. Chưa fit calibration hoặc train vòng 2; model báo cáo không đổi.

Xem [báo cáo bổ sung](../C/bao_cao_phase7_improvement_round1_va_chan_doan_vong2.md) và [kế hoạch tiếp theo](../../plan/ke_hoach_cai_thien_phase7_sau_vong1.md). G7.0–G7.3 và chữ ký/cross-review giữ trạng thái như nguồn hiện hành, không suy ra PASS từ số unit test hoặc việc hoàn thành tài liệu.
