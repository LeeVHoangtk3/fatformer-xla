# Danh mục nguồn và vị trí file

Chốt ngày **10/11/2026**. Mọi đường dẫn dưới đây tính từ root repo nếu nằm trong code block; các liên kết tương đối mở trực tiếp từ thư mục báo cáo. Nguồn số ưu tiên JSON/CSV và arrays đã hậu kiểm; README cung cấp diễn giải. Kế hoạch quy định cách làm, không thay thế số đo.

## 1. Bản chuẩn Task 7.15 — dùng cho kết quả chính

- [Hồ sơ hiện hành](../../../DATASET/logs/phase7/task_7.15/current/README.md).
- [Phân tích hậu kiểm](../../../DATASET/logs/phase7/task_7.15/current/analysis/README.md).
- [Tám tiêu chí G7.2](../../../DATASET/logs/phase7/task_7.15/current/analysis/gate_summary.csv).
- [Kết quả bảy điều kiện CFE](../../../DATASET/logs/phase7/task_7.15/current/analysis/condition_summary.csv).
- [Kết quả 18 tập in-dist](../../../DATASET/logs/phase7/task_7.15/current/analysis/per_subset_indist.csv).
- [Calibration GPU](../../../DATASET/logs/phase7/task_7.15/current/calibration/anchored_v2.json).
- [Latency bản chuẩn](../../../DATASET/logs/phase7/task_7.15/current/analysis/latency_summary.csv).
- [Acceptance của run](../../../DATASET/logs/phase7/task_7.15/current/raw/acceptance.json).
- [Package reference và vị trí Drive](../../../DATASET/logs/phase7/task_7.15/current/artifacts/package_reference.json).
- [Notebook Task 7.15 đã nộp](../../../notebooks/task/phase7_corrected/phase7_corrected_rerun_colab.ipynb).

```text
Package SHA-256:
248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0

Source checkpoint SHA-256:
5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c

Package trên Drive:
MyDrive/Fatformer/checkpoint/phase7/corrected_20261109_102954_3d28ad/unified_fatformer_v2.pth

Source checkpoint trên Drive:
MyDrive/Fatformer/checkpoint/phase7/anchored_v2_run1/fatformer_anchored_v2_best.pth
```

Không có weights mới tương ứng ở local; đây là vị trí/hash theo hồ sơ GPU đã hậu kiểm. Không thay bằng file local khác hash, package P0 hoặc incumbent refit.

## 2. Task 7.16 — hồ sơ tài liệu và xác nhận

- [Trạng thái Task 7.16](../../../DATASET/logs/phase7/task_7.16/README.md).
- [View báo cáo tài liệu](../../../DATASET/logs/phase7/task_7.16/final_phase7_report.json) — không phải benchmark GPU mới.
- [Catalog nguồn](../../../DATASET/logs/phase7/task_7.16/evidence_catalog.json) và [bảng giá trị nguồn](../../../DATASET/logs/phase7/task_7.16/source_values.csv).
- [Xác nhận người thật còn chờ](../../../DATASET/logs/phase7/task_7.16/human_signoff.json).
- [XML 103 case pipeline](../../../DATASET/logs/phase7/task_7.16/validation/targeted_tests.xml) và [XML 53 case component](../../../DATASET/logs/phase7/task_7.16/validation/component_tests.xml).

Đây là hồ sơ lịch sử và kết quả kiểm tra tại thời điểm tạo. Không xem manifest kiểm tài liệu cũ là đã tự kiểm mọi thay đổi tài liệu về sau, bao gồm thư mục Markdown mới này.

## 3. Pilot train vòng 1

- [Báo cáo pilot](../../../DATASET/logs/phase7/improvement_round1/runs/pilot_20261109_182359/README.md).
- [Verification](../../../DATASET/logs/phase7/improvement_round1/runs/pilot_20261109_182359/analysis/verification.json).
- [Import manifest](../../../DATASET/logs/phase7/improvement_round1/runs/pilot_20261109_182359/import_manifest.json).
- [Notebook pilot](../../../notebooks/task/phase7_improvement_round1/phase7_improvement_round1_colab.ipynb).

```text
Run: pilot_20261109_182359
Code: 26650142e76f790c0642c091a05dfc2dc3770cef
Ứng viên: p0_legacy_full, epoch 5
P0 source SHA-256:
9d97613c5f271801a30ff514326557f262a372abeb4d52ce864c78f20fc1069b
Drive:
MyDrive/Fatformer/phase7/improvement_round1/pilot_20261109_182359/p0_legacy_full/best.pth
```

27 file raw nằm trong thư mục `raw/` của run. Báo cáo pilot có các dòng “chưa có evaluation” là trạng thái tại thời điểm đó; evaluation đã hoàn tất ở mục4, không dùng các dòng lịch sử làm trạng thái hiện hành.

## 4. Matched evaluation vòng 1

- [Báo cáo hậu kiểm](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/README.md).
- [Gates hai model](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/gates.csv).
- [Metrics tái tính](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/recomputed_metrics.json).
- [So sánh paired](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/comparison.csv) và [CI verification](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/paired_verification.json).
- [Tái lập dự đoán incumbent](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/incumbent_reproduction.json).
- [Quyết định](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/decision.json).
- [Latency](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/latency.csv).
- [XML 25 test](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/validation/local_tests.xml).
- [Notebook evaluation](../../../notebooks/task/phase7_pilot_evaluation/phase7_pilot_evaluation_colab.ipynb).

```text
Run: p0_eval_20261110_065901
Code: 977b282852120a7c5d1cb5d3cc110955edf00ce7
P0 package SHA-256:
e26a16273d3da6f851a4fe41c994307b585339e9f5f7f9ac58c58a3621c863e2
Incumbent refit package SHA-256:
de36061f4cc10e8b5a7b599842c160bfc5109e2a29bef2d659cf77388aaf1a5e
Drive:
MyDrive/Fatformer/phase7/improvement_round1/evaluation/p0_eval_20261110_065901/
  candidate_p0/package/unified.pth
  incumbent_refit/package/unified.pth
```

Nguồn tiếp nhận: `wait/evaluation_results_without_weights/`, 90 file raw. Không đổi hai package đánh giá trên thành package Task 7.15 hiện hành.

## 5. Validation diagnostics vòng 2

- [Báo cáo hậu kiểm và đính chính](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/README.md).
- [Verification](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/analysis/verification.json).
- [Kiểm chứng diễn giải generator/phân bố điểm](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/analysis/interpretation.json).
- [Báo cáo tái tính](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/analysis/recomputed/).
- [XML 25 test](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/analysis/validation/local_tests.xml).
- [Notebook diagnostics đã nộp](../../../notebooks/task/phase7_validation_diagnostics/phase7_validation_diagnostics_colab.ipynb).

```text
Run: val_diag_20261110_142828
Code: 0dabd725e5b7a351514a9b7ee34f4bd6014b584f
Nguồn tiếp nhận: wait/validation_diagnostics_without_weights/
Drive:
MyDrive/Fatformer/phase7/improvement_round2/diagnostics/val_diag_20261110_142828/
```

16 file raw được giữ nguyên. Các đoạn “chưa fit calibration mới” trong README diagnostics là trạng thái trước run ở mục6, không phải trạng thái cuối hiện tại. Notebook có output/bản sao evidence có thể chỉ lưu local; không mặc định đã push Git.

## 6. Calibration transfer vòng 2

- [Báo cáo chi tiết](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/README.md).
- [Verification](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/analysis/verification.json).
- [P0 summary.csv](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/raw/candidate_p0/comparison/summary.csv).
- [Incumbent summary.csv](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/raw/incumbent/comparison/summary.csv).
- [Báo cáo tái tính](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/analysis/recomputed/).
- [Quyết định không thay deployment](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/analysis/decision.json).
- [Định hướng sau kết quả](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/analysis/post_result_plan.json).
- [Import manifest](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/import_manifest.json).
- [XML 30 test audit/transfer](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/analysis/validation/local_tests.xml).
- [Hướng dẫn calibration notebook](../../../notebooks/task/phase7_calibration_transfer/README.md) và [template notebook](../../../notebooks/task/phase7_calibration_transfer/phase7_calibration_transfer_colab.ipynb).

```text
Run: calib_transfer_20261110_155948
Code: 253942a5262432fbed7ce7ca8b12b4e455bfc2ce
Nguồn tiếp nhận: wait/calibration_transfer_without_weights/
Log đã sắp xếp:
DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/
```

56 file raw được giữ nguyên. Run hoàn tất theo kết quả/source/metadata; notebook thực thi chưa được nộp để đối chiếu cell/output. Không biến template thành notebook đã chạy. NPZ có thể không được Git theo dõi; dùng gói local/Drive khi cần kiểm arrays. Không có package triển khai mới từ run này.

## 7. Cách dùng khi viết báo cáo

1. Dùng [file 01](01_bao_cao_tong_hop.md) làm phần nội dung chính, chọn bảng cần thiết từ [file 02](02_bang_so_lieu.md).
2. Kèm tên run, dữ liệu, điều kiện, metric, ngưỡng và đơn vị khi chép số. Ghi package hash rút gọn trong phần trình bày, hash đầy đủ ở phụ lục.
3. Dẫn đến CSV/JSON tương ứng khi cần độ chính xác hoặc tính chênh lệch. Không tính từ số làm tròn hai chữ số nếu đã có giá trị gốc.
4. Dùng [file 03](03_quy_trinh_va_kiem_chung.md) cho phương pháp và phạm vi kiểm chứng, [file 04](04_ket_luan_va_huong_tiep_theo.md) cho thảo luận/kết luận.
5. Khi có run mới, tạo mục kết quả mới với trạng thái hậu kiểm riêng; không sửa raw hoặc tự thay số của model chuẩn. Chỉ cập nhật kết luận lựa chọn khi có bằng chứng chấp nhận tương ứng.
