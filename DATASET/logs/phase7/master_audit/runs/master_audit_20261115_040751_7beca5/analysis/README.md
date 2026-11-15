# Hậu kiểm master full data audit — 15/11/2026

Run **`master_audit_20261115_040751_7beca5`**, source commit **`1f39ca8fdc11890fa1daed8cf885506c4b79eba8`**. **Đã hoàn tất thực thi trong phạm vi danh mục: 8 archive, 37/37 job, 1.618.061 lượt dự đoán. G7.2 vẫn FAIL (6/8); G7.3 PENDING.** Không dùng trạng thái hoàn tất thực thi thay cho đạt mục tiêu chất lượng.

Hồ sơ đã đồng bộ vào [logs/master_audit](../../../../../../../DATASET/logs/phase7/master_audit/README.md): [raw run status](../../../../../../../DATASET/logs/phase7/master_audit/runs/master_audit_20261115_040751_7beca5/raw/run_status.json), [verification.json](../../../../../../../DATASET/logs/phase7/master_audit/runs/master_audit_20261115_040751_7beca5/analysis/verification.json), [manifest SHA-256](../../../../../../../DATASET/logs/phase7/master_audit/runs/master_audit_20261115_040751_7beca5/import_manifest.json). Nguồn nhận ban đầu vẫn được giữ nguyên tại `wait/master_audit_20261115_040751_7beca5`. Không sửa weights, calibration, threshold, raw report hoặc nhãn GPU. SQLite/NPZ được giữ local và loại khỏi Git; clone Git cần phục hồi chúng từ bộ kết quả gốc để tái tính metric từng ảnh.

## Phạm vi xác minh

- Đối chiếu 179 hash source/config/manifest/notebook với bytes trong Git commit thực tế; fingerprint và manifest hashes khớp.
- SQLite integrity đạt; count, tổng byte, nhãn, split và subset của đủ 8 archive khớp báo cáo. Có 944.327 bản ghi archive và 815.689 hash file phân biệt. Các biến thể, replay và real reference có thể phụ thuộc nhau; đây không phải số ảnh gốc độc lập.
- Đủ đúng 37 NPZ và job JSON, khớp checksum, package/archive SHA, member/ID/nhãn/thứ tự từ inventory. Tổng 1.618.061 lượt là số dự đoán trên các split/condition, không cộng vào một accuracy chung.
- Tính lại pooled/per-subset metric của 37 job và core18 ở đủ 7 điều kiện từ xác suất từng ảnh; accuracy và AUC còn được kiểm bằng phép tính trực tiếp. CSV 175 dòng khớp bản tái xuất.
- Score/label/threshold nhất quán; softmax tái tính từ logits nằm trong tolerance. G7.2 được tính lại đúng; median/p95/overhead latency khớp 200 cặp số đo thô mỗi batch.
- JUnit ghi **204 tests, 0 failures, 0 errors, 0 skips**, 12 file liên quan, không phải full repository suite.

Giới hạn hậu kiểm: không giải mã/forward lại toàn kho ảnh Drive ở local; SHA archive là bằng chứng trong run, không phải checksum mới đo lại từ Drive. Không bootstrap lại CI95 trong lần hậu kiểm này. Đây là post-hoc remeasurement, không phải phép thử mù mới.

## Định danh và protocol

- Package SHA-256: `248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`.
- Source checkpoint SHA-256: `5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c`.
- T = `2.7832491397857666`; threshold = `0.7918508648872375`; router tắt, fallback clean cố định.
- NVIDIA L4, Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, batch16; backbone autocast FP16, logits/softmax FP32, TF32 tắt. Không train và không refit calibration.

## Coverage archive

| Archive | Bản ghi ảnh | Trạng thái |
|---|---:|---|
| progan_train | 144,024 | completed |
| progan_val | 8,000 | completed |
| diffusion_staging | 3,600 | completed |
| data_v2 | 14,400 | completed |
| cfe | 2,000 | completed |
| test_gans | 90,329 | completed |
| test_diffusion | 20,000 | completed |
| test_degraded | 661,974 | completed |

Không gộp `train_select` thành một split mới: đó là view của train + val_select. CFE calibration 400 và evaluation 1.600 được giữ riêng; v2 train/val_select/val_calib giữ đúng manifest.

## CFE evaluation — 1.600 ảnh mỗi điều kiện

| Điều kiện | Accuracy % | Real đúng % | Fake đúng % | AUC % | Fake@Real90 % |
|---|---:|---:|---:|---:|---:|
| clean | 74.38 | 95.62 | 53.12 | 88.66 | 64.12 |
| q30 | 65.81 | 91.38 | 40.25 | 79.88 | 42.62 |
| q50 | 66.50 | 92.88 | 40.12 | 81.56 | 47.50 |
| q75 | 66.44 | 94.00 | 38.88 | 83.32 | 48.62 |
| webp40 | 66.31 | 89.00 | 43.62 | 79.01 | 41.75 |
| blur15 | 71.06 | 88.12 | 54.00 | 84.21 | 50.75 |
| downup05 | 77.56 | 87.88 | 67.25 | 86.99 | 64.88 |

Đối chiếu theo sample ID với GPU benchmark hiện hành Task7.15: **cả 7 điều kiện có max delta probability = 0, số nhãn thay đổi = 0**. Lượt này xác nhận lại kết quả; không phải cải thiện model hay nâng accuracy do chạy lại.

## Core18 — 82.003 lượt ảnh mỗi điều kiện

| Điều kiện | Macro balanced ACC % | Pooled ACC % | Real đúng % | Fake đúng % | AUC % |
|---|---:|---:|---:|---:|---:|
| clean | 90.10 | 93.46 | 90.41 | 96.52 | 98.41 |
| q30 | 71.42 | 70.91 | 85.45 | 56.37 | 79.67 |
| q50 | 74.29 | 74.15 | 87.67 | 60.63 | 82.97 |
| q75 | 78.37 | 78.88 | 89.15 | 68.60 | 87.38 |
| webp40 | 71.52 | 71.18 | 82.49 | 59.86 | 80.18 |
| blur15 | 78.12 | 80.61 | 81.69 | 79.52 | 89.43 |
| downup05 | 81.85 | 83.85 | 77.50 | 90.21 | 92.94 |

Clean pooled ACC 93,46% không thay macro balanced ACC 90,10%. Q30 pooled ACC 70,91%, Fake đúng 56,37% cho thấy suy giảm trước nén; không được dùng kết quả clean để mô tả mọi điều kiện. 5 supplemental không góp vào core18/G7.2.

## Các split chẩn đoán — không gộp vào test

| Archive/split | Điều kiện | N | Accuracy % | Real đúng % | Fake đúng % |
|---|---|---:|---:|---:|---:|
| progan_train/train | clean | 144,024 | 96.72 | 93.44 | 100.00 |
| progan_val/validation | clean | 8,000 | 95.33 | 90.65 | 100.00 |
| diffusion_staging/train | clean | 3,060 | 72.84 | 98.56 | 47.12 |
| diffusion_staging/val | clean | 540 | 73.33 | 99.26 | 47.41 |
| data_v2/train | clean | 12,240 | 82.55 | 99.25 | 62.46 |
| data_v2/val_calib | clean | 1,080 | 82.22 | 98.81 | 62.24 |
| data_v2/val_calib | q30 | 1,080 | 75.56 | 97.12 | 49.59 |
| data_v2/val_select | clean | 1,080 | 82.69 | 98.98 | 63.06 |
| data_v2/val_select | q30 | 1,080 | 75.46 | 97.29 | 49.18 |
| cfe/calibration | clean | 400 | 76.75 | 95.00 | 58.50 |

Những metric này đo model cố định trên dữ liệu train/validation/calibration, không chứng minh khả năng tổng quát hóa hoặc được dùng chọn lại threshold theo kết quả đã xem.

## Test degraded lưu sẵn — protocol lịch sử riêng

| Biến thể lưu sẵn | N | Accuracy % | Real đúng % | Fake đúng % | AUC % |
|---|---:|---:|---:|---:|---:|
| blur_s1 | 110,329 | 86.39 | 90.25 | 82.54 | 92.88 |
| blur_s2 | 110,329 | 77.09 | 84.26 | 69.93 | 85.56 |
| down_up | 110,329 | 83.27 | 84.48 | 82.05 | 90.88 |
| jpeg_q30 | 110,329 | 70.89 | 86.35 | 55.42 | 80.11 |
| jpeg_q50 | 110,329 | 79.16 | 86.29 | 72.04 | 87.44 |
| jpeg_q70 | 110,329 | 82.54 | 87.72 | 77.36 | 90.52 |

`condition=clean` trong các job này nghĩa là không tạo degradation thêm lần hai; ảnh vẫn là biến thể cũ theo `split`. Không gộp Q70/blur1/blur2 hoặc thứ tự xử lý cũ với CFE post-crop Q75/blur1.5.

## Gates và giới hạn còn mở

- **G7.2 đạt 6/8, FAIL tổng thể**. Clean Fake@Real90 = 64,125% <65%; worst CFE clean subset Fake ACC = 25/81 = 30,8642% <45%. Q30 Fake@Real90 đã đạt mốc40%, không phải một tiêu chí còn trượt.
- **G7.3 PENDING**: một backbone forward/batch và overhead GPU đạt; chưa có best-by-condition reference.
- G7.0/G7.1 **NOT_REEVALUATED trong run này**; không đóng trạng thái lịch sử bằng full-run của package cuối.
- Exact-byte overlap v2–CFE/clean test, v2 cross-split, CFE cross-split và xung đột nhãn đều không phát hiện vi phạm trong SQLite. **pHash và generator-disjoint audit toàn kho chưa chạy**; không gọi đây là chứng minh zero leakage 100%.
- Không có ground truth vị trí giả mạo hoặc nghiệm thu Gradio Colab UI trong run benchmark này; không dùng full-data inference để xác nhận chức năng định vị hay giám định pháp y.

## Latency mới trên L4

| Batch | Median ms/ảnh | FPS | Median overhead ms/ảnh |
|---:|---:|---:|---:|
| 1 | 46.1488 | 21.67 | 0.227768 |
| 8 | 9.6205 | 103.94 | -0.006121 |
| 16 | 9.7838 | 102.21 | -0.007638 |
| 32 | 10.6378 | 94.00 | 0.003779 |

Batch1 package 46,15 ms/ảnh và overhead 0,227768 ms/ảnh đều thuộc run mới này, không ghi đè số đo lịch sử 44,98/0,188055. Delta âm nhỏ ở batch8/16 được giữ đúng số đo, không chứng minh wrapper làm backbone nhanh hơn. Latency dùng tensor tổng hợp, không gồm đọc ảnh/queue/UI.

## Lưu ý độ chính xác số

So softmax float64 tái tính từ logits với score GPU FP32, chênh lớn nhất **2,4598×10⁻⁷**. Có **891 lượt** đổi nhãn trong phép tái tính float64 sát ngưỡng; đây không phải 891 lỗi nhãn nội tại của run. Nhãn GPU lưu sẵn khớp hoàn toàn phép so sánh score/threshold GPU; giữ chúng làm nguồn cho metric CUDA. Không tự thay nhãn bằng hậu kiểm CPU hoặc suy ra Windows CPU backbone phải cho cùng số đổi nhãn.

Kết luận: đã có bằng chứng hoàn tất danh mục full-data được cấu hình và metric nhất quán với dự đoán, không cần chạy lại toàn notebook chỉ vì lỗi tên ảnh trước đây. Model/package được giữ nguyên; độ phủ thực nghiệm tăng nhưng các phần chưa đạt và pending vẫn phải được công bố.
