# BẢNG TỔNG HỢP HIỆU CHUẨN ĐO THẬT — TASK 7.4
> **Evidence level**: `measured` | Nguồn: held-out calibration split

| Model | Nguồn fit | n_calib | T | T* (FPR=5%) | T* (FPR=1%) | ECE trước | ECE sau | Real ACC | Fake ACC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchored` | `logits_anchored_clean_calib_split.npz` | 400 | 1.4336 | 0.9664 | 0.9844 | 14.39% | 11.02% | 95.00% | 21.50% |
| `baseline` | `logits_baseline_clean_calib_split.npz` | 400 | 1.6402 | 0.9127 | 0.9990 | 34.00% | 33.54% | 95.00% | 34.00% |

> Kiểm tra partition_validation trong từng JSON để xác nhận phân vùng. Kết quả sửa sau hậu kiểm được ghi rõ trong recovery; không coi là phép thử mù mới.

Anchored v2: [Task 7.15 current calibration](../task_7.15/current/calibration/anchored_v2.json). Legacy v2 results are archived under ../task_7.15/archive/.
