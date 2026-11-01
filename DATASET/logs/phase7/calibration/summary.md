# BẢNG TỔNG HỢP HIỆU CHUẨN ĐO THẬT — TASK 7.4
> **Evidence level**: `measured` | Nguồn: held-out calibration split

| Model | Nguồn fit | n_calib | T | T* (FPR=5%) | T* (FPR=1%) | ECE trước | ECE sau | Real ACC | Fake ACC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchored` | `logits_anchored_clean_calib_split.npz` | 400 | 1.4336 | 0.9664 | 0.9844 | 14.39% | 11.02% | 95.00% | 21.50% |
| `baseline` | `logits_baseline_clean_calib_split.npz` | 400 | 1.6402 | 0.9127 | 0.9990 | 34.00% | 33.54% | 95.00% | 34.00% |

> `T*` được fit trên calibration split độc lập; không dùng evaluation split để chọn ngưỡng.
