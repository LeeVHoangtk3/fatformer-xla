# BẢNG TỔNG HỢP HIỆU CHUẨN ĐO THẬT — TASK 7.4
> **Evidence level**: `measured` | Nguồn: held-out calibration split

| Model | Nguồn fit | n_calib | T | T* (FPR=5%) | T* (FPR=1%) | ECE trước | ECE sau | Real ACC | Fake ACC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchored_v2` | `logits_anchored_v2_calib_fp16.npz` | 400 | 2.7832 | 0.7919 | 0.8683 | 14.38% | 5.39% | 95.00% | 58.50% |

> Kiểm tra partition_validation trong từng JSON để xác nhận phân vùng. Kết quả sửa sau hậu kiểm được ghi rõ trong recovery; không coi là phép thử mù mới.
