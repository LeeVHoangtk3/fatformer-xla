# BẢNG TỔNG HỢP HIỆU CHUẨN ĐO THẬT — TASK 7.4
> **Evidence level**: `measured` | Nguồn: held-out calibration split

| Model | Nguồn fit | n_calib | T | T* (FPR=5%) | T* (FPR=1%) | ECE trước | ECE sau | Real ACC | Fake ACC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchored_v2` | `logits_anchored_v2_clean_calib.npz` | 400 | 2.7864 | 0.7953 | 0.8680 | 14.72% | 6.06% | 95.00% | 59.00% |

> Kiểm tra partition_validation trong từng JSON để xác nhận phân vùng. Kết quả sửa sau hậu kiểm được ghi rõ trong recovery; không coi là phép thử mù mới.
