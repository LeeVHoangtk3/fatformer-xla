# BẢNG TỔNG HỢP HIỆU CHUẨN ĐO THẬT — TASK 7.4
> **Evidence level**: `measured` | Nguồn: held-out calibration split

| Model | Nguồn fit | n_calib | T | T* (FPR=5%) | T* (FPR=1%) | ECE trước | ECE sau | Real ACC | Fake ACC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchored_v2` | `logits_anchored_v2_calibration_400.npz` | 400 | 1.4670 | 0.9274 | 0.9728 | 14.29% | 11.56% | 95.00% | 58.50% |

> Kiểm tra partition_validation trong từng JSON để xác nhận phân vùng. Kết quả sửa sau hậu kiểm được ghi rõ trong recovery; không coi là phép thử mù mới.
