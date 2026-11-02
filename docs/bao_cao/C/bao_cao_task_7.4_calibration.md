# BÁO CÁO NGHIỆM THU TASK 7.4
## Hiệu chuẩn nhiệt độ và ngưỡng FPR — zero data leakage

> **Trạng thái**: `[x] ĐÃ NGHIỆM THU ĐO THẬT`
> **Calibration split**: 400 ảnh (200 Real, 200 Fake)
> **Evaluation split**: 1.600 ảnh độc lập
> **Evidence level**: `measured`

## 1. Khế ước hiệu chuẩn

- Temperature scaling tối ưu NLL trên calibration split.
- Ngưỡng `T*` được chọn riêng tại FPR mục tiêu 1%, 5% và 10%.
- Calibration và evaluation sử dụng sample ID không giao nhau.
- Không fit nhiệt độ hoặc ngưỡng trên tập evaluation.
- Task này không chạy cross-validation; trường `cv` trong cả hai artifact bằng `null`.

Artifact:

- [`tools/calibrate_and_select.py`](../../../tools/calibrate_and_select.py)
- [`baseline.json`](../../../DATASET/logs/phase7/calibration/baseline.json)
- [`anchored.json`](../../../DATASET/logs/phase7/calibration/anchored.json)
- [`summary.md`](../../../DATASET/logs/phase7/calibration/summary.md)

## 2. Kết quả đo thật

| Mô hình | T | T* tại FPR 5% | ECE trước | ECE sau | NLL trước | NLL sau | Real ACC | Fake ACC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | **1.6402** | **0.9127** | 34.00% | **33.54%** | 3.4835 | **2.1392** | 95.00% | 34.00% |
| Anchored | **1.4336** | **0.9664** | 14.39% | **11.02%** | 0.7868 | **0.6659** | 95.00% | 21.50% |

### Ngưỡng theo FPR

| Mô hình | T*@1% | FPR đo | T*@5% | FPR đo | T*@10% | FPR đo |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.9990 | 1.00% | 0.9127 | 5.00% | 0.0258 | 10.00% |
| Anchored | 0.9844 | 1.00% | 0.9664 | 5.00% | 0.9375 | 10.00% |

## 3. Toàn vẹn dữ liệu

- [x] Mỗi calibration artifact ghi `calibration_source` và SHA-256 nguồn logits.
- [x] Mỗi split có 400 mẫu cân bằng 200/200.
- [x] 8/8 kiểm thử trong `tests/test_phase7_acceptance.py` xác nhận quy tắc phân vùng và không chồng lấn sample ID.
- [x] Evaluation v3 tiêu thụ tham số cố định từ calibration JSON, không suy ra ngưỡng từ batch đang chấm.
- [x] Reliability diagram được sinh cho Baseline và Anchored.

## 4. Diễn giải

Temperature scaling giảm NLL và ECE cho cả hai mô hình. Mức giảm ECE của Baseline nhỏ (`0.46 pp`), vì vậy không được mô tả là cải thiện mạnh. Các ngưỡng trên chỉ hợp lệ với checkpoint và calibration source có SHA-256 tương ứng trong artifact.

## 5. Kết luận

Task 7.4 đạt nghiệm thu đo thật với calibration/evaluation tách biệt. Không có tuyên bố 5-fold CV trong nghiệm thu này.
