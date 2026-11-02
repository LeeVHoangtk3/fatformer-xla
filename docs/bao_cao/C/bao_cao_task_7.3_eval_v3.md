# BÁO CÁO NGHIỆM THU TASK 7.3
## Evaluation v3 — khế ước đo lường trên CFE

> **Trạng thái**: `[x] ĐÃ NGHIỆM THU ĐO THẬT`
> **Môi trường**: NVIDIA L4 (Google Colab)
> **Dữ liệu**: CommunityForensics-Eval, 2.000 ảnh
> **Commit sinh số đo**: `851be5ec5e0bab73413deb202290a8a514bac0fc`
> **Evidence level**: `measured`

## 1. Phạm vi nghiệm thu

Evaluation v3 đã thay thế đường đo cũ bằng một khế ước thống nhất:

- Nạp checkpoint qua canonical loader, bắt buộc `missing=0`, `unexpected=0` và `partial=false`.
- Không dùng `strict=False`, không lọc key theo shape và không ép số adapter block.
- Không dùng ngưỡng phụ thuộc thành phần batch (`adaptive_thresh`, `mean_w`).
- Báo cáo AUC, AP, TPR tại FPR 1%/5%/10%, ECE, NLL và bootstrap CI 95%.
- Ghi đầy đủ provenance: checkpoint SHA-256, Git commit, `degrade_order` và trạng thái nạp.

Artifact chính:

- [`tools/eval_communityforensics.py`](../../../tools/eval_communityforensics.py)
- [`eval_v3_baseline_clean.json`](../../../DATASET/logs/phase7/eval_v3_baseline_clean.json)
- [`eval_v3_baseline_clean.md`](../../../DATASET/logs/phase7/eval_v3_baseline_clean.md)

## 2. Kết quả đo thật

| Mô hình | Điều kiện | ROC-AUC (CI 95%) | AP (CI 95%) | TPR@FPR=5% | Fake@Real90 | Canonical load |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | Clean | **76.20%** [73.97, 78.49] | 78.20% [75.78, 80.96] | 30.75% | 44.25% | `ok (0/0, n=3)` |
| Baseline | Q30 | **42.33%** [39.58, 44.99] | 50.48% [47.58, 53.82] | 9.75% | 14.88% | `ok (0/0, n=3)` |
| Anchored | Clean | **79.25%** [77.00, 81.43] | 72.77% [69.37, 77.07] | 12.00% | 38.38% | `ok (0/0, n=3)` |
| Anchored | Q30 | **45.66%** [42.72, 48.36] | 47.96% [44.86, 51.34] | 5.75% | 9.25% | `ok (0/0, n=3)` |
| WiSE-FT v2 | Clean | **81.78%** [79.88, 83.74] | 81.63% [79.22, 84.27] | 34.75% | 48.88% | `ok (0/0, n=3)` |
| WiSE-FT v2 | Q30 | **43.67%** [41.04, 46.24] | 49.71% [46.79, 53.06] | 6.75% | 12.25% | `ok (0/0, n=3)` |

Tổng cộng đã sinh 30 cặp báo cáo JSON/Markdown đo thật: 6 báo cáo cho ba cấu hình chính và 24 báo cáo cho 12 ứng viên WiSE-FT trên Clean/Q30.

## 3. Kiểm tra chất lượng

- [x] JSON dùng `report_schema="v3"` và `evidence_level="measured"`.
- [x] Sáu báo cáo chính sử dụng 2.000 ảnh CFE.
- [x] `degrade_order="post_resize"` được ghi trong JSON và Markdown.
- [x] Canonical loader trả `0/0 keys` cho các checkpoint hợp lệ.
- [x] Có TPR@FPR 1%/5%/10%, ECE, NLL và bootstrap CI 95%.
- [x] Ba ca kiểm thử toán học qua `--test_dummy` đạt; chỉ dùng để kiểm tra engine, không dùng làm benchmark.
- [x] Không còn artifact 40 ảnh dummy được trình bày như kết quả đo thật.

## 4. Kết luận

Task 7.3 đạt nghiệm thu đo thật. Harness v3 là nguồn đo chính thức cho Task 7.4–7.6. Kết quả Task 7.6 cho thấy G7.1 không đạt, nhưng điều đó không làm mất hiệu lực của khế ước đo lường.
