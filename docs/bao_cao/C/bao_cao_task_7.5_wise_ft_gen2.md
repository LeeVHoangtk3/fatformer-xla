# BÁO CÁO NGHIỆM THU TASK 7.5
## WiSE-FT Gen 2 — quét lưới alpha × beta_srm

> **Trạng thái**: `[x] ĐÃ NGHIỆM THU ĐO THẬT`
> **Lưới**: 12/12 ứng viên
> **Winner**: `wise_ft_v2_a55_b35`
> **SHA-256**: `eb170fd62700b3613124b8c1519bc796c659ba1e4733be9abb1fb9f5801e429c`
> **Commit sinh checkpoint và số đo**: `851be5ec5e0bab73413deb202290a8a514bac0fc`

## 1. Phương pháp

WiSE-FT Gen 2 chỉ hợp nhất hai checkpoint biểu diễn cùng topology adapter `n=3`, indices `{7, 15, 23}`:

- Baseline: 1.116 keys.
- Anchored: 1.200 keys.
- Keys chung được nội suy theo `alpha`.
- Nhánh SRM đầu ra được co theo `beta_srm`; fixed buffers được giữ nguyên.
- Mọi ứng viên phải qua canonical verification với `missing=0`, `unexpected=0`, `partial=false`.

Lưới đo:

- `alpha ∈ {0.35, 0.45, 0.55, 0.65}`.
- `beta_srm ∈ {0.25, 0.35, 0.50}`.

Artifact:

- [`tools/wise_ft_merge.py`](../../../tools/wise_ft_merge.py)
- [`tools/sweep_wise_ft_candidates.py`](../../../tools/sweep_wise_ft_candidates.py)
- [`wise_ft_v2_sweep.csv`](../../../DATASET/logs/phase7/wise_ft_v2_sweep.csv)
- [`fatformer_wise_ft_v2_best_metadata.json`](../../../DATASET/checkpoints/phase7/fatformer_wise_ft_v2_best_metadata.json)

## 2. Kết quả quét lưới đo thật

| alpha | beta_srm | Real Clean | Fake@Real90 Clean | AUC Clean | Real Q30 | Fake@Real90 Q30 | AUC Q30 | Chọn |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0.35 | 0.25 | 94.50% | 47.88% | 80.14% | 100.00% | 13.38% | 43.06% | — |
| 0.35 | 0.35 | 94.50% | 48.88% | 80.70% | 100.00% | 14.12% | 43.01% | — |
| 0.35 | 0.50 | 94.25% | 49.00% | 81.58% | 100.00% | 14.00% | 42.96% | — |
| 0.45 | 0.25 | 94.75% | 47.50% | 80.71% | 100.00% | 12.38% | 43.34% | — |
| 0.45 | 0.35 | 94.62% | 48.25% | 81.28% | 100.00% | 12.62% | 43.29% | — |
| 0.45 | 0.50 | 94.38% | 48.38% | 82.14% | 100.00% | 13.25% | 43.24% | — |
| 0.55 | 0.25 | 95.00% | 47.62% | 81.23% | 100.00% | 11.88% | 43.68% | — |
| **0.55** | **0.35** | **94.88%** | **48.88%** | **81.78%** | **100.00%** | **12.25%** | **43.67%** | **SELECTED** |
| 0.55 | 0.50 | 94.50% | 48.75% | 82.62% | 100.00% | 12.75% | 43.56% | — |
| 0.65 | 0.25 | 95.50% | 46.62% | 81.66% | 100.00% | 10.38% | 44.26% | — |
| 0.65 | 0.35 | 95.50% | 47.62% | 82.20% | 100.00% | 11.38% | 44.21% | — |
| 0.65 | 0.50 | 95.12% | 47.88% | 82.97% | 100.00% | 11.25% | 44.19% | — |

## 3. Checkpoint chiến thắng

- File: `DATASET/checkpoints/phase7/fatformer_wise_ft_v2_best.pth`.
- Cấu hình: `alpha=0.55`, `beta_srm=0.35`.
- SHA-256: `eb170fd62700b3613124b8c1519bc796c659ba1e4733be9abb1fb9f5801e429c`.
- Dung lượng: `2208.28 MB`.
- Layout: `n=3`, indices `{7, 15, 23}`, `partial=false`.
- Verification: `missing=0`, `unexpected=0`, status `ok`.

Ứng viên được chọn theo quy tắc Pareto đã khóa của sweep, không phải vì đạt giá trị cực đại riêng lẻ ở mọi cột. Checkpoint WiSE-FT Gen 1 không được ghi đè và không được dùng làm kết quả hiện hành.

## 4. Kiểm tra chất lượng

- [x] 12/12 ứng viên có `evidence_level=measured`.
- [x] 12/12 ứng viên qua verify `0/0` với 1.200 keys.
- [x] CSV có đúng một dòng `selected=True`.
- [x] Metadata winner khớp alpha, beta, SHA-256 và commit nguồn.
- [x] `tests/test_wise_ft_merge_verify.py`: 2/2 pass.
- [x] `tools/wise_ft_merge.py --test_dummy`: pass.
- [x] Không dùng bảng synthetic để tuyển chọn.

## 5. Kết luận và liên hệ G7.1

Task 7.5 đạt nghiệm thu đo thật. Winner `wise_ft_v2_a55_b35` được chuyển sang Task 7.6 làm checkpoint trước huấn luyện. Task 7.6 xác định **G7.1 FAIL** vì Q30 chưa đạt ngưỡng; đây là kết quả năng lực của mô hình, không phải lỗi của quy trình sweep.
