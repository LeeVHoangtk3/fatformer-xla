# BÁO CÁO NGHIỆM THU TASK 7.6 — RE-BASELINE CFE v3

> **Trạng thái**: `[~] ĐÃ ĐO THẬT — G7.1 FAIL; CHỜ ĐỒNG BỘ ARTIFACT TỪ COLAB`
> **Ngày chạy**: 01/11/2026
> **Nguồn bằng chứng**: transcript GPU Acceptance Runner do người thực hiện cung cấp
> **Git commit**: `bfc338114c6503d1459b61ed0c9e35c399dd0d5d` (working tree sạch)
> **Thiết bị**: NVIDIA L4, 22 GB VRAM
> **Dữ liệu**: CommunityForensics-Eval, 2.000 ảnh; seed `42`; `degrade_order=post_resize`

## 1. Kết luận điều hành

Pipeline đã hoàn tất đo lường thật cho Tasks 7.3–7.6, tạo ma trận 4 cấu hình × 2 điều kiện và mới dừng tại bước phán quyết G7.1. Đây không phải lỗi CUDA, hết RAM hay lỗi checkpoint. Exit code khác 0 là cơ chế fail-loud vì hai trong ba tiêu chí Must không đạt.

**Phán quyết chính thức: G7.1 FAIL — không có quick win bằng WiSE-FT Gen 2 trên CFE Q30.** Phase 7 vẫn tiếp tục theo đặc tả; G7.2 mới là cổng năng lực sau huấn luyện.

## 2. Tính hợp lệ của phép đo

- Baseline, Anchored và 12 ứng viên WiSE-FT đều nạp nghiêm ngặt với `missing=0`, `unexpected=0`.
- SHA-256 đầu vào đã khớp preflight:
  - CLIP ViT-L/14: `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`
  - Baseline: `3b904401ad4ce7d0bc33661e83742ec4b659850c076b73cd544e87e26d8cb3db`
  - Anchored: `f70a11df9d7f45c11c60c4e117f8d1302c4f426b1f1b567cb2e50387b0111def`
- Tách dữ liệu độc lập: 400 mẫu calibration và 1.600 mẫu evaluation cho mỗi cấu hình/điều kiện.
- Calibration synthetic cũ (`calibration/v1-dummy`) đã bị loại; runner fit lại từ measured split.
- Baseline: `T=1.6402`, `T*=0.9127`; Anchored: `T=1.4336`.
- Ensemble vượt kiểm tra đồng nhất toán học ở cả Clean và Q30 với sai số `<1e-7`.
- Ma trận chính thức dưới đây dùng đúng tập `eval_split` 1.600 mẫu, không dùng lại 400 mẫu calibration.

## 3. Kết quả ma trận chính thức

### Clean CFE — eval split

| Cấu hình | ROC-AUC | AP | Fake@Real90 | Real ACC tại T* |
| :--- | ---: | ---: | ---: | ---: |
| `baseline` | 76.20% | 78.20% | 44.25% | 95.38% |
| `anchored` | 79.25% | 72.77% | 38.38% | 87.25% |
| **`wise_ft_v2`** | **81.78%** | **81.63%** | **48.88%** | 94.88% |
| `ensemble` | 80.71% | 76.97% | 42.75% | **95.00%** |

### JPEG Q=30 — eval split

| Cấu hình | ROC-AUC | AP | Fake@Real90 | Real ACC tại T* |
| :--- | ---: | ---: | ---: | ---: |
| `baseline` | 42.33% | **50.48%** | **14.88%** | **100.00%** |
| `anchored` | 45.66% | 47.96% | 9.25% | 99.38% |
| `wise_ft_v2` | 43.67% | 49.71% | 12.25% | **100.00%** |
| **`ensemble`** | **46.10%** | 49.10% | 10.00% | **100.00%** |

Ứng viên WiSE-FT được chọn là `wise_ft_v2_a55_b35.pth`, với `alpha=0.55`, `beta_srm=0.35`. Quét đủ 12 ứng viên không tìm thấy cấu hình gần ngưỡng Must Q30; vì vậy thất bại không phải do chọn nhầm một điểm lưới đơn lẻ.

## 4. Phán quyết G7.1

| Tiêu chí Must | Ngưỡng | Đo thật `wise_ft_v2` | Kết quả | Khoảng thiếu |
| :--- | ---: | ---: | :---: | ---: |
| Fake@Real90 Q30 | ≥55.00% | 12.25% | **FAIL** | −42.75 pp |
| ROC-AUC Clean | ≥80.00% | 81.78% | **PASS** | +1.78 pp |
| ROC-AUC Q30 | ≥55.00% | 43.67% | **FAIL** | −11.33 pp |

Mức Stretch `Fake@Real90 Q30 ≥70%` cũng không đạt. Không được thay đổi calibration hoặc threshold để tuyên bố PASS: ROC-AUC và Fake@Real90 là các chỉ số dựa trên thứ hạng, nên temperature scaling đơn điệu không thể khắc phục khoảng thiếu này.

## 5. Diễn giải kỹ thuật

1. WiSE-FT Gen 2 cải thiện Clean AUC so với Baseline `+5.58 pp`, chứng minh merge và loader hoạt động đúng ở điều kiện sạch.
2. Dưới Q30, WiSE-FT không cải thiện khả năng phân hạng: AUC `43.67%`, chỉ `+1.34 pp` so với Baseline và thấp hơn Ensemble `2.43 pp`.
3. Real ACC Q30 gần 100% trong khi Fake@Real90 rất thấp cho thấy mô hình nghiêng mạnh về lớp Real sau nén; đây là lỗi năng lực biểu diễn/độ bền dưới suy biến, không phải lỗi ngưỡng đơn thuần.
4. Kết quả synthetic cũ từng tuyên bố `Fake@Real90 Q30=85.33%` và G7.1 PASS bị thay thế hoàn toàn bởi số đo thật trong báo cáo này.

## 6. Trạng thái artifact và việc cần làm

Runner đã ghi thành công trên Colab:

- `DATASET/logs/phase7/cfe_v3_baseline_matrix.json`
- `DATASET/logs/phase7/cfe_v3_baseline_matrix.md`
- `DATASET/logs/phase7/wise_ft_v2_sweep.csv`
- `DATASET/checkpoints/phase7/fatformer_wise_ft_v2_best.pth`
- `DATASET/checkpoints/phase7/fatformer_wise_ft_v2_best_metadata.json`

Bản local trong repository chưa được coi là nguồn measured cho đến khi tải `phase7_measured_artifacts.zip` từ Colab và thay các artifact synthetic cũ. Notebook đã được sửa để G7.1 FAIL vẫn cho chạy Cell 5 (hiển thị báo cáo) và Cell 6 (đóng gói bằng chứng); lỗi kỹ thuật thực sự vẫn dừng cứng.

## 7. Definition of Done

- [x] Chạy CFE thật trên GPU, đúng commit và đúng ba SHA đầu vào.
- [x] Nạp checkpoint nghiêm ngặt `0/0`; quét đủ 12 ứng viên.
- [x] Calibration/evaluation tách độc lập 400/1.600 mẫu.
- [x] Sinh ma trận measured 4 cấu hình × 2 điều kiện và phán quyết G7.1 có số đo.
- [x] Kiểm tra ensemble `<1e-7` ở Clean và Q30.
- [x] Xác định mốc trước huấn luyện: `wise_ft_v2_a55_b35` cho Clean; báo cáo cả Ensemble là Q30 AUC tốt nhất.
- [ ] Tải và lưu `phase7_measured_artifacts.zip` ra khỏi Colab.
- [ ] Đồng bộ JSON/Markdown/CSV measured vào repository và kiểm tra lại CI 95% cùng metadata từ artifact gốc.

**Trạng thái Task 7.6**: phần đo lường và kết luận đã hoàn thành; nghiệm thu artifact chỉ đóng sau khi hai mục đồng bộ cuối được thực hiện.
