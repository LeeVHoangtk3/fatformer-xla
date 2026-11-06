# BÁO CÁO NGHIỆM THU MA TRẬN 4 CẤU HÌNH TRÊN CFE & PHÁN QUYẾT G7.1
> **Evidence level**: `measured` | **Gate verdict**: `FAIL`

## MA TRẬN TÁI LẬP MỐC SÀN 4 CẤU HÌNH TRÊN COMMUNITYFORENSICS-EVAL
> **Dự án**: FatFormer-XLA | **Nhiệm vụ**: Task 7.6 (Phase 7 — D4)
> **Git Commit**: `851be5ec5e0bab73413deb202290a8a514bac0fc` | **Khung đo lường**: `eval_communityforensics.py` (v3.0)
> **Thứ tự suy thoái**: `post_resize` | **Cấu hình hiệu chuẩn**: $T=1.6402, T^*=0.9127$

---

## 1. Ma Trận Đối Đầu 4 Cấu Hình (4 Models × 2 Conditions × 6 Metrics)

### 1.1. Điều kiện Ảnh Sạch Phòng Lab (Clean CFE)

| Cấu hình | Trạng thái nạp | ROC-AUC (Clean) | AP (Clean) | TPR@FPR=1% | TPR@FPR=5% | **Fake@Real90 (Clean)** | Real ACC ($T^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | `ok (0/0)` | 76.20% | 78.20% | 20.25% | 30.75% | **44.25%** | 95.38% |
| **`anchored`** | `ok (0/0)` | 79.25% | 72.77% | 1.50% | 12.00% | **38.38%** | 87.25% |
| **`wise_ft_v2` (Gen 2)** | `ok (0/0)` | **81.78%** | **81.63%** | **18.38%** | **34.75%** | **48.88%** | **94.88%** |
| **`ensemble` (1+2)** | `ok (2 models)` | 80.71% | 76.97% | 2.50% | 31.37% | **42.75%** | 95.00% |

> **Khoảng tin cậy 95% CI (Clean)**:
> - `wise_ft_v2`: AUC `[79.88%, 83.74%]`, Fake@Real90 `[42.49%, 55.2%]`
> - `ensemble`: AUC `[78.51%, 82.76%]`, Fake@Real90 `[38.24%, 51.62%]`

### 1.2. Điều kiện Nén Sâu Mạng Xã Hội (JPEG Q=30 CFE)

| Cấu hình | Trạng thái nạp | ROC-AUC (Q30) | AP (Q30) | TPR@FPR=1% | TPR@FPR=5% | **Fake@Real90 (Q30)** | Real ACC ($T^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | `ok (0/0)` | 42.33% | 50.48% | 4.38% | 9.75% | **14.88%** | 100.00% |
| **`anchored`** | `ok (0/0)` | 45.66% | 47.96% | 0.75% | 5.75% | **9.25%** | 99.38% |
| **`wise_ft_v2` (Gen 2)** | `ok (0/0)` | **43.67%** | **49.71%** | **3.75%** | **6.75%** | **12.25%** | **100.00%** |
| **`ensemble` (1+2)** | `ok (2 models)` | 46.10% | 49.10% | 1.50% | 6.62% | **10.00%** | 100.00% |

> **Khoảng tin cậy 95% CI (Q30)**:
> - `wise_ft_v2`: AUC `[41.04%, 46.24%]`, Fake@Real90 `[8.32%, 15.6%]`
> - `ensemble`: AUC `[43.14%, 48.88%]`, Fake@Real90 `[7.5%, 12.46%]`

---

## 2. Bảng Đối Chiếu Với Báo Cáo Ngày 26/10 & Phân Tích Nguyên Nhân (§5.3)

Chỉ tính chênh lệch trực tiếp khi hai phía dùng cùng metric. `Fake ACC` tại một
ngưỡng cố định và `Fake@Real90` là hai đại lượng khác nhau, nên không được trừ
cho nhau hoặc diễn giải thành mức cải thiện.

| Cấu hình | Điều kiện | Metric cùng định nghĩa | Số 26/10 | Harness v3 | Chênh lệch | Giải thích |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **`baseline`** | Clean | ROC-AUC | 76.21% | 76.20% | -0.01 pp | Gần như tái lập; sai khác trong nhiễu bootstrap/làm tròn. |
| **`anchored`** | Clean | ROC-AUC | 79.28% | 79.25% | -0.03 pp | Gần như tái lập; loader và tập dữ liệu nhất quán. |
| **`baseline`** | Q30 | ROC-AUC | 50.00% | 42.33% | -7.67 pp | Harness v3 cố định `post_resize`, seed và eval split; số mới cho thấy thứ hạng dưới nén thấp hơn mức ngẫu nhiên. |
| **`anchored`** | Q30 | ROC-AUC | 50.07% | 45.66% | -4.41 pp | Khác protocol/ngưỡng không giải thích AUC; kết quả v3 xác nhận không có năng lực phân hạng bền vững dưới Q30. |
| **`anchored`** | Clean | Real ACC | 48.10% | 87.25% | +39.15 pp | Calibration held-out thay ngưỡng tay `0.40`, giảm mạnh false positive trên ảnh Real. |
| **`anchored`** | Q30 | Real ACC | 58.40% | 99.38% | +40.98 pp | Cùng tác động thay ngưỡng; Real ACC tăng không đồng nghĩa AUC/Fake@Real90 tăng. |
| **`wise_ft` Gen 1 → Gen 2** | Clean/Q30 | — | `invalid_layout` | `0/0 keys` | Không so trực tiếp | Gen 1 nạp thiếu adapter nên bị rút tư cách làm baseline; Gen 2 là checkpoint khác và phải báo số độc lập. |

Các số `Fake ACC` ngày 26/10 chỉ dùng để truy vết lịch sử. Mốc v3 chính thức
dùng ROC-AUC, TPR@FPR và Fake@Real90 trên eval split để tránh kết luận sai do
dịch ngưỡng.

---

## 3. Phán Quyết Cổng Chất Lượng G7.1 (§5.4)

- **Cấu hình xuất sắc nhất**: **`wise_ft_v2`**
- **Fake@Real90 (JPEG Q=30)**: **12.25%** (Yêu cầu Must: $\ge 55\%$, Stretch: $\ge 70\%$)
- **AUC (Clean)**: **81.78%** (Yêu cầu Must: $\ge 80\%$)
- **AUC (JPEG Q=30)**: **43.67%** (Yêu cầu Must: $\ge 55\%$)

### KẾT LUẬN:
**CỔNG G7.1: [ ] FAIL**
*(Must: FAIL | Stretch: FAIL)*
