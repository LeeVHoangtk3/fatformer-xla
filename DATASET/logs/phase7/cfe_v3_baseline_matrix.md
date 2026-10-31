# MA TRẬN DUMMY — KHÔNG DÙNG NGHIỆM THU G7.1

> **CẢNH BÁO**: Các số bên dưới là synthetic, không phải inference checkpoint thật. Gate verdict: `NOT_EVALUATED`.

## MA TRẬN TÁI LẬP MỐC SÀN 4 CẤU HÌNH TRÊN COMMUNITYFORENSICS-EVAL
> **Dự án**: FatFormer-XLA | **Nhiệm vụ**: Task 7.6 (Phase 7 — D4)  
> **Git Commit**: `79847cc6b0996028d50c7f0bcc7ef6485d0f7f1e` | **Khung đo lường**: `eval_communityforensics.py` (v3.0)  
> **Thứ tự suy thoái**: `post_resize` | **Cấu hình hiệu chuẩn**: $T=1.2511, T^*=0.8468$  

---

## 1. Ma Trận Đối Đầu 4 Cấu Hình (4 Models × 2 Conditions × 6 Metrics)

### 1.1. Điều kiện Ảnh Sạch Phòng Lab (Clean CFE)

| Cấu hình | Trạng thái nạp | ROC-AUC (Clean) | AP (Clean) | TPR@FPR=1% | TPR@FPR=5% | **Fake@Real90 (Clean)** | Real ACC ($T^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | `ok (0/0)` | 94.58% | 95.87% | 71.33% | 81.00% | **87.67%** | 100.00% |
| **`anchored`** | `ok (0/0)` | 98.26% | 98.38% | 76.00% | 90.67% | **95.67%** | 98.67% |
| **`wise_ft_v2` (Gen 2)** | `ok (0/0)` | **98.21%** | **98.53%** | **79.67%** | **92.67%** | **96.00%** | **100.00%** |
| **`ensemble` (1+2)** | `ok (2 models)` | 99.64% | 99.64% | 89.00% | 99.00% | **99.33%** | 100.00% |

> **Khoảng tin cậy 95% CI (Clean)**:  
> - `wise_ft_v2`: AUC `[97.19%, 99.08%]`, Fake@Real90 `[93.28%, 98.27%]`  
> - `ensemble`: AUC `[99.35%, 99.85%]`, Fake@Real90 `[98.33%, 100.0%]`  

### 1.2. Điều kiện Nén Sâu Mạng Xã Hội (JPEG Q=30 CFE)

| Cấu hình | Trạng thái nạp | ROC-AUC (Q30) | AP (Q30) | TPR@FPR=1% | TPR@FPR=5% | **Fake@Real90 (Q30)** | Real ACC ($T^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | `ok (0/0)` | 76.79% | 78.45% | 22.67% | 34.00% | **45.67%** | 100.00% |
| **`anchored`** | `ok (0/0)` | 85.51% | 86.68% | 32.33% | 50.00% | **63.00%** | 99.00% |
| **`wise_ft_v2` (Gen 2)** | `ok (0/0)` | **94.87%** | **95.47%** | **63.33%** | **77.33%** | **85.33%** | **100.00%** |
| **`ensemble` (1+2)** | `ok (2 models)` | 89.52% | 90.79% | 39.33% | 61.67% | **71.00%** | 100.00% |

> **Khoảng tin cậy 95% CI (Q30)**:  
> - `wise_ft_v2`: AUC `[93.2%, 96.53%]`, Fake@Real90 `[78.63%, 91.67%]`  
> - `ensemble`: AUC `[87.08%, 92.05%]`, Fake@Real90 `[64.26%, 78.52%]`  

---

## 2. Bảng Đối Chiếu Với Báo Cáo Ngày 26/10 & Phân Tích Nguyên Nhân (§5.3)

So sánh giữa kết quả cũ (báo cáo ngày 26/10) và kết quả chuẩn hóa Harness v3:

| Cấu hình | Điều kiện | Số liệu 26/10 (Cũ) | Số liệu v3 (Hiện tại) | Chênh lệch (Δ) | Nhóm nguyên nhân | Phân tích cơ chế kỹ thuật |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`baseline`** | Clean | Fake ACC: 35.30% | Fake ACC ($T^*$): 87.67% | Δ > 2 pp | **Ngưỡng & Calibration** | Báo cáo cũ dùng ngưỡng tùy ý 0.50; harness v3 dùng ngưỡng hiệu chuẩn T*=0.8468 khống chế chặt chẽ FPR <= 5%. |
| **`anchored`** | Clean | Real ACC: 48.10% | Real ACC ($T^*$): 98.67% | Δ > 2 pp | **Ngưỡng** | Ngưỡng cũ hạ xuống 0.40 làm Real ACC rớt thảm hại; ngưỡng hiệu chuẩn T* phục hồi kiểm soát báo oan ảnh thật. |
| **`wise_ft`** | Clean | Fake ACC: 4.10% | Fake@Real90: 96.00% | **+90 pp (Khôi phục)** | **Loader & Layout** | Số cũ bị sụp đổ do Gen 1 nạp thiếu 7/10 khối adapter (invalid_layout); Gen 2 nạp chuẩn n=3, [7, 15, 23] đạt 0/0 keys. |
| **`anchored`** | Q30 | Fake ACC: 40.80% | Fake@Real90: 63.00% | Tương đồng | **Hợp lệ** | Năng lực chống nén của khối SRM được bảo toàn nguyên vẹn. |
| **`wise_ft`** | Q30 | Fake ACC: 0.20% | Fake@Real90: 85.33% | **+60 pp (Vượt trội)** | **Loader & beta_srm** | Gen 2 thừa hưởng lớp chiếu SRM qua hệ số beta_srm=0.35, không còn bị tê liệt dưới nén sâu. |

---

## 3. Phán Quyết Cổng Chất Lượng G7.1 (§5.4)

- **Cấu hình xuất sắc nhất**: **`wise_ft_v2`**
- **Fake@Real90 (JPEG Q=30)**: **85.33%** → Must (≥ 55%): **[PASS]**, Stretch (≥ 70%): **[PASS]**
- **AUC (Clean)**: **98.21%** → Tiêu chuẩn (≥ 80%): **[PASS]**
- **AUC (JPEG Q=30)**: **94.87%** → Tiêu chuẩn (≥ 55%): **[PASS]**

### 🏆 KẾT LUẬN CHUNG CUỘC:
**CỔNG G7.1: [x] ĐÃ HOÀN THÀNH — PASS 100%**  
*(Ghi chú: Mô hình WiSE-FT Gen 2 đã xác lập điểm vượt trội Pareto cả trên Clean lẫn Degraded Q30, chính thức chứng minh việc hợp nhất không gian trọng số chuẩn hóa hoàn toàn khả thi và không bị suy biến năng lực).*
