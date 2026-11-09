# BÁO CÁO KIỂM CHUẨN KHÓA SỔ PHASE 7 (FINAL LOCKED BENCHMARK)

> **Mã nhiệm vụ**: Task 7.14 (Phase 7 — D9–D10)  
> **Phiên bản báo cáo**: `report_schema: v3` (Evidence: measured)  
> **Đối tượng chấm**: `fatformer_anchored_v2_best.pth` (SHA-256: `5cd49252ae941897...`)  
> **Operating point**: frozen_reference_operating_point ($T = 1.6402$, $T^* = 0.9127$)

---

## 1. Phán Quyết Hai Cổng Chặn G7.2 & G7.3 (Master Plan §II.3)

### 🚧 Cổng G7.2 — Năng Lực Thật & Độ Bền Ngoại Vi
- Phán quyết: **FAIL (Hợp lệ theo Điều khoản kết quả âm §II.3)**

| Tiêu chuẩn đo lường | Mục tiêu Must | Kết quả đo thực tế | Trạng thái |
| :--- | :---: | :---: | :---: |
| **CFE Clean: Fake @ Real = 90%** | $\ge 65.0\%$ | **64.12%** | ❌ FAIL |
| **CFE Clean: ROC-AUC** | $\ge 78.0\%$ | **88.67%** | ✅ PASS |
| **CFE Q30: ROC-AUC** | $\ge 60.0\%$ | **79.87%** | ✅ PASS |
| **CFE Q30: Fake @ Real = 90%** | $\ge 40.0\%$ | **42.62%** | ✅ PASS |
| **18 Tập In-Distribution: Macro ACC** | $\ge 90.0\%$ | **90.46%** | ✅ PASS |
| **18 Tập In-Distribution: Real ACC** | $\ge 90.0\%$ (cứng) | **90.67%** | ✅ PASS |
| **Worst-subset Fake ACC (CFE)** | $\ge 45.0\%$ | **28.40%** | ❌ FAIL |
| **ECE sau hiệu chuẩn (Q30)** | $\le 0.08$ | **0.1349** | ❌ FAIL |

### 🚧 Cổng G7.3 — Hợp Nhất & Hiệu Năng Vận Hành
- Phán quyết: **CHƯA ĐẠT (Độ trễ overhead và delta chưa được kiểm chứng đầy đủ)**
- **Độ trễ backbone trên GPU**: `44.40 ms/ảnh`; chưa đo overhead package trên GPU thực tế, không dùng số này để chấm mục tiêu độ trễ thêm.
- **Kiến trúc mô hình**: Một checkpoint duy nhất, một lần `forward()`, không cần metadata điều kiện ngoại vi.

---

## 2. Kết Quả Đo 18 Tập In-Distribution Chuẩn Hóa
- **Tổng số ảnh đo đạc**: 82,003
- **Macro ACC**: 90.46% | **Micro ACC**: 93.43%
- **Real ACC**: 90.67% | **Fake ACC**: 96.19%
- **ROC-AUC**: 98.28%

---

## 3. Độ Bền Trước 6 Biến Thể Suy Biến Vật Lý (CFE)
| Điều kiện suy biến | Mô tả kỹ thuật | ROC-AUC (%) | Fake@Real90 (%) | Macro ACC (%) |
| :--- | :--- | :---: | :---: | :---: |
| `Clean` | Chuẩn mốc sàn phòng lab | 88.67 | 64.12 | 73.75 |
| `q75` | Biến thể Q75 | 83.31 | 48.75 | 66.19 |
| `q50` | Biến thể Q50 | 81.55 | 47.62 | 66.31 |
| `q30` | Biến thể Q30 | 79.87 | 42.62 | 65.31 |
| `webp40` | Biến thể WEBP40 | 79.00 | 41.88 | 65.94 |
| `blur15` | Biến thể BLUR15 | 84.22 | 50.62 | 71.00 |
| `downup05` | Biến thể DOWNUP05 | 87.00 | 64.75 | 77.38 |

---

## 4. Ba Câu Kết Luận Chung Cuộc
1. **Mô hình có tốt hơn Baseline có ý nghĩa thống kê không?** Chưa có đối chiếu paired cùng protocol đủ để kết luận. Số đo hiện tại và operating point được ghi ở trên; không suy diễn ưu thế thống kê từ một lần chạy.
2. **Miền dữ liệu nào còn yếu nhất?** Nén JPEG sâu chất lượng thấp ($Q=30$) và các mô hình sinh hình ảnh khuếch tán mới lạ ngoài miền huấn luyện (OOD) vẫn là thử thách lớn nhất.
3. **Có đạt chuẩn G7.2 / G7.3 không?** Xem từng tiêu chí và phán quyết trong JSON. G7.3 còn chờ kiểm chứng package, overhead trên GPU thực tế và delta theo điều kiện. Điều khoản kết quả âm không thay cho kiểm chứng phương pháp.
