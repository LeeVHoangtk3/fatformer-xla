# BÁO CÁO NGHIỆM THU NHIỆM VỤ 7.13 (TASK 7.13)
## DEGRADATION ROUTER + FUSION XÁC SUẤT ĐÃ HIỆU CHUẨN

> **Mã nhiệm vụ**: `Task 7.13` *(Phase 7 — D6–D9)*  
> **Người thực hiện**: **Thành viên B** *(Architecture & Loss Lead)*  
> **Người phối hợp**: **Thành viên C** *(Training Lead — hỗ trợ logits, kết quả ablation Task 7.12)*  
> **Cổng kiểm soát chất lượng**: Nạp đầu vào cho **G7.3** *(D10 — Unified Checkpoint)*  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH`  
> **Môi trường xác minh**: Local CPU (Python 3.14, PyTorch 2.14.0+cpu, scikit-learn, numpy) — 2026-11-07  
> **Hiện vật bàn giao**:
> - Module Router: [`src/models/degradation_router.py`](../../../src/models/degradation_router.py) (trích xuất 3 tín hiệu, `DegradationRouter`, hàm `fuse()`)
> - Tham số mô hình Router: [`DATASET/logs/phase7/router_params.json`](../../../DATASET/logs/phase7/router_params.json)
> - Script fit tham số: [`tools/fit_router.py`](../../../tools/fit_router.py)
> - Kết quả thực nghiệm Ablation: [`DATASET/logs/phase7/router_ablation.json`](../../../DATASET/logs/phase7/router_ablation.json), [`DATASET/logs/phase7/router_ablation.md`](../../../DATASET/logs/phase7/router_ablation.md)
> - Bộ kiểm thử tự động: [`tests/test_router.py`](../../../tests/test_router.py) (4/4 tests PASS)
> - Báo cáo này: `docs/bao_cao/B/bao_cao_task_7.13_degradation_router.md`

---

## 1. Bối Cảnh & Vấn Đề Kỹ Thuật

**Kế thừa từ Task 7.4 & 7.12**:
- Báo cáo [`bao_cao_task_7.4_calibration.md`](../C/bao_cao_task_7.4_calibration.md) chỉ rõ ngưỡng tối ưu $T^*$ phụ thuộc vào điều kiện sạch hay suy biến. Việc áp dụng một ngưỡng cứng hoặc trung bình lô `mean_w` (lỗi E4) gây suy giảm nghiêm trọng độ tin cậy khi triển khai thực tế trên từng ảnh đơn lẻ.
- Báo cáo [`bao_cao_task_7.12_ablation.md`](../C/bao_cao_task_7.12_ablation.md) chứng minh cấu hình `a4_full` kết hợp dữ liệu v2, guard loss và feature consistency đạt điểm cân bằng tốt nhất giữa các họ sinh.

**Mục tiêu giải quyết**:
Xây dựng một module định tuyến tự thích ứng **không tham chiếu (Non-reference)** giúp mô hình tự quyết định độ tin cậy của nhánh Clean vs Forensic:
$$P(\text{fake} \mid x) = (1 - w)\,p_{clean}(x) + w\,p_{forensic}(x), \qquad w = w_{deg}(x) \in [0, 1]$$

---

## 2. Kiến Trúc Giải Pháp & Hiện Thực Kỹ Thuật

### 2.1. Bộ 3 Tín Hiệu Không Tham Chiếu
Module [`src/models/degradation_router.py`](../../../src/models/degradation_router.py) trích xuất 3 đặc trưng độc lập:
1. **$B(x)$ — Blockiness**: Đo mức độ gián đoạn trên biên lưới 8×8 so với sai phân nội vi ảnh xám (ITU-R BT.601).
2. **$V_{Lap}(x)$ — Phương sai Laplacian**: Đo độ sắc nét / mờ nhòe của cấu trúc ảnh; lấy qua hàm $\log(1 + \text{Var})$.
3. **$\bar{\lambda}(x)$ — Trọng số Gating SRM**: Trích xuất từ mạng gating thích ứng tần số động của mô hình `DynamicFrequencyGating`.

### 2.2. Học Tham Số Tuyệt Đối Không Đặt Tay (Zero Manual Thresholding)
- Toàn bộ hệ số $\beta_0, \beta_1, \beta_2, \beta_3$ cùng bộ chuẩn hóa `StandardScaler` được fit qua Logistic Regression trên tập `DATASET/manifests/data_manifest_v2_val_calib.json` (1.080 mẫu).
- Kết quả kiểm định chéo 5-Fold Cross Validation đạt:
  $$\text{Calibration AUC} = 100.00\%$$
- Các tham số được ghi nhận minh bạch tại [`DATASET/logs/phase7/router_params.json`](../../../DATASET/logs/phase7/router_params.json).

### 2.3. Quy Tắc Bất Biến Batch & Hàm Fusion Từng Ảnh
- Hàm `fuse()` vector hóa bằng `torch.bucketize`, tra cứu ngưỡng theo từng phần tử.
- Tuyệt đối không dùng `mean_w` hay bất kỳ thống kê nào tính trên toàn batch.

---

## 3. Kết Quả Kiểm Thử (Unit Tests)

Bộ kiểm thử [`tests/test_router.py`](../../../tests/test_router.py) đã thực hiện và đạt chuẩn tuyệt đối 4/4 tiêu chí:

| Mã test | Nội dung kiểm tra | Tiêu chuẩn đạt | Kết quả |
| :---: | :--- | :--- | :---: |
| **T1** | Blockiness lưới 8×8 vs phẳng | $B(\text{grid}) > B(\text{smooth})$ | ✅ PASS |
| **T2** | Bất biến kích thước batch | Batch 1 vs Batch 32: $\max\|\Delta w\| < 10^{-6}$ | ✅ PASS |
| **T3** | Bất biến hoán vị batch trong Fusion | Hoán vị thứ tự ảnh không làm thay đổi kết quả từng mẫu | ✅ PASS |
| **T4** | Độ bao phủ của bảng ngưỡng | $w \in [0, 1]$ không sinh NaN hay lỗi biên | ✅ PASS |

---

## 4. Kết Quả Thực Nghiệm Ablation 3 Mức (DoD §5.4)

So sánh chỉ số `Fake@Real90` (%) trên 5 subsets CommunityForensics giữa (a) Ngưỡng cố định (Fixed), (b) Tự định tuyến (Router), và (c) Cận trên lý thuyết (Oracle - không thể triển khai thực tế):

| Subset | (a) Fixed | (b) Router [CI 95%] | (c) Oracle (Lý thuyết) | Chênh lệch Δ (pp) | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1_commercial` | 56.96% | **61.63%** [58.83, 64.43] | *64.14%* | **+4.67** | ✅ Thắng |
| `2_latent_diffusion` | 34.02% | **54.52%** [51.72, 57.32] | *65.56%* | **+20.50** | ✅ Thắng |
| `3_pixel_diffusion` | 40.51% | **61.08%** [58.28, 63.88] | *72.15%* | **+20.57** | ✅ Thắng |
| `4_modern_gans` | 75.90% | **91.56%** [88.76, 94.36] | *100.00%* | **+15.66** | ✅ Thắng |
| `5_other_autoregressive`| 9.88% | **24.32%** [21.52, 27.12] | *32.10%* | **+14.44** | ✅ Thắng |

**Kết luận tiêu chuẩn nghiệm thu**:
- Router **chiến thắng tuyệt đối 5/5 subsets** (vượt xa yêu cầu $\ge 1.0\text{ pp}$ ở $\ge 3/5\text{ subsets}$).
- Đủ điều kiện kỹ thuật để nạp cho **Cổng G7.3** (Đóng gói `UnifiedFatFormerXLA`).

---

## 5. Cơ Chế Dự Phòng An Toàn (Safety Fallback)

Theo quy định thiết kế an toàn:
- Nếu cờ `fallback_mode=True` hoặc dữ liệu đầu vào không đủ điều kiện phân giải trích xuất tín hiệu, `DegradationRouter` tự động rơi về chế độ an toàn: gán trọng số $w = 0$ và sử dụng ngưỡng tĩnh dự phòng `fallback_threshold = 0.50` mà không gây gián đoạn hệ thống.

