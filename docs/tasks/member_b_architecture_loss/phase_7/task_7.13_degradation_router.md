# TASK 7.13 — `DegradationRouter` + FUSION XÁC SUẤT ĐÃ HIỆU CHUẨN

- **Mã nhiệm vụ**: `Task 7.13` *(Phase 7 — D6–D9)*
- **Người thực hiện**: **Thành viên B (Architecture & Loss Lead)**
- **Người phối hợp**: **Thành viên C (Training Lead)** — cung cấp `val_calib`, logits, ngưỡng từ Task 7.4
- **Môi trường thực thi**: 🖥️ **Local CPU**
- **Độ ưu tiên**: 🟠 Cao — điều kiện của G7.3 (một checkpoint, tự định tuyến)
- **Cổng liên quan**: nạp cho **G7.3** (D10)
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.13

---

## 1. Mục tiêu

Xây dựng **một** checkpoint tự quyết định "nên tin nhánh sạch hay nhánh forensic" cho từng ảnh, dựa trên **tín hiệu đo được từ chính ảnh** — thay cho việc đặt tay hằng số `mean_w` của cả lô (lỗi E4).

**Kiến trúc fusion**

$$P(\text{fake}\mid x) = (1 - w)\,p_{clean}(x) + w\,p_{forensic}(x), \qquad w = w_{deg}(x) \in [0,1]$$

- `p_clean`, `p_forensic` = xác suất **đã hiệu chuẩn** (Task 7.4), **không** dùng logit thô.
- `w_deg(x)` = xác suất ảnh bị suy biến, fit bằng **logistic regression 3 biến** trên `val_calib`.
- Ngưỡng quyết định là **hàm của `w`** (bảng 2–3 mức fit trên calibration), **không** phải hằng số và **không** phải trung bình của batch.

**Vì sao "không tham chiếu"?** Router phải chạy được trên **một ảnh đơn lẻ** trong production. Mọi tín hiệu dùng ảnh gốc làm tham chiếu (ví dụ so sánh với ảnh clean) đều không hợp lệ.

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | `DynamicFrequencyGating` | [`src/models/gating.py`](../../../../src/models/gating.py) | Nguồn `λ̄(x)` — trả `[1,B,1]`, lấy trung bình |
| I2 | `val_calib` | Task 7.7 — `data_manifest_v2.json` split `val_calib` | Tập **duy nhất** được fit router |
| I3 | `calibration.json` | Task 7.4 — `logs/phase7/calibration/<model>.json` | `T`, `T*`, ngưỡng theo FPR |
| I4 | Logits 2 nhánh | Task 7.6 — ma trận baseline | Fit và đánh giá |
| I5 | Chia subset CFE | `DATASET/test_communityforensics` (600/600/400/200/200) | Đo "thắng ≥ 1 pp ở 3/5 subset" |
| I6 | Checkpoint | `DATASET/checkpoints/phase7/fatformer_anchored_v2_best.pth` | Model để trích `λ̄` |

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | Module router | `src/models/degradation_router.py` | `blockiness()`, `laplacian_var()`, `extract_signals()`, `DegradationRouter` |
| O2 | Tham số | `logs/phase7/router_params.json` | Hệ số + ngưỡng + `n_calib` + `schema` |
| O3 | Fusion | `src/models/unified_fatformer.py` (khung) | `fuse()` dùng `w_deg`, không dùng `mean_w` |
| O4 | Ablation | `logs/phase7/router_ablation.{json,md}` | 3 dòng: fixed / router / oracle, kèm CI |
| O5 | Test | `tests/test_router.py` | 4 test (xem §5.5) |
| O6 | Báo cáo | `docs/bao_cao/B/bao_cao_task_7.13_degradation_router.md` | Kết luận thắng/thua **trung thực** |

---

## 4. Ràng buộc

1. **Không đặt tay hằng số**: mọi hệ số `β₀..β₃` và mọi ngưỡng phải sinh từ fit trên `val_calib`.
2. **Không fit trên test**: `val_calib` và 5 subset CFE là hai tập rời. Nếu vi phạm ⇒ huỷ kết quả (theo kế hoạch §VI).
3. **Không dùng `mean_w` của batch** trong bất kỳ đường suy luận nào.
4. **Tín hiệu phải bất biến với batch size**: chạy 1 ảnh và chạy 32 ảnh phải cho cùng `w` (sai số `1e-6`).
5. **Fallback bắt buộc**: nếu router không thắng, hệ thống phải chạy được với ngưỡng cố định — không được để G7.3 phụ thuộc vào một module "có thể thắng".
6. Chỉ sửa `src/models/`, `tools/`, `configs/`, `tests/` — không sửa `src/datasets/`.

---

## 5. Các bước thực hiện

### Bước 1 — Ba tín hiệu không tham chiếu (D6–D7, ~2,5 h)

**S1. Blockiness `B(x)`** — dấu vết JPEG 8×8:

```python
def blockiness(x):  # x: [B,3,H,W] in [0,1]
    g = 0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]          # [B,H,W]
    dh = (g[:,:,8::8] - g[:,:,7:-1:8]).abs().mean(dim=(1,2)) # biên khối dọc
    dv = (g[:,8::8,:] - g[:,7:-1:8,:]).abs().mean(dim=(1,2)) # biên khối ngang
    nh = (g[:,:,1:] - g[:,:,:-1]).abs().mean(dim=(1,2))      # toàn ảnh dọc
    nv = (g[:,1:,:] - g[:,:-1,:]).abs().mean(dim=(1,2))
    return ((dh + dv) / 2) / ((nh + nv) / 2 + 1e-6) - 1.0    # ~0 nếu không có block
```

**S2. Phương sai Laplacian `V_Lap(x)`** — độ "sắc nét":

```python
lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=x.dtype).view(1,1,3,3).repeat(3,1,1,1)
v = F.conv2d(g.unsqueeze(1), lap, padding=1, groups=1).var(dim=(1,2,3))
return torch.log1p(v)     # log để ổn định thang đo
```

**S3. `λ̄(x)`** — đầu ra gating đã học:

```python
lam = model.gating(feat)            # [1,B,1]
lam_bar = lam.mean(dim=0).squeeze(-1)   # [B]
```

> Lưu ý: `λ̄` cần **forward một phần** (đến gating). Nếu kiến trúc không cho lấy rẻ, chấp nhận forward đầy đủ và lấy `λ̄` từ hook — nhưng **không** được thay bằng hằng số.

**Chuẩn hoá**: fit `StandardScaler` **trên `val_calib`** (lưu `mean/std` vào `router_params.json`), không dùng thống kê của test.

### Bước 2 — Fit logistic regression (D7, ~2 h)

```python
X = np.stack([B, V_Lap, lam_bar], axis=1)      # [N,3] đã chuẩn hoá
y = is_degraded                                  # 0/1 từ nhãn điều kiện của val_calib
clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(X, y)
```

- `is_degraded` lấy từ **nhãn điều kiện** của `val_calib` (ảnh gốc clean vs ảnh đã qua pipeline suy biến của Task 7.8), **không** suy ra từ dự đoán model.
- Báo cáo `AUC` của router trên `val_calib` (5-fold CV) và **histogram `w`** cho clean vs degraded — nếu hai histogram chồng lên nhau gần như hoàn toàn ⇒ router vô dụng, ghi nhận trước khi tốn công ablation.

`router_params.json`:

```json
{
  "schema": "degradation_router/v1",
  "features": ["blockiness", "log1p_lapvar", "lambda_bar"],
  "scaler": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
  "coef": [0.0, 0.0, 0.0],
  "intercept": 0.0,
  "n_calib": 0,
  "calib_auc": 0.0,
  "threshold_table": [
    {"w_min": 0.0, "w_max": 0.33, "threshold": 0.5},
    {"w_min": 0.33, "w_max": 0.66, "threshold": 0.5},
    {"w_min": 0.66, "w_max": 1.0, "threshold": 0.5}
  ]
}
```

**Ngưỡng theo `w`**: chia 3 khoảng theo **quantile** của `w` trên `val_calib` (không chia đều 0–1 nếu phân phối lệch). Trong mỗi khoảng, chọn ngưỡng thoả `FPR ≤ 5%` trên ảnh **Real** của khoảng đó. Nếu một khoảng có < 50 mẫu Real ⇒ gộp với khoảng bên cạnh và ghi rõ.

### Bước 3 — Fusion (D8, ~1,5 h)

```python
def fuse(p_clean, p_forensic, w, threshold_table):
    p = (1 - w) * p_clean + w * p_forensic
    thr = threshold_for(w, threshold_table)   # theo từng ảnh, không theo batch
    return p, (p >= thr).long(), thr
```

- `p_clean`, `p_forensic` phải là **xác suất đã hiệu chuẩn** (`softmax(logits / T)` với `T` từ Task 7.4).
- `threshold_for` tra bảng theo `w` của **từng ảnh** — vector hoá bằng `torch.bucketize`.
- Tuyệt đối không có `mean_w`, `mean_thresh`, hay bất kỳ phép trung bình nào trên batch.

### Bước 4 — Ablation 3 mức (D8–D9, ~1,5 h)

| Mức | Cấu hình | Vai trò |
| :--- | :--- | :--- |
| (a) **Fixed** | `w ≡ 0` hoặc `w ≡ 1` (chọn theo validation), ngưỡng cố định toàn cục | Baseline công bằng |
| (b) **Router** | `w = w_deg(x)`, ngưỡng theo bảng | Đối tượng cần chứng minh |
| (c) **Oracle** | Biết trước điều kiện thật (clean/Q30), chọn nhánh tương ứng | **Cận trên**, không triển khai được |

Chấm trên **5 subset CFE** (600/600/400/200/200) × {clean, Q30}, bootstrap CI 1.000 lần.

**Luật kết luận**: router chỉ được coi là **thắng** nếu `Fake@Real90` cao hơn mức (a) **≥ 1 pp** ở **≥ 3/5 subset**, với CI 95% không chồng lên nhau ở ít nhất 2 subset. Ngược lại ⇒ **báo cáo trung thực là không thắng** và G7.3 dùng ngưỡng cố định.

### Bước 5 — Test (D8, ~0,5 h)

`tests/test_router.py`:

| # | Test | Kỳ vọng |
| :---: | :--- | :--- |
| T1 | `blockiness` ảnh tổng hợp có lưới 8×8 | Giá trị **cao hơn** ảnh phẳng cùng mean |
| T2 | `w` bất biến theo batch | 1 ảnh vs 32 ảnh: `max|Δw| < 1e-6` |
| T3 | `fuse` không dùng mean | Đảo thứ tự ảnh trong batch ⇒ kết quả từng ảnh không đổi |
| T4 | `threshold_for` phủ toàn miền | `w ∈ {0, 0.5, 1}` đều trả ngưỡng hợp lệ, không `NaN` |

---

## 6. Definition of Done

- [ ] `src/models/degradation_router.py` có 3 tín hiệu + `DegradationRouter`; không hằng số đặt tay.
- [ ] `router_params.json` có hệ số, scaler, bảng ngưỡng, `n_calib`, `calib_auc`.
- [ ] `fuse()` dùng `w_deg(x)` **từng ảnh**; grep `mean_w` trong `src/models/` trả về rỗng.
- [ ] `logs/phase7/router_ablation.{json,md}` có **3 dòng** (fixed / router / oracle) + CI 95%.
- [ ] Kết luận nêu rõ: thắng hay không thắng, kèm số pp chênh lệch trên từng subset.
- [ ] `pytest -q tests/test_router.py` — 4/4 xanh.
- [ ] Có fallback: nếu router không thắng, `unified_fatformer.py` chạy được với ngưỡng cố định.
- [ ] `router_params.json` được nhúng vào metadata checkpoint ở Task 7.15.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu | Kill criteria |
| :--- | :---: | :---: | :--- | :--- |
| Router sụp về 1 nhánh (`w` gần như hằng) | **Cao** | G7.3 vô nghĩa | Histogram `w` clean vs Q30 trước khi ablation | `w` gần hằng ⇒ bỏ router, dùng ngưỡng cố định |
| `λ̄` không tách được clean/Q30 vì gating học kém | Trung bình | Giảm AUC router | Báo cáo AUC từng tín hiệu đơn lẻ; nếu `λ̄` vô dụng ⇒ dùng 2 biến, ghi rõ | — |
| Fit ngưỡng trên cùng tập đánh giá | Trung bình | Số liệu ảo | `val_calib` ≠ 5 subset CFE; kiểm tra bằng hash danh sách ảnh | Fit và báo cáo cùng tập ⇒ huỷ |
| Blockiness bị nhiễu do ảnh đã resize 224 | Trung bình | Tín hiệu yếu | Đo trên ảnh gốc nếu có; nếu chỉ có 224, ghi rõ hạn chế | — |
| Oracle bị hiểu nhầm là kết quả thật | Cao | Phản biện bắt lỗi | Nhãn **ORACLE — không triển khai được** trong mọi bảng | — |
| Quá 8 h công ⇒ trễ D9 | Trung bình | Trễ G7.3 | Ưu tiên (a)+(b); oracle chỉ chạy 1 subset nếu thiếu thời gian | — |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2
- Hiệu chuẩn: [`../../member_c_training_eval/phase_7/task_7.4_calibration_threshold_fpr.md`](../../member_c_training_eval/phase_7/task_7.4_calibration_threshold_fpr.md)
- Đóng gói: [`../../member_c_training_eval/phase_7/task_7.15_unified_packaging.md`](../../member_c_training_eval/phase_7/task_7.15_unified_packaging.md)
- Benchmark cuối: [`../../member_c_training_eval/phase_7/task_7.14_locked_benchmark_phase7.md`](../../member_c_training_eval/phase_7/task_7.14_locked_benchmark_phase7.md)
- Chỉ mục giai đoạn: [`README.md`](README.md)
