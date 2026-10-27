# TASK 7.10 — SELECTION METRIC v2: WORST-GROUP + RÀNG BUỘC REAL ACC

- **Mã nhiệm vụ**: `Task 7.10` *(Phase 7 — D5–D6)*
- **Người thực hiện**: **Thành viên B (Architecture & Loss Lead)**
- **Người phối hợp**: **Thành viên C (Training Lead)** — chạy validation trong Trainer, đọc CSV
- **Môi trường thực thi**: 🖥️ **Local CPU**
- **Độ ưu tiên**: 🟠 Cao — quyết định checkpoint nào được chọn ở Task 7.11
- **Cổng liên quan**: không có cổng riêng; ảnh hưởng trực tiếp G7.2
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.10

---

## 1. Mục tiêu

Thay điểm chọn checkpoint hiện tại bằng **worst-group score** có ràng buộc Real.

**Vấn đề của logic hiện tại** ([`src/training/trainer.py`](../../../../src/training/trainer.py) dòng 204–231):

```python
val_score = 0.40 * mean(clean_scores) + 0.40 * mean(robust_scores) + 0.20 * mean(diff_scores)
loader_score = 0.5 * acc + 0.5 * ap
```

Ba lỗi cấu trúc:

1. **Trung bình che lấp nhóm tệ nhất**: một nhóm có `Real ACC = 40%` vẫn được bù bởi `Fake ACC = 99%`, và `acc` gộp còn che tiếp. Checkpoint "tốt" theo composite có thể là checkpoint tệ nhất trên Real-Q30.
2. **AP trộn vào điểm chọn**: AP đo xếp hạng, không đo ngưỡng vận hành. Chọn theo AP rồi triển khai ở ngưỡng 0,5 là hai bài toán khác nhau.
3. **Không có ràng buộc Real**: mô hình có thể đạt composite cao bằng cách hy sinh Real — đúng thứ đã xảy ra ở Gen 1.

**Định nghĩa mới**

$$g \in \{\text{clean}, \text{Q30}\} \times \{\text{real}, \text{fake}\}$$

$$\text{recall}_{c,y} = \frac{\#\{\hat{y}=y \mid y, c\}}{\#\{y, c\}}$$

$$\text{worst\_group} = \min_{c \in \{\text{clean}, \text{Q30}\}} \ \min_{y \in \{\text{real}, \text{fake}\}} \text{recall}_{c,y}$$

> Trong một nhóm chỉ có **một** nhãn, "balanced accuracy" thu về chính **recall** của nhãn đó. Vì vậy `worst_group_score = min` của 4 recall là cách viết tương đương và kiểm chứng được.

**Luật lưu checkpoint**

| Điều kiện | Tệp được lưu | Ghi chú |
| :--- | :--- | :--- |
| `worst_group > best_worst_group` **và** `recall_real_clean ≥ 0,90` | `best.pth` | Ứng viên chính thức |
| `worst_group > best_worst_group` nhưng `recall_real_clean < 0,90` | `best_effort.pth` | **Không** được dùng làm kết quả chính; vẫn giữ để phân tích |
| Cả hai | cập nhật **độc lập** | Không ghi đè nhau |

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | `Trainer.fit()` + `validate()` | [`src/training/trainer.py`](../../../../src/training/trainer.py) | Nơi thay logic chọn |
| I2 | `evaluate_dataloader` | `src/training/trainer.py` (hoặc utils) | Trả `acc/ap/auc/r_acc/f_acc` |
| I3 | Transform Q30 | [`src/datasets/transforms.py`](../../../../src/datasets/transforms.py) — `RandomJPEGCompression` | Dùng để tạo nhánh Q30 trong bộ nhớ |
| I4 | `val_select` | Task 7.7 — `DATASET/manifests/data_manifest_v2.json` | Tập chọn, **tách** khỏi `val_calib` |
| I5 | Ngưỡng vận hành | Task 7.4 — `logs/phase7/calibration/<model>.json` | Ngưỡng dùng khi tính recall |

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | Hàm metric | `src/training/metrics.py` (mới) hoặc trong `trainer.py` | `group_recalls()`, `worst_group_score()` |
| O2 | Trainer sửa đổi | `src/training/trainer.py` | Bỏ composite, dùng worst-group + ràng buộc Real |
| O3 | Log CSV | `<run_dir>/val_groups.csv` | 6 cột: `epoch, clean_r, clean_f, q30_r, q30_f, worst` |
| O4 | Test | `tests/test_selection_metric.py` | Test mảng giả + test ràng buộc lưu |
| O5 | Báo cáo | `docs/bao_cao/B/bao_cao_task_7.10_selection_v2.md` | So sánh checkpoint được chọn theo cũ vs mới |

---

## 4. Ràng buộc

1. **Không dùng `acc` gộp** để chọn checkpoint — chỉ dùng 4 recall nhóm.
2. **Q30 phải tất định**: dùng `quality=30` cố định, **không** random, để các epoch so sánh được với nhau.
3. **Không fit gì trên `val_select`**: chỉ đọc. Ngưỡng lấy từ `calibration.json` của Task 7.4.
4. **Hiệu năng**: validation không chậm quá **2×**; dùng tối đa **300 ảnh con** (stratified) cho nhánh Q30 nếu tập lớn.
5. **Backward-compatible**: nếu `val_loader` là loader đơn (không phải dict) ⇒ giữ hành vi cũ + cảnh báo, không crash.
6. **Không ghi đè** `best.pth` bằng ứng viên không đạt ràng buộc Real.

---

## 5. Các bước thực hiện

### Bước 1 — Hàm metric thuần (D5, ~1 h)

```python
def group_recalls(y_true, y_pred, condition):
    """condition: mảng {0:clean, 1:q30} cùng độ dài.
    Trả {clean_r, clean_f, q30_r, q30_f}; nhóm rỗng ⇒ nan (không phải 0)."""

def worst_group_score(recalls, nan_policy="raise"):
    """min của 4 recall. Nếu có nan: raise (không được che giấu nhóm thiếu dữ liệu)."""
```

**Quyết định thiết kế**: nhóm rỗng ⇒ `nan` + `raise`, **không** trả `0`. Trả `0` sẽ khiến checkpoint "vô tình" bị loại vì lý do kỹ thuật; trả `nan` buộc người chạy phải sửa split.

### Bước 2 — Sinh nhánh Q30 trong bộ nhớ (D5, ~1 h)

1. Lấy **300 ảnh** stratified từ `val_select` (150 real / 150 fake, seed 42).
2. Áp `JPEG(quality=30)` **một lần**, cache tensor đã giải mã để không nén lại mỗi epoch.
3. Chấm cả 2 nhánh (clean + Q30) trong **cùng một lượt** `validate()`.
4. Ghi rõ trong log: `n_eval_clean`, `n_eval_q30`, `q30_source=in_memory_jpeg30`.

> **Không** dùng `RandomJPEGCompression(min_q=30, max_q=95)` — nó ngẫu nhiên, làm recall dao động giữa các epoch và khiến việc chọn checkpoint nhiễu.

### Bước 3 — Thay logic trong `Trainer.fit()` (D5–D6, ~1 h)

```python
recalls = group_recalls(all_y, all_pred, all_cond)
worst = worst_group_score(recalls)

passes_real_guard = recalls["clean_r"] >= 0.90

if worst > self.best_worst_group:
    self.best_worst_group = worst
    if passes_real_guard:
        save("best.pth"); is_best = True
    else:
        save("best_effort.pth"); is_best = False
```

Xoá đoạn `val_score = 0.40*... + 0.40*... + 0.20*...` (dòng 204–210) và `if val_score > self.best_val_score` (dòng 227). Giữ `val_score` như **metric phụ để log**, không dùng để quyết định.

### Bước 4 — Log 6 cột (D6, ~0,5 h)

`val_groups.csv`:

```csv
epoch,clean_r,clean_f,q30_r,q30_f,worst
1,0.9412,0.9880,0.7120,0.8010,0.7120
2,0.9350,0.9901,0.7680,0.8420,0.7680
```

Thêm cột `best_effort` (bool) và `threshold` để truy vết. In ra console 1 dòng/epoch theo cùng định dạng.

### Bước 5 — Test (D6, ~1 h)

`tests/test_selection_metric.py`:

| # | Test | Kỳ vọng |
| :---: | :--- | :--- |
| T1 | Mảng giả 4 nhóm đã biết | `worst` đúng giá trị min, không làm tròn |
| T2 | Nhóm rỗng | `raise`, không trả 0 |
| T3 | `worst` tăng nhưng `clean_r = 0,85` | Ghi `best_effort`, **không** ghi `best.pth` |
| T4 | `worst` tăng và `clean_r = 0,92` | Ghi `best.pth` |
| T5 | Loader đơn (không dict) | Không crash, cảnh báo, dùng hành vi cũ |

### Bước 6 — Đo hiệu năng (D6, ~0,5 h)

Đo `time(validate)` trước/sau trên cùng `val_select`. Ngưỡng: `≤ 2×`.

Nếu vượt: giảm `n_eval_q30` xuống 200, **không** giảm xuống dưới 100 (CI quá rộng). Ghi số đo vào báo cáo.

---

## 6. Definition of Done

- [ ] `worst_group_score` được cài đúng định nghĩa §1, có test T1.
- [ ] `Trainer.fit()` **không còn** dùng composite để quyết định lưu checkpoint.
- [ ] `best.pth` chỉ được ghi khi `clean_r ≥ 0,90`; ngược lại ghi `best_effort.pth`.
- [ ] `val_groups.csv` có đủ 6 cột bắt buộc (+ 2 cột truy vết).
- [ ] `pytest -q tests/test_selection_metric.py` — 5/5 xanh.
- [ ] Validation chậm **≤ 2×**; số đo trong báo cáo.
- [ ] So sánh được: với cùng một run, checkpoint được chọn theo logic cũ vs mới **khác nhau hay không** — ghi rõ trong báo cáo.
- [ ] Loader đơn không bị crash.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
| :--- | :---: | :---: | :--- |
| Ràng buộc `clean_r ≥ 0,90` quá chặt ⇒ không epoch nào đạt `best.pth` | **Cao** | Không có checkpoint chính thức | Vẫn có `best_effort.pth`; báo C ngay từ epoch 2; xem xét nới xuống 0,88 **chỉ khi** có biên bản |
| Q30 in-memory khác Q30 lúc test (do thứ tự resize/nén) | Trung bình | Chọn sai checkpoint | Dùng **cùng** thứ tự `post_resize` như Task 7.3; ghi rõ trong log |
| 300 ảnh ⇒ CI ±5 pp | Cao | Nhiễu chọn | Báo cáo CI; nếu 2 epoch sát nhau (< 1 pp) ⇒ chọn epoch có `q30_r` cao hơn |
| Nhóm Fake-Q30 quá dễ ⇒ `worst` luôn do Real-Q30 quyết định | Trung bình | Mất tác dụng 4 nhóm | Log cả 4; nếu 3 epoch liên tiếp `worst` luôn là `q30_r` ⇒ ghi nhận, không đổi metric giữa chừng |
| Sửa trainer làm hỏng run cũ | Thấp | Mất tái lập | Giữ nhánh `else` cho loader đơn; không đổi tên hàm công khai |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2
- Loss được chọn bởi metric này: [`task_7.9_loss_v2_realguard.md`](task_7.9_loss_v2_realguard.md)
- Ngưỡng vận hành: [`../../member_c_training_eval/phase_7/task_7.4_calibration_threshold_fpr.md`](../../member_c_training_eval/phase_7/task_7.4_calibration_threshold_fpr.md)
- Task train: [`../../member_c_training_eval/phase_7/task_7.11_train_anchored_v2.md`](../../member_c_training_eval/phase_7/task_7.11_train_anchored_v2.md)
- Chỉ mục giai đoạn: [`README.md`](README.md)
