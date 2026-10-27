# TASK 7.3 — EVALUATION v3: KHẾ ƯỚC ĐO LƯỜNG, BỎ `strict=False`, BỎ NGƯỠNG THEO BATCH

- **Mã nhiệm vụ**: `Task 7.3` *(Phase 7 — D2–D3)*
- **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**
- **Người phối hợp**: **Thành viên B (Architecture Lead)** — `canonical_loader` từ Task 7.2
- **Môi trường thực thi**: 🖥️ **Local** + ☁️ **T4**
- **Độ ưu tiên**: 🔴 Khẩn cấp — **chặn mọi số liệu Phase 7**
- **Cổng liên quan**: **G7.0** (cuối D2)
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.3
- **Khế ước bắt buộc**: [`ke_hoach_test_communityforensics_eval.md`](../../../plan/ke_hoach_test_communityforensics_eval.md) — **Phụ lục A**

---

## 1. Mục tiêu

Viết lại đường đo của [`tools/eval_communityforensics.py`](../../../../tools/eval_communityforensics.py) để mọi con số xuất ra **tái lập được, kiểm toán được, và không thể "đẹp" do lỗi kỹ thuật**.

**Sáu khuyết tật phải vá (Phụ lục A.6)**

| Mã | Khuyết tật hiện tại | Hệ quả |
| :---: | :--- | :--- |
| E1 | `strict=False` + lọc theo shape | Nạp thiếu mà không báo |
| E2 | `len(indices) > 8 → 3` | Gen 1 âm thầm mất 7/10 khối |
| E3 | `adaptive_thresh` = trung bình `mean_w` của **tập đang chấm** | Ngưỡng phụ thuộc thành phần lô ⇒ số liệu ảo |
| E4 | Hệ số router đặt tay | Không tái lập, không học từ dữ liệu |
| E5 | Nhiệt độ hard-code; thiếu `TPR@FPR 1/10`; thiếu CI | Không đạt chuẩn học thuật |
| E6 | `wise_ft_merge.py` (thuộc Task 7.5) | Xem Task 7.5 |

**Nguyên tắc**: một con số chỉ được in ra khi nó kèm **điều kiện vận hành** (`FPR`, `T*`, `clean/Q30`, `degrade_order`), **trạng thái nạp**, và **CI 95%**.

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | Eval hiện tại | [`tools/eval_communityforensics.py`](../../../../tools/eval_communityforensics.py) | Đối tượng sửa (đang có thay đổi chưa commit của người dùng — **đọc kỹ trước khi sửa**) |
| I2 | `canonical_loader` | `src/models/canonical_loader.py` (Task 7.2 — tạo mới) | Đường nạp **duy nhất** |
| I3 | Tập test | `DATASET/test_communityforensics` (2.000 ảnh, 5 subset: 600/600/400/200/200) | Đo |
| I4 | `calibration.json` | `logs/phase7/calibration/<model>.json` (Task 7.4) | `T`, `T*`, ngưỡng |
| I5 | Khế ước | Phụ lục A — §A.1, §A.2, §A.5, §A.7 | Định nghĩa metric & schema |

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | Eval v3 | `tools/eval_communityforensics.py` | Không còn `strict=False`, không còn `adaptive_thresh` |
| O2 | Schema output | `logs/phase7/eval_v3_<model>_<cond>.json` | `report_schema: "v3"`, đúng Phụ lục A.7 |
| O3 | Báo cáo Markdown | `logs/phase7/eval_v3_<model>_<cond>.md` | Có cột `TPR@FPR=1/5/10`, dòng "Trạng thái nạp" |
| O4 | Test dummy | `--test_dummy` chạy được | Không cần dữ liệu thật |
| O5 | Báo cáo | `docs/bao_cao/C/bao_cao_task_7.3_eval_v3.md` | Diff hành vi cũ/mới trên Baseline |

---

## 4. Ràng buộc

1. **Không `strict=False`** ở bất kỳ đường nạp nào thuộc pipeline báo cáo (Phụ lục A.5).
2. **Không suy ngưỡng từ tập đang chấm**: ngưỡng phải đến từ `--threshold` hoặc `calibration.json`.
3. **Không dùng `mean_w` của lô** — router phải lấy `router_params.json` (Task 7.13); nếu file chưa có, **raise** với thông báo rõ.
4. **Mọi con số kèm CI**: bootstrap 1.000 lần, seed cố định (`--seed 42`).
5. **Không báo cáo "Overall ACC"** như chỉ số chính (Phụ lục A.4).
6. **`degrade_order` mặc định `post_resize`** và phải ghi vào header JSON + Markdown.
7. **Không xoá** các hàm cũ ngay: giữ sau cờ `--legacy_eval` nhưng in cảnh báo `[DEPRECATED]` — để đối chiếu, không để báo cáo.
8. **Không sửa `src/`** ngoài việc gọi `canonical_loader` (đã có từ 7.2).

---

## 5. Các bước thực hiện

### Bước 1 — Thay đường nạp (D2, ~1 h)

Xoá toàn bộ khối nạp hiện tại (gồm `strict=False`, lọc shape, và fallback `len(indices) > 8 → 3`) và thay bằng:

```python
from src.models.canonical_loader import load_canonical
model, load_report = load_canonical(ckpt_path, model_args)   # raise nếu partial
assert load_report["missing"] == 0 and load_report["unexpected"] == 0
```

Nếu `IncompleteLayoutError` ⇒ **dừng**, in:

```
[REFUSED] <path> không biểu diễn được layout <dropped>.
Đây là invalid_layout — không được báo cáo.
```

và `sys.exit(2)`.

### Bước 2 — Bỏ `adaptive_thresh` (D2, ~0,5 h)

```python
# XOÁ:
# adaptive_thresh = ...
# preds = (probs >= mean_thresh) ...
```

Thay bằng:

```python
threshold = args.threshold if args.threshold is not None else calib["T_star_threshold"]
assert 0.0 < threshold < 1.0
```

Ghi vào JSON: `"threshold": {"value": ..., "source": "calibration.json" | "cli"}`.

### Bước 3 — Router không dùng `mean_w` (D2, ~0,5 h)

`compute_unified_router_predictions` phải:

1. Đọc `router_params.json` (**bắt buộc**; thiếu ⇒ raise).
2. Tính `w` **từng ảnh** (Task 7.13), không lấy trung bình.
3. Tra ngưỡng theo `w` (bảng trong `router_params.json`).

Nếu chạy chế độ **không** router ⇒ ghi `router: {"used": false, "reason": "..."}`.

### Bước 4 — Bổ sung metric (D2–D3, ~2 h)

Theo Phụ lục A.1:

| Metric | Cách tính | Ghi chú |
| :--- | :--- | :--- |
| `auc` | `sklearn.metrics.roc_auc_score` trên xác suất **đã hiệu chuẩn** | |
| `ap` | `average_precision_score` | Xử lý chia 0: nếu không có positive ⇒ `null`, **không** `0` |
| `tpr_at_fpr` | `{"0.01", "0.05", "0.10"}` | Dùng `roc_curve` rồi nội suy; **không** làm tròn FPR trước khi nội suy |
| `real_acc`, `fake_acc` | Tại `T*` cố định | Phải in kèm `T*` |
| `ece` | 10 bin, công thức chuẩn | Trước **và** sau hiệu chuẩn |
| `nll` | `log_loss` | |
| `ci95` | Bootstrap 1.000 lần, resample **theo ảnh** | Cho `auc`, `ap`, `tpr@0.05` tối thiểu |

Viết hàm thuần trong `tools/eval_communityforensics.py` (hoặc `src/evaluation/metrics_v3.py` nếu muốn tái sử dụng) và test bằng mảng giả:

- 4 positive, 4 negative, dự đoán hoàn hảo ⇒ `auc=1, ap=1, tpr@fpr=1`.
- Dự đoán đảo ngược ⇒ `auc=0`.
- Toàn negative ⇒ `ap=null`, không crash.

### Bước 5 — Metadata & schema (D3, ~0,5 h)

Mỗi JSON phải có:

```json
{
  "report_schema": "v3",
  "git_commit": "...",
  "timestamp_utc": "...",
  "degrade_order": "post_resize",
  "dataset": {"name": "communityforensics_eval", "n": 2000, "degrade": "clean"},
  "checkpoints": [{"path": "...", "sha256": "...", "n_keys": 1200,
                   "adapter_blocks": [7,15,23], "n_used": 3,
                   "missing": 0, "unexpected": 0, "status": "ok"}],
  "calibration": {"source": "val_calibration", "n_calib": 540,
                  "T": 1.0, "T_star": 0.5, "fpr_target": 0.05},
  "metrics": {"auc": 0.0, "ap": 0.0,
              "tpr_at_fpr": {"0.01": 0.0, "0.05": 0.0, "0.10": 0.0},
              "real_acc": 0.0, "fake_acc": 0.0, "ece": 0.0,
              "ci95": {"auc": [0.0, 0.0], "tpr_at_fpr_0.05": [0.0, 0.0]}},
  "per_subset": []
}
```

`git_commit` lấy bằng `subprocess.check_output(["git", "rev-parse", "HEAD"])`; nếu repo bẩn ⇒ thêm `"dirty": true` và in cảnh báo.

### Bước 6 — `--degrade_order` (D3, ~0,5 h)

```python
parser.add_argument("--degrade_order", choices=["post_resize", "pre_resize"],
                    default="post_resize")
```

- `post_resize`: nén **sau** khi resize 224 (khớp dữ liệu 224×224 hiện có) — **mặc định**.
- `pre_resize`: nén trên ảnh gốc rồi mới resize (chỉ dùng khi có ảnh độ phân giải gốc).
- Ghi giá trị vào JSON + tiêu đề Markdown: `Điều kiện: Q30 (post_resize)`.

### Bước 7 — Markdown & kiểm thử (D3, ~0,5 h)

1. Thêm cột `TPR@FPR=10%` vào bảng.
2. Thêm dòng **"Trạng thái nạp"**: `ok` / `invalid_layout` / `partial`, kèm `missing/unexpected`.
3. Chạy:

```bash
python tools/eval_communityforensics.py --test_dummy
python tools/eval_communityforensics.py --checkpoint <Baseline> \
  --data_root DATASET/test_communityforensics --condition clean \
  --calibration logs/phase7/calibration/baseline.json --seed 42
```

4. Kiểm tra: không còn dòng `Filtered out-of-graph` khi nạp 3 checkpoint hợp lệ; `missing=0, unexpected=0`.
5. So sánh với số cũ (nếu có) và **giải thích chênh lệch** do loader / ngưỡng / metric trong báo cáo.

---

## 6. Definition of Done

- [ ] `grep -n "strict=False" tools/eval_communityforensics.py` → **rỗng**.
- [ ] `grep -n "adaptive_thresh\|mean_w" tools/eval_communityforensics.py` → **rỗng** (hoặc chỉ còn trong nhánh `--legacy_eval` có cảnh báo).
- [ ] `--test_dummy` chạy thành công, không cần checkpoint.
- [ ] Chạy thật trên Baseline cho JSON đúng `report_schema: "v3"` với **đủ** trường Phụ lục A.7.
- [ ] Có `tpr_at_fpr` cho **cả 1/5/10%**, `ece`, `nll`, `ci95` cho ít nhất `auc` + `tpr@0.05`.
- [ ] `degrade_order` xuất hiện trong JSON **và** Markdown.
- [ ] Bảng Markdown có cột `TPR@FPR=10%` + dòng "Trạng thái nạp".
- [ ] Không còn cảnh báo "Filtered out-of-graph" khi nạp 3 checkpoint hợp lệ.
- [ ] Test mảng giả: 3 case ở §5.4 xanh.
- [ ] Báo cáo nêu rõ **chênh lệch** giữa eval cũ và v3 trên cùng checkpoint, và nguyên nhân.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
| :--- | :---: | :---: | :--- |
| Đang có thay đổi chưa commit của người dùng trong file | **Cao** | Ghi đè mất công sức | Đọc `git diff` trước; chỉ sửa phần cần; **không** `git checkout` |
| `canonical_loader` chưa xong ⇒ bị chặn | Trung bình | Trễ G7.0 | Làm song song phần metric (§5.4) trong lúc chờ 7.2 |
| Bootstrap 1.000 lần quá chậm trên 2.000 ảnh | Thấp | +vài phút | Vector hoá bằng numpy; chỉ bootstrap AUC/AP/TPR chính |
| `ap=null` làm vỡ bảng Markdown | Trung bình | Báo cáo lỗi | Định dạng `—`, ghi chú "không có positive" |
| Ngưỡng từ `calibration.json` khác `T*` dùng lúc fit | Trung bình | Lệch số | Ghi rõ `threshold.source`; test assert khớp `fpr_target` |
| `git_commit` khi repo bẩn gây hiểu nhầm tái lập | Trung bình | Không tái lập được | Ghi `dirty: true` + `git diff --stat` hash |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) — §I.4 (E1–E5)
- Khế ước đo lường: [`../../../plan/ke_hoach_test_communityforensics_eval.md`](../../../plan/ke_hoach_test_communityforensics_eval.md) — Phụ lục A
- Loader: [`../../member_b_architecture_loss/phase_7/task_7.2_canonical_loader.md`](../../member_b_architecture_loss/phase_7/task_7.2_canonical_loader.md)
- Hiệu chuẩn: [`task_7.4_calibration_threshold_fpr.md`](task_7.4_calibration_threshold_fpr.md)
- Chỉ mục giai đoạn: [`README.md`](README.md)
