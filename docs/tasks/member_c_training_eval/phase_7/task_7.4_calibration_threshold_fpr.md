# TASK 7.4 — HIỆU CHUẨN NHIỆT ĐỘ & NGƯỠNG CÓ RÀNG BUỘC FPR + ECE

- **Mã nhiệm vụ**: `Task 7.4` *(Phase 7 — D3)*
- **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**
- **Người phối hợp**: **Thành viên B** (tiêu thụ ngưỡng trong router — Task 7.13)
- **Môi trường thực thi**: 🖥️ **Local CPU**
- **Độ ưu tiên**: 🔴 Khẩn cấp — **chặn G7.0 và Task 7.5/7.13**
- **Cổng liên quan**: **G7.0**
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.4
- **Khế ước**: [`ke_hoach_test_communityforensics_eval.md`](../../../plan/ke_hoach_test_communityforensics_eval.md) — Phụ lục A.2

---

## 1. Mục tiêu

Sinh ra **một tệp cấu hình hiệu chuẩn duy nhất cho mỗi checkpoint**, đủ để mọi báo cáo sau đó dùng lại mà không phải "chọn ngưỡng sau khi đã xem kết quả".

**Ba đầu ra của hiệu chuẩn**

| Ký hiệu | Ý nghĩa | Dùng ở đâu |
| :--- | :--- | :--- |
| `T` | Nhiệt độ tối thiểu hoá NLL (làm mượt xác suất) | Tính `AUC/AP/ECE` |
| `T*` | **Ngưỡng** trên xác suất đã hiệu chuẩn, thoả `FPR ≤ fpr_target` | Tính `Real ACC`, `Fake ACC`, quyết định nhị phân |
| `ECE` | Sai số hiệu chuẩn kỳ vọng (10 bin), trước & sau | Báo cáo G7.2 (`ECE ≤ 0,08`) |

> ⚠️ **Quan trọng**: `T*` trong tài liệu này là **ngưỡng quyết định** (threshold), **không** phải nhiệt độ. Cách đặt tên này giữ tương thích với schema Phụ lục A.7 (`T` và `T_star` là hai trường riêng). Mọi báo cáo phải ghi rõ: `T* = ngưỡng tại FPR mục tiêu`.

**Vì sao phải ràng buộc FPR?** Ngưỡng 0,5 mặc định giả định xác suất đã cân bằng. Trên tập lệch (300 real / 300 fake mỗi subset nhưng phân bố điểm khác nhau), ngưỡng 0,5 có thể cho `FPR = 18%` — tức báo oan 1/5 ảnh thật. Ràng buộc FPR biến ngưỡng thành **quyết định vận hành có thể công bố**, không phải con số tuỳ ý.

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | Script hiện tại | [`tools/calibrate_and_select.py`](../../../../tools/calibrate_and_select.py) | Đối tượng nâng cấp |
| I2 | `val_calib` | Task 7.7 — split `val_calib` trong `data_manifest_v2.json` | Nguồn fit **ưu tiên** |
| I3 | 2.000 ảnh CFE | `DATASET/test_communityforensics` | Nguồn fit **dự phòng** (5-fold CV) |
| I4 | Logits từng checkpoint | Sinh bởi eval v3 (Task 7.3) hoặc chấm riêng | Fit `T`, `T*` |
| I5 | Checkpoint cần hiệu chuẩn | Baseline, Anchored, Gen 2 (Task 7.5), `anchored_v2` (Task 7.11) | Mỗi file một JSON |

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | Calibration JSON | `logs/phase7/calibration/<model>.json` | §5.3 |
| O2 | Script nâng cấp | `tools/calibrate_and_select.py` | `--fit_temperature`, `--fpr_target`, `--cv_folds` |
| O3 | Hình | `logs/phase7/calibration/<model>_reliability.png` | Reliability diagram trước/sau |
| O4 | Bảng tổng | `logs/phase7/calibration/summary.md` | Mọi model × {T, T*, ECE trước, ECE sau, n_calib} |
| O5 | Báo cáo | `docs/bao_cao/C/bao_cao_task_7.4_calibration.md` | Nguồn fit, cảnh báo |

---

## 4. Ràng buộc (theo Phụ lục A.2)

1. **Fit và báo cáo tách rời**: nếu fit trên `val_calib` ⇒ báo cáo trên CFE. Nếu **buộc** phải dùng CFE ⇒ **5-fold CV**, báo cáo **cả 5 fold**, và **không** được fit trên fold đang chấm.
2. **Không chọn ngưỡng sau khi xem kết quả**: ngưỡng là **quantile** của điểm Real trong tập fit, không phải "ngưỡng đẹp nhất trên bảng kết quả".
3. **Không dùng trung bình lô** để suy ngưỡng.
4. Ghi **`calibration_source`, `n_calib`, `fold_id`** (nếu CV) trong mọi JSON.
5. `ECE` phải tính **cả trước và sau** hiệu chuẩn — nếu chỉ báo sau, không chứng minh được tác dụng.
6. **Không fit lại** `T` cho từng điều kiện (clean/Q30) — dùng **cùng** `T`, nhưng `T*` **được phép khác** theo điều kiện và phải ghi rõ.

---

## 5. Các bước thực hiện

### Bước 1 — Fit nhiệt độ `T` (D3, ~1 h)

```python
# logits: [N, 2] (real=0, fake=1); y: [N]
T = nn.Parameter(torch.ones(1))
opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)
def closure():
    opt.zero_grad()
    loss = F.cross_entropy(logits / T.clamp(min=1e-3), y)
    loss.backward(); return loss
opt.step(closure)
T = float(T.clamp(0.05, 10.0))
```

Ghi `T` **trước khi** kẹp và sau khi kẹp nếu khác (cảnh báo nếu bị kẹp).

### Bước 2 — Fit ngưỡng `T*` theo FPR (D3, ~1 h)

```python
p_fake = softmax(logits / T)[:, 1]
p_real = p_fake[y == 0]

thr = float(np.quantile(p_real, 1.0 - fpr_target))   # FPR <= target
# kiểm tra lại bằng số thực:
fpr = float((p_real >= thr).mean())
assert fpr <= fpr_target + 1e-9, (fpr, fpr_target)
```

- Làm cho `fpr_target ∈ {0.01, 0.05, 0.10}`; **lưu cả 3**.
- `T_default` (dùng để báo cáo `Real/Fake ACC`) = `T*` tại `fpr_target = 0.05`.
- Nếu `|{p_real ≥ thr}|` quá nhỏ (< 5 mẫu) ⇒ cảnh báo CI rộng.

### Bước 3 — ECE & reliability (D3, ~0,5 h)

```python
def ece(probs, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(probs, bins[1:-1])
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        conf = probs[m].mean(); acc = y[m].mean()
        e += (m.sum() / len(probs)) * abs(conf - acc)
    return float(e)
```

Vẽ reliability diagram: trục x = confidence trung bình mỗi bin, trục y = accuracy, đường chéo = hoàn hảo. Hai đường: trước / sau.

### Bước 4 — Chế độ 5-fold CV (D3, ~1 h)

Chỉ dùng khi **không** có `val_calib` đủ lớn (≥ 300 mẫu mỗi lớp):

```bash
python tools/calibrate_and_select.py --cv_folds 5 --fpr_target 0.05 \
  --data_root DATASET/test_communityforensics
```

- StratifiedKFold theo `label` **và** `subset`.
- Với mỗi fold: fit `T`, `T*` trên 4 fold, chấm fold còn lại.
- Xuất `folds: [{fold_id, T, T_star, auc, tpr@0.05, n_calib}]` + **trung bình ± độ lệch chuẩn**.
- **Cấm** báo cáo chỉ fold tốt nhất.

### Bước 5 — Ghi JSON & bảng tổng (D3, ~0,5 h)

```json
{
  "schema": "calibration/v1",
  "model": "baseline",
  "checkpoint": {"path": "...", "sha256": "..."},
  "calibration_source": "val_calib",
  "n_calib": 540,
  "class_balance": {"real": 270, "fake": 270},
  "T": 1.2763,
  "T_star": {"0.01": 0.71, "0.05": 0.52, "0.10": 0.44},
  "T_star_default": 0.52,
  "fpr_target": 0.05,
  "fpr_measured": {"0.01": 0.0095, "0.05": 0.0491, "0.10": 0.0998},
  "real_acc": 0.0,
  "fake_acc": 0.0,
  "ece_before": 0.0,
  "ece_after": 0.0,
  "nll_before": 0.0,
  "nll_after": 0.0,
  "fold_id": null
}
```

`summary.md`: bảng `model | source | n_calib | T | T*@5% | ECE_before | ECE_after | real_acc | fake_acc`.

### Bước 6 — Kiểm tra tính độc lập với batch (D3, ~0,3 h)

Test bắt buộc: lấy `T*` từ JSON, chấm **cùng một ảnh** trong 3 ngữ cảnh lô khác nhau (1 ảnh, lô toàn Real, lô 50/50). Dự đoán phải **giống hệt**. Nếu khác ⇒ còn phụ thuộc lô ⇒ chưa đạt.

---

## 6. Definition of Done

- [ ] Mỗi checkpoint mục tiêu có `logs/phase7/calibration/<model>.json` đúng schema §5.5.
- [ ] `T*` có đủ 3 mức FPR (`0.01/0.05/0.10`) và `fpr_measured ≈ fpr_target` (sai số ≤ 0,5 pp).
- [ ] `ECE` báo cáo **trước và sau**; có reliability diagram.
- [ ] `calibration_source` và `n_calib` ghi rõ; nếu CV ⇒ có `fold_id` + cả 5 fold.
- [ ] Không có đường nào fit trên tập đang chấm (kiểm bằng danh sách `sha256` ảnh).
- [ ] Test độc lập batch (§5.6) xanh: cùng ảnh ⇒ cùng dự đoán ở 3 ngữ cảnh lô.
- [ ] `summary.md` tổng hợp mọi model.
- [ ] Báo cáo `bao_cao_task_7.4_calibration.md` nêu rõ nguồn fit và mọi cảnh báo (kẹp `T`, CI rộng, mất cân bằng lớp).

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu | Kill criteria |
| :--- | :---: | :---: | :--- | :--- |
| `val_calib` quá nhỏ (< 300/lớp) | Trung bình | CI rộng, `T` nhiễu | Chuyển sang 5-fold CV CFE, báo cáo cả 5 fold | < 100/lớp ⇒ không đủ, báo C |
| `T` bị kẹp ở biên (0,05 hoặc 10) | Trung bình | Hiệu chuẩn vô hiệu | Ghi cảnh báo; kiểm tra phân bố logit | Kẹp ở cả 2 biên ⇒ dữ liệu có vấn đề |
| Fit và báo cáo cùng tập | **Cao nếu làm ẩu** | Số liệu ảo | Lưu `sha256` tập fit; script assert rời nhau | Phát hiện trùng ⇒ huỷ kết quả |
| Quantile với mẫu nhỏ cho `FPR` thực tế > mục tiêu | Cao | Vượt ràng buộc | Dùng `quantile` + kiểm tra lại bằng đếm thực; nếu vượt ⇒ tăng ngưỡng 1 bước | `fpr_measured > fpr_target + 1 pp` ⇒ làm lại |
| `ECE` 10 bin trên 2.000 mẫu nhiễu | Trung bình | Kết luận sai | Báo cáo kèm CI bootstrap cho ECE | — |
| Nhiều checkpoint ⇒ 4 file JSON lệch phiên bản | Thấp | Không so sánh được | Cùng `schema` + cùng `n_calib`; ghi `git_commit` | — |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) — §IV.2
- Khế ước: [`../../../plan/ke_hoach_test_communityforensics_eval.md`](../../../plan/ke_hoach_test_communityforensics_eval.md) — A.2, A.7
- Eval tiêu thụ ngưỡng: [`task_7.3_eval_v3_measurement_contract.md`](task_7.3_eval_v3_measurement_contract.md)
- Router tiêu thụ ngưỡng: [`../../member_b_architecture_loss/phase_7/task_7.13_degradation_router.md`](../../member_b_architecture_loss/phase_7/task_7.13_degradation_router.md)
- Chỉ mục giai đoạn: [`README.md`](README.md)
