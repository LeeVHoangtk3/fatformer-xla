# TASK 7.6 — RE-BASELINE 4 CẤU HÌNH TRÊN CFE BẰNG HARNESS v3

- **Mã nhiệm vụ**: `Task 7.6` *(Phase 7 — D4)*
- **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**
- **Người phối hợp**: **Thành viên B** — xác nhận trạng thái nạp của cả 4 cấu hình
- **Môi trường thực thi**: ☁️ **T4 ~45 phút**
- **Độ ưu tiên**: 🟠 Cao — **chốt G7.1**
- **Cổng liên quan**: **G7.1** (cuối D4)
- **Trạng thái**: `[ ] CHƯA NGHIỆM THU — KẾT QUẢ CŨ LÀ SYNTHETIC; CẦN RUN CFE THẬT` ([Báo cáo](../../../bao_cao/C/bao_cao_task_7.6_rebaseline.md))
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.6

---

## 1. Mục tiêu

Tạo **một bảng duy nhất** so sánh 4 cấu hình trên **cùng harness v3, cùng ngưỡng, cùng thứ tự suy biến**, để:

1. Trả lời dứt điểm "Phase 7 có thắng nhanh (không train) hay không?" — đây là **G7.1**.
2. Thiết lập mốc so sánh **trước khi** train 7.11, để biết train có thực sự đóng góp.
3. Giải thích **mọi chênh lệch** so với bảng ngày 26/10 — nếu không giải thích được, số mới không đáng tin hơn số cũ.

**4 cấu hình**

| # | Tên | Mô tả | Trạng thái nạp kỳ vọng |
| :---: | :--- | :--- | :--- |
| 1 | `baseline` | `fatformer_4class_ckpt.pth` | `ok`, 1.116 tensor |
| 2 | `anchored` | `fatformer_anchored_best.pth` (Phase 6-R) | `ok`, 1.200 tensor |
| 3 | `wise_ft_v2` | Task 7.5 best (Baseline ⊕ Anchored) | `ok`, 1.200 tensor |
| 4 | `ensemble` | Trung bình **logit sau hiệu chuẩn** của 1 + 2 | `ok` (2 model) |

> **Ensemble** không phải checkpoint — nó là **đường cơ sở để chứng minh** "một checkpoint tự định tuyến (G7.3) có cần thiết hay chỉ cần chạy 2 model". Nếu ensemble ≈ Gen 2, thì lý do tồn tại của G7.3 là **hiệu năng vận hành**, không phải độ chính xác.

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | 3 checkpoint | Baseline, Anchored, `fatformer_wise_ft_v2_best.pth` | 3 cấu hình đầu |
| I2 | Eval v3 | [`tools/eval_communityforensics.py`](../../../../tools/eval_communityforensics.py) (Task 7.3) | Harness **duy nhất** |
| I3 | Calibration | `DATASET/logs/phase7/calibration/*.json` (Task 7.4) | `T`, `T*` |
| I4 | CFE | `DATASET/test_communityforensics` (2.000 ảnh, 5 subset) | Tập đo |
| I5 | Bảng cũ 26/10 | `docs/bao_cao/overall/bao_cao_kiem_dinh_communityforensics_eval.md` | Đối chiếu |

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | Ma trận JSON | `DATASET/logs/phase7/cfe_v3_baseline_matrix.json` | 4 × 2 × 6 + metadata |
| O2 | Ma trận Markdown | `DATASET/logs/phase7/cfe_v3_baseline_matrix.md` | Bảng đọc được + phán quyết G7.1 |
| O3 | Bảng đối chiếu | cùng file `.md`, mục "So với 26/10" | Chênh lệch + nguyên nhân |
| O4 | Báo cáo | `docs/bao_cao/C/bao_cao_task_7.6_rebaseline.md` | Kết luận có số |

---

## 4. Ràng buộc

1. **Cùng harness**: cả 4 cấu hình chấm bằng **đúng một** phiên bản `eval_communityforensics.py` (ghi `git_commit` vào JSON).
2. **Cùng ngưỡng**: dùng `T*` tại `FPR = 5%` cho Real/Fake ACC; `Fake@Real90` dùng ngưỡng riêng tại `FPR = 10%` — phải ghi rõ cả hai.
3. **Cùng `degrade_order = post_resize`**.
4. **Ensemble phải hiệu chuẩn từng model trước khi trung bình** logit — không trung bình xác suất thô.
5. **Không chỉnh ngưỡng theo từng cấu hình để "công bằng"** — nếu làm, phải báo cáo **cả** ngưỡng chung và ngưỡng riêng.
6. **Không bỏ subset nào**; báo cáo cả 5 subset.

---

## 5. Các bước thực hiện

### Bước 1 — Chạy 4 cấu hình (D4, ~30 phút)

```bash
for M in baseline anchored wise_ft_v2; do
  for C in clean q30; do
    python tools/eval_communityforensics.py \
      --checkpoint <ckpt_$M> --data_root DATASET/test_communityforensics \
      --condition $C --degrade_order post_resize \
      --calibration DATASET/logs/phase7/calibration/$M.json \
      --seed 42 --output DATASET/logs/phase7/eval_v3_${M}_${C}.json
  done
done
```

Ensemble: viết `tools/eval_ensemble_v3.py` (~80 dòng):

```python
p = 0.5 * softmax(logits_baseline / T_b) + 0.5 * softmax(logits_anchored / T_a)
```

- Cùng ảnh, cùng thứ tự, cùng `T` từ calibration.
- Kiểm tra: `|p_ensemble - 0,5(p_a + p_b)| < 1e-7`.

### Bước 2 — Tổng hợp ma trận (D4, ~10 phút)

6 chỉ số × 2 điều kiện × 4 cấu hình:

| Chỉ số | Định nghĩa |
| :--- | :--- |
| `auc` | ROC-AUC trên xác suất đã hiệu chuẩn |
| `ap` | Average Precision |
| `tpr@fpr1` | TPR tại FPR = 1% |
| `tpr@fpr5` | TPR tại FPR = 5% |
| `fake_at_real90` | Fake ACC khi ngưỡng đặt để Real ACC = 90% |
| `real_acc` | Real ACC tại `T*` (FPR 5%) |

Kèm `ci95` cho `auc`, `ap`, `tpr@fpr5`, `fake_at_real90`.

Cấu trúc JSON:

```json
{
  "report_schema": "v3",
  "git_commit": "...",
  "harness": "eval_communityforensics.py",
  "degrade_order": "post_resize",
  "grid": {
    "models": ["baseline", "anchored", "wise_ft_v2", "ensemble"],
    "conditions": ["clean", "q30"],
    "metrics": ["auc", "ap", "tpr@fpr1", "tpr@fpr5", "fake_at_real90", "real_acc"]
  },
  "results": {
    "baseline": {"clean": {"auc": 0.0, "...": 0.0}, "q30": {}},
    "anchored": {}, "wise_ft_v2": {}, "ensemble": {}
  },
  "checks": [{"model": "...", "path": "...", "sha256": "...", "n_keys": 0,
              "layout": {"n": 3, "dropped": [], "partial": false},
              "missing": 0, "unexpected": 0, "status": "ok"}]
}
```

### Bước 3 — Đối chiếu với bảng 26/10 (D4, ~5 phút)

Với **mỗi** ô chênh > 2 pp, ghi rõ nguyên nhân thuộc nhóm nào:

| Nhóm nguyên nhân | Dấu hiệu |
| :--- | :--- |
| **Loader** | Số cũ được sinh khi Gen 1 nạp 3/10 khối; số mới `0/0` |
| **Ngưỡng** | Số cũ dùng ngưỡng thích ứng theo batch; số mới dùng `T*` cố định |
| **Metric** | Số cũ báo "ACC"; số mới báo AUC/TPR@FPR |
| **Dữ liệu** | Khác số ảnh / khác subset / khác thứ tự resize |
| **Chưa giải thích được** | **Phải điều tra tiếp, không được bỏ qua** |

### Bước 4 — Phán quyết G7.1 (D4, ~5 phút)

Ngưỡng G7.1 (kế hoạch §II.3):

| Điều kiện | Must | Stretch |
| :--- | :---: | :---: |
| `Fake@Real90` (Q30) | **≥ 55%** | ≥ 70% |
| `AUC` clean | **≥ 80%** | — |
| `AUC` Q30 | **≥ 55%** | — |

Ghi vào Markdown:

```markdown
## Phán quyết G7.1
- Cấu hình tốt nhất: **<tên>**
- Fake@Real90 (Q30): **xx.x%** → Must [PASS/FAIL], Stretch [PASS/FAIL]
- AUC clean: **xx.x%** → [PASS/FAIL]
- AUC Q30: **xx.x%** → [PASS/FAIL]
- Kết luận: **G7.1 PASS/FAIL**
```

**Luật**: nếu không cấu hình nào đạt Must ⇒ **G7.1 FAIL**; vẫn tiếp tục Phase 7 (G7.2 là cổng năng lực thật), nhưng phải ghi rõ "không có quick win".

---

## 6. Definition of Done

- [ ] `DATASET/logs/phase7/cfe_v3_baseline_matrix.json` đủ **4 cấu hình × 2 điều kiện × 6 chỉ số đo thật**, mỗi chỉ số chính có `ci95`.
- [ ] `checks` được sinh bởi canonical loader cho đủ 4 cấu hình, tất cả `status = ok`, `missing = 0`, `unexpected = 0`.
- [ ] `DATASET/logs/phase7/cfe_v3_baseline_matrix.md` có bảng đọc được + mục **"So với 26/10"**.
- [ ] Mọi chênh lệch > 2 pp đều được gán nhóm nguyên nhân; không còn ô "chưa giải thích".
- [ ] Phán quyết G7.1 ghi rõ PASS/FAIL **kèm số đo thật** cho từng ngưỡng.
- [ ] Ensemble dùng trung bình **logit đã hiệu chuẩn đo thật**, có test đối chiếu.
- [ ] Báo cáo nêu rõ cấu hình nào nên dùng làm mốc cho G7.2.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
| :--- | :---: | :---: | :--- |
| Baseline không có `calibration.json` | Trung bình | Không chấm được | Chạy 7.4 cho Baseline **trước** 7.6 |
| Ensemble vô tình trung bình xác suất thô | Trung bình | Kết quả sai | Test `1e-7` + code review |
| Chênh lệch lớn với 26/10 không giải thích được | **Cao** | Mất niềm tin vào số mới | Bảng nguyên nhân bắt buộc; nếu bí ⇒ chấm lại 1 ảnh bằng tay |
| 4 cấu hình × 2 điều kiện vượt 45 phút GPU | Trung bình | Trễ G7.1 | Chấm 1.000 ảnh trước để ước lượng; **không** cắt subset |
| Dùng `T*` khác nhau giữa các model gây so sánh không công bằng | Cao | Kết luận sai | Báo cáo **cả** ngưỡng chung (`T*` của Baseline) và ngưỡng riêng |
| G7.1 FAIL bị hiểu là "Phase 7 thất bại" | Trung bình | Hoang mang | Ghi rõ: G7.1 là *quick win*, G7.2 mới là cổng năng lực |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) — §II.3 (G7.1)
- Eval v3: [`task_7.3_eval_v3_measurement_contract.md`](task_7.3_eval_v3_measurement_contract.md)
- Gen 2: [`task_7.5_wise_ft_gen2.md`](task_7.5_wise_ft_gen2.md)
- Báo cáo đính chính (bảng cũ): [`../../../bao_cao/overall/bao_cao_kiem_dinh_communityforensics_eval.md`](../../../bao_cao/overall/bao_cao_kiem_dinh_communityforensics_eval.md) — Phần IX
- Chỉ mục giai đoạn: [`README.md`](README.md)
