# TASK 7.14 — LOCKED BENCHMARK PHASE 7 (IN-DIST 82K + CFE + 6 BIẾN THỂ)

- **Mã nhiệm vụ**: `Task 7.14` *(Phase 7 — D9–D10)*
- **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**
- **Người phối hợp**: **Thành viên A** (đối chiếu danh sách ảnh, chống lẫn test) + **Thành viên B** (router/fusion)
- **Môi trường thực thi**: ☁️ **T4 / A100**
- **Độ ưu tiên**: 🔴 Khẩn cấp — **chốt G7.3 và toàn bộ số liệu luận văn**
- **Cổng liên quan**: **G7.3** (cuối D10)
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.14

---

## 1. Mục tiêu

Chấm **một lượt, một lần, không chọn lọc** để tạo ra **bộ số liệu cuối cùng** của Phase 7:

1. **In-dist**: 18 tập của bài báo gốc (**82.003 ảnh**) — kiểm tra không quên miền cũ.
2. **Out-of-dist**: CFE **clean** và **Q30** (2.000 ảnh, 5 subset).
3. **6 biến thể suy biến** trên CFE — đo độ bền theo từng loại suy biến.
4. **Per-generator**: bảng theo 4 nhóm generator × 5 subset.

Đây là lần chấm **duy nhất** được dùng để viết kết quả. Mọi lần chấm sau (nếu có) phải ghi rõ là "hậu nghiệm" và **không** thay thế bảng này.

**6 biến thể suy biến** (định nghĩa cố định, không đổi sau khi chạy)

| Mã | Tham số | Mô phỏng |
| :---: | :--- | :--- |
| `q75` | JPEG quality 75 | Nén nhẹ, mạng xã hội |
| `q50` | JPEG quality 50 | Nén trung bình |
| `q30` | JPEG quality 30 | Nén mạnh (chuẩn G7.1/G7.2) |
| `webp40` | WebP quality 40 | Tải lại qua nền tảng web |
| `blur15` | Gaussian σ = 1,5 | Ảnh chụp lại / mất nét |
| `downup05` | Down-up ratio 0,5 | Thu nhỏ rồi phóng to |

Tất cả áp dụng **`post_resize`** (sau khi đưa về 224) — khớp dữ liệu hiện có.

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | Checkpoint cuối | `DATASET/checkpoints/phase7/fatformer_anchored_v2_best.pth` (hoặc `_best_effort.pth`) | Đối tượng chấm |
| I2 | Router | `logs/phase7/router_params.json` (Task 7.13) | Fusion (nếu thắng) |
| I3 | Calibration | `logs/phase7/calibration/anchored_v2.json` (Task 7.4) | `T`, `T*` |
| I4 | 18 tập in-dist | `DATASET/test_*` (82.003 ảnh) | Không quên miền cũ |
| I5 | CFE | `DATASET/test_communityforensics` | OOD |
| I6 | Eval v3 | [`tools/eval_communityforensics.py`](../../../../tools/eval_communityforensics.py) | Harness |
| I7 | Script in-dist | [`tools/eval_locked_benchmark.py`](../../../../tools/eval_locked_benchmark.py) | Nền để mở rộng |

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | JSON cuối | `logs/phase7/final_phase7_report.json` | `report_schema: v3`, đủ 3 khối |
| O2 | Markdown cuối | `logs/phase7/final_phase7_report.md` | Bảng + hình + phán quyết |
| O3 | ROC | `logs/phase7/roc_phase7.png` | 4 đường: clean, q30, in-dist, macro |
| O4 | Per-generator | `logs/phase7/per_generator_phase7.{csv,md}` | 5 subset × 4 nhóm × 2 điều kiện |
| O5 | Báo cáo | `docs/bao_cao/C/bao_cao_task_7.14_locked_benchmark.md` | Kết luận G7.2/G7.3 |

---

## 4. Ràng buộc

1. **Một lượt chấm duy nhất**: cấu hình chốt trước khi chạy. Nếu phải sửa lỗi kỹ thuật ⇒ chấm lại **toàn bộ**, ghi rõ lý do, không vá từng ô.
2. **`report_schema: "v3"`** + `git_commit` + `sha256` từng checkpoint (Phụ lục A.5).
3. **Không có số nào của Gen 1** trong báo cáo này (kể cả trong bảng so sánh). Nếu cần nhắc, đặt ở phụ lục với nhãn `invalid_layout`.
4. **In-dist phải báo Macro ACC**, không chỉ overall; và **Real ACC riêng** (G7.2 yêu cầu `Real ≥ 90%`).
5. **Không gộp subset**: mọi bảng phải giữ 5 subset × 4 nhóm generator.
6. **Mọi chỉ số có CI 95%** (bootstrap 1.000).
7. **Không đổi ngưỡng sau khi xem kết quả** — ngưỡng lấy từ `calibration.json` đã khoá.

---

## 5. Các bước thực hiện

### Bước 1 — Khoá cấu hình (D9, ~0,3 h)

Viết `logs/phase7/LOCKED_CONFIG.json` **trước** khi chấm:

```json
{
  "checkpoint": {"path": "...", "sha256": "..."},
  "router": {"mode": "router|fixed", "params_sha256": "..."},
  "calibration": {"T": 0.0, "T_star": {"0.01": 0.0, "0.05": 0.0, "0.10": 0.0}},
  "degrade_order": "post_resize",
  "variants": ["q75", "q50", "q30", "webp40", "blur15", "downup05"],
  "seed": 42,
  "locked_at": "2026-11-06T00:00:00Z"
}
```

Sau khi ghi file này, **không** sửa. Mọi thay đổi ⇒ chấm lại từ đầu.

### Bước 2 — In-dist 18 tập (D9, ~1,5 h)

```bash
python tools/eval_locked_benchmark.py \
  --checkpoint <ckpt> --data_root DATASET \
  --datasets <18 tên> --calibration logs/phase7/calibration/anchored_v2.json \
  --output logs/phase7/indist_phase7.json
```

- Dùng `load_canonical` (không `strict=False`).
- Với mỗi tập: `n`, `acc`, `real_acc`, `fake_acc`, `auc`, `tpr@fpr5`, `ci95`.
- Tổng hợp: **Macro ACC** (trung bình đều 18 tập) + **Micro ACC** (gộp) — báo cáo **cả hai**, nhưng Macro là chỉ số chính.
- Kiểm tra: tổng `n = 82.003`; nếu lệch ⇒ dừng, đối chiếu với A.

### Bước 3 — CFE clean + Q30 (D9, ~1 h)

Dùng đúng harness Task 7.3, cùng `T`, `T*`:

```bash
python tools/eval_communityforensics.py --checkpoint <ckpt> \
  --data_root DATASET/test_communityforensics --condition clean \
  --degrade_order post_resize --calibration <calib> --seed 42 \
  --output logs/phase7/cfe_clean_phase7.json

python tools/eval_communityforensics.py ... --condition q30 ...
```

### Bước 4 — 6 biến thể (D9–D10, ~1,5 h)

```bash
for v in q75 q50 q30 webp40 blur15 downup05; do
  python tools/eval_communityforensics.py --checkpoint <ckpt> \
    --data_root DATASET/test_communityforensics --condition $v \
    --degrade_order post_resize --calibration <calib> --seed 42 \
    --output logs/phase7/cfe_${v}_phase7.json
done
```

`q30` **phải trùng** kết quả bước 3 trong `0,1 pp` — nếu không, có phi tất định ⇒ dừng, điều tra.

### Bước 5 — Per-generator (D10, ~1 h)

Bảng chéo 5 subset × 4 nhóm generator × 2 điều kiện:

```csv
subset,generator_group,n_real,n_fake,auc_clean,tpr5_clean,fake_at_real90_clean,auc_q30,tpr5_q30,fake_at_real90_q30
1_commercial,diffusion,300,300,...
```

- Nhóm generator: `diffusion`, `gan`, `autoregressive`, `other` (theo metadata CFE).
- Ô có < 30 mẫu ⇒ ghi `—` + ghi chú "quá ít mẫu", **không** tính.
- Tìm **nhóm tệ nhất**: đây là nội dung chính của mục Hạn chế.

### Bước 6 — Tổng hợp & phán quyết (D10, ~1 h)

**Cấu trúc `final_phase7_report.json`**

```json
{
  "report_schema": "v3",
  "git_commit": "...",
  "locked_config": {...},
  "checkpoints": [{"path": "...", "sha256": "...", "n_keys": 0,
                   "adapter_blocks": [7,15,23], "n_used": 3,
                   "missing": 0, "unexpected": 0, "status": "ok"}],
  "calibration": {"source": "val_calib", "n_calib": 0, "T": 0.0, "T_star": {}},
  "indist": {"n_total": 82003, "macro_acc": 0.0, "micro_acc": 0.0,
             "real_acc": 0.0, "per_dataset": []},
  "cfe": {"clean": {}, "q30": {}, "variants": {}},
  "per_generator": [],
  "gates": {
    "G7.2": {"pass": false, "detail": {}},
    "G7.3": {"pass": false, "detail": {}}
  }
}
```

**Phán quyết G7.2** (kế hoạch §II.3)

| Điều kiện | Ngưỡng | Kết quả |
| :--- | :---: | :--- |
| `Fake@Real90` (CFE Q30) | ≥ 65% | |
| `AUC` (CFE clean) | ≥ 78% | |
| `AUC` (CFE Q30) | ≥ 60% | |
| Macro ACC in-dist | ≥ 90% | |
| `Real ACC` in-dist | ≥ 90% (cứng) | |
| `ECE` sau hiệu chuẩn | ≤ 0,08 | |

**Phán quyết G7.3**

| Điều kiện | Ngưỡng | Kết quả |
| :--- | :---: | :--- |
| Một checkpoint, một `forward()` | bắt buộc | |
| Chênh lệch vs ensemble/fixed | ≤ 1,5 pp | |
| Độ trễ trung bình | < 5 ms/ảnh (GPU) | |

Nếu G7.3 không đạt vì router: ghi rõ **đã fallback về ngưỡng cố định** (Task 7.13 §5.4).

### Bước 7 — Hình & báo cáo (D10, ~0,5 h)

- `roc_phase7.png`: 4 đường ROC (in-dist, clean, q30, macro) với AUC trong legend.
- `final_phase7_report.md`: bảng lớn + phán quyết + **3 câu kết luận**:
  1. Mô hình có tốt hơn Baseline **có ý nghĩa** không (CI)?
  2. Điều kiện nào còn yếu nhất?
  3. Có đạt G7.2/G7.3 không, nếu không thì thiếu bao nhiêu pp?

---

## 6. Definition of Done

- [ ] `LOCKED_CONFIG.json` được ghi **trước** khi chấm, có `sha256` checkpoint + router.
- [ ] In-dist: `n_total = 82.003`, có `macro_acc` **và** `real_acc`, CI cho từng tập.
- [ ] CFE clean + Q30 + **6 biến thể** đều có, `q30` trùng bước 3 trong 0,1 pp.
- [ ] `per_generator_phase7.csv` có 5 subset × 4 nhóm × 2 điều kiện; ô < 30 mẫu ghi `—`.
- [ ] `final_phase7_report.json` đúng `report_schema: v3`, có `gates` cho G7.2 và G7.3.
- [ ] `roc_phase7.png` có 4 đường + AUC trong legend.
- [ ] **Không** có số nào của Gen 1 trong báo cáo chính.
- [ ] Mọi bảng có cột CI 95%.
- [ ] Kết luận nêu rõ **nhóm tệ nhất** và **khoảng cách tới ngưỡng** (pp).
- [ ] `docs/bao_cao/C/bao_cao_task_7.14_locked_benchmark.md` hoàn tất, sẵn sàng trích dẫn.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
| :--- | :---: | :---: | :--- |
| Chấm xong mới phát hiện sai cấu hình | Trung bình | Phải chấm lại, tốn GPU | `LOCKED_CONFIG.json` + pre-flight 10 phút trước khi chấm |
| 82.003 ảnh vượt thời gian GPU | **Cao** | Trễ G7.3 | Chấm 1 tập nhỏ trước để ước lượng; dùng A100 nếu có |
| Lẫn ảnh test vào in-dist (trùng tên) | Trung bình | Số liệu sai | Đối chiếu `sha256` danh sách ảnh với A |
| `q30` hai lần cho kết quả khác | Trung bình | Phi tất định | Cùng seed, `torch.use_deterministic_algorithms`; nếu lệch > 0,1 pp ⇒ dừng |
| Chọn lọc kết quả vô thức (bỏ tập xấu) | Cao | Mất học thuật | Bảng cố định 18 tập + 5 subset; **không** được bỏ |
| Router thắng trên `val_select` nhưng thua ở đây | Trung bình | G7.3 trượt | Fallback ngưỡng cố định đã chuẩn bị |
| CI quá rộng ⇒ không kết luận | Cao | Không bảo vệ được | Báo cáo CI; chỉ kết luận khi chênh > nửa độ rộng CI |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) — §II.3 (G7.2, G7.3), §IX
- Khế ước: [`../../../plan/ke_hoach_test_communityforensics_eval.md`](../../../plan/ke_hoach_test_communityforensics_eval.md) — Phụ lục A
- Train: [`task_7.11_train_anchored_v2.md`](task_7.11_train_anchored_v2.md)
- Router: [`../../member_b_architecture_loss/phase_7/task_7.13_degradation_router.md`](../../member_b_architecture_loss/phase_7/task_7.13_degradation_router.md)
- Đóng gói: [`task_7.15_unified_packaging.md`](task_7.15_unified_packaging.md)
- Chỉ mục giai đoạn: [`README.md`](README.md)
