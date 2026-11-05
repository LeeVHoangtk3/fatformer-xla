# TASK 7.11 — HUẤN LUYỆN ANCHORED RECOVERY v2

- **Mã nhiệm vụ**: `Task 7.11` *(Phase 7 — D6–D7)*
- **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**
- **Người phối hợp**: **Thành viên A** (dữ liệu, kiểm toán lần 2) + **Thành viên B** (loss, selection, sửa lỗi train)
- **Môi trường**: kế hoạch T4 / L4 / A100; preflight/notebook lưu L4, benchmark lưu Blackwell. Chưa xác định đầy đủ GPU của từng phiên; chưa có bằng chứng cấu hình này chạy được trên T4 16 GB.
- **Độ ưu tiên**: 🔴 Khẩn cấp — **quyết định G7.2**
- **Cổng liên quan**: **G7.2** (cuối D8, sau Task 7.12)
- **Trạng thái**: `[~] Có log 5 epoch; epoch 4 tốt nhất trên val_select (Worst-Group 46.67%, Clean Real 97.97%). Chưa đủ bằng chứng nghiệm thu toàn bộ DoD; G7.2 chưa kết luận.` Xem [báo cáo kết quả](../../../bao_cao/C/bao_cao_task_7.11_train_v2.md).
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.11

---

## 1. Mục tiêu

Train **một** checkpoint duy nhất từ Baseline, dùng **dữ liệu mới + loss mới + chọn checkpoint mới**, nhắm mục tiêu:

$$\text{Fake@Real90} \ge 65\%,\quad \text{AUC} \ge 78\%,\quad \text{AUC}_{Q30} \ge 60\%,\quad \text{Macro in-dist} \ge 90\%,\quad \text{Real} \ge 90\%,\quad \text{ECE} \le 0{,}08$$

Đây là **cổng năng lực thật** của Phase 7. Nếu trượt, Phase 7 vẫn hợp lệ **chỉ khi** có checkpoint Pareto tốt nhất + ablation đòn bẩy thất bại + mục Hạn chế trung thực (kế hoạch §II.3).

**Vì sao warm start từ Baseline (không phải từ Anchored)?**

- Baseline là checkpoint **gốc**, không nhiễm dữ liệu Phase 6.
- Anchored đã qua một vòng fine-tune; warm start từ Anchored sẽ **cộng dồn** sai lệch của Phase 6 và khó quy trách nhiệm khi G7.2 trượt.
- SRM/Gating được thêm vào với **Zero-Init** (`SRM proj = 0`, `λ ≡ 1,0`) nhằm giữ đầu ra khởi tạo gần Baseline. Phải kiểm tra Step-0 identity trong run mới; không suy ra đóng góp riêng của dữ liệu hay loss từ thiết kế khởi tạo.

Đây là thiết kế khởi tạo và mục tiêu của task, không thay cho kiểm chứng từng run. Chưa có ablation thì chưa tách được đóng góp của dữ liệu, từng loss hay thành phần train được.

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Bắt buộc | Kiểm tra trước khi train |
| :---: | :--- | :--- | :---: | :--- |
| I1 | Manifest v2 | `DATASET/manifests/data_manifest_v2.json` (Task 7.7) | ✅ | ≥ 12.000 mẫu, `violations = 0` |
| I2 | Degradation v2 | `configs/degradation_v2.yaml` (Task 7.8) | ✅ | `clean_prob = 0.60` |
| I3 | Loss v2 | `src/training/loss.py` (Task 7.9) | ✅ | 3 test loss xanh |
| I4 | Selection v2 | `src/training/trainer.py` (Task 7.10) | ✅ | `worst_group` hoạt động |
| I5 | Baseline | `DATASET/pretrained/fatformer_4class_ckpt.pth` | ✅ | `verify 0/0` |
| I6 | Config | `configs/train_phase7_anchored_v2.yaml` | ✅ | khớp bảng §3 |
| I7 | GPU | T4 (16 GB) hoặc tốt hơn | ✅ | ≥ 4 h khả dụng |

**Chốt chặn**: nếu `DATASET/logs/phase7/data_isolation_report.json` **không** có `violations = 0` ⇒ **không được chạy**.

---

## 3. Cấu hình huấn luyện hiện tại

| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| Warm start | Baseline + Zero-Init SRM/Gating | `SRM proj = 0`, `λ ≡ 1,0` |
| `num_vit_adapter` | **3** (`{7,15,23}`) | Không đổi — ràng buộc vị trí |
| Backbone | **Đóng băng** | Chỉ train adapter + SRM + gating |
| LR adapter cũ | `1e-5` | Cosine schedule; warmup 5% chưa triển khai |
| LR SRM / Gating | `1e-4` | |
| Batch | 16, `grad_accum = 2` | Hiệu dụng 32; khác mục tiêu batch 64 trong kế hoạch gốc |
| AMP | FP16 | `torch.cuda.amp` |
| Epoch | 5; có thể cắt xuống 4 trước khi train | Early-stop patience 2 chưa triển khai |
| Dữ liệu | `train` 85%, `clean_prob = 0.60` | `val_select` để chọn |
| Loss | `0,3 CE_clean + 0,3 CE_deg + 0,7 KD + 0,2 Cons + 0,3 Feat + 0,5 Guard` | Task 7.9 |
| Seed | 42 | |
| Checkpoint | Mỗi epoch lên Drive | `--resume` bắt buộc |

**Ngân sách**: nếu tổng GPU < 8 h ⇒ **4 epoch**, ghi rõ trong báo cáo (kế hoạch §V.4).

---

## 4. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | Checkpoint chính | `DATASET/checkpoints/phase7/fatformer_anchored_v2_best.pth` | Chỉ ghi khi `clean_r ≥ 0,90` |
| O2 | Checkpoint dự phòng | `..._best_effort.pth` | `worst_group` cao nhất, kể cả không đạt ràng buộc Real |
| O3 | Lịch sử train | `<run_dir>/train_history.csv` | 6 cột nhóm + loss components + LR |
| O4 | Đường hội tụ | `<run_dir>/m7_convergence.png` | 4 recall + worst + loss |
| O5 | Benchmark bước | `DATASET/logs/phase7/train_benchmark_100steps.json` | ms/step, VRAM, ước lượng/epoch |
| O6 | Báo cáo | `docs/bao_cao/C/bao_cao_task_7.11_train_v2.md` | Thời gian thực, epoch dừng, lý do |

---

## 5. Các bước thực hiện

### Bước 1 — Pre-flight (D6, ~0,5 h)

Checklist tự động (viết `tools/preflight_phase7_train.py`):

```text
[ ] manifest v2 tồn tại, n ≥ 12.000
[ ] isolation report: violations == 0
[ ] degradation_v2.yaml: clean_prob == 0.60
[ ] loss test: loss_v2 + smoke integration (26 cases)
[ ] selection test: selection_metric + smoke + regression (31 cases)
[ ] baseline load_canonical: missing=0 unexpected=0
[ ] val_calib != val_select (so sha256 danh sách ảnh)
[ ] GPU free memory ≥ 14 GB (T4)
```

In PASS/FAIL/SKIP. Chế độ train chỉ `ready_to_train=true` khi đủ tám PASS; CPU không đạt gate GPU. Audit có SKIP không được dùng làm bằng chứng sẵn sàng. Kiểm tra cả ảnh stage/SHA256, audit bound đúng manifest/config và split path/SHA256; không chỉ đếm records.

```bash
python -m tools.preflight_phase7_train --config configs/train_phase7_anchored_v2.yaml --output DATASET/logs/phase7/preflight_train.json
```

### Bước 2 — Benchmark 100 step (D6, ~15 phút GPU)

```bash
python -m src.training.train_crossdomain --config configs/train_phase7_anchored_v2.yaml \
  --benchmark_steps 100 --benchmark_warmup 5 \
  --save_dir scratch/phase7_benchmark --output DATASET/logs/phase7/train_benchmark_100steps.json
```

Ghi `ms_per_step` theo mini-batch, `peak_vram_gib`/`peak_reserved_gib`, `steps_per_epoch`, `est_minutes_per_epoch`, `est_total_minutes`. Benchmark không ghi checkpoint/CSV train và không được kết hợp resume/max_steps. Ước lượng chưa tính validation/backup.

**Luật**: nếu `est_total_minutes > 300` trên T4 ⇒ giảm epoch xuống 4 **trước khi** bắt đầu (không dừng giữa chừng).

### Bước 3 — Train (D6–D7, ~3–5 h GPU)

```bash
python -m src.training.train_crossdomain --config configs/train_phase7_anchored_v2.yaml \
  --resume_from DATASET/checkpoints/phase7/checkpoint_latest.pth \
  --save_dir DATASET/checkpoints/phase7 \
  --history_csv DATASET/checkpoints/phase7/train_history.csv
```

Lượt mới bỏ `--resume_from` và dùng run directory mới. Resume chỉ nhận full snapshot ở epoch hoàn tất; model-only best và partial `--max_steps` không phải resume. Tắt giữa epoch sẽ chạy lại epoch bị ngắt. Notebook mới kiểm checksum, dùng config gốc trong archive và tự chọn full latest nếu có; không cam kết bit-exact qua môi trường GPU khác.

**Theo dõi mỗi epoch**:

| Cột | Ý nghĩa | Ngưỡng cảnh báo |
| :--- | :--- | :--- |
| `clean_r` | Real ACC clean | < 0,85 ⇒ rủi ro quên |
| `clean_f` | Fake ACC clean | < 0,90 ⇒ underfit |
| `q30_r` | Real ACC Q30 | < 0,75 ⇒ guard chưa hiệu quả |
| `q30_f` | Fake ACC Q30 | < 0,70 ⇒ robustness kém |
| `worst` | min 4 giá trị | phải tăng ≥ 2 pp sau 2 epoch |
| `loss_total` | + 5 thành phần | `guard` không được = 0 suốt 3 epoch |

**Dừng sớm có kiểm soát (theo dõi thủ công; chưa có early-stop tự động)**:

- Nếu sau 3 epoch `worst` không tăng ≥ 1 pp ⇒ dừng, ghi rõ, chuyển sang đòn bẩy dự phòng.
- Nếu `loss` NaN ⇒ dừng ngay, kiểm tra `fake_index`, LR, và `guard` (xem Task 7.9 §7).

### Bước 4 — Đòn bẩy dự phòng (D7, chỉ khi cần)

Theo kế hoạch §VI, **chỉ bật khi** `worst_group` không tăng ≥ 2 pp sau 6 epoch (hoặc 4 nếu cắt):

| Đòn bẩy | Cấu hình | Chi phí | Điều kiện bật |
| :--- | :--- | :--- | :--- |
| Mở khoá nhẹ | `resblocks.20–23`, LR `1e-6` | +20% thời gian | Ưu tiên 1 |
| LoRA | `r = 8`, `alpha = 16`, trên `qkv` + `proj` | +35% thời gian | Nếu mở khoá không đủ |

**Bắt buộc**: chạy **một** đòn bẩy tại một thời điểm, cùng seed, ghi vào báo cáo như một **run riêng**. Không trộn nhiều thay đổi.

### Bước 5 — Lưu & vẽ (D7, ~0,5 h)

1. `best.pth` chỉ khi `clean_r ≥ 0,90`; nếu không, `best_effort.pth` (Task 7.10).
2. `m7_convergence.png`: 2 panel — (a) 4 recall + worst theo epoch, (b) `loss_total` + 5 thành phần.
3. Ghi `train_history.csv` với **đủ** cột: `epoch, clean_r, clean_f, q30_r, q30_f, worst, loss_total, ce_clean, ce_deg, kd, feat, guard, lr, minutes`.
4. Chạy `load_canonical` trên checkpoint cuối: in `missing/unexpected/layout` và **lưu vào báo cáo**.

---

## 6. Definition of Done

- [x] Preflight đã lưu đủ 8 PASS: L4 21.66 GiB trống, 14.400 ảnh staged khớp SHA256, `ready_to_train=true` tại `DATASET/logs/anchored_v2_run1/preflight_train.json`. Đây là bằng chứng trước train, không xác minh checkpoint cuối.
- [x] Có JSON benchmark 100 step trên Blackwell: 162.04 ms/step, 2.07 phút/epoch ước lượng, peak allocated 18.23 GiB. Notebook lưu phiên L4 và bỏ qua benchmark khi resume; chưa làm rõ lịch sử từng phiên. Không suy số đo này sang L4/T4.
- [x] `train_history.csv` có 42 cột, đủ 5 epoch và các trường nhóm/loss/LR/thời gian. `grad_norm` NaN tại epoch 1, 4; cần điều tra, không coi log đủ cột là bằng chứng train hoàn toàn ổn định.
- [x] `m7_convergence.png` có 2 panel, đọc được và khớp CSV tại `DATASET/logs/anchored_v2_run1/`.
- [ ] `best.pth` **hoặc** `best_effort.pth` tồn tại và được kiểm chứng. CSV hỗ trợ chọn epoch 4 (Worst-Group 46.67%, Clean Real 97.97%); chưa trực tiếp kiểm tệp `.pth` trên Drive. Cần xác minh tệp best chứa đúng trọng số epoch đó.
- [ ] Checkpoint cuối `verify 0/0`, `layout.n = 3`, `dropped = []`. Chưa có output canonical load sau train trong bộ hiện vật; không dùng kết quả load baseline thay thế.
- [x] Thời gian mỗi epoch và tổng có trong CSV: 130.54, 129.79, 130.40, 129.93, 720.83 giây; tổng 1,241.50 giây (~20.69 phút). Đây là train+validation, không phải toàn bộ wall-clock phiên hoặc thời gian backup.
- [x] Log không cho thấy worst-group trì trệ theo ngưỡng theo dõi: tăng từ 0% đến 46.67% tại epoch 4, còn 38% tại epoch 5. Chưa có bằng chứng chạy đòn bẩy riêng; không suy ra đã đạt G7.2.
- [x] Log ghi đủ epoch 1–5, khớp horizon 5 epoch của cấu hình runtime. Notebook đã lưu có resume; không khẳng định đây là một phiên chạy liên tục.
- [x] Báo cáo kết quả đã ghi cả giới hạn và phần cần bổ sung: [báo cáo Task 7.11](../../../bao_cao/C/bao_cao_task_7.11_train_v2.md).

**Phần nghiệm thu còn mở**: kiểm chứng tệp checkpoint và canonical load sau train; điều tra gradient NaN; làm rõ lịch sử GPU/benchmark/resume. Các chỉ số validation hiện tại chưa thay cho phép đo G7.2 hoặc ablation Task 7.12.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu | Kill criteria |
| :--- | :---: | :---: | :--- | :--- |
| Backbone đóng băng ⇒ trần năng lực thấp | **Cao** | G7.2 trượt | Chuẩn bị 2 đòn bẩy; bật khi `worst` bình trị | Sau 6 epoch `worst` không tăng ≥ 2 pp ⇒ bật đòn bẩy |
| Colab hết hạn giữa phiên | Trung bình | Mất tiến độ | Checkpoint mỗi epoch lên Drive + `--resume` | — |
| OOM trên T4 (batch 32 + AMP) | Trung bình | Không chạy được | `grad_accum = 2`, `batch = 16` nếu cần; **không** đổi LR | — |
| `guard` phạt nhầm do nhãn Real sai | Trung bình | Học lệch | Theo dõi `guard_excess_mean`; nếu > 1,0 từ epoch 1 ⇒ audit 100 ảnh Real | — |
| Chọn checkpoint theo `worst` nhưng `clean_r` không đạt 90% | Cao | Không có `best.pth` | Vẫn có `best_effort`; báo C từ epoch 2 | — |
| Loss tổng lớn (2,1) gây bất ổn | Trung bình | NaN / phân kỳ | Log grad norm; nếu > 2× baseline ⇒ giảm LR 2×, ghi rõ | NaN ⇒ dừng, sửa |
| Dữ liệu v2 lệch phân phối so với test | Trung bình | Tăng ảo | So sánh histogram `family` train vs CFE test trong báo cáo | — |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) — §II.3 (G7.2), §V.4
- Dữ liệu: [`../../member_a_data_infra/phase_7/task_7.7_kho_du_lieu_v2.md`](../../member_a_data_infra/phase_7/task_7.7_kho_du_lieu_v2.md)
- Suy biến: [`../../member_a_data_infra/phase_7/task_7.8_degradation_pipeline_v2.md`](../../member_a_data_infra/phase_7/task_7.8_degradation_pipeline_v2.md)
- Loss: [`../../member_b_architecture_loss/phase_7/task_7.9_loss_v2_realguard.md`](../../member_b_architecture_loss/phase_7/task_7.9_loss_v2_realguard.md)
- Selection: [`../../member_b_architecture_loss/phase_7/task_7.10_selection_metric_v2.md`](../../member_b_architecture_loss/phase_7/task_7.10_selection_metric_v2.md)
- Ablation: [`task_7.12_ablation_mini_v2.md`](task_7.12_ablation_mini_v2.md)
- Chỉ mục giai đoạn: [`README.md`](README.md)
