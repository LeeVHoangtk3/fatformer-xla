# TASK 7.12 — ABLATION MINI 4 CẤU HÌNH (CÙNG SEED 42)

- **Mã nhiệm vụ**: `Task 7.12` *(Phase 7 — D7–D8)*
- **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)** — chạy
- **Người phối hợp**: **Thành viên B (Architecture & Loss Lead)** — review kết luận đòn bẩy
- **Môi trường thực thi**: ☁️ **T4**
- **Độ ưu tiên**: 🟠 Cao — quyết định **G7.2** và nội dung mục "Đóng góp" của luận văn
- **Cổng liên quan**: **G7.2** (cuối D8)
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.12

---

## 1. Mục tiêu

Trả lời **một** câu hỏi có thể bảo vệ trước hội đồng:

> "Nếu kết quả tốt lên, điều gì thực sự tạo ra nó — dữ liệu mới, `RealGuardLoss`, hay `FeatureConsistencyLoss`?"

**4 cấu hình**, **cùng seed 42**, **cùng manifest v2**, **cùng số epoch**, **cùng siêu tham số** — chỉ khác thành phần loss:

| # | Tên | Dữ liệu | `L_CE` | `L_KD` | `L_Feat` | `L_Guard` | Ý nghĩa |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| A1 | `+data` | v2 | ✅ | ✅ | ✗ | ✗ | Đòn bẩy **dữ liệu** |
| A2 | `+data+guard` | v2 | ✅ | ✅ | ✗ | ✅ | Thêm **RealGuard** |
| A3 | `+data+featcons` | v2 | ✅ | ✅ | ✅ | ✗ | Thêm **FeatureConsistency** |
| A4 | `full` | v2 | ✅ | ✅ | ✅ | ✅ | Đầy đủ (trùng Task 7.11) |

> **A4 phải trùng Task 7.11** (cùng seed, cùng config). Nếu lệch > 1 pp ⇒ có phi tất định ⇒ phải điều tra **trước khi** kết luận.

**Quy ước đọc kết quả**

- `Δ(A2 − A1)` = đóng góp của **RealGuard**.
- `Δ(A3 − A1)` = đóng góp của **FeatureConsistency**.
- `Δ(A4 − max(A2, A3))` = đóng góp **tương tác** (nếu âm ⇒ hai loss triệt tiêu nhau — phải báo cáo).

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | Manifest v2 | `DATASET/manifests/data_manifest_v2.json` | Dữ liệu chung |
| I2 | Degradation v2 | `configs/degradation_v2.yaml` | Chung |
| I3 | Loss v2 | `src/training/loss.py` (Task 7.9) | Có cờ bật/tắt từng thành phần |
| I4 | Selection v2 | `src/training/trainer.py` (Task 7.10) | Chọn checkpoint |
| I5 | Config gốc | `configs/train_phase7_anchored_v2.yaml` | Sinh 4 config con |
| I6 | Checkpoint A4 | `DATASET/checkpoints/phase7/fatformer_anchored_v2_best.pth` | Đối chiếu |

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | 4 config | `configs/ablation_v2/a1_data.yaml` … `a4_full.yaml` | Chỉ khác trọng số loss |
| O2 | Bảng kết quả | `logs/phase7/ablation_mini_v2.csv` | 4 dòng × 8 chỉ số |
| O3 | JSON | `logs/phase7/ablation_mini_v2.json` | Kèm `ci95` + `delta` |
| O4 | Hình | `logs/phase7/m7_ablation.png` | Bar chart 4 cấu hình × 4 chỉ số chính |
| O5 | Báo cáo | `docs/bao_cao/C/bao_cao_task_7.12_ablation.md` | Kết luận đòn bẩy nào đóng góp nhiều nhất |

---

## 4. Ràng buộc

1. **Cùng seed 42**, cùng thứ tự dữ liệu (`DataLoader` seed), cùng `num_workers`. Sai khác duy nhất: trọng số loss.
2. **Cùng số epoch** cho cả 4 cấu hình. Nếu cắt ngân sách ⇒ cắt **đều** (3 epoch cho cả 4), không cắt riêng.
3. **Cùng luật chọn checkpoint** (`worst_group` + ràng buộc `clean_r ≥ 0,90`).
4. **Không chọn epoch tốt nhất cho từng cấu hình** — dùng checkpoint do Trainer lưu.
5. **Không chạy thêm cấu hình thứ 5** để "cứu" kết quả. Nếu cần, đó là Phase 8.
6. **Báo cáo cả kết quả âm**: nếu `Δ` âm, ghi rõ; không được chỉ báo cáo cấu hình tốt.

---

## 5. Các bước thực hiện

### Bước 1 — Sinh 4 config (D7, ~0,5 h)

```yaml
# configs/ablation_v2/a1_data.yaml
inherits: ../train_phase7_anchored_v2.yaml
loss:
  weight_feat: 0.0
  weight_guard: 0.0
```

Tương tự:

| File | `weight_feat` | `weight_guard` |
| :--- | ---: | ---: |
| `a1_data.yaml` | 0,0 | 0,0 |
| `a2_data_guard.yaml` | 0,0 | 0,5 |
| `a3_data_featcons.yaml` | 0,3 | 0,0 |
| `a4_full.yaml` | 0,3 | 0,5 |

Viết script kiểm tra: 4 config chỉ khác **đúng 2 khoá** này; mọi khoá khác phải giống hệt (so hash của dict đã loại 2 khoá).

### Bước 2 — Chốt ngân sách (D7, ~0,2 h)

| Ngân sách GPU còn lại | Số epoch | Ghi chú |
| :--- | :---: | :--- |
| ≥ 8 h | 6 | Đầy đủ |
| 4–8 h | 3 | Cắt đều, ghi rõ |
| < 4 h | 3, chỉ A1 và A4 | Cắt xuống **2 cấu hình** (theo kế hoạch §V.4) |

Nếu cắt xuống 2 cấu hình: chỉ trả lời được "dữ liệu + full", **không** tách được guard/featcons. Ghi rõ hạn chế này.

### Bước 3 — Chạy 4 cấu hình (D7–D8, ~3–6 h GPU)

```bash
for cfg in a1_data a2_data_guard a3_data_featcons a4_full; do
  python -m src.training.train_crossdomain \
    --config configs/ablation_v2/${cfg}.yaml \
    --seed 42 --save_dir DATASET/checkpoints/phase7/ablation/${cfg} \
    --history_csv logs/phase7/ablation_${cfg}_history.csv
done
```

Theo dõi: mỗi run phải có **cùng** `n_train`, `n_val`, và `steps_per_epoch` (kiểm tra từ log).

### Bước 4 — Chấm 8 chỉ số (D8, ~1 h GPU)

Trên `val_select` (Q30 in-memory) + CFE clean/Q30:

| # | Chỉ số | Nguồn |
| :---: | :--- | :--- |
| 1 | `worst_group` | `val_select` |
| 2 | `clean_r` | `val_select` |
| 3 | `q30_r` | `val_select` |
| 4 | `q30_f` | `val_select` |
| 5 | `fake_at_real90` (Q30) | CFE |
| 6 | `auc_clean` | CFE |
| 7 | `auc_q30` | CFE |
| 8 | `ece` | CFE (sau hiệu chuẩn) |

Kèm `ci95` cho 5, 6, 7.

CSV:

```csv
config,epochs,worst_group,clean_r,q30_r,q30_f,fake_at_real90_q30,auc_clean,auc_q30,ece_after
a1_data,6,...
a2_data_guard,6,...
a3_data_featcons,6,...
a4_full,6,...
```

### Bước 5 — Phân tích đòn bẩy (D8, ~0,5 h)

```python
delta_guard    = a2.worst - a1.worst
delta_featcons = a3.worst - a1.worst
delta_interact = a4.worst - max(a2.worst, a3.worst)
```

**Luật kết luận** (theo Phụ lục A.8 — chênh lệch phải lớn hơn nửa độ rộng CI):

| Điều kiện | Kết luận |
| :--- | :--- |
| `Δ ≥ +2 pp` và CI không chồng | Đòn bẩy **có đóng góp** |
| `-1 < Δ < +2 pp` | **Không kết luận được** — ghi "không có bằng chứng" |
| `Δ ≤ −1 pp` | Đòn bẩy **phản tác dụng** — ghi rõ, xem xét bỏ |

Trả lời dứt khoát: **đòn bẩy nào đóng góp nhiều nhất**, kèm số.

### Bước 6 — Hình & báo cáo (D8, ~0,5 h)

`m7_ablation.png`: 4 nhóm cột (A1–A4) × 4 chỉ số (`worst_group`, `fake@real90`, `auc_q30`, `clean_r`), có error bar CI.

Báo cáo phải có:

1. Bảng 4 × 8.
2. 3 giá trị `Δ` + phán quyết.
3. Câu trả lời 1 dòng: "Đòn bẩy đóng góp nhiều nhất là **…**, +x,x pp trên `worst_group`."
4. Nếu cắt epoch/2 cấu hình: ghi rõ ảnh hưởng tới khả năng kết luận.

---

## 6. Definition of Done

- [ ] 4 config tồn tại, **chỉ khác** `weight_feat` và `weight_guard` (đã kiểm bằng hash).
- [ ] 4 run cùng seed 42, cùng số epoch, cùng `steps_per_epoch` (log chứng minh).
- [ ] `logs/phase7/ablation_mini_v2.csv` có **4 dòng × 8 chỉ số**.
- [ ] `ablation_mini_v2.json` có `ci95` cho ít nhất 3 chỉ số chính và 3 giá trị `delta`.
- [ ] `m7_ablation.png` có error bar, đọc được.
- [ ] A4 khớp Task 7.11 trong **1 pp**; nếu lệch ⇒ có ghi chú điều tra.
- [ ] Kết luận nêu **đòn bẩy đóng góp nhiều nhất** kèm số pp; kết quả âm được báo cáo trung thực.
- [ ] Nếu cắt ngân sách: ghi rõ số epoch và hệ quả lên khả năng kết luận.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
| :--- | :---: | :---: | :--- |
| Phi tất định ⇒ 4 run không so sánh được | **Cao** | Kết luận vô nghĩa | Cùng seed + `torch.use_deterministic_algorithms(True)` nếu có thể; ghi rõ nếu không |
| A4 lệch Task 7.11 | Trung bình | Nghi ngờ pipeline | So `steps_per_epoch`, `n_train`, loss khởi đầu; nếu lệch > 1 pp ⇒ dừng, điều tra |
| 4 run vượt ngân sách GPU | **Cao** | Trễ G7.2 | Chốt ngân sách **trước**; cắt đều xuống 3 epoch |
| `Δ` nằm trong nhiễu (CI rộng) | Cao | Không kết luận được | Báo cáo "không có bằng chứng", **không** suy diễn |
| Guard và FeatCons triệt tiêu nhau | Trung bình | A4 < max(A2,A3) | Báo cáo `delta_interact` âm; đề xuất Phase 8 tách |
| Chọn checkpoint khác nhau giữa các run do ngưỡng | Trung bình | So sánh sai | Cùng `val_select`, cùng ngưỡng từ Task 7.4 |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) — §IV.2, §V.4
- Train chính: [`task_7.11_train_anchored_v2.md`](task_7.11_train_anchored_v2.md)
- Loss: [`../../member_b_architecture_loss/phase_7/task_7.9_loss_v2_realguard.md`](../../member_b_architecture_loss/phase_7/task_7.9_loss_v2_realguard.md)
- Khế ước CI: [`../../../plan/ke_hoach_test_communityforensics_eval.md`](../../../plan/ke_hoach_test_communityforensics_eval.md) — A.8
- Chỉ mục giai đoạn: [`README.md`](README.md)
