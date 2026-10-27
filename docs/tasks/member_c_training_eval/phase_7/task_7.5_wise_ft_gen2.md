# TASK 7.5 — WiSE-FT **GEN 2** + LƯỚI `α × β_srm` + `--verify`

- **Mã nhiệm vụ**: `Task 7.5` *(Phase 7 — D3–D4)*
- **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**
- **Người phối hợp**: **Thành viên B** — review công thức trộn & xác nhận `0/0`
- **Môi trường thực thi**: 🖥️ **Local CPU** (merge) + ☁️ **T4 ~25 phút** (chấm 12 ứng viên)
- **Độ ưu tiên**: 🟠 Cao — quyết định **G7.1**
- **Cổng liên quan**: **G7.1** (cuối D4)
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.5

---

## 1. Mục tiêu

Sinh **Gen 2** đúng nghĩa: trộn **Baseline** và **Anchored** — hai checkpoint **cùng `num_vit_adapter = 3`**, cùng tập khối `{7,15,23}` — thay vì trộn Baseline với Gen 1 (10 khối, không biểu diễn được).

**Vì sao Gen 1 không thể tái sử dụng**

| | Baseline | Anchored | Gen 1 (cũ) |
| :--- | :--- | :--- | :--- |
| Khối adapter | `{7,15,23}` | `{7,15,23}` | `{2,5,7,8,11,14,15,17,20,23}` |
| `n` khả biểu | 3 | 3 | **không có** |
| Trộn với Baseline | ✅ hợp lệ | ✅ hợp lệ | ❌ tạo 10 khối, không nạp được |

Vì vậy: **Gen 2 = Baseline ⊕ Anchored**, cả hai `n=3`. Không có ngoại lệ.

**Hai trục tìm kiếm**

- `α`: tỷ lệ trộn trên **key chung** — `θ = (1−α)·θ_baseline + α·θ_anchored`.
- `β_srm`: hệ số co cho **projection cuối của SRM** (key chỉ có ở Anchored) — `θ = β_srm · θ_anchored`.

Lưới: `α ∈ {0,35; 0,45; 0,55; 0,65}` × `β_srm ∈ {0,25; 0,35; 0,50}` = **12 ứng viên**.

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | Baseline (1.116 tensor, không SRM) | `DATASET/checkpoints/fatformer_4class_ckpt.pth` | Thành phần `θ_baseline` |
| I2 | Anchored (1.200 tensor, có SRM + Gating) | `DATASET/checkpoints/phase6_r/fatformer_anchored_best.pth` | Thành phần `θ_anchored` |
| I3 | Script hiện tại | [`tools/wise_ft_merge.py`](../../../../tools/wise_ft_merge.py) | Đối tượng vá (đang có thay đổi chưa commit — **đọc diff trước**) |
| I4 | `canonical_loader` | Task 7.2 | Dùng cho `--verify` |
| I5 | `calibration.json` | Task 7.4 | `T`, `T*` để chấm |
| I6 | CFE clean + Q30 | `DATASET/test_communityforensics` | Chấm 12 ứng viên |

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | Script vá | `tools/wise_ft_merge.py` | 5 vá §5.1, có `--verify` |
| O2 | 12 checkpoint ứng viên | `DATASET/checkpoints/phase7/candidates/wise_ft_v2_a<a>_b<b>.pth` | Không ghi đè tên cũ |
| O3 | Bảng sweep | `logs/phase7/wise_ft_v2_sweep.csv` | 12 dòng × đầy đủ chỉ số |
| O4 | Checkpoint thắng | `DATASET/checkpoints/phase7/fatformer_wise_ft_v2_best.pth` | `verify 0/0` |
| O5 | Metadata | `..._best_metadata.json` | `alpha, beta_srm, sha256, layout, n_keys` |
| O6 | Báo cáo | `docs/bao_cao/C/bao_cao_task_7.5_wise_ft_gen2.md` | Bảng 12 dòng + phán quyết G7.1 |

---

## 4. Ràng buộc

1. **Chỉ trộn hai checkpoint cùng `num_vit_adapter`**. Assert trước khi trộn: `adapter_blocks(baseline) == adapter_blocks(anchored) == {7,15,23}`.
2. **Không tái sử dụng tên file cũ** `fatformer_wise_ft_best.pth` — tên đó đã gắn với `invalid_layout`.
3. **Mọi ứng viên phải qua `--verify`** trước khi được chấm. `missing ≠ 0` hoặc `unexpected ≠ 0` ⇒ `sys.exit(1)`, không chấm.
4. **Chọn theo luật công bố trước**: tối đa `Fake@Real90`; hoà (chênh < 0,5 pp) ⇒ ưu tiên `AUC Q30` cao hơn.
5. **Không chọn theo tập test rồi báo cáo lại trên chính tập đó** — chọn trên `val_select` + Q30 của `val_select`; báo cáo trên CFE.
6. **Metadata bắt buộc** có `alpha`, `beta_srm`, `sha256` của **hai file nguồn**, `layout`, `n_keys`, `git_commit`.

---

## 5. Các bước thực hiện

### Bước 1 — Vá `tools/wise_ft_merge.py` (D3, ~2 h)

#### (a) Bỏ `srm.*` / `gating.*` top-level khỏi output

Hiện tại nhánh `target_only_keys` copy **mọi** key còn lại bằng `t_targ.clone()`, kể cả `srm.*` và `gating.*` ở cấp cao nhất. Model không có các module đó ⇒ `unexpected keys`.

```python
TOP_LEVEL_DROP_PREFIXES = ("srm.", "gating.")

def _is_top_level_module_key(k: str) -> bool:
    kl = k.lower()
    return any(kl.startswith(p) for p in TOP_LEVEL_DROP_PREFIXES)
```

Trong cả hai vòng lặp, `if _is_top_level_module_key(k): dropped_top_level += 1; continue`.
Ghi `dropped_top_level_keys` vào `stats` và **assert** danh sách này nằm trong whitelist đã biết (không bỏ nhầm `srm_block.*`).

> **Phân biệt quan trọng**: `srm_block.*` (nằm **trong** block) là tham số thật, phải giữ. Chỉ bỏ `srm.*` **cấp gốc**.

#### (b) `is_fixed_buffer` khớp chính xác

Hiện tại:

```python
return any(b in k_lower for b in ["k1", "k2", "k3", "srm_filter", "haar_kernel"])
```

`"k1" in "srm_block1.conv.weight"` → **True** (vì `block1` chứa `k1`). Hậu quả: tensor học được bị coi là buffer cố định và **copy nguyên**, không nhân `β_srm`, không trộn `α`.

Sửa:

```python
_FIXED_BUFFER_RE = re.compile(
    r"(^|\.)(srm_block\d*\.)?(k[123]|srm_filter|haar_kernel)$"
)

def is_fixed_buffer(key: str) -> bool:
    return bool(_FIXED_BUFFER_RE.search(key.lower()))
```

Test bắt buộc:

| Key | Kỳ vọng |
| :--- | :---: |
| `srm_block.k1` | True |
| `srm_block1.k2` | True |
| `srm_block1.conv.weight` | **False** |
| `clip_model.visual.transformer.resblocks.7.attn.adapter_k1.weight` | **False** |

#### (c) Thêm `--verify`

```python
if args.verify:
    from src.models.canonical_loader import load_canonical
    model, rep = load_canonical(output_path, model_args)
    print(f"[VERIFY] missing={len(rep['missing'])} unexpected={len(rep['unexpected'])} "
          f"layout={rep['layout']}")
    if rep["missing"] or rep["unexpected"] or rep["layout"]["partial"]:
        sys.exit(1)
```

Verify phải chạy **trước** khi ghi metadata và **trước** khi chấm.

#### (d) Lưới `α × β_srm`

```bash
python tools/wise_ft_merge.py \
  --orig_checkpoint DATASET/checkpoints/fatformer_4class_ckpt.pth \
  --target_checkpoint DATASET/checkpoints/phase6_r/fatformer_anchored_best.pth \
  --grid_search --alpha_grid 0.35,0.45,0.55,0.65 \
  --beta_grid 0.25,0.35,0.50 \
  --output_dir DATASET/checkpoints/phase7/candidates \
  --verify
```

Sửa tham số `--m4_checkpoint` (tên cũ, gây hiểu nhầm) thành `--target_checkpoint`; giữ alias `--m4_checkpoint` với `deprecated=True`.

**(e) Metadata**

```json
{
  "schema": "wise_ft_v2/v1",
  "alpha": 0.45,
  "beta_srm": 0.35,
  "sources": {
    "baseline": {"path": "...", "sha256": "...", "n_keys": 1116},
    "anchored": {"path": "...", "sha256": "...", "n_keys": 1200}
  },
  "layout": {"n": 3, "indices": [7,15,23], "dropped": [], "partial": false},
  "n_keys": 1200,
  "dropped_top_level_keys": 0,
  "verification": {"missing": 0, "unexpected": 0, "status": "ok"},
  "git_commit": "..."
}
```

**Xoá `sweep_threshold_constrained`** (dead code — không được gọi ở đâu, gây hiểu nhầm là đã có hiệu chuẩn).

### Bước 2 — Chạy 12 ứng viên trên CPU (D3, ~1 h)

```bash
for a in 0.35 0.45 0.55 0.65; do
  for b in 0.25 0.35 0.50; do
    python tools/wise_ft_merge.py --orig_checkpoint ... --target_checkpoint ... \
      --alpha $a --srm_scale $b --verify \
      --output_path DATASET/checkpoints/phase7/candidates/wise_ft_v2_a${a}_b${b}.pth
  done
done
```

Kiểm tra nhanh: 12/12 file tồn tại, mỗi file ~2 GB, `verify` in `missing=0 unexpected=0`.

### Bước 3 — Chấm 12 ứng viên (D4, ~25 phút T4)

Với mỗi ứng viên, chấm trên **`val_select`** (không phải CFE):

- clean + Q30 (`post_resize`), `T` và `T*` từ Task 7.4;
- 2 chỉ số quyết định: `Fake@Real90` và `AUC Q30`.

> `Fake@Real90` = **Fake ACC khi ngưỡng được đặt sao cho Real ACC = 90%** (tức `FPR = 10%`). Đây là chỉ số của G7.1, không phải `Fake ACC @ 0,5`.

Ghi `logs/phase7/wise_ft_v2_sweep.csv`:

```csv
alpha,beta_srm,n_keys,verify,real_acc_clean,fake_at_real90_clean,auc_clean,
real_acc_q30,fake_at_real90_q30,auc_q30,selected
```

### Bước 4 — Chọn & lưu (D4, ~0,5 h)

```python
key = (fake_at_real90_clean, auc_q30)      # ưu tiên 1, hoà thì ưu tiên 2
best = max(candidates, key=key)
```

- Hoà `fake_at_real90` trong 0,5 pp ⇒ so `auc_q30`.
- Lưu `fatformer_wise_ft_v2_best.pth` + `_best_metadata.json`.
- Chạy lại `--verify` trên file best, `missing=0 unexpected=0`.

### Bước 5 — Báo cáo (D4, ~0,5 h)

Bảng 12 dòng + 4 cột chỉ số; nêu rõ **ứng viên thắng và biên độ**. Nếu mọi ứng viên đều **kém hơn Baseline** ⇒ ghi rõ: "WiSE-FT Gen 2 không cải thiện; G7.1 chỉ đạt nhờ Anchored/Ensemble" và **không** chọn Gen 2 làm kết quả chính.

---

## 6. Definition of Done

- [ ] `is_fixed_buffer` khớp **chính xác**, có 4 test ở §5.1(b) xanh.
- [ ] Output **không còn** key `srm.*` / `gating.*` top-level; `dropped_top_level_keys` được ghi.
- [ ] `--verify` tồn tại, `sys.exit(1)` khi `missing/unexpected/partial` khác 0.
- [ ] Lưới đủ **12** ứng viên (`α` 4 × `β` 3), có `--alpha_grid`/`--beta_grid`.
- [ ] Metadata đủ `alpha, beta_srm, sha256 (2 nguồn), layout, n_keys, git_commit`.
- [ ] `logs/phase7/wise_ft_v2_sweep.csv` có **12 dòng** + cột `selected`.
- [ ] Best checkpoint `verify 0/0`, `layout.partial = false`.
- [ ] **Không** ghi đè `fatformer_wise_ft_best.pth` cũ.
- [ ] `sweep_threshold_constrained` đã bị xoá (grep rỗng).
- [ ] Báo cáo nêu rõ phán quyết G7.1 pass/fail **có số**.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
| :--- | :---: | :---: | :--- |
| Còn key top-level bị bỏ nhầm (`srm_block.*`) | Trung bình | Mất tham số SRM | Whitelist + assert; so `n_keys` với 1.200 |
| `β_srm = 0` làm SRM vô hiệu | Thấp (lưới ≥ 0,25) | Mất nhánh forensic | Lưới đã chặn; báo cáo AUC Q30 |
| Chọn trên CFE rồi báo cáo trên CFE | **Cao** | Số liệu ảo | Chọn trên `val_select`; ghi rõ trong CSV |
| Gen 2 thắng trên `val_select` nhưng thua trên CFE | Trung bình | G7.1 trượt | Báo cáo **cả hai**; kết luận theo CFE |
| 12 file × 2 GB = 24 GB đĩa | Trung bình | Hết dung lượng | Xoá ứng viên thua sau khi chốt; giữ CSV + hash |
| `torch.load` cần `weights_only=False` | Trung bình | Không nạp được | Xử lý trong `canonical_loader`, không lặp lại ở đây |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) — §I.1, §I.4 (E6)
- Bảng checkpoint: [`../../../plan/tong_hop_checkpoints_va_lich_su_thuc_nghiem.md`](../../../plan/tong_hop_checkpoints_va_lich_su_thuc_nghiem.md) §5.4, §7
- Hiệu chuẩn: [`task_7.4_calibration_threshold_fpr.md`](task_7.4_calibration_threshold_fpr.md)
- Re-baseline: [`task_7.6_rebaseline_cfe_v3.md`](task_7.6_rebaseline_cfe_v3.md)
- Chỉ mục giai đoạn: [`README.md`](README.md)
