# TASK 7.1 — AUDIT CẤU TRÚC CHECKPOINT & ĐO LẠI GEN 1 (`n=3` vs `n=8`)

- **Mã nhiệm vụ**: `Task 7.1` *(Phase 7 — D1–D2)*
- **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**
- **Người phối hợp**: **Thành viên B (Architecture Lead)** — bàn giao script audit cho B ở D2 để B refactor theo `canonical_loader`
- **Môi trường thực thi**: 🖥️ **Local CPU** + ☁️ **T4 ~10 phút**
- **Độ ưu tiên**: 🔴 Khẩn cấp — **mở khoá toàn bộ Phase 7**
- **Cổng liên quan**: **G7.0** (toàn vẹn đo lường, cuối D2)
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.1

---

## 1. Mục tiêu

Biến **mọi giả thuyết** ở kế hoạch §I thành **số liệu đo được**, và trả lời dứt điểm một câu hỏi:

> "Gen 1 (WiSE-FT cũ) có thật sự sụp năng lực, hay chỉ sụp vì loader chọn `n=3` và bỏ 7/10 khối adapter?"

Kết quả của task này quyết định:

- Gen 1 bị **cách ly vĩnh viễn** (nhãn `invalid_layout`) hay được **phục hồi một phần**;
- Toàn bộ số liệu Phase 6-R có còn hiệu lực hay không;
- Cách viết mục **Hạn chế** của luận văn.

**Bằng chứng đã có** (kế hoạch §I.1, đo bằng `torch.load`):

| Checkpoint | Số tensor | Khối adapter | Tensor/khối | SRM | Ghi chú |
| :--- | ---: | :--- | ---: | :---: | :--- |
| Baseline | 1.116 | `{7,15,23}` | 66 | Không | Hợp lệ |
| M4 | 1.684 | `{2,5,8,11,14,17,20,23}` | 94 | Có | Hợp lệ (`n=8`) |
| **Gen 1** | **1.816** | **`{2,5,7,8,11,14,15,17,20,23}`** | 66 tại `{7,15,23}`, 94 tại còn lại | Có | **Không biểu diễn được** |
| Anchored | 1.200 | `{7,15,23}` | 94 | Có | Hợp lệ |

Vì `n` chỉ sinh được `{(i+1)·24/n − 1}` với `n ∈ {3,4,6,8,12,24}`, **không** `n` nào cho hợp 10 khối trên. Loader cũ "xử lý" bằng `len(indices) > 8 → 3`, tức nạp 3/10 khối.

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | 4 checkpoint `.pth` | `DATASET/checkpoints/**` | Đối tượng audit |
| I2 | Tập test CFE (2.000 ảnh, 224×224) | `DATASET/test_communityforensics/` | Đo lại Gen 1 |
| I3 | `ViT-L-14.pt` | `DATASET/` / `checkpoints/` | Dựng model |
| I4 | Constructor hiện tại | [`src/models/clip/model.py`](../../../../src/models/clip/model.py) dòng 403–411 | Quy tắc `(i+1)·n_spilt−1` |
| I5 | Eval cũ | [`tools/eval_communityforensics.py`](../../../../tools/eval_communityforensics.py) | **Không sửa**; chỉ đọc để hiểu lỗi E1/E2 |
| I6 | Audit cũ | `logs/checkpoint_keys_audit.json`, [`tools/audit_checkpoints.py`](../../../../tools/audit_checkpoints.py) | Mẫu tham khảo |

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | Script audit | `tools/audit_checkpoint_compat.py` | Xuất JSON theo schema §5.1 |
| O2 | Audit 4 checkpoint | `logs/phase7/checkpoint_structure_audit.json` | §5.1 |
| O3 | Đo lại Gen 1 | `logs/phase7/wise_ft_legacy_reinterpretation.json` | 2 lần chấm: `n=3`, `n=8` |
| O4 | Kết luận | `docs/bao_cao/C/bao_cao_task_7.1_checkpoint_audit.md` | **1 trang**, có bảng + phán quyết |
| O5 | Hình | `logs/phase7/gen1_n3_vs_n8.png` | So sánh 3 chỉ số giữa 2 layout |

---

## 4. Ràng buộc

1. **Không sửa** `tools/eval_communityforensics.py` trong task này — việc đó thuộc Task 7.3.
2. **`strict=False` chỉ được phép trong đường chẩn đoán** của task này, và phải:
   - in banner `[DIAGNOSTIC-ONLY] partial load`,
   - ghi `partial=true` + danh sách `dropped` vào JSON,
   - **không** dùng để sinh bất kỳ số liệu chính thức nào.
   Sau khi Task 7.2 xong, script phải chuyển sang `load_canonical(..., allow_partial=True)`.
3. **Cùng điều kiện đo** khi so `n=3` vs `n=8`: cùng 2.000 ảnh **clean**, cùng `T=1.0`, cùng ngưỡng `0,5`, cùng `batch_size`.
4. **Không dùng số của Gen 1** làm kết quả chính thức — kể cả khi `n=8` cho kết quả cao.
5. **Không chấm Q30** cho Gen 1 ở task này (tiết kiệm GPU); chỉ clean.

---

## 5. Các bước thực hiện

### Bước 1 — Viết `tools/audit_checkpoint_compat.py` (D1, ~2 h)

CLI:

```bash
python tools/audit_checkpoint_compat.py \
  --checkpoints DATASET/checkpoints/*.pth \
  --layers 24 \
  --output logs/phase7/checkpoint_structure_audit.json
```

**Schema JSON bắt buộc**

```json
{
  "schema": "checkpoint_structure_audit/v1",
  "generated_at": "2026-10-27T00:00:00Z",
  "torch_version": "2.x",
  "checkpoints": [
    {
      "path": "...",
      "sha256": "...",
      "size_bytes": 0,
      "n_keys": 0,
      "adapter_blocks": [],
      "expressible_n": [],
      "layout": {"n": null, "indices": [], "dropped": [], "partial": true},
      "blocks_dropped": {},
      "has_srm_per_block": {"7": false, "15": false, "23": false},
      "tensor_count_per_block": {"7": 66, "15": 66, "23": 66},
      "alias_equality": {"image_encoder_vs_clip": 1.0},
      "dead_keys": [],
      "has_gating": false,
      "has_top_level_srm": false
    }
  ]
}
```

**Quy tắc tính từng trường**

| Trường | Cách tính |
| :--- | :--- |
| `adapter_blocks` | Quét key `*(resblocks|transformer.resblocks).{i}.*adapter*`; hỗ trợ cả `clip_model.visual.*` và `image_encoder.*` |
| `expressible_n` | Các `n ∈ {3,4,6,8,12,24}` mà `{(i+1)·24/n−1} ⊇ adapter_blocks` |
| `blocks_dropped` | Với **từng** `n` khả biểu: `adapter_blocks − S_n`; key là chuỗi `n` |
| `has_srm_per_block` | Đếm tensor `srm_block.*` trong từng block; `true` nếu > 0 |
| `alias_equality` | Tỷ lệ tensor `image_encoder.X` có `clip_model.visual.X` **giá trị bằng nhau** (`torch.equal`), trên tổng số key `image_encoder.*` |
| `dead_keys` | Key có trong checkpoint nhưng không tồn tại ở **bất kỳ** model nào dựng được từ 4 layout ứng viên |

**Kỳ vọng**:

| Checkpoint | `expressible_n` | Kết luận |
| :--- | :--- | :--- |
| Baseline | `[3]` | `partial=false` |
| M4 | `[8]` (hoặc `[3,8]` nếu `{7,15,23} ⊂ {2,5,8,...}` — **không** đúng, nên chỉ `[8]`) | `partial=false` |
| Anchored | `[3]` | `partial=false` |
| **Gen 1** | `[3,8]` nhưng **`dropped` khác rỗng ở cả hai** | `partial=true`; `n=8` bỏ `{7,15}`, `n=3` bỏ `{2,5,8,11,14,17,20}` |

### Bước 2 — Chạy audit 4 checkpoint (D1, ~0,5 h)

```bash
python tools/audit_checkpoint_compat.py --checkpoints <4 file> \
  --output logs/phase7/checkpoint_structure_audit.json
```

Kiểm tra bằng mắt: `n_keys` khớp bảng §1; `alias_equality` ≈ 1,0; `dead_keys` rỗng.

### Bước 3 — Đo lại Gen 1: `n=3` vs `n=8` (D2, ~1 h code + 10 phút T4)

1. Viết `tools/eval_gen1_reinterpretation.py` (hoặc thêm cờ vào eval cũ — **không** sửa mặc định):
   - Dựng model với `num_vit_adapter=3` ⇒ nạp phần nạp được ⇒ chấm.
   - Dựng model với `num_vit_adapter=8` ⇒ nạp phần nạp được ⇒ chấm.
   - Cả hai ghi rõ `n_loaded_blocks`, `dropped`.
2. Chấm **2.000 ảnh clean**, `T=1.0`, ngưỡng `0,5`, `batch_size=32`.
3. Xuất:

```json
{
  "schema": "gen1_reinterpretation/v1",
  "checkpoint": "...",
  "split": "cfe_clean_2000",
  "runs": [
    {"layout": "n3", "loaded_blocks": [7,15,23], "dropped": [2,5,8,11,14,17,20],
     "acc": 0.0, "real_acc": 0.0, "fake_acc": 0.0, "auc": 0.0},
    {"layout": "n8", "loaded_blocks": [2,5,8,11,14,17,20,23], "dropped": [7,15],
     "acc": 0.0, "real_acc": 0.0, "fake_acc": 0.0, "auc": 0.0}
  ],
  "verdict": "artifact_loader | different_basin | inconclusive"
}
```

### Bước 4 — Viết kết luận 1 trang (D2, ~0,5 h)

`docs/bao_cao/C/bao_cao_task_7.1_checkpoint_audit.md` phải trả lời **đúng 4 câu**:

1. **Cấu trúc**: Gen 1 có bao nhiêu khối, có biểu diễn được không? (bảng)
2. **Số đo**: `n=8` có cao hơn `n=3` **có ý nghĩa** không? (kèm CI; nếu chênh < 3 pp ⇒ coi là không)
3. **Phán quyết**: `artifact_loader` / `different_basin` / `inconclusive`.
4. **Hệ quả**: Gen 1 dùng được hay không; số liệu Phase 6-R nào bị vô hiệu.

**Luật phán quyết**:

| Điều kiện | `verdict` |
| :--- | :--- |
| `n=8` hơn `n=3` ≥ 10 pp Fake ACC, CI không chồng | `artifact_loader` — Gen 1 vẫn `invalid_layout`, nhưng nguyên nhân là loader |
| `n=8` và `n=3` chênh < 3 pp, cả hai đều thấp | `different_basin` — trộn WiSE-FT tạo basin khác, không phải lỗi loader |
| 3 pp ≤ chênh < 10 pp, hoặc CI chồng | `inconclusive` — không đủ bằng chứng, giữ `invalid_layout` |

Trong **mọi** trường hợp, Gen 1 bị gắn `invalid_layout` và **không** được dùng làm kết quả chính thức.

---

## 6. Definition of Done

- [ ] `tools/audit_checkpoint_compat.py` chạy được trên 4 checkpoint, không crash.
- [ ] `logs/phase7/checkpoint_structure_audit.json` có **đủ 4** checkpoint và **đủ** các trường ở §5.1 (không trường nào `null` ngoài `layout.n` của Gen 1).
- [ ] `n_keys` khớp bảng §1: 1.116 / 1.684 / 1.816 / 1.200.
- [ ] `logs/phase7/wise_ft_legacy_reinterpretation.json` có 2 run `n3`/`n8` + `verdict`.
- [ ] `logs/phase7/gen1_n3_vs_n8.png` tồn tại, có nhãn trục rõ ràng.
- [ ] Báo cáo 1 trang trả lời đủ 4 câu ở §5.4.
- [ ] **Không** số nào của Gen 1 xuất hiện như kết quả chính thức ở bất kỳ tài liệu nào (grep `Gen 1` + số trong `docs/`).
- [ ] Ghi rõ Gen 1 = `invalid_layout` trong báo cáo và trong `README_C.md`.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
| :--- | :---: | :---: | :--- |
| `n=8` không dựng được model vì `{2,5,7,...}` chồng lấn | **Cao** | Không đo được | Chấp nhận nạp một phần: `n=8` ⇒ bỏ `{7,15}`; ghi rõ `dropped` |
| Kết luận `artifact_loader` bị hiểu thành "Gen 1 dùng được" | **Cao** | Sai học thuật | Câu chốt bắt buộc: "vẫn `invalid_layout`, chỉ giải thích nguyên nhân" |
| Chênh lệch do khác random (augmentation) | Trung bình | Kết luận sai | Chấm **clean**, không augmentation; cùng seed, cùng batch |
| Tốn GPU hơn 10 phút | Thấp | Trễ D2 | Chấm 1.000 ảnh nếu cần, ghi rõ; **không** đổi sang Q30 |
| Bàn giao script cho B bị trùng lặp logic | Trung bình | 2 nguồn sự thật | D2 bàn giao; Task 7.2 refactor script thành wrapper của `canonical_loader` |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) — §I.1, §I.2, §I.5
- Task loader kế thừa script: [`../../member_b_architecture_loss/phase_7/task_7.2_canonical_loader.md`](../../member_b_architecture_loss/phase_7/task_7.2_canonical_loader.md)
- Lịch sử checkpoint: [`../../../plan/tong_hop_checkpoints_va_lich_su_thuc_nghiem.md`](../../../plan/tong_hop_checkpoints_va_lich_su_thuc_nghiem.md) §5.4
- Chỉ mục giai đoạn: [`README.md`](README.md)
