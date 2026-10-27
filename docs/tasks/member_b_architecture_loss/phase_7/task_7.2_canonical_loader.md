# TASK 7.2 — CANONICAL LOADER + ADAPTER-INDEX TƯỜNG MINH

- **Mã nhiệm vụ**: `Task 7.2` *(Phase 7 — D1–D3)*
- **Người thực hiện**: **Thành viên B (Architecture & Loss Lead)**
- **Người phối hợp**: **Thành viên C (Training Lead)** — chạy smoke `train_crossdomain.py`
- **Môi trường thực thi**: 🖥️ **Local CPU**
- **Độ ưu tiên**: 🔴 Khẩn cấp — **chặn G7.0 và Task 7.3**
- **Cổng liên quan**: **G7.0** (toàn vẹn đo lường, cuối D2)
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.2

---

## 1. Mục tiêu

Loại bỏ **vĩnh viễn** khả năng "nạp thiếu mà im lặng" — nguyên nhân gốc khiến Gen 1 bị kết luận sai là "sụp năng lực" trong khi thực chất chỉ nạp được 3/10 khối adapter.

**Bằng chứng xuất phát** (đo trực tiếp bằng `torch.load`, ghi ở kế hoạch §I.1):

| Checkpoint | Số tensor | Khối adapter | Tensor/khối | SRM | Dung lượng |
| :--- | ---: | :--- | ---: | :---: | ---: |
| Baseline | 1.116 | `{7,15,23}` | 66 | Không | 1,97 GB |
| M4 | 1.684 | `{2,5,8,11,14,17,20,23}` | 94 | Có | 7,50 GB |
| **Gen 1 (WiSE-FT cũ)** | **1.816** | **`{2,5,7,8,11,14,15,17,20,23}`** | 66/94 | Có | 4,69 GB |
| Anchored | 1.200 | `{7,15,23}` | 94 | Có | 1,97 GB |

**Ràng buộc toán học**: `num_vit_adapter = n` chỉ biểu diễn được các khối `{(i+1)·24/n − 1}`, với `n ∈ {3,4,6,8,12,24}`. Với `n=3` ⇒ `{7,15,23}`; `n=8` ⇒ `{2,5,8,11,14,17,20,23}`. **Không tồn tại `n`** sinh ra hợp `{2,5,7,8,11,14,15,17,20,23}`. Vì vậy Gen 1 **không thể** nạp đầy đủ bằng constructor hiện tại — loader cũ đã "giải quyết" bằng `len(indices) > 8 → 3`, tức là **âm thầm bỏ 7/10 khối**.

> **Kiểm tra alias đã loại trừ giả thuyết khác**: 100% tensor `image_encoder.*` bằng `clip_model.visual.*` trong cả 4 file ⇒ sụp năng lực **không** phải do alias key. Vấn đề thuần tuý là **layout adapter**.

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | 4 checkpoint | `DATASET/checkpoints/**` (Baseline, M4, Gen 1, Anchored) | Đối tượng kiểm thử |
| I2 | Constructor hiện tại | [`src/models/clip/model.py`](../../../../src/models/clip/model.py) dòng 403–411 | Nơi áp đặt quy tắc `(i+1)·n_spilt−1` |
| I3 | Model factory | [`src/models/clip_models.py`](../../../../src/models/clip_models.py), [`src/models/__init__.py`](../../../../src/models/__init__.py) | Cách dựng model từ `args` |
| I4 | Eval hiện tại | [`tools/eval_communityforensics.py`](../../../../tools/eval_communityforensics.py) | Nơi phát hiện lỗi E1/E2 |
| I5 | Smoke test có sẵn | [`tools/test_src_load.py`](../../../../tools/test_src_load.py) | Chống hồi quy sau khi đổi constructor |
| I6 | Kết quả Task 7.1 | `logs/phase7/checkpoint_structure_audit.json` | Số liệu đối chiếu |

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | Module loader | `src/models/canonical_loader.py` | 5 hàm công khai (xem §5.2) |
| O2 | Constructor sửa đổi | `src/models/clip/model.py` | Tham số `adapter_block_indices: Optional[List[int]]` |
| O3 | Bộ test | `tests/test_canonical_loader.py` | 5 test (a)–(e) |
| O4 | Audit tool | `tools/audit_checkpoint_compat.py` | Gọi **chính** loader, không logic riêng |
| O5 | Báo cáo audit | `logs/phase7/checkpoint_structure_audit.json` | Theo schema §5.3 |
| O6 | Báo cáo | `docs/bao_cao/B/bao_cao_task_7.2_canonical_loader.md` | Kết luận 4/4 checkpoint |

---

## 4. Ràng buộc

1. **`strict=True` là mặc định duy nhất**. Không có cờ `strict=False` trong bất kỳ chữ ký công khai nào.
2. **Fail loud**: `dropped ≠ ∅` ⇒ raise `IncompleteLayoutError` (hoặc trả `partial=True` **chỉ khi** caller truyền `allow_partial=True`), không bao giờ im lặng.
3. **Một nguồn sự thật cho alias**: giữ `clip_model.visual.*`, bỏ `image_encoder.*`. Ghi rõ lý do trong docstring.
4. **Không đổi hành vi mặc định**: khi không truyền `adapter_block_indices`, quy tắc `(i+1)·n−1` phải giữ nguyên — để checkpoint cũ vẫn nạp được.
5. **Không phá `train_crossdomain.py`**: sau khi sửa constructor phải chạy `tools/test_src_load.py` + smoke 1 batch.
6. **Không hard-code đường dẫn checkpoint** trong module.

---

## 5. Các bước thực hiện

### Bước 1 — Mở constructor (D1, ~1,5 h)

Trong `VisionTransformer.__init__` (dòng ~392), thêm tham số:

```python
def __init__(self, input_resolution, patch_size, width, layers, heads,
             output_dim, args=None,
             adapter_block_indices: Optional[List[int]] = None):
```

Logic:

```python
if adapter_block_indices is not None:
    idx = sorted(set(int(i) for i in adapter_block_indices))
    assert len(idx) == len(adapter_block_indices), "adapter_block_indices bị trùng"
    assert all(0 <= i < layers for i in idx), f"index ngoài [0,{layers})"
    add_adapter = [False] * layers
    for i in idx:
        add_adapter[i] = True
else:
    # giữ nguyên hành vi cũ: (i+1)*n_spilt - 1
    ...
```

Đồng thời thêm `self.adapter_block_indices = idx` để loader đối chiếu ngược.

> **Bẫy**: `args` có thể chứa `num_vit_adapter` khác với độ dài danh sách. Nếu truyền `adapter_block_indices` thì **bỏ qua** `num_vit_adapter` — ghi rõ trong docstring, nếu không sẽ có 2 nguồn sự thật.

### Bước 2 — Viết `canonical_loader.py` (D1–D2, ~3 h)

**API bắt buộc**

```python
def extract_state_dict(ckpt) -> dict:
    """Nhận dict | path | object có .state_dict(); hỗ trợ 'model'/'state_dict'/'model_state';
    strip tiền tố 'module.'."""

def infer_adapter_blocks(sd: dict) -> set[int]:
    """Quét key dạng 'clip_model.visual.transformer.resblocks.{i}.attn.adapter*'
    (và biến thể 'image_encoder.*'), trả về tập index."""

def resolve_layout(blocks: set[int], layers: int = 24) -> dict:
    """Chọn n ∈ {3,4,6,8,12,24} phủ được NHIỀU khối nhất.
    Trả {n, indices, dropped, partial, expressible}."""

def canonicalize(sd: dict, model_sd: dict) -> tuple[dict, dict]:
    """Chuẩn hoá key + đối chiếu shape. Trả (clean_sd, report)."""

def load_canonical(ckpt_path, model_args, allow_partial=False) -> tuple[nn.Module, dict]:
    """Dựng model → resolve layout → nạp strict → assert 0/0 → trả (model, report)."""
```

**Thuật toán `resolve_layout`** (phải cài đúng thứ tự ưu tiên):

1. Với mỗi `n` trong `{24,12,8,6,4,3}`: tính `S_n = {(i+1)·24/n − 1}`.
2. `cover_n = |blocks ∩ S_n|`, `dropped_n = blocks − S_n`.
3. Chọn `n` có `dropped_n = ∅` **và** `n` lớn nhất có thể. Nếu nhiều `n` cùng phủ đủ, chọn `n` **nhỏ nhất** (ít tham số "thừa" hơn) — ghi rõ quy tắc này trong docstring.
4. Nếu không `n` nào phủ đủ: chọn `n` có `dropped_n` nhỏ nhất, trả `partial=True`.

**Kỳ vọng trên 4 checkpoint**

| Checkpoint | `expressible` | Kết quả phải trả |
| :--- | :--- | :--- |
| Baseline | `{7,15,23}` | `n=3, dropped=∅, partial=False` |
| M4 | `{2,5,8,11,14,17,20,23}` | `n=8, dropped=∅, partial=False` |
| Anchored | `{7,15,23}` | `n=3, dropped=∅, partial=False` |
| **Gen 1** | 10 khối | `partial=True`, `dropped={7,15}` với `n=8`; hoặc `n=3` với `dropped={2,5,8,11,14,17,20}` ⇒ **raise `IncompleteLayoutError`** |

**Quy tắc `canonicalize`**:

1. Bỏ mọi key bắt đầu bằng `image_encoder.` (giữ `clip_model.visual.*`).
2. Bỏ `srm.*`, `gating.*` ở **top-level** nếu model không có module tương ứng (nhưng **giữ** `srm_block.*` đã nằm trong block — đây là điểm script `wise_ft_merge.py` làm sai).
3. Đối chiếu shape: mọi key chung phải cùng shape; lệch ⇒ ghi vào `shape_mismatch` và raise.
4. `missing = model_keys − ckpt_keys`, `unexpected = ckpt_keys − model_keys`.

### Bước 3 — `load_canonical` (D2, ~1 h)

```python
def load_canonical(ckpt_path, model_args, allow_partial=False):
    sd, meta = extract_state_dict(torch.load(ckpt_path, map_location="cpu"))
    blocks = infer_adapter_blocks(sd)
    layout = resolve_layout(blocks, layers=model_args.get("layers", 24))
    if layout["partial"] and not allow_partial:
        raise IncompleteLayoutError(
            f"{ckpt_path}: không biểu diễn được {sorted(layout['dropped'])} "
            f"với n ∈ {{3,4,6,8,12,24}}. Đây là invalid_layout.")
    model = build_model(**model_args,
                        adapter_block_indices=layout["indices"] if not layout["partial"] else None)
    clean_sd, report = canonicalize(sd, model.state_dict())
    missing, unexpected = model.load_state_dict(clean_sd, strict=True), None
    report.update(path=str(ckpt_path), layout=layout,
                  sha256=sha256_file(ckpt_path), n_keys=len(sd))
    assert not report["shape_mismatch"], report
    return model, report
```

### Bước 4 — Test (D2–D3, ~2 h)

Tạo `tests/` (chưa tồn tại trong repo) và `tests/test_canonical_loader.py`:

| # | Test | Kỳ vọng |
| :---: | :--- | :--- |
| (a) | Baseline + Anchored | `missing=0, unexpected=0`, `partial=False` |
| (b) | M4 | `n=8`, `0/0` |
| (c) | Gen 1 | raise `IncompleteLayoutError`; với `allow_partial=True` trả `partial=True` + `dropped` khác rỗng |
| (d) | State dict giả có key lạ | `unexpected > 0` và raise |
| (e) | State dict giả có shape lệch | `shape_mismatch` khác rỗng và raise |

Chạy:

```bash
pytest -q tests/test_canonical_loader.py
```

### Bước 5 — Đồng bộ audit tool & chống hồi quy (D3, ~0,5 h)

1. `tools/audit_checkpoint_compat.py` **import** `canonical_loader`, không copy logic. Xuất `logs/phase7/checkpoint_structure_audit.json` theo schema:

```json
{
  "schema": "checkpoint_structure_audit/v1",
  "checkpoints": [{
    "path": "...", "sha256": "...", "n_keys": 1816,
    "adapter_blocks": [2,5,7,8,11,14,15,17,20,23],
    "expressible_n": [3,8],
    "layout": {"n": 8, "indices": [2,5,8,11,14,17,20,23],
               "dropped": [7,15], "partial": true},
    "has_srm_per_block": true,
    "alias_equality": {"image_encoder_vs_clip": 1.0},
    "dead_keys": []
  }]
}
```

2. Chạy:

```bash
grep -rn "strict=False" tools/ src/    # phải rỗng
python tools/test_src_load.py
python -m pytest -q tests/test_canonical_loader.py
```

3. Smoke 1 batch với `train_crossdomain.py` (chỉ 1 step, `--max_steps 1` nếu có, hoặc `tools/smoke_test_pipeline.py`).

---

## 6. Definition of Done

- [ ] `src/models/canonical_loader.py` tồn tại với đúng 5 hàm công khai ở §5.2; mỗi hàm có docstring nêu rõ kiểu trả về.
- [ ] `src/models/clip/model.py` nhận `adapter_block_indices`, có `assert` chống trùng và vượt `layers`; hành vi mặc định **không đổi**.
- [ ] `pytest -q tests/test_canonical_loader.py` — 5/5 xanh.
- [ ] Baseline / M4 / Anchored nạp **0/0**; Gen 1 bị **từ chối** hoặc đánh dấu `partial=True` kèm `dropped`.
- [ ] `grep -rn "strict=False" tools/ src/` trả về **rỗng**.
- [ ] `logs/phase7/checkpoint_structure_audit.json` đúng schema, có `sha256` từng file.
- [ ] `tools/test_src_load.py` xanh + smoke 1 batch `train_crossdomain.py` không lỗi.
- [ ] Báo cáo `docs/bao_cao/B/bao_cao_task_7.2_canonical_loader.md` kết luận rõ 4/4 checkpoint.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
| :--- | :---: | :---: | :--- |
| Đổi constructor làm hỏng `train_crossdomain.py` | Trung bình | Chặn 7.11 | Giữ default cũ; chạy `tools/test_src_load.py` + smoke trước khi merge |
| `resolve_layout` chọn `n` sai (phủ đủ nhưng khác ý đồ) | Trung bình | Nạp "đúng" nhưng sai kiến trúc | Ghi rõ quy tắc ưu tiên; test (a)(b) khoá kỳ vọng `n` |
| Bỏ `image_encoder.*` làm mất key không có trong `clip_model.visual.*` | Thấp | Thiếu tham số | So sánh tập key trước/sau; nếu có key chỉ tồn tại ở `image_encoder.*` ⇒ giữ lại + cảnh báo |
| `torch.load` cần `weights_only=False` trên PyTorch mới | Trung bình | Không nạp được | Bọc try/except, ghi rõ phiên bản PyTorch trong báo cáo |
| Test (e) shape lệch khó tạo | Thấp | Test giả | Tạo state dict từ model rồi `unsqueeze` 1 tensor |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) — §I.1, §I.2
- Task audit tiền đề: [`../../member_c_training_eval/phase_7/task_7.1_audit_checkpoint_gen1.md`](../../member_c_training_eval/phase_7/task_7.1_audit_checkpoint_gen1.md)
- Task tiêu thụ loader: [`../../member_c_training_eval/phase_7/task_7.3_eval_v3_measurement_contract.md`](../../member_c_training_eval/phase_7/task_7.3_eval_v3_measurement_contract.md)
- Chỉ mục giai đoạn: [`README.md`](README.md)
