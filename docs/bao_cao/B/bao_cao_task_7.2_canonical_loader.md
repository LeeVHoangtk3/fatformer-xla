# BÁO CÁO NGHIỆM THU NHIỆM VỤ 7.2 (TASK 7.2)
## CANONICAL LOADER + ADAPTER-INDEX TƯỜNG MINH

> **Mã nhiệm vụ**: `Task 7.2` *(Phase 7 — D1–D3)*  
> **Người thực hiện**: **Thành viên B** *(Architecture & Loss Lead)*  
> **Người phối hợp**: **Thành viên C** *(Training Lead)*  
> **Cổng kiểm soát chất lượng**: **Cổng G7.0 (Toàn vẹn đo lường)**
> **Trạng thái**: `[~] ĐÃ TRIỂN KHAI — CHỜ ĐÓNG repo-wide strict=False VÀ TÁI LẬP SMOKE LOCAL`
> **Môi trường xác minh**: Google Colab CPU (Python 3.13, torch 2.x+cpu) — chạy thực tế 4 checkpoint + smoke test forward pass  
> **Hiện vật bàn giao**:
> - Module loader chính: [`src/models/canonical_loader.py`](../../../src/models/canonical_loader.py)
> - Constructor sửa đổi: [`src/models/clip/model.py`](../../../src/models/clip/model.py) — `VisionTransformer.__init__`
> - Bộ test: [`tests/test_canonical_loader.py`](../../../tests/test_canonical_loader.py)
> - Audit tool (refactored): [`tools/audit_checkpoint_compat.py`](../../../tools/audit_checkpoint_compat.py)
> - Dữ liệu audit v1: [`DATASET/logs/phase7/checkpoint_structure_audit.json`](../../../DATASET/logs/phase7/checkpoint_structure_audit.json)
> - Báo cáo này: `docs/bao_cao/B/bao_cao_task_7.2_canonical_loader.md`

---

## 1. Bối Cảnh & Vấn Đề Giải Quyết

**Kế thừa từ Task 7.1** ([`bao_cao_task_7.1_checkpoint_audit.md`](./../../C/../C/bao_cao_task_7.1_checkpoint_audit.md)): Đã xác nhận checkpoint Gen 1 (`fatformer_wise_ft_best.pth`) là **chimeric checkpoint** với 10 khối adapter không biểu diễn được bởi bất kỳ `n ∈ {3,4,6,8,12,24}` nào. Loader cũ (dùng `strict=False`) đã **âm thầm bỏ 7/10 khối** khi nạp Gen 1, dẫn đến kết luận sai về năng lực mô hình.

**Vấn đề cần giải quyết**: Loại bỏ vĩnh viễn khả năng "nạp thiếu mà im lặng" — đảm bảo mọi checkpoint bất hợp lệ đều bị từ chối rõ ràng với thông báo đầy đủ.

---

## 2. Kiến Trúc Giải Pháp

### 2.1. Luồng xử lý `canonical_loader.py`

```
Checkpoint path/dict
    │
    ▼
extract_state_dict()    ← Chuẩn hoá: nested dicts, strip module.*, FileNotFoundError
    │
    ▼
infer_adapter_blocks()  ← Quét key regex → tập block indices
    │
    ▼
resolve_layout()        ← Chọn n ∈ {3,4,6,8,12,24}: exact match ưu tiên n nhỏ nhất
    │                      Nếu không có exact: partial=True
    ├── partial=True ──── raise IncompleteLayoutError (trừ allow_partial=True)
    │
    ▼
build_model(args với adapter_block_indices)   ← Dựng model tương ứng
    │
    ▼
canonicalize(sd, model.state_dict())         ← Drop image_encoder.*, top-level srm./gating.*
    │                                           Shape mismatch → raise ShapeMismatchError
    ▼
model.load_state_dict(clean_sd, strict=...)  ← strict=True (mặc định); strict=False chỉ với allow_partial
    │
    ▼
(model, report)   ← report gồm: layout, missing/unexpected counts, sha256, n_keys
```

### 2.2. Thay đổi `VisionTransformer.__init__`

Tham số `adapter_block_indices: Optional[List[int]]` được thêm vào constructor. Logic:

- Nếu `adapter_block_indices` truyền vào: dùng danh sách tường minh, **bỏ qua** `num_vit_adapter` (tránh 2 nguồn sự thật).
- Nếu có danh sách tường minh thì danh sách đó là nguồn sự thật; `num_vit_adapter` legacy bị bỏ qua theo đúng contract.
- Hành vi mặc định (`adapter_block_indices=None`, `args.adapter_block_indices=None`): giữ nguyên quy tắc `(i+1)·n-1`.

---

## 3. Kết Quả Kiểm Tra 4 Checkpoint

Kết quả **chạy thực tế** tại `scratch/verify_task_7_2.py` trên Google Colab CPU (2026-10-31). Kết quả lưu tại [`logs/phase7/task_7.2_canonical_load_verification.json`](../../../logs/phase7/task_7.2_canonical_load_verification.json).

| Checkpoint | n_keys | Layout | missing | unexpected | dropped_top_level | sha256 | Trạng thái |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- | :---: |
| **Baseline** (`fatformer_4class_ckpt.pth`) | 1116 | `n=3 {7,15,23}` | **0** | **0** | 0 | `3b904401...` | ✅ LOADED |
| **Anchored** (`fatformer_anchored_recovery_best.pth`) | 1200 | `n=3 {7,15,23}` | **0** | **0** | 0 | `f70a11df...` | ✅ LOADED |
| **M4** (`crossdomain_model_best_m4.pth`) | 1684 | `n=8 {2,5,8,11,14,17,20,23}` | **0** | **0** | 14* | `b5ad8396...` | ✅ LOADED |
| **Gen 1** (`fatformer_wise_ft_best.pth`) | — | `partial=True, dropped=[7,15]` | — | — | — | — | ✅ REJECTED |

> *M4 `dropped_top_level=14`: các key `language_guided_alignment.*` và `srm./gating.*` cấp cao nhất trong checkpoint không tồn tại trong model. Đây là hành vi đúng — `canonicalize()` bỏ các key thừa trước khi `load_state_dict`, nên `unexpected=0`.

### Giải thích `resolve_layout` cho Gen 1:

Tập khối `{2,5,7,8,11,14,15,17,20,23}` không khớp bất kỳ `n` nào:
- `n=8` → `S_8 = {2,5,8,11,14,17,20,23}` → `dropped = {7,15}` → `partial=True`
- `n=3` → `S_3 = {7,15,23}` → `dropped = {2,5,8,11,14,17,20}` → `partial=True`

Thuật toán chọn `n=8` (dropped nhỏ nhất: 2 block). Gen 1 bị **từ chối** và không bao giờ được âm thầm ánh xạ sang `n=3` hay `n=8`.

---

## 4. Kết Quả Kiểm Tra Chức Năng (pytest)

Bộ test gốc gồm **12 test cases**, đã chạy trên Colab CPU (2026-10-31): **12 passed in 4.41s**. Sau review bổ sung test thứ 13 cho `allow_partial=True`; test mới đã qua compile nhưng local hiện thiếu `pytest`, nên cần tái chạy trọn bộ trước khi đóng DoD.

| Test | Nội dung | Kỳ vọng |
| :--- | :--- | :---: |
| `test_loads_exact_explicit_layout[7,15,23]` | Baseline/Anchored layout | `missing=0, unexpected=0` |
| `test_loads_exact_explicit_layout[2,5,8,11,14,17,20,23]` | M4 layout | `missing=0, unexpected=0` |
| `test_constructor_falls_back_to_legacy_adapter_count` | `num_vit_adapter=3` fallback | `{7,15,23}` |
| `test_constructor_uses_explicit_adapter_indices` | Explicit `[2,5,8,...]` | M4 indices |
| `test_constructor_explicit_indices_override_legacy_count` | Explicit indices và `num_vit_adapter` legacy cùng tồn tại | Explicit indices thắng |
| `test_gen1_layout_can_be_inspected_only_with_allow_partial` | Forensic partial load | `partial=True`, có `dropped/unexpected` |
| `test_strict_loading_reports_key_mismatch[missing]` | Key bị thiếu | `CheckpointCompatibilityError` |
| `test_strict_loading_reports_key_mismatch[unexpected]` | Key thừa | `CheckpointCompatibilityError` |
| `test_gen1_layout_is_rejected_without_remapping` | Gen 1 chimeric | `IncompleteLayoutError` |
| `test_original_namespace_is_not_mutated` | `model_args` immutable | không thay đổi |
| `test_caller_adapter_index_must_match_checkpoint` | Caller index ≠ checkpoint | `IncompleteLayoutError` |
| `test_wrong_adapter_prefix_does_not_guess_layout` | Prefix không nhận diện | `IncompleteLayoutError` |
| `test_shape_mismatch_is_reported` | Shape lệch | `ShapeMismatchError` |

**Lệnh kiểm tra**:
```bash
pytest -q tests/test_canonical_loader.py
```

---

## 5. Đồng Bộ Audit Tool (O4)

`tools/audit_checkpoint_compat.py` đã được refactored theo yêu cầu O4:
- **Import** `extract_state_dict`, `infer_adapter_blocks`, `resolve_layout` từ `canonical_loader`.
- **Không** có logic layout resolution hoặc regex adapter detection riêng nữa.
- Các metadata audit-specific (alias equality, SRM per-block, dead keys, SHA-256) vẫn nằm trong tool.

Điểm khác biệt so với phiên bản Task 7.1: Tool hiện gọi `canonical_loader` làm nguồn sự thật duy nhất, đảm bảo không có sự lệch lạc giữa kết quả audit và hành vi loader thực tế.

---

## 6. Kiểm Tra DoD

| DoD Item | Trạng thái | Ghi chú |
| :--- | :---: | :--- |
| `src/models/canonical_loader.py` tồn tại với 5 hàm công khai | ✅ | `extract_state_dict`, `infer_adapter_blocks`, `resolve_layout`, `canonicalize`, `load_canonical` |
| `src/models/clip/model.py` nhận `adapter_block_indices` | ✅ | Dòng 393, đầy đủ assert |
| `pytest -q tests/test_canonical_loader.py` — 13 test cases hiện tại | ⏳ | 12 test cũ đã pass Colab; test mới compile pass, chờ môi trường có `pytest` |
| Baseline/M4/Anchored nạp `0/0`; Gen 1 bị từ chối | ✅ | **Xác nhận thực tế** — xem [`task_7.2_canonical_load_verification.json`](../../../DATASET/logs/phase7/task_7.2_canonical_load_verification.json) |
| `grep -rn "strict=False" tools/ src/` trả về rỗng | ❌ | Vẫn còn nhiều đường nạp legacy; canonical eval v3 đã strict nhưng DoD repo-wide chưa đạt |
| `logs/phase7/checkpoint_structure_audit.json` đúng schema, có `sha256` | ✅ | Đã tồn tại từ Task 7.1 |
| `tools/audit_checkpoint_compat.py` import canonical_loader | ✅ | Refactored Task 7.2 |
| `tools/test_src_load.py` xanh + smoke forward pass | ✅ | **TEST 1** (strict=True, use_srm=False): output `[1,2]` ✅ **TEST 2** (strict=False, SRM+Gating): output `[1,2]` ✅ (Colab CPU, 2026-10-31) |
| Báo cáo kết luận rõ 4/4 checkpoint | ✅ | Mục 3 báo cáo này |

---

## 7. Rủi ro đã giải quyết

| Rủi ro | Kết quả |
| :--- | :--- |
| Constructor phá `train_crossdomain.py` | ✅ Hành vi mặc định giữ nguyên (fallback `num_vit_adapter`) |
| `resolve_layout` chọn `n` sai | ✅ Ưu tiên rõ ràng: exact match → n nhỏ nhất; ghi rõ trong docstring |
| Bỏ `image_encoder.*` làm mất key | ✅ Chỉ bỏ khi có counterpart `clip_model.visual.*` trong cả sd và model_sd |
| `torch.load` cần `weights_only=False` | ✅ `extract_state_dict` có try/except tương thích đa phiên bản PyTorch |

---

## 8. Kết Luận Nghiệm Thu

Task 7.2 đã hoàn thành loader lõi và kiểm chứng 4 checkpoint. Tuy nhiên chưa được tuyên bố hoàn thành 100% vì tiêu chí repo-wide `strict=False` và smoke tái lập trên môi trường local chưa đóng. Đây là đầu vào cần thiết cho **G7.0**, không phải phán quyết G7.1.

| Hạng mục | Kết quả thực tế |
| :--- | :--- |
| pytest | **12 test cũ passed**; bộ 13 test hiện tại chờ tái chạy |
| Baseline load | LOADED, missing=0, unexpected=0, sha256=`3b904401...` |
| Anchored load | LOADED, missing=0, unexpected=0, sha256=`f70a11df...` |
| M4 load | LOADED, missing=0, unexpected=0, sha256=`b5ad8396...` |
| Gen1 | REJECTED (`IncompleteLayoutError`, dropped=[7,15]) |
| DoD #4 | **PASS** |
| test_src_load TEST 1 | build_model OK, strict=True OK, forward `[1,2]` ✅ |
| test_src_load TEST 2 | build_model SRM+Gating OK, strict=False OK, forward `[1,2]` ✅ |
| DoD #7 | **PASS** |

Bàn giao cho Thành viên C:
- `load_canonical()` sẵn sàng được dùng trong `eval_communityforensics.py` (Task 7.3).
- Gen 1 bị từ chối đúng, Baseline/Anchored/M4 nạp đúng kiến trúc theo `adapter_block_indices` tường minh.
- Kết quả xác minh lưu tại [`DATASET/logs/phase7/task_7.2_canonical_load_verification.json`](../../../DATASET/logs/phase7/task_7.2_canonical_load_verification.json).
