# BÁO CÁO NGHIỆM THU NHIỆM VỤ 7.9 (TASK 7.9)
## LOSS V2: `RealGuardLoss` + `FeatureConsistencyLoss`

> **Mã nhiệm vụ**: `Task 7.9` *(Phase 7 — D4–D5)*  
> **Người thực hiện**: **Thành viên B** *(Architecture & Loss Lead)*  
> **Người phối hợp**: **Thành viên C** *(Training Lead — sử dụng loss trong vòng huấn luyện Phase 7)*  
> **Cổng kiểm soát chất lượng**: — *(nạp đầu vào cho **G7.2**)*  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH`  
> **Môi trường xác minh**: Google Colab CPU (Python 3.13.18, pytest 8.4.2, torch 2.x+cpu) — 2026-11-01  
> **Hiện vật bàn giao**:
> - Loss: [`src/training/loss.py`](../../../src/training/loss.py) — `RealGuardLoss`, `FeatureConsistencyLoss`, cập nhật `CrossDomainTotalLoss`
> - Model: [`src/models/clip_models.py`](../../../src/models/clip_models.py) — cờ `return_features` trong `CLIPModel.forward`
> - Bộ test: [`tests/test_loss_v2.py`](../../../tests/test_loss_v2.py)
> - Config: [`configs/train_phase7_anchored_v2.yaml`](../../../configs/train_phase7_anchored_v2.yaml)
> - Báo cáo này: `docs/bao_cao/B/bao_cao_task_7.9_loss_v2_realguard.md`

---

## 1. Bối Cảnh & Vấn Đề Giải Quyết

**Kế thừa từ Task 6.3-B** ([`bao_cao_task_6.3_b_distillation_and_consistency_loss.md`](./bao_cao_task_6.3_b_distillation_and_consistency_loss.md)): `CrossDomainTotalLoss` gồm `AugCE + KD (Teacher-Correct) + PairedConsistency` chỉ ràng buộc ở mức **logit**. Mô hình student có xu hướng đẩy logit Fake của ảnh **Real** bị suy biến lên cao hơn teacher → tăng False Positive trên Real.

**Kế thừa từ Task 7.2** ([`bao_cao_task_7.2_canonical_loader.md`](./bao_cao_task_7.2_canonical_loader.md)): `load_canonical()` là đường nạp chuẩn; mọi thay đổi model ở task này phải giữ nguyên kết quả `DoD #4` của 7.2.

**Vấn đề cần giải quyết**:
1. Chặn student vượt teacher trên logit Fake của mẫu Real (**RealGuard**).
2. Đồng nhất biểu diễn **feature** clean ↔ degraded, không chỉ logit (**FeatureConsistency**).
3. Hoàn toàn tương thích ngược: run cũ (weight mới = 0) cho kết quả y hệt.

---

## 2. Kiến Trúc Giải Pháp

### 2.1. `RealGuardLoss`

Với tập mẫu Real `R` (`targets == 0`), `k = fake_index = 1`:

```
L_guard = mean_{i∈R} [ ReLU( z_s[i,k] − stopgrad(z_t[i,k]) ) ]²
```

- `z_s`: logit student trên ảnh degraded; `z_t`: logit teacher trên ảnh clean (detach).
- ReLU → chỉ có gradient khi `z_s > z_t`; không kéo logit xuống vô hạn.
- Batch không có Real → trả `0` (giữ graph), `guard_active=False`, không NaN.

### 2.2. `FeatureConsistencyLoss`

```
L_feat = mean(1 − cos(f_deg, stopgrad(f_clean)))
       + T² · KL( softmax(stopgrad(z_clean)/T) ‖ softmax(z_deg/T) ),   T = 2.0
```

- `f_*` = `aug_image_features` lấy qua `CLIPModel.forward(..., return_features=True)`.
- Stop-gradient trên nhánh clean để nhánh degraded học theo, không kéo ngược.

### 2.3. `CrossDomainTotalLoss` (cập nhật)

| Chế độ | Điều kiện | Công thức |
| :--- | :--- | :--- |
| **Legacy** | `weight_feat = weight_guard = 0` (mặc định) | `0.7·[0.7·CE_c + 0.3·CE_d] + L_KD + L_cons` — **y hệt Phase 6-R** |
| **Phase 7** | có weight mới > 0 | `0.3·CE_c + 0.3·CE_d + 0.7·L_KD + 0.2·L_cons + w_feat·L_feat + w_guard·L_guard` |

Dict trả về bổ sung `loss_feat`, `loss_guard`, `guard_active` để log.

### 2.4. `CLIPModel.forward(image, return_dual=False, return_features=False)`

- Mặc định `False` → hành vi cũ không đổi.
- `True` → trả `(logits, aug_image_features)`; feature shape `(B, 768)`.

### 2.5. Config Phase 7

`configs/train_phase7_anchored_v2.yaml`: `weight_feat=0.3`, `weight_guard=0.5`, `alpha_kd=0.7`, `beta_cons=0.2`, `T=2.0`, `fake_index=1` (REAL=0, FAKE=1 theo `data_manifest.json`).

---

## 3. Kết Quả Kiểm Tra Chức Năng (pytest)

Chạy thực tế trên Colab CPU (2026-11-01): **4 passed in 13.32s**.

| Test | Nội dung | Kỳ vọng | Kết quả |
| :--- | :--- | :--- | :---: |
| `test_a_backward_compat_allclose` | weight mới = 0 so với công thức Phase 6-R | `allclose` | ✅ PASSED |
| `test_b_guard_zero_when_student_below_teacher` | `z_s < z_t` | `L_guard = 0` | ✅ PASSED |
| `test_c_guard_positive_and_gradient_pulls_down` | `z_s > z_t` | `L_guard > 0`, gradient kéo `z_s` xuống | ✅ PASSED |
| `test_d_all_fake_no_nan` | Batch toàn Fake | Không NaN, `guard_active=False` | ✅ PASSED |

**Lệnh kiểm tra**:
```bash
python -m pytest tests/test_loss_v2.py -v
```

---

## 4. Kiểm Tra Hồi Quy (Regression) Task 7.2

Do sửa `clip_models.py`, chạy lại `scratch/verify_task_7_2.py` với code mới (Colab CPU, 2026-11-01):

| Checkpoint | n_keys | Layout | missing | unexpected | sha256 | Trạng thái |
| :--- | :---: | :--- | :---: | :---: | :--- | :---: |
| **Baseline** | 1116 | `n=3 {7,15,23}` | 0 | 0 | `3b904401...` | ✅ LOADED |
| **Anchored** | 1200 | `n=3 {7,15,23}` | 0 | 0 | `f70a11df...` | ✅ LOADED |
| **M4** | 1684 | `n=8 {2,5,...,23}` | 0 | 0 | `b5ad8396...` | ✅ LOADED |
| **Gen 1** | — | `partial=True, dropped=[7,15]` | — | — | — | ✅ REJECTED |

→ **`DoD #4: PASS`**, sha256 trùng khớp báo cáo 7.2.

**Smoke forward (Baseline)**:
- `forward: (1, 2)` | `return_features: (1, 2) (1, 768)`
- `torch.allclose(out, out_return_features) = True` → cờ mới không làm đổi logit.

---

## 5. Kiểm Tra DoD

| DoD Item | Trạng thái | Ghi chú |
| :--- | :---: | :--- |
| `RealGuardLoss` chỉ phạt khi `z_s > z_t` trên mẫu Real | ✅ | Test (b), (c) |
| Batch không có Real không NaN | ✅ | Test (d) |
| `FeatureConsistencyLoss` cosine + KL, stop-grad nhánh clean | ✅ | Mục 2.2 |
| `CrossDomainTotalLoss` tương thích ngược khi weight = 0 | ✅ | Test (a) allclose |
| `CLIPModel` xuất feature, không đổi logit cũ | ✅ | `allclose = True`, feature `(1, 768)` |
| Config Phase 7 với đủ 5 trọng số | ✅ | `train_phase7_anchored_v2.yaml` |
| Không phá Task 7.2 | ✅ | `DoD #4: PASS` |

---

## 6. Rủi Ro Đã Giải Quyết

| Rủi ro | Kết quả |
| :--- | :--- |
| Thay đổi loss làm lệch run cũ | ✅ `legacy_mode` giữ nguyên công thức; test allclose |
| Guard kéo logit Real xuống vô hạn | ✅ ReLU + so với teacher, chỉ phạt phần vượt |
| NaN khi batch toàn Fake | ✅ Trả `0 * sum` giữ graph; test (d) |
| Sửa `forward` phá loader/eval | ✅ Mặc định `return_features=False`; regression 7.2 PASS |
| Local Windows không chạy được PyTorch (Smart App Control / PowerShell `DriveNotFound`) | ✅ Xác minh toàn bộ trên Colab |

---

## 7. Ghi Chú Tái Lập Trên Colab

- Code: `MyDrive/xla_ckpt/xla_code.zip` (chứa `src/`, `tests/`, `configs/`, `scratch/`), giải nén vào `/content/xla`.
- CLIP: `ViT-L-14.pt` đã lưu trên Drive `MyDrive/xla_ckpt/ViT-L-14.pt`, symlink vào `/content/xla/pretrained/ViT-L-14.pt` (code tìm tại `pretrained/`).
- Checkpoint: symlink 4 file `.pth` từ Drive vào đúng đường dẫn `scratch/verify_task_7_2.py` yêu cầu.

---

## 8. Kết Luận Nghiệm Thu

Task 7.9 **hoàn thành**. Loss v2 sẵn sàng, tương thích ngược, không ảnh hưởng loader 7.2.

| Hạng mục | Kết quả thực tế |
| :--- | :--- |
| pytest `test_loss_v2.py` | **4 passed** (13.32s) |
| Regression 7.2 `DoD #4` | **PASS** |
| Forward + `return_features` | `(1,2)` + `(1,768)`, logit không đổi |

Bàn giao cho Thành viên C:
- Gọi `model(x, return_features=True)` cho cả nhánh clean và degraded, truyền `features_clean`, `features_deg`, `weight_feat`, `weight_guard` vào `CrossDomainTotalLoss`.
- Đọc trọng số từ `configs/train_phase7_anchored_v2.yaml`; log thêm `loss_feat`, `loss_guard`, `guard_active`.
- Đầu vào cho **Task 7.10** (selection metric v2) và cổng **G7.2**.

---

## 9. Tích Hợp Đồng Bộ Huấn Luyện (End-to-End Integration)

Đã hoàn tất nối Loss v2 trực tiếp vào pipeline huấn luyện của dự án tại [`src/training/train_crossdomain.py`](../../../src/training/train_crossdomain.py):
1. **Trích xuất đặc trưng tự động**: Trong `CrossDomainTrainer.train_epoch()`, khi `weight_feat > 0` hoặc `weight_guard > 0`, student model được gọi với `return_features=True` cho cả hai nhánh clean và degraded, trích xuất `features_clean` và `features_deg` để cung cấp cho `CrossDomainTotalLoss`.
2. **Nạp cấu hình YAML & CLI**: `parse_arguments()` đã được bổ sung nạp tự động `weight_feat`, `weight_guard`, `fake_index` từ file YAML (`loss.*`) và hỗ trợ cờ CLI tương ứng (`--weight_feat`, `--weight_guard`, `--fake_index`).
3. **Giám sát Loss phụ**: Vòng lặp huấn luyện ghi nhận và log chi tiết `loss_feat`, `loss_guard`, và tỷ lệ kích hoạt `guard_active_ratio`.
4. **Kiểm thử tích hợp khép kín (Smoke Test)**:
   - File test: [`tests/test_smoke_loss_v2_integration.py`](../../../tests/test_smoke_loss_v2_integration.py)
   - Kết quả: **3/3 PASSED (100%)**
     * `test_smoke_loss_v2_trainer_integration`: PASSED (Forward + Backward + Optimizer Step với đầy đủ 5 thành phần loss).
     * `test_smoke_config_yaml_loss_v2_parsing`: PASSED (Đọc chuẩn `train_phase7_anchored_v2.yaml`).
     * `test_smoke_backward_compat_legacy_mode`: PASSED (Khi weights=0, hoàn toàn không gọi `return_features`, bảo toàn tài nguyên).
