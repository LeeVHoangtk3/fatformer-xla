# TASK 7.9 — LOSS v2: `RealGuardLoss` + `FeatureConsistencyLoss`

- **Mã nhiệm vụ**: `Task 7.9` *(Phase 7 — D4–D5)*
- **Người thực hiện**: **Thành viên B (Architecture & Loss Lead)**
- **Người phối hợp**: **Thành viên C (Training Lead)** — tích hợp vào `train_crossdomain.py`, smoke trên T4
- **Môi trường thực thi**: 🖥️ **Local CPU** (code + test) + ☁️ **T4** (smoke 1 batch)
- **Độ ưu tiên**: 🟠 Cao — chặn Task 7.11
- **Cổng liên quan**: không có cổng riêng; là đòn bẩy chính của G7.2
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.9

---

## 1. Mục tiêu

Bổ sung **hai ràng buộc còn thiếu** vào `CrossDomainTotalLoss`, nhắm đúng hai thất bại quan sát được của Phase 6-R:

| Thất bại quan sát được | Nguyên nhân | Đòn bẩy |
| :--- | :--- | :--- |
| `Real` bị dự đoán thành `Fake` khi ảnh bị nén mạnh (Real ACC tụt) | Không có ràng buộc **một chiều** bảo vệ Real | `RealGuardLoss` |
| Đặc trưng trôi dạt giữa clean và degraded ⇒ cùng một ảnh cho 2 quyết định khác nhau | Không có ràng buộc ở **không gian đặc trưng** | `FeatureConsistencyLoss` |

**Công thức tổng (theo kế hoạch §IV.2)**

$$\mathcal{L} = 0{,}3\,\mathcal{L}_{CE}(x_{clean}) + 0{,}3\,\mathcal{L}_{CE}(x_{deg}) + 0{,}7\,\mathcal{L}_{KD} + 0{,}3\,\mathcal{L}_{Feat} + 0{,}5\,\mathcal{L}_{Guard}$$

### `RealGuardLoss`

$$\mathcal{L}_{Guard} = \frac{1}{|B_{real}|}\sum_{i\in B_{real}} \Big[\mathrm{ReLU}\big(z^{fake}_{s}(x^{deg}_i) - z^{fake}_{t}(x^{clean}_i)\big)\Big]^2$$

- `z^fake` = **logit** của lớp Fake (không phải xác suất).
- `s` = student (model đang train), `t` = teacher (Baseline, `eval()` + `no_grad`).
- Dùng `ReLU` để **chỉ phạt khi student vượt teacher** trên ảnh Real. Nếu student đã "ngoan" hơn teacher thì loss = 0 — không kéo logit xuống vô hạn, không tạo gradient đối kháng.
- Nếu `|B_real| = 0` ⇒ trả `0.0` **kèm cờ** `guard_active=False` (không chia cho 0, không NaN).

### `FeatureConsistencyLoss`

$$\mathcal{L}_{Feat} = \big(1 - \cos(f_s(x_{deg}),\ \mathrm{sg}[f_s(x_{clean})])\big) + \mathrm{KL}\big(p_{clean}\,\|\,p_{deg}\big)$$

- `f` = vector **trước classifier** (`aug_image_features`), `sg` = stop-gradient.
- `p_clean`, `p_deg` = softmax có nhiệt độ `T=2.0`, `p_clean` **detach**.
- Mục tiêu: nhánh degraded học **cùng biểu diễn** với nhánh clean, thay vì học một shortcut riêng.

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | Loss hiện tại | [`src/training/loss.py`](../../../../src/training/loss.py) | `CrossDomainTotalLoss` (dòng ~247), `CrossDomainDistillationLoss`, `PairedConsistencyLoss` |
| I2 | Cặp ảnh `(clean, degraded)` | Task 7.8 — `configs/degradation_v2.yaml` | Nguồn `x_clean`, `x_deg` |
| I3 | Teacher | Baseline `fatformer_4class_ckpt.pth` | `z_t^{fake}`, `p_clean` |
| I4 | Trainer hiện tại | [`src/training/trainer.py`](../../../../src/training/trainer.py), [`src/training/train_crossdomain.py`](../../../../src/training/train_crossdomain.py) | Nơi gọi loss |
| I5 | Config mẫu | [`configs/train_m6_r_anchored.yaml`](../../../../configs/train_m6_r_anchored.yaml) | Mẫu để tạo config v2 |

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | 2 module loss | `src/training/loss.py` | `RealGuardLoss`, `FeatureConsistencyLoss` — mặc định `weight=0` |
| O2 | Tích hợp | `src/training/loss.py` | `CrossDomainTotalLoss.forward(...)` nhận thêm `logits_clean`, `logits_deg`, `features_clean`, `features_deg` |
| O3 | Cờ forward | `src/models/clip/model.py` / `clip_models.py` | `return_features=True` trả thêm `aug_image_features` |
| O4 | Config | `configs/train_phase7_anchored_v2.yaml` | 5 trọng số + `T` + `label_smoothing` |
| O5 | Test | `tests/test_loss_v2.py` | 3 test (a)(b)(c) |
| O6 | Báo cáo | `docs/bao_cao/B/bao_cao_task_7.9_loss_v2.md` | Công thức, test, số đo gradient |

---

## 4. Ràng buộc

1. **Backward-compatible tuyệt đối**: khi `weight_feat = weight_guard = 0`, `loss_total` phải **trùng** công thức cũ trong sai số `1e-6`. Đây là điều kiện để không phá các run cũ.
2. **Không NaN**: mọi phép chia phải có mẫu số bảo vệ (`max(n, 1)`), `cos` phải chuẩn hoá `eps=1e-8`.
3. **`return_features` mặc định `False`** — không đổi `forward()` hiện tại.
4. **Teacher luôn `eval()` + `no_grad()`**; không backprop qua teacher.
5. **Không đổi** `PairedConsistencyLoss` và `CrossDomainDistillationLoss` hiện có.
6. **Trọng số không cần cộng bằng 1** — nhưng phải log **từng thành phần** để phát hiện mất cân bằng (xem §7).

---

## 5. Các bước thực hiện

### Bước 1 — `RealGuardLoss` (D4, ~2 h)

```python
class RealGuardLoss(nn.Module):
    """Chỉ phạt khi student vượt teacher trên logit Fake của mẫu Real."""
    def __init__(self, fake_index: int = 1):
        super().__init__()
        self.fake_index = fake_index

    def forward(self, logits_student_deg, logits_teacher_clean, is_real):
        # is_real: bool tensor [B]
        if is_real.sum() == 0:
            return logits_student_deg.sum() * 0.0, {"guard_active": False, "n_real": 0}
        z_s = logits_student_deg[is_real, self.fake_index]
        z_t = logits_teacher_clean[is_real, self.fake_index].detach()
        excess = torch.relu(z_s - z_t)
        loss = (excess ** 2).mean()
        return loss, {"guard_active": True, "n_real": int(is_real.sum()),
                      "guard_excess_mean": excess.mean().detach()}
```

**Kiểm tra `fake_index`**: phải lấy từ mapping nhãn của dataset (`REAL=0, FAKE=1` theo `data_manifest.json`), **không** giả định. Ghi hằng số vào config.

### Bước 2 — `FeatureConsistencyLoss` (D4, ~1,5 h)

```python
class FeatureConsistencyLoss(nn.Module):
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.T = temperature

    def forward(self, f_deg, f_clean, logits_deg, logits_clean):
        cos = F.cosine_similarity(f_deg, f_clean.detach(), dim=-1, eps=1e-8)
        loss_cos = (1.0 - cos).mean()
        p_clean = F.softmax(logits_clean.detach() / self.T, dim=-1)
        log_p_deg = F.log_softmax(logits_deg / self.T, dim=-1)
        loss_kl = F.kl_div(log_p_deg, p_clean, reduction="batchmean") * (self.T ** 2)
        return loss_cos + loss_kl, {"feat_cos": cos.mean().detach(), "feat_kl": loss_kl.detach()}
```

> Nhân `T²` để KL không bị thu nhỏ khi `T > 1` (chuẩn Hinton). Ghi rõ trong docstring.

### Bước 3 — Nối vào `CrossDomainTotalLoss` (D4–D5, ~1,5 h)

Chữ ký mới (giữ default để code cũ chạy):

```python
def forward(self, logits_clean, logits_deg, targets, teacher_logits=None,
            features_clean=None, features_deg=None,
            weight_feat: float = 0.0, weight_guard: float = 0.0):
```

Logic:

```python
ce_clean = self.ce_loss(logits_clean, targets)
ce_deg   = self.ce_loss(logits_deg, targets)
kd       = self.kd_loss(...)              # giữ nguyên
cons     = self.cons_loss(...)            # giữ nguyên
feat = guard = torch.zeros((), device=...)
if weight_feat > 0:
    feat, _ = self.feat_loss(features_deg, features_clean, logits_deg, logits_clean)
if weight_guard > 0:
    guard, _ = self.guard_loss(logits_deg, teacher_logits, is_real=(targets == 0))
total = (0.3*ce_clean + 0.3*ce_deg + 0.7*kd + 0.2*cons
         + weight_feat*feat + weight_guard*guard)
return {"loss_total": total, "ce_clean": ..., "ce_deg": ..., "kd": ...,
        "cons": ..., "feat": ..., "guard": ..., "guard_active": ...}
```

> ⚠️ **Chú ý quan trọng**: công thức cũ là `0.7·[0.7·CE_clean + 0.3·CE_deg] = 0.49/0.21`, không phải `0.3/0.3`. Khi `weight_feat=weight_guard=0`, để test allclose với bản cũ, phải **giữ nhánh cũ** (`self.weight_ce * (0.7*ce_clean + 0.3*ce_deg)`), và chỉ dùng `0.3/0.3` khi có ít nhất một weight mới > 0. Cài bằng một cờ `legacy_mode` nội bộ, **không** để lộ ra ngoài.

### Bước 4 — `return_features` (D5, ~0,5 h)

1. Trong `VisionTransformer.forward(..., return_full)` và wrapper `clip_models`, thêm `return_features: bool = False`.
2. Khi `True`, trả `(logits, aug_image_features)`; khi `False`, hành vi **giữ nguyên**.
3. `train_crossdomain.py`: gọi `model(x_deg, return_features=True)` khi `weight_feat > 0` hoặc `weight_guard > 0`; ngược lại dùng đường cũ.

### Bước 5 — Test (D5, ~1 h)

`tests/test_loss_v2.py`:

| # | Test | Kỳ vọng |
| :---: | :--- | :--- |
| (a) | `weight_feat=weight_guard=0` | `loss_total` **allclose** với `CrossDomainTotalLoss` bản cũ (`atol=1e-6`) trên 3 batch giả ngẫu nhiên (seed cố định) |
| (b) | Mẫu Real có `z_s < z_t` | `guard == 0` và gradient của `z_s^{fake}` **bằng 0** |
| (c) | Mẫu Real có `z_s > z_t` | `guard > 0` và `∂guard/∂z_s^{fake} > 0` (gradient **kéo logit xuống**) |

Test bổ sung (khuyến nghị): batch **toàn Fake** ⇒ `guard == 0`, không NaN, `guard_active == False`.

### Bước 6 — Config & smoke (D5, ~0,5 h)

`configs/train_phase7_anchored_v2.yaml`:

```yaml
loss:
  weight_ce: 0.3
  weight_ce_deg: 0.3
  label_smoothing: 0.05
  temperature: 2.0
  alpha_kd: 0.7
  beta_cons: 0.2
  weight_feat: 0.3
  weight_guard: 0.5
  fake_index: 1
```

Smoke trên T4:

```bash
python -m src.training.train_crossdomain --config configs/train_phase7_anchored_v2.yaml --max_steps 20
```

Kiểm tra: không NaN, `guard_active` True ở ≥ 1 step, `feat_cos` khởi đầu > 0,9 (vì clean/deg cùng ảnh).

---

## 6. Definition of Done

- [ ] `RealGuardLoss` và `FeatureConsistencyLoss` có trong `src/training/loss.py`, **mặc định `weight=0`**.
- [ ] `CrossDomainTotalLoss.forward` nhận `logits_clean/logits_deg/teacher_logits/features_*` và trả dict có **đủ 7 khoá** (`loss_total, ce_clean, ce_deg, kd, cons, feat, guard`).
- [ ] `pytest -q tests/test_loss_v2.py` — 3/3 xanh (a)(b)(c), cộng test toàn-Fake.
- [ ] `weight=0` ⇒ `allclose` với bản cũ (`atol=1e-6`) trên 3 batch.
- [ ] `return_features` mặc định `False`; `forward()` cũ không đổi (test smoke so sánh output trước/sau).
- [ ] Smoke 20 step trên T4: không NaN, không OOM, `feat_cos` được log.
- [ ] `configs/train_phase7_anchored_v2.yaml` tồn tại, khớp §5.6.
- [ ] Báo cáo ghi **grad norm** 100 step đầu để chứng minh loss không bùng nổ.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
| :--- | :---: | :---: | :--- |
| **Tổng trọng số tăng 1,2 → 2,1** ⇒ grad norm tăng, LR hiệu dụng đổi | **Cao** | Train phân kỳ / mất ổn định | Log grad norm 100 step; nếu > 2× baseline ⇒ giảm LR 2× hoặc chia `loss_total` cho 2,1 và **ghi rõ** trong config |
| `guard` phạt nhầm mẫu Real bị gán nhãn sai | Trung bình | Học lệch | Kiểm tra phân bố `guard_excess_mean`; nếu > 1,0 ngay từ epoch 1 ⇒ audit nhãn |
| `feat_cos` sập về 0 (học collapse) | Trung bình | Mất đa dạng | Theo dõi `feat_cos` mỗi epoch; nếu < 0,5 ⇒ giảm `weight_feat` |
| `return_features` làm chậm forward | Thấp | +5–10% thời gian | Chỉ bật khi cần; đo trong smoke |
| `fake_index` sai | **Cao nếu giả định** | Guard vô nghĩa | Đọc từ config + assert `0 <= fake_index < n_classes` |
| KL dùng `reduction="mean"` thay `batchmean` | Trung bình | Loss lệch theo batch size | Dùng `batchmean` + test với 2 batch size khác nhau |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2
- Cặp dữ liệu đầu vào: [`../../member_a_data_infra/phase_7/task_7.8_degradation_pipeline_v2.md`](../../member_a_data_infra/phase_7/task_7.8_degradation_pipeline_v2.md)
- Task chọn checkpoint dùng loss này: [`task_7.10_selection_metric_v2.md`](task_7.10_selection_metric_v2.md)
- Task train tiêu thụ: [`../../member_c_training_eval/phase_7/task_7.11_train_anchored_v2.md`](../../member_c_training_eval/phase_7/task_7.11_train_anchored_v2.md)
- Chỉ mục giai đoạn: [`README.md`](README.md)
