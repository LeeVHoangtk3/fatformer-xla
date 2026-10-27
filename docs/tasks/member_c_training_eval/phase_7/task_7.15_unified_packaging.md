# TASK 7.15 — ĐÓNG GÓI `UnifiedFatFormerXLA` + BÁO CÁO NGHIỆM THU

- **Mã nhiệm vụ**: `Task 7.15` *(Phase 7 — D11)*
- **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**
- **Người phối hợp**: **Thành viên B** — review wrapper, đảm bảo `forward()` không đổi ngữ nghĩa
- **Môi trường thực thi**: 🖥️ **Local CPU** + ☁️ GPU (đo độ trễ)
- **Độ ưu tiên**: 🟠 Cao — điều kiện cuối của **G7.3**
- **Cổng liên quan**: **G7.3**
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.15

---

## 1. Mục tiêu

Biến kết quả nghiên cứu thành **một artifact dùng được**: một file `.pth`, một lệnh CLI, một `forward()`.

**Ba yêu cầu của G7.3**

| Yêu cầu | Ngưỡng | Cách chứng minh |
| :--- | :---: | :--- |
| Một checkpoint, một `forward()` | bắt buộc | `ls` chỉ 1 file `.pth`; API không có nhánh "chọn model" |
| Chênh lệch vs ensemble/fixed | ≤ 1,5 pp | Chấm cùng tập, so trực tiếp |
| Độ trễ | < 5 ms/ảnh (GPU) | Đo 200 ảnh, báo cáo trung vị + p95 |

**Điểm khác biệt so với "đóng gói cho có"**

- Metadata (calibration, router, phiên bản) **nằm trong** file `.pth`, không phụ thuộc file ngoài.
- CLI **không** cần biết gì về kiến trúc — chỉ nhận ảnh và in kết quả.
- Nếu router không thắng (Task 7.13), wrapper **tự** dùng ngưỡng cố định — không crash, không cần cờ đặc biệt.

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | Checkpoint | `DATASET/checkpoints/phase7/fatformer_anchored_v2_best.pth` (hoặc `_best_effort.pth`) | Trọng số |
| I2 | Router params | `logs/phase7/router_params.json` | Fusion (nếu dùng) |
| I3 | Calibration | `logs/phase7/calibration/*.json` | `T`, `T*` |
| I4 | Loader | `src/models/canonical_loader.py` (Task 7.2 — tạo mới) | Nạp an toàn |
| I5 | Model | [`src/models/clip_models.py`](../../../../src/models/clip_models.py) | Kiến trúc |
| I6 | Kết quả G7.3 | `logs/phase7/final_phase7_report.json` | Số đối chiếu |

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | Wrapper | `src/models/unified_fatformer.py` | `UnifiedFatFormerXLA` + `predict()` |
| O2 | CLI | `tools/infer_unified.py` | Ảnh → `{prob_fake, label, w_deg, threshold}` |
| O3 | Checkpoint hợp nhất | `DATASET/checkpoints/phase7/unified_fatformer_v2.pth` | Trọng số + metadata |
| O4 | Metadata ngoài | `..._metadata.json` | Bản sao đọc được bằng mắt |
| O5 | Báo cáo độ trễ | `logs/phase7/latency_phase7.json` | trung vị, p95, batch 1/8/32 |
| O6 | Báo cáo | `docs/bao_cao/C/bao_cao_task_7.15_unified_packaging.md` | Nghiệm thu G7.3 |

---

## 4. Ràng buộc

1. **Một file `.pth` duy nhất** cho suy luận. Không được yêu cầu "nạp thêm router.pth".
2. **Metadata nhúng trong checkpoint** dưới khoá `metadata`; CLI đọc từ đó, không đọc file rời.
3. **`predict()` là API duy nhất** — không có `predict_clean`, `predict_forensic` công khai.
4. **Không phụ thuộc `mean_w`** hay thống kê batch (Task 7.13).
5. **Xử lý ảnh đầu vào** đúng như lúc eval: `Resize(224) → CenterCrop(224) → ToTensor → Normalize`, **không** augmentation.
6. **Fallback**: thiếu `router_params` ⇒ tự dùng ngưỡng cố định, in cảnh báo `[WARN] router disabled`, vẫn trả kết quả.
7. **Không sửa** trọng số trong quá trình đóng gói — `sha256` của tensor phải khớp checkpoint gốc.

---

## 5. Các bước thực hiện

### Bước 1 — Wrapper `UnifiedFatFormerXLA` (D11, ~1,5 h)

```python
class UnifiedFatFormerXLA(nn.Module):
    def __init__(self, backbone, router=None, calibration=None, router_enabled=False):
        super().__init__()
        self.backbone = backbone
        self.router = router
        self.calibration = calibration or {}
        self.router_enabled = router_enabled

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """x: [B,3,224,224] đã normalize. Trả dict:
        prob_fake [B], label [B], w_deg [B] | None, threshold [B]"""
        logits, feats = self.backbone(x, return_features=True)
        T = self.calibration.get("T", 1.0)
        p_clean = torch.softmax(logits / T, dim=-1)[:, 1]

        if self.router_enabled and self.router is not None:
            w = self.router(feats, x)                       # [B] từng ảnh
            p_forensic = p_clean                            # cùng backbone, khác nhánh gating
            p = (1 - w) * p_clean + w * p_forensic
            thr = self.router.threshold_for(w)              # vector [B]
        else:
            w = None
            p = p_clean
            thr = torch.full_like(p, self.calibration["T_star"]["0.05"])

        return {"prob_fake": p, "label": (p >= thr).long(),
                "w_deg": w, "threshold": thr}
```

> **Ghi chú thiết kế**: vì chỉ có **một** backbone, `p_clean` và `p_forensic` chia sẻ logits; router điều chỉnh **ngưỡng** và **trọng số** chứ không chạy 2 model. Nếu B quyết định dùng 2 nhánh thật (ví dụ gating mở/tắt SRM), phải chứng minh độ trễ < 5 ms — nếu không, giữ thiết kế một nhánh và ghi rõ.

### Bước 2 — Đóng gói checkpoint (D11, ~0,5 h)

```python
payload = {
    "model": model.state_dict(),
    "metadata": {
        "schema": "unified_fatformer/v1",
        "layout": {"n": 3, "indices": [7, 15, 23], "dropped": [], "partial": False},
        "calibration": calib_dict,          # T, T_star, source, n_calib
        "router": router_dict,              # None nếu không dùng
        "router_enabled": bool,
        "git_commit": "...",
        "source_checkpoint_sha256": "...",
        "created_at": "..."
    }
}
torch.save(payload, "DATASET/checkpoints/phase7/unified_fatformer_v2.pth")
```

Kiểm tra:

```python
assert all(torch.equal(payload["model"][k], src_sd[k]) for k in payload["model"])
```

### Bước 3 — CLI (D11, ~1 h)

```bash
python tools/infer_unified.py --checkpoint DATASET/checkpoints/phase7/unified_fatformer_v2.pth \
  --images samples/*.jpg --output logs/phase7/infer_demo.json
```

Xuất:

```json
{
  "checkpoint": "...", "layout": {"n": 3, "partial": false},
  "results": [{"path": "a.jpg", "prob_fake": 0.87, "label": "fake",
               "w_deg": 0.42, "threshold": 0.51, "ms": 3.1}]
}
```

Yêu cầu CLI:

- `--images` nhận nhiều file hoặc thư mục.
- `--device {cpu,cuda}`, `--batch_size`.
- In bảng gọn ra stdout, ghi JSON nếu có `--output`.
- Không crash khi ảnh hỏng: in `[SKIP] path: lý do`.

### Bước 4 — Đo độ trễ (D11, ~0,5 h)

```python
# 200 ảnh, batch 1/8/32, warmup 20, đo bằng torch.cuda.Event
```

Ghi `latency_phase7.json`:

```json
{
  "device": "T4",
  "batch": {"1": {"median_ms": 0.0, "p95_ms": 0.0},
            "8": {}, "32": {}},
  "per_image_ms_batch1": 0.0,
  "meets_5ms": false
}
```

> G7.3 yêu cầu **< 5 ms/ảnh**. Nếu batch 1 vượt, báo cáo cả batch 8/32 và ghi rõ điều kiện đạt. **Không** đổi đơn vị để "đạt".

### Bước 5 — Đối chiếu G7.3 (D11, ~0,5 h)

1. Chấm `unified_fatformer_v2.pth` trên 2.000 ảnh CFE bằng chính `predict()`.
2. So với `final_phase7_report.json`:
   - `|Fake@Real90_unified − Fake@Real90_report| ≤ 1,5 pp`
   - `|AUC_unified − AUC_report| ≤ 1,5 pp`
3. Nếu lệch > 1,5 pp ⇒ wrapper đang làm sai (tiền xử lý, `T`, ngưỡng) ⇒ sửa, không chấp nhận.

### Bước 6 — Báo cáo nghiệm thu (D11, ~0,5 h)

`docs/bao_cao/C/bao_cao_task_7.15_unified_packaging.md` gồm:

1. Sơ đồ luồng: ảnh → preprocess → backbone → (router) → `prob_fake` → `label`.
2. Bảng G7.3: 3 yêu cầu, số đo, PASS/FAIL.
3. Bảng độ trễ (batch 1/8/32, median + p95).
4. **1 lệnh mẫu** chạy được, copy-paste.
5. Danh sách file artifact + `sha256`.

---

## 6. Definition of Done

- [ ] `src/models/unified_fatformer.py` tồn tại, chỉ có **một** API công khai `predict()`.
- [ ] `DATASET/checkpoints/phase7/unified_fatformer_v2.pth` chứa **cả** `model` và `metadata`.
- [ ] `sha256` của tensor khớp checkpoint nguồn (assert trong script đóng gói).
- [ ] `tools/infer_unified.py` chạy được trên **20 ảnh mẫu**, xuất JSON + bảng stdout.
- [ ] Ảnh hỏng không làm crash CLI.
- [ ] Thiếu `router` ⇒ fallback ngưỡng cố định, có cảnh báo, vẫn chạy.
- [ ] `latency_phase7.json` có median + p95 cho batch 1/8/32.
- [ ] Chênh lệch vs `final_phase7_report.json` ≤ **1,5 pp** trên cả `Fake@Real90` và `AUC`.
- [ ] Độ trễ < 5 ms/ảnh **hoặc** ghi rõ không đạt + điều kiện đạt.
- [ ] Báo cáo nghiệm thu có bảng G7.3 PASS/FAIL **kèm số**.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
| :--- | :---: | :---: | :--- |
| Tiền xử lý trong CLI khác lúc eval | **Cao** | Lệch > 1,5 pp | Dùng **cùng** hàm transform; test so sánh 20 ảnh |
| Router hai nhánh làm vượt 5 ms | Trung bình | G7.3 trượt | Giữ một nhánh; nếu hai nhánh, đo lại và báo cáo |
| `.pth` quá lớn (> 2,5 GB) | Thấp | Khó chia sẻ | Chỉ lưu `state_dict` float32; không lưu optimizer |
| Metadata lệch với checkpoint thật | Trung bình | Kết quả sai | Sinh metadata **từ** checkpoint khi đóng gói, không copy tay |
| `predict()` trả xác suất thô chưa hiệu chuẩn | Cao | Ngưỡng sai | Test: `p` tính từ `softmax(logits/T)`, assert `T` được áp |
| Người dùng chạy trên CPU và thấy chậm | Trung bình | Hiểu nhầm G7.3 trượt | Báo cáo **cả** CPU và GPU, nêu rõ ngưỡng áp cho GPU |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) — §II.3 (G7.3)
- Locked benchmark: [`task_7.14_locked_benchmark_phase7.md`](task_7.14_locked_benchmark_phase7.md)
- Router: [`../../member_b_architecture_loss/phase_7/task_7.13_degradation_router.md`](../../member_b_architecture_loss/phase_7/task_7.13_degradation_router.md)
- Calibration: [`task_7.4_calibration_threshold_fpr.md`](task_7.4_calibration_threshold_fpr.md)
- Chỉ mục giai đoạn: [`README.md`](README.md)
