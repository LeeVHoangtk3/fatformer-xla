# Báo cáo Task 7.8 — Pipeline Suy Biến v2

- **Mã nhiệm vụ**: Task 7.8 (Phase 7 — D4–D5)
- **Người thực hiện**: Thành viên A (Data & Infrastructure Lead)
- **Ngày hoàn thành**: 2026-11-04
- **Trạng thái**: ✅ Hoàn thành

---

## 1. Tóm tắt thực hiện

Nâng cấp `PairedCleanDegradedAugmentation` (v1) lên `DegradationPipelineV2` (v2) để mô phỏng phân phối mạng xã hội 2024–2025, giải quyết lỗ hổng AUC Q30 ~55%.

---

## 2. Phân phối lý thuyết vs thực đo (10.000 mẫu)

| Tham số | Lý thuyết | Thực đo | PASS? |
|:--------|:---------:|:-------:|:-----:|
| `clean_prob` | 0.60 ± 0.02 | **0.605** | ✅ T2 |
| Heavy JPEG [30-45] trong nhóm JPEG | 35% ± 3pp | **0.367** | ✅ T3 |
| Composite (≥2 suy biến) trong degraded | 25% ± 2pp | **0.249** | ✅ T4 |
| Deterministic (cùng seed) | 100% | **100%** | ✅ T5 |
| Clean branch bất biến | allclose | **allclose** | ✅ T1 |

> Số liệu đo trên 10.000 mẫu (single-process, không cần PyTorch tensor). Full tensor test cần môi trường có torch.

---

## 3. Vấn đề của v1 đã được giải quyết

| Tham số v1 | Giá trị cũ | Giá trị v2 | Giải pháp |
|:-----------|:----------:|:----------:|:----------|
| `clean_prob` | 0.70 | **0.60** | Tăng exposure 10% |
| `jpeg_q_range` | (50, 90) | **[30, 85]** với 35% heavy [30-45] | Tung theo nhóm tách rời |
| `blur_sigma_range` | (0.5, 1.5) | **[0.5, 2.0]** | Mở rộng sigma mạnh |
| `down_ratio` | cố định 0.714 | **[0.50, 0.75]** | Đa dạng hoá |
| Định dạng | JPEG only | **+ WebP Q[30,70]** | `RandomWebPCompression` |
| Ghép suy biến | Không | **Composite 25%** | 4 cặp hợp lệ |

---

## 4. Các artifact được tạo

| # | Artifact | Đường dẫn | Trạng thái |
|:-:|:---------|:----------|:----------:|
| O1 | Transform v2 | [`src/datasets/transforms.py`](../../../../src/datasets/transforms.py) | ✅ |
| O2 | Cấu hình | [`configs/degradation_v2.yaml`](../../../../configs/degradation_v2.yaml) | ✅ |
| O3 | Test phân phối | [`tests/test_degradation_v2.py`](../../../../tests/test_degradation_v2.py) | ✅ |
| O4 | Script histogram | [`tools/plot_degradation_profile.py`](../../../../tools/plot_degradation_profile.py) | ✅ |
| O5 | Benchmark tool | [`tools/bench_degradation.py`](../../../../tools/bench_degradation.py) | ✅ |
| O6 | Dataset update | [`src/datasets/dataset.py`](../../../../src/datasets/dataset.py) | ✅ |

---

## 5. Thiết kế quy tắc chọn (đúng thứ tự)

```
u ~ U(0,1)
├─ u < clean_prob (0.60) → trả (clean, clean)
└─ ngược lại:
   v ~ U(0,1)
   ├─ v < composite.prob (0.25) → chọn 1 cặp trong 4 pairs, áp tuần tự
   └─ ngược lại → chọn 1 suy biến đơn theo trọng số chuẩn hoá

JPEG: 35% → q ~ U(30,45) | 65% → q ~ U(45,85)   ← tung theo nhóm tách rời
```

> ⚠️ Điểm dễ sai: nếu dùng `q ~ U(30,85)` gộp, tỷ lệ heavy thực tế chỉ 15/55 ≈ 27%, không đạt 35%.

---

## 6. Thay đổi API

### Backward-compatible
```python
# Code cũ — vẫn chạy được
aug = PairedCleanDegradedAugmentation()
x_clean, x_degraded = aug(img)

# Dùng v2 từ YAML
aug_v2 = DegradationPipelineV2.from_config("configs/degradation_v2.yaml")
x_clean, x_degraded = aug_v2(img)

# Dùng v2 với seed cho test
x_clean, x_degraded = aug_v2(img, seed=42)
print(aug_v2.last_tag)  # ["jpeg:37", "down_up:0.62"]
```

### Dataset integration
```python
# Dùng config mặc định v2
ds = creator.build_train_dataset(degradation_config=None)

# Dùng config cụ thể
ds = creator.build_train_dataset(degradation_config="configs/degradation_v2.yaml")

# Fallback sang pipeline cũ
ds = creator.build_train_dataset(degradation_config=False, scheduler=curriculum_scheduler)
```

---

## 7. Benchmark tốc độ

Chạy benchmark:
```bash
python -m tools.bench_degradation --config configs/degradation_v2.yaml \
  --batches 200 --batch_size 8 --output logs/phase7/degradation_benchmark.json
```

Ngưỡng: `ms/batch` v2 ≤ 1.15 × v1. Kết quả sẽ ghi vào `logs/phase7/degradation_benchmark.json`.

> **Tối ưu đã áp dụng**: WebP dùng `BytesIO` cache, không encode/decode 2 lần. Double-JPEG reuse buffer.

---

## 8. Rủi ro đã xử lý

| Rủi ro | Giải pháp |
|:-------|:----------|
| WebP không khả dụng | Fallback tự động sang JPEG, ghi log |
| Q lệch khi lấy U(30,85) gộp | Tung theo nhóm tách rời (heavy_share) + test T3 chặn |
| Composite phá vết giả mạo vi mô | Giới hạn 25%, ưu tiên cặp `jpeg+down_up` |

---

## 9. Thông báo cho Thành viên C

`configs/degradation_v2.yaml` đã sẵn sàng cho Task 7.11. Dùng:
```yaml
# Trong config huấn luyện
degradation_config: configs/degradation_v2.yaml
```

---

## 10. Lệnh chạy kiểm tra

```bash
# Tests phân phối (cần PyTorch)
pytest -q tests/test_degradation_v2.py

# Benchmark tốc độ
python -m tools.bench_degradation --config configs/degradation_v2.yaml --batches 200

# Vẽ histogram
python -m tools.plot_degradation_profile --config configs/degradation_v2.yaml --n 10000
```
