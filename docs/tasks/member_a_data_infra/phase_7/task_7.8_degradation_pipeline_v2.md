# TASK 7.8 — PIPELINE SUY BIẾN v2 (JPEG 30–85, WebP, DOUBLE-JPEG, GHÉP 2 SUY BIẾN)

- **Mã nhiệm vụ**: `Task 7.8` *(Phase 7 — D4–D5)*
- **Người thực hiện**: **Thành viên A (Data & Infrastructure Lead)**
- **Người phối hợp**: **Thành viên B (Architecture Lead)** — đảm bảo phân phối suy biến khớp với `RealGuardLoss` / `FeatureConsistencyLoss` (Task 7.9)
- **Môi trường thực thi**: 🖥️ **Local CPU**
- **Độ ưu tiên**: 🟠 Cao — **chặn Task 7.11** (cấu hình train dùng `degradation_v2.yaml`)
- **Cổng liên quan**: không có cổng riêng; ảnh hưởng trực tiếp G7.2
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.8

---

## 1. Mục tiêu

Nâng `PairedCleanDegradedAugmentation` (hiện có trong [`src/datasets/transforms.py`](../../../../src/datasets/transforms.py)) từ mức "suy biến nhẹ, thiên lệch về ảnh sạch" lên mức **mô phỏng phân phối mạng xã hội 2024–2025**, để mô hình không sụp khi gặp ảnh nén mạnh — đây chính là lỗ hổng khiến AUC Q30 chỉ còn ~55%.

**Vấn đề của pipeline hiện tại**

| Tham số hiện tại | Giá trị | Vấn đề |
| :--- | :--- | :--- |
| `clean_prob` | 0,70 | 70% mẫu không hề bị suy biến ⇒ mô hình gần như chỉ học miền sạch |
| `jpeg_q_range` | (50, 90) | Không bao giờ chạm Q30–45 — đúng vùng test khó nhất |
| `blur_sigma_range` | (0,5; 1,5) | Thiếu blur mạnh (mạng xã hội nén + blur đồng thời) |
| `down_ratio` | 160/224 ≈ 0,714 | Cố định, không đa dạng |
| Định dạng | Chỉ JPEG | Không có WebP — định dạng phổ biến của ảnh tải lại |
| Ghép suy biến | Không | Thực tế luôn là **chuỗi** suy biến, không phải một |

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | `PairedCleanDegradedAugmentation` hiện tại | [`src/datasets/transforms.py`](../../../../src/datasets/transforms.py) (dòng ~241) | Nền để mở rộng, **giữ API** `__call__(img) -> (clean, degraded)` |
| I2 | Các transform đơn lẻ hiện có | `RandomJPEGCompression`, `RandomGaussianBlur`, `RandomDownUpSampling` | Tái sử dụng, không viết lại |
| I3 | Manifest v2 (Task 7.7) | `DATASET/manifests/data_manifest_v2.json` | Nguồn ảnh để đo phân phối & tốc độ |
| I4 | Cấu hình hiện tại | `configs/train_m6_r_anchored.yaml` | Mẫu định dạng YAML |

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung bắt buộc |
| :---: | :--- | :--- | :--- |
| O1 | Transform v2 | `src/datasets/transforms.py` | `PairedCleanDegradedAugmentation` nhận `config: dict` (hoặc đường dẫn YAML) |
| O2 | Cấu hình | `configs/degradation_v2.yaml` | Toàn bộ tham số, không hard-code |
| O3 | Test phân phối | `tests/test_degradation_v2.py` | 5 test (xem §5.4) |
| O4 | Hình histogram | `logs/phase7/degradation_profile.png` | Q distribution, nhóm nặng, blur σ, composite |
| O5 | Báo cáo tốc độ | `logs/phase7/degradation_benchmark.json` | `ms/batch` trước/sau, `%` chênh lệch |
| O6 | Báo cáo | `docs/bao_cao/A/bao_cao_task_7.8_degradation_v2.md` | Bảng phân phối + benchmark |

---

## 4. Ràng buộc

1. **Backward-compatible API**: `__call__(img) -> Tuple[Tensor, Tensor]`, cả hai `(3,224,224)`, cùng nhãn. Code cũ gọi không tham số vẫn phải chạy (dùng default = giá trị v2).
2. **Tất định khi cần**: hỗ trợ `seed` để test tái lập; trong train dùng RNG của `random`/`torch` như hiện tại.
3. **Không suy biến ảnh clean**: nhánh `clean` tuyệt đối chỉ `Resize → CenterCrop → ToTensor → Normalize`.
4. **Không làm chậm > 15%** (`time per batch`) so với pipeline hiện tại — đo trên cùng 200 batch, cùng `num_workers`.
5. **Không đổi** `src/models/`, `src/training/`; chỉ `src/datasets/`, `configs/`, `tests/`, `tools/`.
6. **Không dùng torchvision mới** nếu môi trường không có; WebP dùng `PIL` (`img.save(buf, format="WEBP", quality=q)`).

---

## 5. Các bước thực hiện

### Bước 1 — Thiết kế phân phối (D4, ~1 h)

Viết `configs/degradation_v2.yaml`:

```yaml
# configs/degradation_v2.yaml — Task 7.8
seed: 42
img_resolution: 224
crop_resolution: 224

clean_prob: 0.60            # 40% mẫu bị suy biến (trước: 30%)

jpeg:
  prob: 0.55
  q_range: [30, 85]
  heavy_share: 0.35         # >= 35% khối lượng JPEG rơi vào [30, 45]
  heavy_range: [30, 45]

webp:
  prob: 0.15
  q_range: [30, 70]

double_jpeg:
  prob: 0.15
  q1_range: [70, 90]
  q2_range: [30, 50]

blur:
  prob: 0.20
  sigma_range: [0.5, 2.0]

down_up:
  prob: 0.20
  ratio_range: [0.50, 0.75]

composite:
  prob: 0.25                # 25% mẫu bị >= 2 suy biến
  pairs:                    # cặp được phép ghép
    - [jpeg, down_up]
    - [jpeg, blur]
    - [webp, down_up]
    - [double_jpeg, blur]
```

**Quy tắc chọn** (phải cài đúng thứ tự, nếu không phân phối sẽ lệch):

1. Tung `u ~ U(0,1)`. Nếu `u < clean_prob` ⇒ trả `(clean, clean)` — **không** suy biến.
2. Ngược lại, tung `v ~ U(0,1)`. Nếu `v < composite.prob` ⇒ chọn **một cặp** trong `composite.pairs` theo trọng số đều, áp dụng **tuần tự** theo thứ tự trong cặp.
3. Nếu không, chọn **một** suy biến đơn theo trọng số `jpeg 0.55 / webp 0.15 / double_jpeg 0.15 / blur 0.20 / down_up 0.20` (chuẩn hoá lại).
4. Với JPEG: 35% lần lấy `q ~ U(30,45)`, 65% còn lại lấy `q ~ U(45,85)` — **tách rời** để bảo đảm tỷ lệ nhóm nặng, không phụ thuộc may mắn.

> **Điểm dễ sai**: nếu lấy `q ~ U(30,85)` rồi hy vọng "35% rơi vào [30,45]" thì tỷ lệ thực tế chỉ là `15/55 ≈ 27%`. Phải **tung theo nhóm** như bước 4.

### Bước 2 — Cài đặt transform (D4, ~2 h)

1. Thêm class `DegradationPipelineV2` (hoặc mở rộng trực tiếp `PairedCleanDegradedAugmentation`) với `from_config(path_or_dict)`.
2. Thêm `RandomWebPCompression(min_q, max_q)`, `RandomDoubleJPEG(q1_range, q2_range)` vào cùng file.
3. Giữ nguyên 3 transform cũ; transform v2 **gọi** chúng thay vì copy logic.
4. Ghi `degradation_tag` vào output phụ (ví dụ `self.last_tag = ["jpeg:37", "down_up:0.62"]`) để test và phân tích có thể đọc — **không** đưa vào tensor trả về.

### Bước 3 — Nối vào dataset (D4, ~0,5 h)

1. `src/datasets/dataset.py`: nhận `degradation_config` (đường dẫn YAML) và truyền vào transform.
2. Nếu `degradation_config` là `null` ⇒ dùng default v2 (không phá code cũ).
3. Smoke: 1 batch qua `DataLoader` với `batch_size=8`, in shape + tag.

### Bước 4 — Test phân phối & tốc độ (D5, ~1,5 h)

Viết `tests/test_degradation_v2.py`:

| # | Test | Điều kiện pass |
| :---: | :--- | :--- |
| T1 | Clean branch bất biến | Ảnh clean nhận được **giống hệt** pipeline chuẩn (allclose với `Resize→Crop→ToTensor→Normalize`) |
| T2 | 10.000 mẫu: tỷ lệ clean | `clean ≈ 0,60 ± 0,02` |
| T3 | 10.000 mẫu: nhóm nặng | Trong các mẫu bị JPEG, `q ∈ [30,45]` chiếm **≥ 35% ± 3 pp** |
| T4 | 10.000 mẫu: composite | `composite ≈ 0,25 ± 0,02` trong nhóm bị suy biến |
| T5 | Tất định | Cùng `seed` ⇒ cùng tag + cùng tensor (allclose) |

Đo tốc độ:

```bash
python -m tools.bench_degradation --config configs/degradation_v2.yaml \
  --batches 200 --num_workers 4 --output logs/phase7/degradation_benchmark.json
```

Ngưỡng: `ms/batch` mới ≤ `1,15 ×` cũ. Nếu vượt: giảm `num_workers` không phải giải pháp — phải tối ưu (tránh encode/decode WebP 2 lần, cache buffer `BytesIO`).

### Bước 5 — Vẽ histogram & báo cáo (D5, ~1 h)

1. Sinh `logs/phase7/degradation_profile.png` với 4 panel: (a) histogram Q, (b) tỷ lệ theo loại suy biến, (c) histogram `σ` blur, (d) tỷ lệ composite.
2. Ghi báo cáo: bảng phân phối lý thuyết vs thực đo (10.000 mẫu), bảng tốc độ trước/sau, các quyết định đã thay đổi so với v1.
3. Cập nhật [`../README_A.md`](../README_A.md) và thông báo C rằng `configs/degradation_v2.yaml` đã sẵn sàng cho Task 7.11.

---

## 6. Definition of Done

- [ ] `configs/degradation_v2.yaml` tồn tại, **không còn tham số suy biến hard-code** trong transform.
- [ ] `clean_prob = 0.60`; `Q ∈ [30,85]`; **≥ 35%** khối lượng JPEG trong `[30,45]` (đo thực, không lý thuyết).
- [ ] Có WebP, double-JPEG, blur `σ ∈ [0.5,2.0]`, down-up `0.5–0.75`, composite 25%.
- [ ] `pytest -q tests/test_degradation_v2.py` — 5/5 xanh.
- [ ] `logs/phase7/degradation_profile.png` có 4 panel; số liệu khớp test.
- [ ] DataLoader chậm hơn **≤ 15%**; số đo ghi trong `degradation_benchmark.json`.
- [ ] API `__call__` không đổi; smoke 1 batch qua `src/datasets/dataset.py` thành công.
- [ ] Báo cáo `docs/bao_cao/A/bao_cao_task_7.8_degradation_v2.md` hoàn tất.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
| :--- | :---: | :---: | :--- |
| Phân phối Q lệch vì lấy `U(30,85)` gộp | **Cao** | Không đạt 35% nhóm nặng | Tung theo nhóm tách rời (bước 1.4) + test T3 chặn |
| WebP không khả dụng trong PIL build | Trung bình | Thiếu 1 loại suy biến | Fallback: bỏ WebP, tăng JPEG nặng; **phải ghi rõ trong báo cáo** |
| Composite làm mất vết giả mạo vi mô | Trung bình | Giảm trần năng lực | Giới hạn 25%, ưu tiên cặp `jpeg+down_up` (ít phá cấu trúc) |
| Chậm > 15% | Trung bình | Trễ 7.11 | Tối ưu buffer encode; đo bằng 200 batch thật, không đo 1 ảnh |
| Ảnh 224×224 nén lại bị blockiness giả | Trung bình | Học sai dấu hiệu | Luôn `post_resize` degrade; ghi rõ thứ tự trong config |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md)
- Kho dữ liệu v2: [`task_7.7_kho_du_lieu_v2.md`](task_7.7_kho_du_lieu_v2.md)
- Loss tiêu thụ cặp `(clean, degraded)`: [`../../member_b_architecture_loss/phase_7/task_7.9_loss_v2_realguard.md`](../../member_b_architecture_loss/phase_7/task_7.9_loss_v2_realguard.md)
- Chỉ mục giai đoạn: [`README.md`](README.md)
