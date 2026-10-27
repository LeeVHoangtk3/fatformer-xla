# TASK 7.7 — KHO DỮ LIỆU ĐA BỘ SINH v2 (CFE-Small PHÂN TẦNG + ẢNH THẬT ĐA NGUỒN)

- **Mã nhiệm vụ**: `Task 7.7` *(Phase 7 — Tái lập đo lường & Hợp nhất, D1–D3)*
- **Người thực hiện**: **Thành viên A (Data & Infrastructure Lead)**
- **Người phối hợp**: **Thành viên C (Training Lead)** — xác nhận manifest trước khi bắt đầu Task 7.11
- **Môi trường thực thi**: ☁️ **Google Drive 5TB / Colab CPU** (stream Parquet) + 🖥️ Local (audit)
- **Độ ưu tiên**: 🔴 Khẩn cấp — **chặn Task 7.11**
- **Cổng liên quan**: không có cổng riêng; đầu ra là **điều kiện tiên quyết** của G7.2
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §IV.2 — Task 7.7

---

## 1. Mục tiêu

Xây dựng kho dữ liệu huấn luyện **12.000–16.000 ảnh** cho Gen 2, thoả đồng thời 3 điều kiện:

1. **Phủ 5 họ kiến trúc 2024–2025**: `commercial`, `latent_diffusion`, `pixel_diffusion`, `modern_gans`, `autoregressive` — mỗi họ tối đa 1.200 ảnh.
2. **Cô lập 100%** với 2.000 ảnh test CommunityForensics và 18 tập test của Task 5.1 — cả ở mức **byte** (SHA-256) lẫn **tri giác** (pHash).
3. **Generator-disjoint**: không một generator nào trong tập train được trùng với generator trong tập test.

Đầu ra phải đủ tin cậy để khi G7.2 trượt, kết luận "dữ liệu không phải nguyên nhân" là **có bằng chứng**, không phải suy đoán.

> **Vì sao phải làm lại kho dữ liệu?** Kho hiện tại (`DATASET/manifests/data_manifest.json`, 3.600 mẫu) chỉ có **một nguồn duy nhất** `source=diffusion_staging`, với 3 generator (`real` 1.800 / `stable_diffusion` 1.200 / `midjourney` 600). Một kho đơn nguồn không thể chứng minh năng lực tổng quát hoá — nó chỉ chứng minh khả năng khớp lại phân phối cũ.

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn / Nguồn | Vai trò | Bắt buộc |
| :---: | :--- | :--- | :--- | :---: |
| I1 | `CommunityForensics-Small` (Parquet) | HF `CommunityForensics/CommunityForensics-Small` | Nguồn ảnh giả chính, có trường `generator` | ✅ |
| I2 | Danh sách generator của tập test CFE | `DATASET/test_communityforensics/` + manifest test | Danh sách **cấm** (deny-list) | ✅ |
| I3 | Danh sách 18 tập test Task 5.1 | [`bao_cao_task_5.1_full_benchmark_18_datasets.md`](../../../bao_cao/C/bao_cao_task_5.1_full_benchmark_18_datasets.md) | Danh sách **cấm** bổ sung | ✅ |
| I4 | Kho replay in-dist | `DATASET/manifests/data_manifest.json` (3.600 mẫu) | Chống quên miền cũ | ✅ |
| I5 | Ảnh thật đa nguồn | COCO / FFHQ / RAISE / LAION / ImageNet | Neo lớp Real | ✅ (≥ 3 nguồn) |
| I6 | `tools/verify_data_isolation.py` | có sẵn — SHA-256 + pHash | Công cụ kiểm toán | ✅ |
| I7 | `tools/build_crossdomain_manifest.py` | có sẵn — sinh manifest | Mẫu định dạng manifest | ✅ |

**Không được dùng làm đầu vào**: bất kỳ ảnh nào đã nằm trong `DATASET/test_communityforensics/` hoặc trong 18 tập Task 5.1 — kể cả ở dạng crop, resize, nén lại, lật, hay đổi định dạng.

---

## 3. Đầu ra

| # | Artifact | Đường dẫn | Nội dung bắt buộc |
| :---: | :--- | :--- | :--- |
| O1 | Manifest v2 | `DATASET/manifests/data_manifest_v2.json` | ≥ 12.000 bản ghi; mỗi bản ghi có `path, sha256, phash, generator, family, split, label, source, width, height` |
| O2 | Báo cáo cô lập | `logs/phase7/data_isolation_report.json` | 3 cặp kiểm tra; `violations = 0` |
| O3 | Script stream | `tools/build_manifest_v2.py` | có `--dry-run`, `--limit_per_family`, `--cache_dir`, `--resume` |
| O4 | Gói dữ liệu | `DATASET/train_crossdomain_v2.tar` (hoặc thư mục) | đã đẩy lên Drive, có `sha256` của gói |
| O5 | Báo cáo | `docs/bao_cao/A/bao_cao_task_7.7_kho_du_lieu_v2.md` | bảng phân bố, thời gian, dung lượng, vi phạm |
| O6 | Hình phân bố | `logs/phase7/data_v2_distribution.png` | histogram theo `family` × `split` × `label` |

---

## 4. Ràng buộc (bất di bất dịch)

1. **Cấm tái sử dụng test**: 2.000 ảnh CFE và 18 tập Task 5.1 là *held-out vĩnh viễn*. Nếu audit phát hiện **1** vi phạm ⇒ **chặn Task 7.11**, không có ngoại lệ.
2. **Generator-disjoint**: danh sách generator deny phải được tính **tự động** từ manifest test, không gõ tay. Ghi rõ số generator bị loại.
3. **Hash bắt buộc**: ảnh thiếu `sha256` hoặc `phash` ⇒ **không được** vào manifest.
4. **Cân bằng theo họ**: chênh lệch số mẫu giữa họ lớn nhất và nhỏ nhất ≤ **1,5×** (sau khi chia split).
5. **Tỷ lệ Real/Fake**: 45–55% Real trong **mỗi** split.
6. **Replay in-dist**: 3.600 ảnh `diffusion_staging` chiếm **20–25%** tổng kho.
7. **Không sửa** `src/models/`, `src/training/` — chỉ `src/datasets/`, `tools/`, `configs/`, `DATASET/`.
8. **Kill criteria**: stream vượt **4 giờ** hoặc **60 GB** ⇒ hạ xuống **8.000 ảnh (600/họ)**, ghi rõ trong báo cáo và báo ngay cho C.

---

## 5. Các bước thực hiện

### Bước 1 — Xây deny-list generator (D1, ~1,5 h)

1. Đọc manifest test CFE và trích tập `generator` duy nhất.
2. Đọc danh mục 18 tập Task 5.1, trích tên generator/họ tương ứng.
3. Chuẩn hoá tên (lowercase, bỏ phiên bản phụ, gộp alias) và lưu `logs/phase7/generator_denylist.json`:

```json
{
  "source": ["cfe_test_manifest", "task_5.1_18_datasets"],
  "n_generators": 0,
  "generators": [],
  "normalization_rules": ["lower", "strip_version_suffix", "alias_table"]
}
```

4. Kiểm tra `--dry-run`: in ra **số generator bị loại** và **số ảnh ước tính còn lại**. Nếu số còn lại < 10.000 ⇒ mở rộng nguồn (thêm họ mới), **không** nới deny-list.

**Bằng chứng**: `tools/build_manifest_v2.py --dry-run` in được bảng `family | n_candidates | n_blocked | n_kept`.

### Bước 2 — Stream 5 họ (D1–D2, ~6 h)

1. Stream Parquet theo batch (không tải toàn bộ), lọc **ngay** theo `family` và deny-list trước khi giải mã ảnh — tiết kiệm băng thông.
2. Giới hạn **1.200 ảnh/họ**, lấy mẫu **đa dạng generator trong cùng họ** (không lấy 1.200 ảnh của một generator).
3. Kiểm tra tính hợp lệ: kích thước ≥ 200×200, giải mã được, không alpha-only, không ảnh 1 màu.
4. Cache ảnh đã lọc vào `DATASET/raw_v2/<family>/` để `--resume` không phải stream lại.

**Bằng chứng**: log stream ghi `n_scanned, n_kept, n_blocked_by_generator, n_rejected_invalid, bytes_downloaded`.

### Bước 3 — Ảnh thật đa nguồn (D2, ~2 h)

1. Thu **3.000–4.000** ảnh thật từ **≥ 3 nguồn** khác nhau (khuyến nghị: RAISE + COCO + FFHQ, thêm LAION/ImageNet nếu cần độ đa dạng nén).
2. Ưu tiên ảnh **đã qua JPEG thật** (không phải PNG được nén lại) để lớp Real có đa dạng lịch sử nén.
3. Ghi `source` riêng cho từng nguồn; **không** gộp thành `real` chung — cần cho phân tích thiên lệch nguồn ở Task 7.14.
4. Khử trùng lặp nội bộ bằng pHash trước khi đưa vào split.

### Bước 4 — Replay in-dist (D2, ~0,5 h)

1. Nạp 3.600 mẫu từ `DATASET/manifests/data_manifest.json`.
2. Giữ nguyên `label` và `generator` gốc, đánh dấu `is_replay: true`.
3. Kiểm tra 3.600 mẫu này **không** trùng với test (cùng pipeline audit).

### Bước 5 — Chia split phân tầng (D2–D3, ~1 h)

| Split | Tỷ lệ | Mục đích | Ràng buộc |
| :--- | ---: | :--- | :--- |
| `train` | 85% | Huấn luyện Task 7.11 | cân bằng `family` × `label` |
| `val_calib` | 7,5% | Fit nhiệt độ + router (7.4, 7.13) | **không** dùng để chọn checkpoint |
| `val_select` | 7,5% | Chọn checkpoint theo worst-group (7.10) | **không** dùng để fit calibration |

- Chia **theo generator** khi có thể (một generator chỉ nằm ở một split) để `val_*` đo được tổng quát hoá.
- Nếu một generator có quá ít mẫu để chia, ghi rõ vào báo cáo và chấp nhận chia theo ảnh — nhưng **phải** khai báo.
- `random_state = 42`, lưu lại danh sách `sha256` của từng split để tái lập.

### Bước 6 — Kiểm toán cô lập 3 cặp (D2–D3, ~1,5 h)

Chạy `tools/verify_data_isolation.py` cho **3 cặp độc lập**:

```bash
# Cặp 1: v2 vs tập test CFE
python tools/verify_data_isolation.py \
  --manifest DATASET/manifests/data_manifest_v2.json \
  --test_dirs DATASET/test_communityforensics \
  --phash_threshold 8 \
  --output_report logs/phase7/isolation_vs_cfe.json

# Cặp 2: v2 vs 18 tập Task 5.1
python tools/verify_data_isolation.py \
  --manifest DATASET/manifests/data_manifest_v2.json \
  --test_dirs <18 thư mục test> \
  --phash_threshold 8 \
  --output_report logs/phase7/isolation_vs_18sets.json

# Cặp 3: v2 vs chính nó (train ↔ val_calib ↔ val_select)
python tools/verify_data_isolation.py \
  --manifest DATASET/manifests/data_manifest_v2.json \
  --phash_threshold 8 \
  --output_report logs/phase7/isolation_internal.json
```

Gộp 3 báo cáo thành `logs/phase7/data_isolation_report.json` với trường tổng `violations`.

> **Lưu ý kỹ thuật**: script hiện có `--phash_threshold` mặc định `3`. Task này yêu cầu **8** (chặt hơn, bắt được near-duplicate sau crop/nén mạnh). Nếu ngưỡng 8 tạo quá nhiều false positive trên ảnh thật cùng nguồn, ghi rõ lý do và dùng **3 cho nội bộ**, **8 cho đối chiếu test** — nhưng phải khai báo trong báo cáo.

### Bước 7 — Đóng gói & bàn giao (D3, ~1,5 h)

1. Đóng gói `.tar` (không nén, để đọc stream) và tính `sha256` của gói.
2. Đẩy lên Drive theo cấu trúc `fatformer-xla/phase7/train_crossdomain_v2.tar`.
3. Viết `docs/bao_cao/A/bao_cao_task_7.7_kho_du_lieu_v2.md` gồm:
   - bảng `family × split × label`;
   - số generator bị loại và lý do;
   - thời gian stream, dung lượng tải, dung lượng lưu;
   - kết quả 3 cặp audit;
   - **tuyên bố rõ**: "0 vi phạm" hoặc "có vi phạm → đã chặn 7.11".
4. Cập nhật [`../README_A.md`](../README_A.md) và gửi C xác nhận.

---

## 6. Definition of Done

- [ ] `DATASET/manifests/data_manifest_v2.json` tồn tại, **≥ 12.000** bản ghi, đủ 8 trường bắt buộc trên mỗi bản ghi.
- [ ] `logs/phase7/data_isolation_report.json` có **`violations = 0`** trên cả 3 cặp.
- [ ] `logs/phase7/generator_denylist.json` được sinh **tự động**, không gõ tay; có ≥ 1 nguồn dẫn xuất rõ ràng.
- [ ] 5 họ đều có mặt; chênh lệch giữa họ ≤ 1,5×; Real/Fake trong mỗi split nằm trong 45–55%.
- [ ] 3.600 ảnh replay chiếm 20–25% tổng kho và có cờ `is_replay: true`.
- [ ] Báo cáo có bảng phân bố + hình `data_v2_distribution.png` + thời gian/dung lượng thực tế.
- [ ] Gói dữ liệu đã lên Drive, có `sha256` ghi trong báo cáo.
- [ ] Thành viên C đã xác nhận manifest (ghi ngày + người xác nhận trong báo cáo).

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu | Kill criteria |
| :--- | :---: | :---: | :--- | :--- |
| Stream quá chậm / tốn dung lượng | Cao | Trễ D4 | Lọc theo metadata trước khi giải mã; cache theo họ; `--resume` | > 4 h hoặc > 60 GB ⇒ hạ 8.000 ảnh |
| Rò rỉ ảnh test vào train | Trung bình | **Phá giá trị học thuật** | 3 cặp audit + deny-list tự động | 1 vi phạm ⇒ chặn 7.11 |
| Deny-list quá rộng, còn < 10.000 ảnh | Trung bình | Không đủ dữ liệu | Mở rộng sang họ mới, **không** nới deny-list | Còn < 8.000 sau khi mở rộng ⇒ báo C đổi chiến lược |
| pHash ngưỡng 8 gây false positive | Trung bình | Mất ảnh thật | Dùng 3 cho nội bộ, 8 cho đối chiếu test; khai báo | — |
| Nguồn ảnh thật không đa dạng nén | Cao | Real dễ bị học vẹt | Bắt buộc ≥ 3 nguồn, ưu tiên JPEG thật | Chỉ có 1 nguồn ⇒ không đạt DoD |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md)
- Task tiêu thụ dữ liệu: [`../../member_c_training_eval/phase_7/task_7.11_train_anchored_v2.md`](../../member_c_training_eval/phase_7/task_7.11_train_anchored_v2.md)
- Task pipeline suy biến: [`task_7.8_degradation_pipeline_v2.md`](task_7.8_degradation_pipeline_v2.md)
- Chỉ mục giai đoạn: [`README.md`](README.md)
- Bảng điều khiển A: [`../README_A.md`](../README_A.md)
