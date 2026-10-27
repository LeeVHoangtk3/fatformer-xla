# KẾ HOẠCH TRIỂN KHAI KIỂM ĐỊNH NGOẠI VI: COMMUNITYFORENSICS-EVAL (CVPR 2025)

> **Dự án**: Nâng cao độ bền vững pháp y cho FatFormer trước biến dạng nén mạng xã hội (FatFormer-XLA)  
> **Giai đoạn**: Milestone 7 (External Unseen Evaluation Benchmark)  
> **Mục tiêu cốt lõi**: Kiểm thử mù Zero-shot (Unseen Testset) trên các mô hình AI tạo sinh thương mại và mã nguồn mở thế hệ mới nhất (Midjourney v6, DALL-E 3, SDXL, Flux.1, DeepFloyd, StyleGAN3...).  
> **Quy mô tập test**: 2.000 ảnh chuẩn hóa PNG Lossless (1.000 Real : 1.000 Fake), chia đều thành 5 tập kiểm chuẩn con cân bằng 50/50.  
> **Nguyên tắc khoa học**: Zero Data Leakage (Kiểm toán SHA-256 đối chiếu `data_manifest.json`), cách ly 100% khỏi quá trình huấn luyện.

---

## I. MỤC TIÊU VÀ Ý NGHĨA HỌC THUẬT

> 🚨 **CẬP NHẬT 27/10/2026 — ĐỌC TRƯỚC**: Kế hoạch này được viết trước khi phát hiện `fatformer_wise_ft_best.pth` là `invalid_layout`. Vì vậy mọi mục tiêu "chứng minh WiSE-FT ưu việt" và mọi cột WiSE-FT trong các bảng mẫu bên dưới **đã hết hiệu lực**. Kế hoạch hiện hành là [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](ke_hoach_phase_7_dieu_chinh_toan_dien.md); khế ước đo lường bắt buộc ở **Phụ lục A**. Thiết kế đích là **một checkpoint tự định tuyến**, không phải hệ thống kép.

1. **Xóa tan nghi vấn "Học thuộc phòng lab" (Overfitting to Lab Benchmarks)**:
   - Các tập test trong giai đoạn trước (ProGAN, StyleGAN, LDM, Glide) đều xuất phát từ các mô hình sinh giai đoạn 2019–2023.
   - Việc đưa vào tập dữ liệu độc lập **CommunityForensics-Eval (CVPR 2025)** chứng minh năng lực tổng quát hóa thực sự của FatFormer-XLA trên các mô hình sinh tiên tiến nhất năm 2024–2025 mà mô hình chưa từng được huấn luyện.
2. ~~**Khẳng định tính ưu việt của Kiến trúc Hệ thống Kép (Dual-Operating Architecture)**~~
   - ~~Chứng minh **WiSE-FT** duy trì độ chính xác đỉnh cao ... **Anchored Recovery** cứu vãn ...~~
   - ✅ **Mục tiêu thay thế**: đo trung thực **Baseline** vs **Anchored** vs **WiSE-FT Gen 2** (Task 7.5) trên cùng harness v3, rồi kiểm chứng giả thuyết "một checkpoint tự định tuyến" ở **G7.3**.
   - ✅ Vẫn giữ đánh giá song song 2 trạng thái: **Clean** và **JPEG Q=30 in-memory**.

---

## II. MA TRẬN PHÂN TẦNG 5 TẬP KIỂM CHUẨN CON (STRATIFIED BENCHMARK MATRIX)

Để đo đạc chính xác ROC-AUC, AP, Real ACC và Fake ACC cho từng công nghệ mà không bị thiên lệch phân phối, dữ liệu được tổ chức thành 5 tập con cân bằng nhị phân 50/50:

```text
test_communityforensics/
├── 1_commercial/            # 600 ảnh (300 Real / 300 Fake): Midjourney v6, DALL-E 3
│   ├── 0_real/              # 150 COCO + 75 RAISE + 75 LAION
│   └── 1_fake/              # 150 Midjourney v6 + 150 DALL-E 3
├── 2_latent_diffusion/      # 600 ảnh (300 Real / 300 Fake): SDXL, Flux.1
│   ├── 0_real/              # 150 COCO + 75 FFHQ + 75 LAION
│   └── 1_fake/              # 150 SDXL + 150 Flux.1
├── 3_pixel_diffusion/       # 400 ảnh (200 Real / 200 Fake): DeepFloyd-IF, Guided-Diff (ADM)
│   ├── 0_real/              # 100 COCO + 50 RAISE + 50 FFHQ
│   └── 1_fake/              # 100 DeepFloyd-IF + 100 ADM
├── 4_modern_gans/           # 200 ảnh (100 Real / 100 Fake): StyleGAN3, GigaGAN
│   ├── 0_real/              # 50 FFHQ + 50 COCO
│   └── 1_fake/              # 50 StyleGAN3 + 50 GigaGAN
└── 5_other_autoregressive/  # 200 ảnh (100 Real / 100 Fake): Muse, LlamaGen/Parti
    ├── 0_real/              # 50 RAISE + 50 LAION
    └── 1_fake/              # 50 Muse + 50 LlamaGen
```

---

## III. CODE TRÍCH XUẤT STREAMING PARQUET (CHUẨN HÓA SCHEMA HF & ZERO LEAKAGE)

Đoạn mã chạy trực tiếp trên Colab/Kaggle, nạp streaming từ `OwensLab/CommunityForensics-Eval` (`split="CompEval"`), đọc dữ liệu byte thô `image_data`, lọc chính xác theo `architecture` và `real_source`, kiểm toán cách ly mã băm SHA-256 từ `data_manifest.json` và đóng gói thành `test_communityforensics.tar`:

```python
import os, io, json, hashlib, tarfile
from collections import defaultdict
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

TARGET_DIR = "/content/dataset_local/test_communityforensics"
DRIVE_TAR_PATH = "/content/drive/MyDrive/Fatformer/datasets/test_communityforensics.tar"
EXISTING_MANIFEST = "/content/drive/MyDrive/Fatformer/manifests/data_manifest.json"
SEED = 42

os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DRIVE_TAR_PATH), exist_ok=True)

# 1. Nạp danh sách SHA-256 cũ để đảm bảo Zero Data Leakage
existing_hashes = set()
if os.path.exists(EXISTING_MANIFEST):
    with open(EXISTING_MANIFEST, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        records = manifest_data.get("records", manifest_data) if isinstance(manifest_data, dict) else manifest_data
        for entry in records:
            if "sha256" in entry:
                existing_hashes.add(entry["sha256"])
    print(f"[*] Đã nạp {len(existing_hashes)} mã băm SHA-256 để kiểm toán cách ly.")

# 2. Định nghĩa chỉ tiêu phân tầng 5 tập con
QUOTAS = {
    "1_commercial": {
        "arch_key": "Commercial",
        "fake_max": 300,
        "real_quotas": {"coco": 150, "raise": 75, "laion": 75}
    },
    "2_latent_diffusion": {
        "arch_key": "LatDiff",
        "fake_max": 300,
        "real_quotas": {"coco": 150, "ffhq": 75, "laion": 75}
    },
    "3_pixel_diffusion": {
        "arch_key": "PixDiff",
        "fake_max": 200,
        "real_quotas": {"coco": 100, "raise": 50, "ffhq": 50}
    },
    "4_modern_gans": {
        "arch_key": "GAN",
        "fake_max": 100,
        "real_quotas": {"ffhq": 50, "coco": 50}
    },
    "5_other_autoregressive": {
        "arch_key": "other",
        "fake_max": 100,
        "real_quotas": {"raise": 50, "laion": 50}
    },
}

for group in QUOTAS:
    os.makedirs(f"{TARGET_DIR}/{group}/0_real", exist_ok=True)
    os.makedirs(f"{TARGET_DIR}/{group}/1_fake", exist_ok=True)

counts = {
    g: {
        "fake": 0,
        "real": {src: 0 for src in cfg["real_quotas"]},
        "models_collected": defaultdict(int)
    }
    for g, cfg in QUOTAS.items()
}

MAX_PER_SPECIFIC_MODEL = 60

print("[*] Đang kết nối Streaming Parquet từ OwensLab/CommunityForensics-Eval (split='CompEval')...")
ds = load_dataset("OwensLab/CommunityForensics-Eval", split="CompEval", streaming=True)
ds = ds.shuffle(seed=SEED, buffer_size=5000)

manifest_records = []
total_collected = 0
pbar = tqdm(total=2000, desc="Đang trích xuất phân tầng 2.000 ảnh")

for item in ds:
    raw_bytes = item.get("image_data")
    if not raw_bytes:
        continue

    img_hash = hashlib.sha256(raw_bytes).hexdigest()
    if img_hash in existing_hashes:
        continue

    label = int(item.get("label", -1))
    arch = str(item.get("architecture", "")).strip()
    model_name = str(item.get("model_name", "unknown")).strip()
    real_src = str(item.get("real_source", "")).lower().strip()

    assigned_group = None
    matched_r_src = None

    if label == 1:
        for g, cfg in QUOTAS.items():
            if arch.lower() == cfg["arch_key"].lower():
                if counts[g]["fake"] < cfg["fake_max"]:
                    if counts[g]["models_collected"][model_name] < MAX_PER_SPECIFIC_MODEL:
                        assigned_group = g
                        break
    elif label == 0:
        for g, cfg in QUOTAS.items():
            for src_key, src_max in cfg["real_quotas"].items():
                if src_key in real_src and counts[g]["real"][src_key] < src_max:
                    assigned_group = g
                    matched_r_src = src_key
                    break
            if assigned_group:
                break

    if assigned_group:
        try:
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            continue

        if label == 1:
            idx = counts[assigned_group]["fake"]
            safe_mname = model_name.replace("/", "_").replace(" ", "_")[:30]
            fname = f"fake_{idx:04d}_{safe_mname}.png"
            rel_path = f"{assigned_group}/1_fake/{fname}"
            counts[assigned_group]["fake"] += 1
            counts[assigned_group]["models_collected"][model_name] += 1
        else:
            idx = sum(counts[assigned_group]["real"].values())
            fname = f"real_{idx:04d}_{matched_r_src}.png"
            rel_path = f"{assigned_group}/0_real/{fname}"
            counts[assigned_group]["real"][matched_r_src] += 1

        save_path = os.path.join(TARGET_DIR, rel_path)
        img.save(save_path, format="PNG")
        existing_hashes.add(img_hash)

        manifest_records.append({
            "rel_path": rel_path,
            "group": assigned_group,
            "label": label,
            "architecture": arch,
            "model_name": model_name,
            "real_source": real_src,
            "sha256": img_hash
        })

        total_collected += 1
        pbar.update(1)

        if total_collected >= 2000:
            break

pbar.close()

manifest_out = os.path.join(TARGET_DIR, "cfe_manifest.json")
with open(manifest_out, "w", encoding="utf-8") as f:
    json.dump(manifest_records, f, indent=2, ensure_ascii=False)

print(f"\n[✓] Đã thu thập đủ {total_collected}/2.000 ảnh! Đang đóng gói .tar...")
with tarfile.open(DRIVE_TAR_PATH, "w", format=tarfile.PAX_FORMAT) as tar:
    tar.add(TARGET_DIR, arcname="test_communityforensics")

print(f"[✓] Đã lưu thành công gói kiểm chuẩn tại: {DRIVE_TAR_PATH}")
```

---

## IV. KHUNG KẾT QUẢ ĐỐI CHỨNG HỌC THUẬT (ĐƯA VÀO CHƯƠNG 5 BÁO CÁO ĐỒ ÁN)

### Bảng A: Kết Quả Kiểm Chuẩn Trên Miền Ảnh Sạch (Clean CommunityForensics-Eval — 2.000 Ảnh)
*~~Mục đích: Chứng minh WiSE-FT bảo toàn độ chính xác đỉnh cao và chống báo oan ảnh thật trên các mô hình 2024–2025.~~*
*⚠️ Mục đích đã sửa: đo trung thực Baseline vs Anchored vs **Gen 2**, báo cáo theo Phụ lục A (AUC/AP/TPR@FPR + CI). Cột "WiSE-FT Mode 1" chỉ còn là chỗ giữ cho Gen 2.*

| Tập Kiểm Chuẩn Con | Họ Kiến Trúc & Mô Hình Đại Diện | Số Mẫu (Real/Fake) | Baseline CVPR 2024 ($T^*=0.50$) | ~~WiSE-FT Mode 1~~ → **Gen 2** (Task 7.5) | Anchored Mode 2 ($T=1.28, T^*=0.40$) | Phân Tích Học Thuật & Pháp Y |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1_commercial** | Midjourney v6, DALL-E 3 | 600 (300/300) | *X.XX%* | *Y.YY%* | *Z.ZZ%* | Khả năng tổng quát hóa trên AI thương mại đóng |
| **2_latent_diffusion**| SDXL, Flux.1 | 600 (300/300) | *X.XX%* | *Y.YY%* | *Z.ZZ%* | Nhận diện dấu vết khuếch tán tiềm ẩn thế hệ mới |
| **3_pixel_diffusion** | DeepFloyd, ADM | 400 (200/200) | *X.XX%* | *Y.YY%* | *Z.ZZ%* | Đánh giá trên nhóm khuếch tán không gian điểm ảnh |
| **4_modern_gans** | StyleGAN3, GigaGAN | 200 (100/100) | *X.XX%* | *Y.YY%* | *Z.ZZ%* | Kiểm tra khả năng duy trì tri thức họ GAN hiện đại |
| **5_other_autoregressive**| Other / Autoregressive | 200 (100/100) | *X.XX%* | *Y.YY%* | *Z.ZZ%* | Kiểm thử Zero-shot trên kiến trúc phi truyền thống |
| 🏆 **SUBSET-MACRO MEAN**| **Trung bình không trọng số (5 họ)** | **2.000** | *...%* | *...%* | *...%* | *Bỏ kỳ vọng "WiSE-FT giữ vững"; chờ số Gen 2* |
| 📊 **OVERALL POOLED MICRO**| **Toàn bộ 2.000 ảnh gộp (Micro ACC)** | **2.000** | *...%* | **...% 🏆** | *...%* | Đánh giá tổng hợp toàn bộ tập mẫu |
| • Clean Real ACC | COCO, FFHQ, RAISE, LAION | 1.000 | *...%* | **...% 🏆** | *...%* | Bảo vệ ảnh thật tự nhiên, chống báo động giả |
| • Clean Fake ACC | Toàn bộ 5 họ AI mới | 1.000 | *...%* | *...%* | **...% 🔥** | Độ nhạy bắt ảnh giả trên miền chưa từng huấn luyện |
| • Clean ROC-AUC / AP | Độc lập ngưỡng cắt $T^*$ | 2.000 | *...%* | **...% 🏆** | *...%* | Năng lực phân tách đặc trưng tổng quát |

---

### Bảng B: Kết Quả Kiểm Chuẩn Dưới Biến Dạng Nén Sâu (JPEG Q=30 CommunityForensics-Eval — 2.000 Ảnh)
*~~Mục đích: Chứng minh Baseline gốc và WiSE-FT sụp đổ khi ảnh AI mới bị nén qua mạng xã hội, trong khi Anchored Recovery cứu vãn ngoạn mục.~~*
*⚠️ Mục đích đã sửa: kiểm tra liệu có mô hình nào nâng được **AUC** dưới Q=30 (không chỉ hạ ngưỡng). AUC Q30 của Anchored hiện là 50,07% ⇒ chưa đạt.*

| Chỉ Số Đo Đạc Dưới Nén Sâu ($Q=30$) | Số Mẫu | Baseline CVPR 2024 | ~~WiSE-FT (Chế Độ 1)~~ → **Gen 2** | Anchored Recovery (Chế Độ 2) | Kết Luận Đối Chứng |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1_commercial Fake ACC ($Q=30$)** | 300 Fake | ~X.X% (Liệt) | ~Y.Y% | **Z.ZZ% 🔥** | Bắt được ảnh thương mại dù bị lưới DCT $8\times8$ phá hủy |
| **2_latent_diffusion Fake ACC ($Q=30$)** | 300 Fake | ~X.X% (Liệt) | ~Y.Y% | **Z.ZZ% 🔥** | Khối SRM 3-Kernels và Gating $\lambda(x)$ phát huy tác dụng |
| 🏆 **SUBSET-MACRO FAKE ACC** | **1.000 Fake** | **~5-7% (Sụp đổ)** | **~6-10%** | **> 35-45% 🏆** | **Khẳng định giá trị bắt buộc của Chế độ Forensic** |
| 🔥 **OVERALL POOLED FAKE ACC** | **1.000 Fake** | **~5-7%** | **~6-10%** | **> 35-45% 🏆** | Toàn bộ 1.000 ảnh Fake gộp (Micro) |
| • Overall Q=30 Macro ACC | 2.000 | ~50% | ~52% | *...%* | Duy trì ranh giới phân loại dưới suy biến khắc nghiệt |
| • Overall Q=30 Real ACC | 1.000 | *...%* | *...%* | *...%* | Bảo vệ ảnh thật dưới nén sâu |
| • Overall Q=30 ROC-AUC / AP | 2.000 | *...%* | *...%* | **...% 🏆** | Bảo toàn thứ tự xếp hạng xác suất khi bị nén sâu |

---

## V. CÔNG CỤ THỰC THI KIỂM ĐỊNH TỰ ĐỘNG: `tools/eval_communityforensics.py`

Lệnh thực thi đánh giá một bước trên GPU:
```bash
python tools/eval_communityforensics.py \
    --data_dir /content/dataset_local/test_communityforensics \
    --clip_path pretrained/ViT-L-14.pt \
    --output_dir /content/drive/MyDrive/Fatformer/logs/cfe_benchmark
```
Script sẽ tự động:
1. Đọc và phân loại 5 tập con.
2. Suy luận trên 3 checkpoint: Baseline gốc, WiSE-FT, Anchored Recovery.
3. Chạy song song 2 trạng thái: Clean và in-memory JPEG $Q=30$.
4. Tự động xuất tệp `communityforensics_benchmark_report.md` và `communityforensics_benchmark_report.json` với đúng định dạng Bảng A & B.

> ⚠️ **Cảnh báo (27/10/2026)**: Mô tả ở mục V phản ánh **bản script cũ**. Bản cập nhật hiện tại **vẫn còn** các khuyết tật E1–E6 (xem Phụ lục A.6 và [báo cáo đính chính](../bao_cao/overall/bao_cao_kiem_dinh_communityforensics_eval.md#ix-đính-chính-kỹ-thuật-ngày-27102026)). Không dùng kết quả của script cho tới khi **G7.0** hoàn tất.

---

## PHỤ LỤC A — KHẾ ƯỚC ĐO LƯỜNG v3 (BẮT BUỘC TỪ 27/10/2026)

Phụ lục này thay thế mọi quy ước báo cáo trước đây cho các bài kiểm chuẩn OOD (CommunityForensics-Eval và các tập mở rộng). Mọi báo cáo không tuân thủ sẽ bị **trả lại, không nghiệm thu**.

### A.1. Chỉ số bắt buộc

| Nhóm | Chỉ số | Ghi chú |
| :--- | :--- | :--- |
| **Chính** | `ROC-AUC`, `AP` | Tính trên xác suất **đã hiệu chuẩn**; AP phải xử lý edge-case chia 0 |
| **Chính** | `TPR@FPR = 1%`, `5%`, `10%` | Đây là chỉ số thay thế "Fake ACC" khi so sánh mô hình |
| **Phụ** | `Real ACC`, `Fake ACC` tại `T*` cố định | Chỉ báo cáo **kèm** giá trị `T*` và nguồn fit |
| **Hiệu chuẩn** | `ECE` (10 bin), `NLL` | Trước và sau hiệu chuẩn |
| **Độ bất định** | Bootstrap CI 95% cho AUC, AP, TPR | 1.000 lần resample, seed cố định |
| **Phân tầng** | 5 subset × 4 nhóm generator × 2 điều kiện | Không gộp mất chi tiết |

### A.2. Tập hiệu chuẩn phải tách rời

- `T`, `T*`, và tham số router **chỉ được fit** trên một trong hai nguồn:
  1. `val_calibration` in-dist (ưu tiên), **hoặc**
  2. 400 ảnh CFE tách khỏi 2.000 ảnh báo cáo; nếu không đủ thì **cross-validation 5-fold** và báo cáo cả 5 fold.
- **Nghiêm cấm**: fit và báo cáo trên cùng một tập; dùng `mean` của tập đang chấm để suy ra ngưỡng; chọn ngưỡng sau khi đã xem kết quả.
- Mọi JSON phải ghi `calibration_source`, `n_calib`, và `fold_id` (nếu dùng CV).

### A.3. Điều kiện vận hành phải công bố

Mỗi con số phải kèm **ngữ cảnh triển khai**:
- `FPR mục tiêu` (1/5/10%) và `T*` tương ứng.
- Điều kiện ảnh: `clean` hoặc `Q30` (và nếu khác, ghi rõ tham số).
- `degrade_order`: `post_resize` (mặc định, khớp dữ liệu 224×224 hiện tại) hoặc `pre_resize` (chỉ dùng cho ảnh độ phân giải gốc).

### A.4. Cấm báo cáo "overall accuracy" đơn lẻ

- Không được dùng "Overall ACC" làm chỉ số chính, không được so sánh hai mô hình chỉ bằng Overall ACC.
- Lý do: Overall ACC phụ thuộc tỷ lệ real/fake của tập và che giấu hiện tượng "báo oan ảnh thật để bắt thêm ảnh giả".
- Nếu buộc phải nêu (ví dụ để đối chiếu báo cáo cũ), phải đặt trong ngoặc kèm nhãn *(chỉ để tham chiếu lịch sử)* và **kèm ngay** `Real ACC` + `Fake ACC` + `T*`.

### A.5. Trạng thái nạp checkpoint phải công bố

Mỗi báo cáo phải có bảng:

| Checkpoint | `n_keys` | `adapter_blocks` | `n` dùng | `missing` | `unexpected` | `status` |
| :--- | ---: | :--- | :---: | ---: | ---: | :--- |
| *(tên file)* | | | | **0** | **0** | `ok` / `invalid_layout` / `partial` |

- `missing > 0` hoặc `unexpected > 0` ⇒ **status ≠ ok** ⇒ kết quả không được dùng làm kết quả chính.
- Không chấp nhận `strict=False` trong bất kỳ đường nạp nào thuộc pipeline báo cáo.
- Mọi JSON phải có `git_commit`, `sha256` của từng checkpoint, `report_schema: "v3"`.

### A.6. Sáu khuyết tật phải vá trước khi báo cáo có hiệu lực

| Mã | Vị trí | Mô tả | Task |
| :--- | :--- | :--- | :---: |
| E1 | `eval_communityforensics.py` | `strict=False` + lọc theo shape | 7.2–7.3 |
| E2 | `eval_communityforensics.py` | fallback `len(indices) > 8 → 3` | 7.2 |
| E3 | `eval_communityforensics.py` | ngưỡng thích ứng theo `mean_w` của tập đang chấm | 7.4 |
| E4 | `eval_communityforensics.py` | hệ số router đặt tay | 7.13 |
| E5 | `eval_communityforensics.py` | nhiệt độ hard-code, thiếu `TPR@FPR 1/10`, thiếu CI | 7.3 |
| E6 | `wise_ft_merge.py` | calibration chết, không `--verify`, buffer detection chuỗi con, không quét `β_srm` | 7.5 |

### A.7. Mẫu JSON tối thiểu

```json
{
  "report_schema": "v3",
  "git_commit": "<sha>",
  "timestamp_utc": "2026-10-27T00:00:00Z",
  "dataset": {"name": "communityforensics_eval", "n": 2000, "degrade": "clean"},
  "checkpoints": [
    {"path": "...", "sha256": "...", "n_keys": 1200, "adapter_blocks": [7,15,23],
     "n_used": 3, "missing": 0, "unexpected": 0, "status": "ok"}
  ],
  "calibration": {"source": "val_calibration", "n": 540, "T": 1.2763, "T_star": 0.40, "fpr_target": 0.05},
  "metrics": {
    "auc": 0.0, "ap": 0.0,
    "tpr_at_fpr": {"0.01": 0.0, "0.05": 0.0, "0.10": 0.0},
    "real_acc": 0.0, "fake_acc": 0.0, "ece": 0.0,
    "ci95": {"auc": [0.0, 0.0], "tpr_at_fpr_0.05": [0.0, 0.0]}
  },
  "per_subset": [{"subset": "1_commercial", "n_real": 300, "n_fake": 300, "auc": 0.0, "tpr_at_fpr_0.05": 0.0}]
}
```

### A.8. Quy tắc trích dẫn trong luận văn / slide

1. Luôn ghi **điều kiện**: `CFE clean, TPR@FPR=5%, AUC` — không ghi "độ chính xác 90%".
2. Không trích dẫn số của checkpoint có `status: invalid_layout` (hiện tại: `fatformer_wise_ft_best.pth`).
3. Khi so sánh với bài báo gốc CVPR 2024, phải nêu rõ khác biệt về metric (paper dùng ACC; ta dùng AUC + TPR@FPR).
4. Mọi bảng phải có cột `CI 95%`; nếu chênh lệch giữa hai mô hình nhỏ hơn nửa độ rộng CI ⇒ **không được kết luận mô hình này tốt hơn**.

