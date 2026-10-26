# KẾ HOẠCH TRIỂN KHAI KIỂM ĐỊNH NGOẠI VI: COMMUNITYFORENSICS-EVAL (CVPR 2025)

> **Dự án**: Nâng cao độ bền vững pháp y cho FatFormer trước biến dạng nén mạng xã hội (FatFormer-XLA)  
> **Giai đoạn**: Milestone 7 (External Unseen Evaluation Benchmark)  
> **Mục tiêu cốt lõi**: Kiểm thử mù Zero-shot (Unseen Testset) trên các mô hình AI tạo sinh thương mại và mã nguồn mở thế hệ mới nhất (Midjourney v6, DALL-E 3, SDXL, Flux.1, DeepFloyd, StyleGAN3...).  
> **Quy mô tập test**: 2.000 ảnh chuẩn hóa PNG Lossless (1.000 Real : 1.000 Fake), chia đều thành 5 tập kiểm chuẩn con cân bằng 50/50.  
> **Nguyên tắc khoa học**: Zero Data Leakage (Kiểm toán SHA-256 đối chiếu `data_manifest.json`), cách ly 100% khỏi quá trình huấn luyện.

---

## I. MỤC TIÊU VÀ Ý NGHĨA HỌC THUẬT

1. **Xóa tan nghi vấn "Học thuộc phòng lab" (Overfitting to Lab Benchmarks)**:
   - Các tập test trong giai đoạn trước (ProGAN, StyleGAN, LDM, Glide) đều xuất phát từ các mô hình sinh giai đoạn 2019–2023.
   - Việc đưa vào tập dữ liệu độc lập **CommunityForensics-Eval (CVPR 2025)** chứng minh năng lực tổng quát hóa thực sự của FatFormer-XLA trên các mô hình sinh tiên tiến nhất năm 2024–2025 mà mô hình chưa từng được huấn luyện.
2. **Khẳng định tính ưu việt của Kiến trúc Hệ thống Kép (Dual-Operating Architecture)**:
   - Đánh giá song song trên 2 trạng thái: **Ảnh Sạch (Clean)** và **Ảnh Nén Sâu Mạng Xã Hội (JPEG $Q=30$)**.
   - Chứng minh **WiSE-FT** duy trì độ chính xác đỉnh cao và bảo vệ ảnh thật trên ảnh sạch, trong khi **Anchored Recovery** cứu vãn sự cố sụp đổ khi ảnh bị nén sâu qua không gian mạng.

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
*Mục đích: Chứng minh WiSE-FT bảo toàn độ chính xác đỉnh cao và chống báo oan ảnh thật trên các mô hình 2024–2025.*

| Tập Kiểm Chuẩn Con | Họ Kiến Trúc & Mô Hình Đại Diện | Số Mẫu (Real/Fake) | Baseline CVPR 2024 ($T^*=0.50$) | WiSE-FT Mode 1 ($T=1.08, T^*=0.50$) | Anchored Mode 2 ($T=1.28, T^*=0.40$) | Phân Tích Học Thuật & Pháp Y |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1_commercial** | Midjourney v6, DALL-E 3 | 600 (300/300) | *X.XX%* | *Y.YY%* | *Z.ZZ%* | Khả năng tổng quát hóa trên AI thương mại đóng |
| **2_latent_diffusion**| SDXL, Flux.1 | 600 (300/300) | *X.XX%* | *Y.YY%* | *Z.ZZ%* | Nhận diện dấu vết khuếch tán tiềm ẩn thế hệ mới |
| **3_pixel_diffusion** | DeepFloyd, ADM | 400 (200/200) | *X.XX%* | *Y.YY%* | *Z.ZZ%* | Đánh giá trên nhóm khuếch tán không gian điểm ảnh |
| **4_modern_gans** | StyleGAN3, GigaGAN | 200 (100/100) | *X.XX%* | *Y.YY%* | *Z.ZZ%* | Kiểm tra khả năng duy trì tri thức họ GAN hiện đại |
| **5_other_autoregressive**| Other / Autoregressive | 200 (100/100) | *X.XX%* | *Y.YY%* | *Z.ZZ%* | Kiểm thử Zero-shot trên kiến trúc phi truyền thống |
| 🏆 **SUBSET-MACRO MEAN**| **Trung bình không trọng số (5 họ)** | **2.000** | *...%* | **...% 🏆** | *...%* | **WiSE-FT giữ vững vị thế ở Chế độ Tiêu chuẩn** |
| 📊 **OVERALL POOLED MICRO**| **Toàn bộ 2.000 ảnh gộp (Micro ACC)** | **2.000** | *...%* | **...% 🏆** | *...%* | Đánh giá tổng hợp toàn bộ tập mẫu |
| • Clean Real ACC | COCO, FFHQ, RAISE, LAION | 1.000 | *...%* | **...% 🏆** | *...%* | Bảo vệ ảnh thật tự nhiên, chống báo động giả |
| • Clean Fake ACC | Toàn bộ 5 họ AI mới | 1.000 | *...%* | *...%* | **...% 🔥** | Độ nhạy bắt ảnh giả trên miền chưa từng huấn luyện |
| • Clean ROC-AUC / AP | Độc lập ngưỡng cắt $T^*$ | 2.000 | *...%* | **...% 🏆** | *...%* | Năng lực phân tách đặc trưng tổng quát |

---

### Bảng B: Kết Quả Kiểm Chuẩn Dưới Biến Dạng Nén Sâu (JPEG Q=30 CommunityForensics-Eval — 2.000 Ảnh)
*Mục đích: Chứng minh Baseline gốc và WiSE-FT sụp đổ khi ảnh AI mới bị nén qua mạng xã hội, trong khi Anchored Recovery cứu vãn ngoạn mục.*

| Chỉ Số Đo Đạc Dưới Nén Sâu ($Q=30$) | Số Mẫu | Baseline CVPR 2024 | WiSE-FT (Chế Độ 1) | Anchored Recovery (Chế Độ 2) | Kết Luận Đối Chứng |
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
