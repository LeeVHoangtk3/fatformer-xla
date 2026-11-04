# SƠ ĐỒ HẠ TẦNG & DANH MỤC TỆP TIN GOOGLE DRIVE 5TB
## BẢN ĐỒ PHÂN BỔ DỮ LIỆU, CHECKPOINTS & KHO HIỆN VẬT THỰC NGHIỆM DỰ ÁN FATFORMER-XLA

> **Tài khoản hạ tầng**: Google AI Pro 5TB Storage  
> **Thư mục gốc trên Google Drive**: `Drive của tôi / Fatformer`  
> **Đường dẫn chuẩn hóa trên Colab/Kaggle**: `/content/drive/MyDrive/Fatformer` (Fallback: `/content/drive/MyDrive/FatFormer_Hub`)  
> **Thời điểm kiểm kê thực tế**: 04/11/2026 (Quét trực tiếp qua Colab Python Scanner)  
> **Phiên bản tài liệu**: `v2.0` (Cập nhật hoàn tất kho dữ liệu v2 Task 7.7)  

---

## I. NGUYÊN TẮC HẠ TẦNG CỐT LÕI (LEAD A PROTOCOL)

1. **Nguyên Tắc I/O Chống Nghẽn SSD Colab (Bất Di Bất Dịch)**:
   - Tuyệt đối **không** để DataLoader của PyTorch đọc trực tiếp từng tệp ảnh nhỏ từ Google Drive qua giao thức FUSE (gây nghẽn I/O, tràn RAM, treo phiên Colab).
   - Luôn đóng gói dữ liệu thành các tệp lưu trữ `.tar` không nén trên Drive. Khi huấn luyện hoặc kiểm thử, copy tệp `.tar` về ổ cứng SSD cục bộ (`/content/` hoặc `/content/fatformer-xla/`) và giải nén sang SSD NVMe trước khi nạp batch.
2. **Nguyên Tắc Phân Định Thư Mục (Directory Boundary)**:
   - `Fatformer/datasets/`: Chuyên trách lưu trữ các tập kiểm chuẩn (Test sets) bị khóa vĩnh viễn và các tập dữ liệu thô/staging lịch sử. Tuyệt đối không đưa dữ liệu train mới vào đây để tránh nhầm lẫn học thuật.
   - `Fatformer/phase7/`: Không gian bàn giao chuyên biệt cho Phase 7 giữa Thành viên A (Data Infra) và Thành viên C (Training Lead).
   - `Fatformer/checkpoint/`: Kho lưu trữ checkpoints đã finetune qua các giai đoạn (Milestone 1–6). *(Lưu ý: tên thư mục là `checkpoint/` số ít)*.
   - `Fatformer/pretrained/`: Chứa các trọng số mốc sàn gốc (Teacher Baseline) và trọng số backbone của OpenAI CLIP.
3. **Quy Tắc Đọc FUSE An Toàn**:
   - Khi quét thư mục Drive qua Python, không dùng `os.walk()` không giới hạn vì sẽ quét đệ quy vô tận vào các thư mục sâu, gây `Input/output error`. Dùng `Path.iterdir()` với độ sâu tối đa 2 tầng và bọc khối `try-except` quanh hàm đọc `stat().st_size`.

---

## II. TOÀN CẢNH CÂY THƯ MỤC THỰC TẾ (LIVE DRIVE TREE)

Dưới đây là kết quả kiểm kê thực chứng trực tiếp từ Google Drive của dự án:

```text
/content/drive/MyDrive/Fatformer/
│
├── 📁 phase7/                                         <-- [KHÔNG GIAN BÀN GIAO CHÍNH PHASE 7 - TASK 7.7]
│   ├── 📄 train_crossdomain_v2.tar (7.777,69 MB)      <-- Gói dữ liệu 14.400 ảnh đa bộ sinh v2 (7,59 GB)
│   ├── 📄 train_crossdomain_v2.tar.sha256 (0,00 MB)   <-- Chữ ký băm SHA-256 xác thực toàn vẹn gói
│   ├── 📄 data_isolation_report.json (0,01 MB)        <-- Báo cáo kiểm toán chốt chặn G7.2 (violations=0)
│   ├── 📄 data_manifest_v2.json (7,78 MB)             <-- Kê khai toàn bộ 14.400 mẫu
│   ├── 📄 data_manifest_v2_train.json (6,61 MB)       <-- Kê khai 12.240 mẫu train (85%)
│   ├── 📄 data_manifest_v2_train_select.json (7,18 MB)<-- Kê khai 13.320 mẫu train + val_select (cho Task 7.11)
│   ├── 📄 data_manifest_v2_val_calib.json (0,60 MB)   <-- Kê khai 1.080 mẫu hiệu chuẩn (cho Task 7.4, 7.13)
│   └── 📄 data_manifest_v2_val_select.json (0,60 MB)  <-- Kê khai 1.080 mẫu chọn checkpoint (cho Task 7.10)
│
├── 📁 pretrained/                                     <-- [TRỌNG SỐ GỐC MỎ NEO & CLIP BACKBONE]
│   ├── 📄 fatformer_4class_ckpt.pth (1.879,74 MB)     <-- Baseline Teacher gốc (warm-start cho Task 7.11)
│   └── 📄 ViT-L-14.pt (889,56 MB)                     <-- Trọng số gốc OpenAI CLIP ViT-L/14
│
├── 📁 checkpoint/                                     <-- [KHO CHECKPOINT HUẤN LUYỆN LỊCH SỬ]
│   ├── 📄 fatformer_plan_a_best_weights.pth (2.207,50 MB) <-- Checkpoint Plan A tốt nhất (đã đo Task 5.1)
│   ├── 📄 fatformer_crossdomain_weights_only.pth (2.207,51 MB)
│   ├── 📄 fatformer_anchored_recovery_best.pth (1.881,42 MB)
│   ├── 📄 fatformer_anchored_recovery_last.pth (1.881,42 MB)
│   ├── 📄 fatformer_anchored_recovery_best_metadata.json (0,00 MB)
│   ├── 📄 fatformer_srm_robust_final.pth (3.354,12 MB)
│   ├── 📄 fatformer_plan_a_final.pth (3.354,11 MB)
│   ├── 📄 fatformer_wise_ft_best.pth (4.473,30 MB)    <-- Mô hình kết hợp không gian trọng số WiSE-FT
│   ├── 📄 calibration_and_selection_report.json (0,00 MB)
│   ├── 📁 checkpoints/
│   ├── 📁 m3/                                         <-- Checkpoints giai đoạn Milestone 3
│   ├── 📁 m4/                                         <-- Checkpoints giai đoạn Milestone 4
│   └── 📁 m4_recovery/                                <-- Checkpoints giai đoạn Milestone 4 Recovery
│
├── 📁 datasets/                                       <-- [KHO TẬP KIỂM CHUẨN TEST BẤT BIẾN & REPLAY]
│   ├── 📄 test_communityforensics.tar (4.977,14 MB)   <-- 2.000 ảnh CFE Test (16 generators)
│   ├── 📄 test_benchmark_gans.tar (19.191,07 MB)      <-- 8 tập test GAN Task 5.1 (62.003 ảnh)
│   ├── 📄 test_benchmark_diffusion.tar (1.244,50 MB)  <-- 10 tập test Diffusion Task 5.1 (20.000 ảnh)
│   ├── 📄 test_degraded.tar (20.018,01 MB)            <-- 18 tập test suy biến vật lý (54.000 lượt đo)
│   ├── 📄 diffusion_staging.tar (1.846,66 MB)         <-- Kho Staging cũ 3.600 ảnh (dùng trích Replay)
│   ├── 📄 progan_train.tar (14.230,22 MB)             <-- Dữ liệu huấn luyện ProGAN 4-class gốc
│   └── 📄 progan_val.tar (798,00 MB)                  <-- Dữ liệu kiểm định ProGAN gốc
│
├── 📁 log/                                            <-- [BÁO CÁO & BẢNG KẾT QUẢ CSV/MD/PNG]
│   ├── 📄 final_full_benchmark_matrix.csv (0,01 MB)   <-- Ma trận đo đạc 18 tập test Task 5.1
│   ├── 📄 task_5.1_master_benchmark_chart.png (0,19 MB)<-- Biểu đồ tổng thể benchmark Task 5.1
│   ├── 📄 task_5.1_clean_report.md & .csv (0,01 MB)   <-- Báo cáo Clean 18 tập test
│   ├── 📄 task_5.1_degraded_report.md & .csv (0,01 MB)<-- Báo cáo 6 biến thể suy biến
│   ├── 📄 baseline_clean_results.csv & .md (0,00 MB)  <-- Kết quả đo mốc sàn Clean
│   ├── 📄 baseline_degraded_results.csv & .md (0,01 MB)
│   ├── 📄 baseline_degraded_chart.png (0,20 MB)
│   ├── 📄 fatformer_q30_comparison.png (0,10 MB)      <-- Biểu đồ đối sánh suy biến JPEG Q30
│   ├── 📄 fatformer_xla_eval_report.md & .csv (0,00 MB)
│   └── 📄 fast_eval_clean_baseline.csv (0,00 MB)
│
├── 📁 logs/                                           <-- [NHẬT KÝ ĐÀO TẠO & KIỂM ĐỊNH THEO GIAI ĐOẠN]
│   ├── 📄 README.md (0,01 MB)
│   ├── 📄 m6_training_history.csv (0,00 MB)           <-- Nhật ký huấn luyện Phase 6
│   ├── 📄 train_val_history.csv (0,00 MB)
│   ├── 📄 val_crossdomain_baseline_metrics.json (0,00 MB)
│   ├── 📁 cfe_benchmark/
│   ├── 📁 inspections/
│   ├── 📁 m1_baseline_freeze/
│   ├── 📁 m3_training_hardening/
│   ├── 📁 m4_full_benchmark/
│   ├── 📁 task_6.6/
│   └── 📁 task_6.6_r/
│
├── 📁 kaggle/                                         <-- [CẤU HÌNH API KAGGLE NỀN TẢNG]
│   └── 📄 kaggle.json (0,00 MB)                       <-- Khóa API đẩy dataset/checkpoints lên Kaggle
│
├── 📄 phase7_measured_artifacts.zip (1.437,34 MB)     <-- Gói nén lưu trữ artifacts đo đạc Phase 7
└── 📄 ke_hoach_phase_7_dieu_chinh_toan_dien.md        <-- Kế hoạch master điều chỉnh Phase 7
```

---

## III. BẢNG TRA CỨU TỆP TIN THEO MỤC ĐÍCH SỬ DỤNG (QUICK LOOKUP MATRIX)

### 1. Phục Vụ Task 7.11 — Huấn Luyện Anchored Recovery v2 (Thành Viên C)

| Mục đích kỹ thuật | Tên tệp trên Google Drive | Đường dẫn tuyệt đối trên Colab | Dung lượng |
| :--- | :--- | :--- | ---: |
| **Gói dữ liệu v2 ($14.400$ ảnh)** | `train_crossdomain_v2.tar` | `/content/drive/MyDrive/Fatformer/phase7/train_crossdomain_v2.tar` | $7.777,69\text{ MB}$ |
| **Kê khai huấn luyện chính thức** | `data_manifest_v2_train_select.json` | `/content/drive/MyDrive/Fatformer/phase7/data_manifest_v2_train_select.json` | $7,18\text{ MB}$ |
| **Kê khai tập hiệu chuẩn độc lập** | `data_manifest_v2_val_calib.json` | `/content/drive/MyDrive/Fatformer/phase7/data_manifest_v2_val_calib.json` | $0,60\text{ MB}$ |
| **Trọng số Teacher mỏ neo (Baseline)** | `fatformer_4class_ckpt.pth` | `/content/drive/MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth` | $1.879,74\text{ MB}$ |
| **Trọng số CLIP Backbone gốc** | `ViT-L-14.pt` | `/content/drive/MyDrive/Fatformer/pretrained/ViT-L-14.pt` | $889,56\text{ MB}$ |

### 2. Phục Vụ Task 7.14 — Locked Benchmark Toàn Diện Phase 7

| Mục đích kiểm thử | Tên tệp trên Google Drive | Đường dẫn tuyệt đối trên Colab | Dung lượng |
| :--- | :--- | :--- | ---: |
| **Tập test CFE ($2.000$ ảnh)** | `test_communityforensics.tar` | `/content/drive/MyDrive/Fatformer/datasets/test_communityforensics.tar` | $4.977,14\text{ MB}$ |
| **Tập test GAN Task 5.1 ($8$ tập)** | `test_benchmark_gans.tar` | `/content/drive/MyDrive/Fatformer/datasets/test_benchmark_gans.tar` | $19.191,07\text{ MB}$ |
| **Tập test Diffusion ($10$ tập)** | `test_benchmark_diffusion.tar` | `/content/drive/MyDrive/Fatformer/datasets/test_benchmark_diffusion.tar` | $1.244,50\text{ MB}$ |
| **Tập test Suy biến vật lý ($6$ biến thể)** | `test_degraded.tar` | `/content/drive/MyDrive/Fatformer/datasets/test_degraded.tar` | $20.018,01\text{ MB}$ |
| **Checkpoint đối chứng tốt nhất** | `fatformer_plan_a_best_weights.pth` | `/content/drive/MyDrive/Fatformer/checkpoint/fatformer_plan_a_best_weights.pth` | $2.207,50\text{ MB}$ |

---

## IV. ĐOẠN MÃ MẪU KẾT NỐI & KIỂM TRA DRIVE AN TOÀN TRÊN COLAB

Để các script trong tương lai hoạt động ổn định và nhất quán giữa các thành viên A, B, C, khuyến nghị sử dụng chuẩn kết nối sau:

```python
from google.colab import drive
from pathlib import Path

# 1. Mount Drive nếu chưa tồn tại
if not Path('/content/drive/MyDrive').exists():
    drive.mount('/content/drive')

# 2. Định danh Drive Root có cơ chế Fallback
DRIVE_ROOT = Path('/content/drive/MyDrive/Fatformer')
if not DRIVE_ROOT.exists():
    for alt in ['/content/drive/MyDrive/FatFormer_Hub', '/content/drive/MyDrive/fatformer']:
        if Path(alt).exists():
            DRIVE_ROOT = Path(alt)
            break

print(f"[✓] Google Drive đã kết nối thành công: {DRIVE_ROOT}")

# 3. Trích xuất đường dẫn các phân vùng cốt lõi
DIR_PHASE7 = DRIVE_ROOT / "phase7"
DIR_DATASETS = DRIVE_ROOT / "datasets"
DIR_CHECKPOINT = DRIVE_ROOT / "checkpoint"
DIR_PRETRAINED = DRIVE_ROOT / "pretrained"

assert (DIR_PHASE7 / "train_crossdomain_v2.tar").is_file(), "Lỗi: Không tìm thấy train_crossdomain_v2.tar!"
assert (DIR_PRETRAINED / "fatformer_4class_ckpt.pth").is_file(), "Lỗi: Không tìm thấy Baseline checkpoint!"
print("[✓] Xác nhận toàn bộ hiện vật tiên quyết đã sẵn sàng!")
```

---

*Tài liệu được lập và chuẩn hóa bởi Thành viên A (Data & Infrastructure Lead).*  
*Lưu trữ tại: `docs/plan/so_do_ha_tang_google_drive_5tb.md`.*
