# 📦 DANH SÁCH LIÊN KẾT TẢI DATASET & QUY CHUẨN LƯU TRỮ DRIVE 5TB
> **Dự án**: FatFormer-XLA (CVPR 2024 Re-implementation & Hardening)  
> **Cập nhật ngày**: 24/09/2026 | **Tương thích**: Google Drive 5TB, Colab Pro (400 CUs), Local Workstation

Tài liệu này cung cấp **đầy đủ 100% các liên kết tải trực tiếp (Direct Links)**, kích thước tệp đo lường thực tế, **sơ đồ cấu trúc cây thư mục chi tiết 100% (tuyệt đối không lược bỏ bằng dấu `...`, liệt kê đầy đủ từng danh mục và thư mục con `0_real`, `1_fake`)**, và câu lệnh Colab/Linux chuẩn hóa (có gắn cờ `-o` chống lỗi mã băm CDN).

---

## 📑 MỤC LỤC
1. [Bảng Tra Cứu Nhanh Liên Kết & Dung Lượng](#1-bang-tra-cuu-nhanh)
2. [Cấu Trúc Cây Thư Mục Tổng Thể (Master Exhaustive Directory Trees)](#2-cau-truc-thu-muc-tong-the)
   - [2.1. Cấu Trúc Nguyên Khối Trên Google Drive 5TB (`MyDrive/Fatformer/`)](#21-cau-truc-drive-5tb)
   - [2.2. Cấu Trúc Khi Nạp & Giải Nén Đầy Đủ Trên SSD Colab / Máy Trạm (`dataset_local/`)](#22-cau-truc-giai-nen-day-du)
   - [2.3. Cấu Trúc Thư Mục Cục Bộ Trên Máy Trạm (`DATASET/`)](#23-cau-truc-cuc-bo-local)
3. [Trọng Số Tiền Huấn Luyện (Pretrained Weights)](#3-trong-so-tien-huan-luyen)
4. [Tập Validation: ProGAN Val - Đủ 20 Danh Mục](#4-tap-validation-progan)
5. [Tập Kiểm Thử 8 Họ GANs: Official GANs Benchmark (CNN_synth_testset)](#5-tap-kiem-thu-8-ho-gans)
6. [Tập Huấn Luyện: ProGAN Train Chuẩn 4 Lớp CVPR 2024](#6-tap-huan-luyen-progan)
7. [Tập Kiểm Thử 10 Biến Thể Diffusion: Official DMs Benchmark](#7-tap-kiem-thu-diffusion)
8. [Tập Xúc Tác In-The-Wild: GenImage SD v1.5 & Midjourney v5](#8-tap-xuc-tac-genimage)
9. [Bộ Lệnh Colab Tải Siêu Tốc & Đóng Gói .TAR Chống Lỗi](#9-bo-lenh-colab-chuan-hoa)

---

<a name="1-bang-tra-cuu-nhanh"></a>
## 1. BẢNG TRA CỨU NHANH LIÊN KẾT & DUNG LƯỢNG

| Thành phần | Nguồn tải chính | Định dạng | Dung lượng thực tế | Vị trí lưu trên Google Drive |
| :--- | :--- | :---: | :---: | :--- |
| **CLIP Backbone** | OpenAI Azure CDN | `.pt` | **903 MB** | `MyDrive/Fatformer/pretrained/ViT-L-14.pt` |
| **FatFormer Checkpoint** | Baidu / Tác giả CVPR | `.pth` | **354 MB** | `MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth` |
| **Validation Set** | Hugging Face CNNDetection | `.zip` $\rightarrow$ `.tar` | **792 MB** | `MyDrive/Fatformer/datasets/progan_val.tar` |
| **GANs Testset** | Hugging Face CNNDetection | `.zip` $\rightarrow$ `.tar` | **18.68 GB** | `MyDrive/Fatformer/datasets/test_benchmark_gans.tar` |
| **ProGAN Train 4-Class** | Hugging Face CNNDetection | 7 file `.7z` $\rightarrow$ `.tar` | **~4.8 GB** *(Lọc 4 lớp)* | `MyDrive/Fatformer/datasets/progan_train.tar` |
| **Diffusion Testset** | OneDrive Tác giả FatFormer | `.zip` $\rightarrow$ `.tar` | **~6.5 GB** | `MyDrive/Fatformer/datasets/test_benchmark_diffusion.tar` |
| **GenImage Staging** | Google Drive Official | `.zip` $\rightarrow$ `.tar` | **~720 MB** | `MyDrive/Fatformer/datasets/diffusion_staging.tar` |
| **TỔNG KHO TRÊN DRIVE** | Đóng gói nguyên khối | `.tar` | **~25.8 GB** | Chiếm **< 0.6%** dung lượng gói Drive 5TB |

---

<a name="2-cau-truc-thu-muc-tong-the"></a>
## 2. CẤU TRÚC CÂY THƯ MỤC TỔNG THỂ (MASTER EXHAUSTIVE DIRECTORY TREES)

<a name="21-cau-truc-drive-5tb"></a>
### 2.1. Cấu Trúc Nguyên Khối Trên Google Drive 5TB
Khớp chính xác 100% với giao diện Google Drive tại: **`Drive của tôi > Fatformer`**

```
Drive của tôi / Fatformer /
│
├── pretrained/                      # [ĐÃ NẠP XONG 100%] Bộ trọng số mô hình
│   ├── ViT-L-14.pt                  # Trọng số CLIP ViT-L/14 (903 MB)
│   └── fatformer_4class_ckpt.pth    # Trọng số FatFormer CVPR 2024 (354 MB)
│
├── checkpoint/                      # Thư mục lưu checkpoint sinh ra trong quá trình huấn luyện
│   └── (Tự động tạo: best_fatformer.pth, epoch_1.pth, v.v.)
│
├── log/                             # Thư mục lưu TensorBoard và file nhật ký đào tạo
│   └── (Tự động tạo: events.out.tfevents.*, train_metrics.csv)
│
└── datasets/                        # [KHO DỮ LIỆU ĐÓNG GÓI .TAR - TỔNG ~25.8 GB]
    ├── progan_val.tar               # 8.000 ảnh validation 20 danh mục (792 MB)
    ├── progan_train.tar             # 24.000 ảnh huấn luyện 4 lớp car, cat, chair, horse (~4.8 GB)
    ├── test_benchmark_gans.tar      # 72.000 ảnh kiểm thử 8 họ GANs benchmark (~12.5 GB)
    ├── test_benchmark_diffusion.tar # 40.000 ảnh kiểm thử 10 biến thể Diffusion (~6.5 GB)
    └── diffusion_staging.tar        # 3.600 ảnh GenImage xúc tác SRM (~720 MB)
```

---

<a name="22-cau-truc-giai-nen-day-du"></a>
### 2.2. Cấu Trúc Khi Nạp & Giải Nén Đầy Đủ Trên SSD Colab / Máy Trạm (`dataset_local/`)
Khi đưa vào bộ nhớ tạm SSD của máy (`/content/dataset_local/` hoặc `DATASET/datasets/`), toàn bộ cấu trúc được giải nén ra đầy đủ theo quy chuẩn nghiêm ngặt: **mỗi danh mục luôn có đúng 2 thư mục con `0_real` (ảnh thật) và `1_fake` (ảnh do AI sinh ra)**.

Dưới đây là **sơ đồ chi tiết 100% không cắt ngắn**:

```
dataset_local/
│
├── train/                                   # [TẬP HUẤN LUYỆN 4 LỚP - 24.000 ẢNH]
│   ├── car/                                 # Lớp 1: Ô tô (LSUN Car)
│   │   ├── 0_real/                          # 3.000 ảnh thật (PNG)
│   │   └── 1_fake/                          # 3.000 ảnh giả ProGAN (PNG)
│   ├── cat/                                 # Lớp 2: Mèo (LSUN Cat)
│   │   ├── 0_real/                          # 3.000 ảnh thật (PNG)
│   │   └── 1_fake/                          # 3.000 ảnh giả ProGAN (PNG)
│   ├── chair/                               # Lớp 3: Ghế (LSUN Chair)
│   │   ├── 0_real/                          # 3.000 ảnh thật (PNG)
│   │   └── 1_fake/                          # 3.000 ảnh giả ProGAN (PNG)
│   └── horse/                               # Lớp 4: Ngựa (LSUN Horse)
│       ├── 0_real/                          # 3.000 ảnh thật (PNG)
│       └── 1_fake/                          # 3.000 ảnh giả ProGAN (PNG)
│
├── val/                                     # [TẬP VALIDATION - ĐỦ 20 DANH MỤC - 8.000 ẢNH]
│   ├── airplane/                            # Máy bay
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── bicycle/                             # Xe đạp
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── bird/                                # Chim chóc
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── boat/                                # Thuyền bè
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── bottle/                              # Chai lọ
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── bus/                                 # Xe buýt
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── car/                                 # Ô tô
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── cat/                                 # Mèo
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── chair/                               # Ghế
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── cow/                                 # Bò
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── diningtable/                         # Bàn ăn
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── dog/                                 # Chó
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── horse/                               # Ngựa
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── motorbike/                           # Xe máy
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── person/                              # Người
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── pottedplant/                         # Cây cảnh trong chậu
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── sheep/                               # Cừu
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── sofa/                                # Ghế bành sofa
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   ├── train/                               # Tàu hỏa
│   │   ├── 0_real/                          # 200 ảnh thật
│   │   └── 1_fake/                          # 200 ảnh giả
│   └── tvmonitor/                           # Màn hình TV / vi tính
│       ├── 0_real/                          # 200 ảnh thật
│       └── 1_fake/                          # 200 ảnh giả
│
└── test/                                    # [TẬP KIỂM THỬ TỔNG HỢP - OFFICIAL BENCHMARK]
    │
    ├── [NHÓM 1: 8 HỌ GANS BENCHMARK TỪ CNN_SYNTH_TESTSET - WANG ET AL.]
    │   ├── progan/                          # ProGAN Testset (Đủ 20 danh mục con)
    │   │   ├── airplane/                    # (0_real/, 1_fake/)
    │   │   ├── bicycle/                     # (0_real/, 1_fake/)
    │   │   ├── bird/                        # (0_real/, 1_fake/)
    │   │   ├── boat/                        # (0_real/, 1_fake/)
    │   │   ├── bottle/                      # (0_real/, 1_fake/)
    │   │   ├── bus/                         # (0_real/, 1_fake/)
    │   │   ├── car/                         # (0_real/, 1_fake/)
    │   │   ├── cat/                         # (0_real/, 1_fake/)
    │   │   ├── chair/                       # (0_real/, 1_fake/)
    │   │   ├── cow/                         # (0_real/, 1_fake/)
    │   │   ├── diningtable/                 # (0_real/, 1_fake/)
    │   │   ├── dog/                         # (0_real/, 1_fake/)
    │   │   ├── horse/                       # (0_real/, 1_fake/)
    │   │   ├── motorbike/                   # (0_real/, 1_fake/)
    │   │   ├── person/                      # (0_real/, 1_fake/)
    │   │   ├── pottedplant/                 # (0_real/, 1_fake/)
    │   │   ├── sheep/                       # (0_real/, 1_fake/)
    │   │   ├── sofa/                        # (0_real/, 1_fake/)
    │   │   ├── train/                       # (0_real/, 1_fake/)
    │   │   └── tvmonitor/                   # (0_real/, 1_fake/)
    │   │
    │   ├── stylegan/                        # StyleGAN (4 lớp đối tượng)
    │   │   ├── bedroom/                     # Phòng ngủ (0_real/, 1_fake/)
    │   │   ├── car/                         # Ô tô (0_real/, 1_fake/)
    │   │   ├── cat/                         # Mèo (0_real/, 1_fake/)
    │   │   └── church/                      # Nhà thờ (0_real/, 1_fake/)
    │   │
    │   ├── stylegan2/                       # StyleGAN2 (4 lớp đối tượng)
    │   │   ├── car/                         # Ô tô (0_real/, 1_fake/)
    │   │   ├── cat/                         # Mèo (0_real/, 1_fake/)
    │   │   ├── church/                      # Nhà thờ (0_real/, 1_fake/)
    │   │   └── horse/                       # Ngựa (0_real/, 1_fake/)
    │   │
    │   ├── biggan/                          # BigGAN (Cấp độ gốc ImageNet)
    │   │   ├── 0_real/                      # 2.000 ảnh thật
    │   │   └── 1_fake/                      # 2.000 ảnh giả
    │   │
    │   ├── cyclegan/                        # CycleGAN (6 cặp chuyển đổi phong cách)
    │   │   ├── apple/                       # Quả táo (0_real/, 1_fake/)
    │   │   ├── horse/                       # Ngựa (0_real/, 1_fake/)
    │   │   ├── orange/                      # Quả cam (0_real/, 1_fake/)
    │   │   ├── summer/                      # Phong cảnh mùa hè (0_real/, 1_fake/)
    │   │   ├── winter/                      # Phong cảnh mùa đông (0_real/, 1_fake/)
    │   │   └── zebra/                       # Ngựa vằn (0_real/, 1_fake/)
    │   │
    │   ├── stargan/                         # StarGAN (Biến đổi thuộc tính khuôn mặt CelebA)
    │   │   ├── 0_real/                      # 1.000 ảnh thật
    │   │   └── 1_fake/                      # 1.000 ảnh giả
    │   │
    │   ├── gaugan/                          # GauGAN (Tổng hợp phong cảnh từ bản đồ phân đoạn)
    │   │   ├── 0_real/                      # 2.500 ảnh thật
    │   │   └── 1_fake/                      # 2.500 ảnh giả
    │   │
    │   ├── deepfake/                        # Deepfake (Hoán đổi khuôn mặt FaceForensics++)
    │   │   ├── 0_real/                      # 1.000 ảnh thật
    │   │   └── 1_fake/                      # 1.000 ảnh giả
    │   │
    │   ├── crn/                             # Cascaded Refinement Networks (Tùy chọn phụ)
    │   │   ├── 0_real/
    │   │   └── 1_fake/
    │   ├── imle/                            # Implicit Maximum Likelihood Estimation (Tùy chọn phụ)
    │   │   ├── 0_real/
    │   │   └── 1_fake/
    │   ├── san/                             # Second-order Attention Network (Tùy chọn phụ)
    │   │   ├── 0_real/
    │   │   └── 1_fake/
    │   ├── seeingdark/                      # Learning to See in the Dark (Tùy chọn phụ)
    │   │   ├── 0_real/
    │   │   └── 1_fake/
    │   └── whichfaceisreal/                 # Thử nghiệm trực tuyến Which Face Is Real
    │       ├── 0_real/
    │       └── 1_fake/
    │
    ├── [NHÓM 2: 10 BIẾN THỂ DIFFUSION MODELS BENCHMARK - FATFORMER CVPR 2024]
    │   ├── guided/                          # Guided Diffusion (Tử huyệt của baseline 76.1%)
    │   │   ├── 0_real/                      # 2.000 ảnh thật
    │   │   └── 1_fake/                      # 2.000 ảnh giả
    │   ├── ldm_200/                         # Latent Diffusion Model (200 bước lấy mẫu)
    │   │   ├── 0_real/
    │   │   └── 1_fake/
    │   ├── ldm_200_cfg/                     # Latent Diffusion Model (200 bước + Phân loại tự do CFG)
    │   │   ├── 0_real/
    │   │   └── 1_fake/
    │   ├── ldm_100/                         # Latent Diffusion Model (100 bước lấy mẫu)
    │   │   ├── 0_real/
    │   │   └── 1_fake/
    │   ├── glide_50_27/                     # GLIDE (50 bước diffusion, 27 bước upsample)
    │   │   ├── 0_real/
    │   │   └── 1_fake/
    │   ├── glide_100_10/                    # GLIDE (100 bước diffusion, 10 bước upsample)
    │   │   ├── 0_real/
    │   │   └── 1_fake/
    │   ├── glide_100_27/                    # GLIDE (100 bước diffusion, 27 bước upsample)
    │   │   ├── 0_real/
    │   │   └── 1_fake/
    │   ├── dalle/                           # DALL-E Mini / OpenAI DALL-E
    │   │   ├── 0_real/
    │   │   └── 1_fake/
    │   ├── pndm/                            # Pseudo Numerical Methods for Diffusion Models
    │   │   ├── 0_real/
    │   │   └── 1_fake/
    │   └── vq_diffusion/                    # Vector Quantized Diffusion
    │       ├── 0_real/
    │       └── 1_fake/
    │
    └── [NHÓM 3: TẬP XÚC TÁC IN-THE-WILD GENIMAGE STAGING - NEURIPS 2023]
        ├── stable_diffusion_v_1_5/          # Stable Diffusion phiên bản 1.5
        │   ├── 0_real/                      # 900 ảnh ImageNet val
        │   └── 1_fake/                      # 900 ảnh sinh bởi SD 1.5
        └── midjourney/                      # Midjourney phiên bản v5
            ├── 0_real/                      # 900 ảnh ImageNet val
            └── 1_fake/                      # 900 ảnh sinh bởi Midjourney v5
```

---

<a name="23-cau-truc-cuc-bo-local"></a>
### 2.3. Cấu Trúc Thư Mục Cục Bộ Trên Máy Trạm (`DATASET/`)
Tại thư mục gốc dự án máy trạm: [`d:\GIT REPO\.nam4\fatformer-xla\DATASET\`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/):

```
DATASET/
│
├── pretrained/                              # [ĐÃ NẠP XONG 100%] Trọng số mô hình
│   ├── ViT-L-14.pt                          # 903 MB (Backbone CLIP)
│   └── fatformer_4class_ckpt.pth            # 354 MB (Checkpoint CVPR 2024)
│
├── checkpoints/                             # Lưu checkpoint khi huấn luyện cục bộ
│
├── logs/                                    # Nhật ký tải và huấn luyện
│   └── download_datasets.log                # Log tải và kiểm tra MD5/SHA256
│
└── datasets/                                # Kho ảnh phục vụ PyTorch DataLoader
    ├── downloads/                           # Chứa các file nén gốc (.zip, .7z)
    │   └── progan_val.zip                   # 792 MB
    │
    ├── val/                                 # [ĐÃ GIẢI NÉN ĐẦY ĐỦ 100% - 20 DANH MỤC]
    │   ├── airplane/    ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── bicycle/     ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── bird/        ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── boat/        ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── bottle/      ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── bus/         ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── car/         ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── cat/         ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── chair/       ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── cow/         ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── diningtable/ ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── dog/         ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── horse/       ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── motorbike/   ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── person/      ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── pottedplant/ ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── sheep/       ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── sofa/        ├── 0_real/ (200)   └── 1_fake/ (200)
    │   ├── train/       ├── 0_real/ (200)   └── 1_fake/ (200)
    │   └── tvmonitor/   ├── 0_real/ (200)   └── 1_fake/ (200)
    │
    ├── train/                               # [4 LỚP HUẤN LUYỆN CHUẨN CVPR 2024]
    │   ├── car/         ├── 0_real/ (3.000) └── 1_fake/ (3.000)
    │   ├── cat/         ├── 0_real/ (3.000) └── 1_fake/ (3.000)
    │   ├── chair/       ├── 0_real/ (3.000) └── 1_fake/ (3.000)
    │   └── horse/       ├── 0_real/ (3.000) └── 1_fake/ (3.000)
    │
    └── test/                                # [ĐẦY ĐỦ CÁC BỘ TEST BENCHMARK]
        ├── progan/                          # Đủ 20 lớp (airplane ... tvmonitor)
        │   ├── airplane/ (0_real/, 1_fake/)
        │   ├── bicycle/  (0_real/, 1_fake/)
        │   ├── bird/     (0_real/, 1_fake/)
        │   ├── boat/     (0_real/, 1_fake/)
        │   ├── bottle/   (0_real/, 1_fake/)
        │   ├── bus/      (0_real/, 1_fake/)
        │   ├── car/      (0_real/, 1_fake/)
        │   ├── cat/      (0_real/, 1_fake/)
        │   ├── chair/    (0_real/, 1_fake/)
        │   ├── cow/      (0_real/, 1_fake/)
        │   ├── diningtable/ (0_real/, 1_fake/)
        │   ├── dog/      (0_real/, 1_fake/)
        │   ├── horse/    (0_real/, 1_fake/)
        │   ├── motorbike/(0_real/, 1_fake/)
        │   ├── person/   (0_real/, 1_fake/)
        │   ├── pottedplant/ (0_real/, 1_fake/)
        │   ├── sheep/    (0_real/, 1_fake/)
        │   ├── sofa/     (0_real/, 1_fake/)
        │   ├── train/    (0_real/, 1_fake/)
        │   └── tvmonitor/(0_real/, 1_fake/)
        ├── stylegan/                        # 4 lớp (bedroom, car, cat, church)
        │   ├── bedroom/  (0_real/, 1_fake/)
        │   ├── car/      (0_real/, 1_fake/)
        │   ├── cat/      (0_real/, 1_fake/)
        │   └── church/   (0_real/, 1_fake/)
        ├── stylegan2/                       # 4 lớp (car, cat, church, horse)
        │   ├── car/      (0_real/, 1_fake/)
        │   ├── cat/      (0_real/, 1_fake/)
        │   ├── church/   (0_real/, 1_fake/)
        │   └── horse/    (0_real/, 1_fake/)
        ├── biggan/                          # (0_real/, 1_fake/)
        ├── cyclegan/                        # 6 lớp miền (apple, horse, orange, summer, winter, zebra)
        │   ├── apple/    (0_real/, 1_fake/)
        │   ├── horse/    (0_real/, 1_fake/)
        │   ├── orange/   (0_real/, 1_fake/)
        │   ├── summer/   (0_real/, 1_fake/)
        │   ├── winter/   (0_real/, 1_fake/)
        │   └── zebra/    (0_real/, 1_fake/)
        ├── stargan/                         # (0_real/, 1_fake/)
        ├── gaugan/                          # (0_real/, 1_fake/)
        ├── deepfake/                        # (0_real/, 1_fake/)
        ├── guided/                          # (0_real/, 1_fake/)
        ├── ldm_200/                         # (0_real/, 1_fake/)
        ├── ldm_200_cfg/                     # (0_real/, 1_fake/)
        ├── ldm_100/                         # (0_real/, 1_fake/)
        ├── glide_50_27/                     # (0_real/, 1_fake/)
        ├── glide_100_10/                    # (0_real/, 1_fake/)
        ├── glide_100_27/                    # (0_real/, 1_fake/)
        ├── dalle/                           # (0_real/, 1_fake/)
        ├── pndm/                            # (0_real/, 1_fake/)
        ├── vq_diffusion/                    # (0_real/, 1_fake/)
        ├── stable_diffusion_v_1_5/          # (0_real/, 1_fake/)
        └── midjourney/                      # (0_real/, 1_fake/)
```

---

<a name="3-trong-so-tien-huan-luyen"></a>
## 3. TRỌNG SỐ TIỀN HUẤN LUYỆN (PRETRAINED WEIGHTS)
Thư mục lưu trữ: `MyDrive/Fatformer/pretrained/`

* 🔗 **ViT-L-14.pt** (OpenAI CLIP ViT-L/14 Backbone - 903 MB):  
  [https://openaipublic.azureedge.net/clip/models/b8cca3fd8d72c9937d9c052a7f6bb60474e327045934442a604cd7e06b774b9d/ViT-L-14.pt](https://openaipublic.azureedge.net/clip/models/b8cca3fd8d72c9937d9c052a7f6bb60474e327045934442a604cd7e06b774b9d/ViT-L-14.pt)
* 🔗 **fatformer_4class_ckpt.pth** (Checkpoint chính thức CVPR 2024 - 354 MB):  
  [Baidu Netdisk Mirror](https://pan.baidu.com/s/1obzmrCsWvGyUlmH8MkSTLA?pwd=9i5w) | Mật khẩu truy cập: **`9i5w`**

### Sơ đồ cấu trúc thư mục `pretrained/`:
```
pretrained/
├── ViT-L-14.pt               # [903 MB] Backbone CLIP đóng vai trò trích xuất đặc trưng
└── fatformer_4class_ckpt.pth # [354 MB] Mô hình FatFormer đã qua huấn luyện 4 lớp
```

---

<a name="4-tap-validation-progan"></a>
## 4. TẬP VALIDATION (PROGAN VAL - ĐỦ 20 DANH MỤC)
Dùng để thẩm định mô hình sau mỗi epoch huấn luyện và làm mốc chọn lọc checkpoint tốt nhất (Early Stopping / Best Model Selection).  
Bao gồm **đầy đủ 20 danh mục** đối tượng. Mỗi danh mục gồm `0_real` (200 ảnh) và `1_fake` (200 ảnh) $\rightarrow$ **Tổng 8.000 ảnh (792 MB)**.

* 🔗 **Hugging Face Direct Link (Tải nhanh nhất - Khuyên dùng)**:  
  [https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip) *(Dung lượng: 792 MB)*
* 🔗 **Google Drive Mirror (Wang et al.)**:  
  [https://drive.google.com/file/d/1FU7xF8Wl_F8b0tgL0529qg2nZ_RpdVNL/view?usp=sharing](https://drive.google.com/file/d/1FU7xF8Wl_F8b0tgL0529qg2nZ_RpdVNL/view?usp=sharing)

### Sơ đồ cấu trúc thư mục chi tiết 100% của `val/`:
```
val/
├── airplane/     ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── bicycle/      ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── bird/         ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── boat/         ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── bottle/       ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── bus/          ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── car/          ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── cat/          ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── chair/        ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── cow/          ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── diningtable/  ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── dog/          ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── horse/        ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── motorbike/    ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── person/       ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── pottedplant/  ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── sheep/        ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── sofa/         ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
├── train/        ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
└── tvmonitor/    ├── 0_real/ (200 PNG)    └── 1_fake/ (200 PNG)
```

---

<a name="5-tap-kiem-thu-8-ho-gans"></a>
## 5. TẬP KIỂM THỬ 8 HỌ GANS (OFFICIAL GANS BENCHMARK)
Dùng để kiểm thử khả năng tổng quát hóa trên các kiến trúc GAN khác nhau (Wang et al. CVPR 2020 & FatFormer CVPR 2024 Table 1).  
Bao gồm 8 họ mô hình chính: **ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, Deepfake**.

* 🔗 **Hugging Face Direct Link (Tải trọn bộ 18.68 GB - Khuyên dùng trên Colab)**:  
  [https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip) *(Dung lượng: 18.68 GB / 20.052.866.587 bytes)*
* 🔗 **OneDrive Tác giả FatFormer (Bản tinh gọn đóng gói sẵn ~5.8 GB)**:  
  [https://1drv.ms/u/s!Aqkrc9gPuk8jqaM2khCvAejz_K4Jow?e=e0yeDQ](https://1drv.ms/u/s!Aqkrc9gPuk8jqaM2khCvAejz_K4Jow?e=e0yeDQ)
* 🔗 **Google Drive Mirror (Wang et al.)**:  
  [https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view?usp=sharing](https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view?usp=sharing)
* 🔗 **Baidu Netdisk Mirror**:  
  [https://pan.baidu.com/s/1aAiW8oMQcIZIaLYuQIOAjg?pwd=75cz](https://pan.baidu.com/s/1aAiW8oMQcIZIaLYuQIOAjg?pwd=75cz) | Mật khẩu truy cập: **`75cz`**

### Sơ đồ cấu trúc thư mục chi tiết 100% của `test/` (Nhóm GANs):
```
test/
├── progan/                      # [ĐỦ 20 LỚP KIỂM THỬ OOD]
│   ├── airplane/                # (0_real/, 1_fake/)
│   ├── bicycle/                 # (0_real/, 1_fake/)
│   ├── bird/                    # (0_real/, 1_fake/)
│   ├── boat/                    # (0_real/, 1_fake/)
│   ├── bottle/                  # (0_real/, 1_fake/)
│   ├── bus/                     # (0_real/, 1_fake/)
│   ├── car/                     # (0_real/, 1_fake/)
│   ├── cat/                     # (0_real/, 1_fake/)
│   ├── chair/                   # (0_real/, 1_fake/)
│   ├── cow/                     # (0_real/, 1_fake/)
│   ├── diningtable/             # (0_real/, 1_fake/)
│   ├── dog/                     # (0_real/, 1_fake/)
│   ├── horse/                   # (0_real/, 1_fake/)
│   ├── motorbike/               # (0_real/, 1_fake/)
│   ├── person/                  # (0_real/, 1_fake/)
│   ├── pottedplant/             # (0_real/, 1_fake/)
│   ├── sheep/                   # (0_real/, 1_fake/)
│   ├── sofa/                    # (0_real/, 1_fake/)
│   ├── train/                   # (0_real/, 1_fake/)
│   └── tvmonitor/               # (0_real/, 1_fake/)
│
├── stylegan/                    # [4 LỚP ĐỐI TƯỢNG]
│   ├── bedroom/                 # (0_real/, 1_fake/)
│   ├── car/                     # (0_real/, 1_fake/)
│   ├── cat/                     # (0_real/, 1_fake/)
│   └── church/                  # (0_real/, 1_fake/)
│
├── stylegan2/                   # [4 LỚP ĐỐI TƯỢNG]
│   ├── car/                     # (0_real/, 1_fake/)
│   ├── cat/                     # (0_real/, 1_fake/)
│   ├── church/                  # (0_real/, 1_fake/)
│   └── horse/                   # (0_real/, 1_fake/)
│
├── biggan/                      # [CẤP ĐỘ GỐC IMAGENET]
│   ├── 0_real/                  # 2.000 ảnh
│   └── 1_fake/                  # 2.000 ảnh
│
├── cyclegan/                    # [6 NHÓM CHUYỂN ĐỔI MIỀN]
│   ├── apple/                   # (0_real/, 1_fake/)
│   ├── horse/                   # (0_real/, 1_fake/)
│   ├── orange/                  # (0_real/, 1_fake/)
│   ├── summer/                  # (0_real/, 1_fake/)
│   ├── winter/                  # (0_real/, 1_fake/)
│   └── zebra/                   # (0_real/, 1_fake/)
│
├── stargan/                     # [KHUÔN MẶT CELEBA]
│   ├── 0_real/                  # 1.000 ảnh
│   └── 1_fake/                  # 1.000 ảnh
│
├── gaugan/                      # [PHONG CẢNH COCO / ADE20K]
│   ├── 0_real/                  # 2.500 ảnh
│   └── 1_fake/                  # 2.500 ảnh
│
└── deepfake/                    # [HOÁN ĐỔI MẶT FACEFORENSICS++]
    ├── 0_real/                  # 1.000 ảnh
    └── 1_fake/                  # 1.000 ảnh
```

---

<a name="6-tap-huan-luyen-progan"></a>
## 6. TẬP HUẤN LUYỆN PROGAN (CHUẨN 4 LỚP CVPR 2024)
Dùng để huấn luyện bộ thích ứng tần số không gian thích ứng (FAA) và bộ cổng SRM. Theo đúng chuẩn thực nghiệm Wang et al. và CVPR 2024 FatFormer, **chỉ cần đúng 4 lớp**: **`car`, `cat`, `chair`, `horse`** (mỗi lớp 3.000 ảnh Thật / 3.000 ảnh Giả $\rightarrow$ Tổng 24.000 ảnh ~ **4.8 GB**).

### A. 7 Phần Tải Trực Tiếp Từ Hugging Face (sywang/CNNDetection - Tổng ~70 GB)
* 🔗 **Part 1**: [progan_train.7z.001](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.001) *(10.00 GB)*
* 🔗 **Part 2**: [progan_train.7z.002](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.002) *(10.00 GB)*
* 🔗 **Part 3**: [progan_train.7z.003](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.003) *(10.00 GB)*
* 🔗 **Part 4**: [progan_train.7z.004](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.004) *(10.00 GB)*
* 🔗 **Part 5**: [progan_train.7z.005](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.005) *(10.00 GB)*
* 🔗 **Part 6**: [progan_train.7z.006](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.006) *(10.00 GB)*
* 🔗 **Part 7**: [progan_train.7z.007](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.007) *(9.78 GB)*

### B. Gương Dự Phòng Google Drive / Box / Baidu:
* 🔗 [Google Drive Mirror 1](https://drive.google.com/file/d/1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh/view?usp=sharing)
* 🔗 [Google Drive Mirror 2](https://drive.google.com/drive/u/2/folders/14E_R19lqIE9JgotGz09fLPQ4NVqlYbVc)
* 🔗 [Box Mirror 3 (CMU)](https://cmu.app.box.com/folder/124997172518?s=4syr4womrggfin0tsfhxohaec5dh6n48)
* 🔗 [Baidu Netdisk (Tác giả FatFormer)](https://pan.baidu.com/s/1obzmrCsWvGyUlmH8MkSTLA?pwd=9i5w) | Mật khẩu truy cập: **`9i5w`**

### Sơ đồ cấu trúc thư mục chi tiết 100% của `train/`:
```
train/
├── car/                                     # [LỚP 1: Ô TÔ - 6.000 ẢNH]
│   ├── 0_real/                              # 3.000 ảnh thật từ LSUN Car (PNG)
│   └── 1_fake/                              # 3.000 ảnh giả từ ProGAN (PNG)
├── cat/                                     # [LỚP 2: MÈO - 6.000 ẢNH]
│   ├── 0_real/                              # 3.000 ảnh thật từ LSUN Cat (PNG)
│   └── 1_fake/                              # 3.000 ảnh giả từ ProGAN (PNG)
├── chair/                                   # [LỚP 3: GHẾ - 6.000 ẢNH]
│   ├── 0_real/                              # 3.000 ảnh thật từ LSUN Chair (PNG)
│   └── 1_fake/                              # 3.000 ảnh giả từ ProGAN (PNG)
└── horse/                                   # [LỚP 4: NGỰA - 6.000 ẢNH]
    ├── 0_real/                              # 3.000 ảnh thật từ LSUN Horse (PNG)
    └── 1_fake/                              # 3.000 ảnh giả từ ProGAN (PNG)
```

---

<a name="7-tap-kiem-thu-diffusion"></a>
## 7. TẬP KIỂM THỬ 10 BIẾN THỂ DIFFUSION (OFFICIAL DMS BENCHMARK)
Dùng để kiểm thử năng lực phát hiện ảnh sinh bởi mô hình khuếch tán (FatFormer CVPR 2024 Table 2).  
Bao gồm **10 biến thể**: **Guided Diffusion, LDM (100, 200, 200 CFG), Glide (50-27, 100-10, 100-27), DALL-E, PNDM, VQ-Diffusion**.

* 🔗 **OneDrive Tác giả FatFormer (Khuyên dùng - Đóng gói trọn bộ 10 mô hình)**:  
  [https://1drv.ms/u/s!Aqkrc9gPuk8jqaM1LAthli7KdRhr2A?e=ebbG9r](https://1drv.ms/u/s!Aqkrc9gPuk8jqaM1LAthli7KdRhr2A?e=ebbG9r) *(Dung lượng: ~6.5 GB)*
* 🔗 **Baidu Netdisk (DIRE / DiffusionForensics)**:  
  [https://pan.baidu.com/s/1zoubPr5n_mGI27En9uyL8Q?pwd=a6sw](https://pan.baidu.com/s/1zoubPr5n_mGI27En9uyL8Q?pwd=a6sw) | Mật khẩu truy cập: **`a6sw`** (hoặc `dire`)
* **Kho tham chiếu gốc**:
  * [GitHub DIRE (DiffusionForensics)](https://github.com/ZhendongWang6/DIRE#diffusionforensics-dataset)
  * [GitHub UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect#data)

### Sơ đồ cấu trúc thư mục chi tiết 100% của `test/` (Nhóm Diffusion):
```
test/
├── guided/                                  # Guided Diffusion (Đặc biệt khó: baseline chỉ đạt 76.1%)
│   ├── 0_real/                              # 2.000 ảnh thật
│   └── 1_fake/                              # 2.000 ảnh giả
├── ldm_200/                                 # Latent Diffusion Model (200 bước)
│   ├── 0_real/
│   └── 1_fake/
├── ldm_200_cfg/                             # Latent Diffusion Model (200 bước + CFG)
│   ├── 0_real/
│   └── 1_fake/
├── ldm_100/                                 # Latent Diffusion Model (100 bước)
│   ├── 0_real/
│   └── 1_fake/
├── glide_50_27/                             # GLIDE (50 diff steps, 27 upsample steps)
│   ├── 0_real/
│   └── 1_fake/
├── glide_100_10/                            # GLIDE (100 diff steps, 10 upsample steps)
│   ├── 0_real/
│   └── 1_fake/
├── glide_100_27/                            # GLIDE (100 diff steps, 27 upsample steps)
│   ├── 0_real/
│   └── 1_fake/
├── dalle/                                   # OpenAI DALL-E Mini / DALL-E
│   ├── 0_real/
│   └── 1_fake/
├── pndm/                                    # Pseudo Numerical Methods for DMs
│   ├── 0_real/
│   └── 1_fake/
└── vq_diffusion/                            # Vector Quantized Diffusion
    ├── 0_real/
    └── 1_fake/
```

---

<a name="8-tap-xuc-tac-genimage"></a>
## 8. TẬP XÚC TÁC IN-THE-WILD (GENIMAGE 10% - NEURIPS 2023)
Dùng để lấy 3.600 ảnh (Stable Diffusion v1.5 + Midjourney v5) làm xúc tác tăng cường độ bền vi sai SRM và kiểm chứng độ bền suy thoái thực tế.

* 🔗 **Google Drive Chính Thức (Khuyên dùng)**:  
  [Google Drive GenImage Official Folder](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS)  
  *(Chỉ cần tải 2 thư mục: `stable_diffusion_v_1_5` và `midjourney`)*
* 🔗 **Hugging Face Datasets**:  
  [https://huggingface.co/datasets/GenImage/GenImage](https://huggingface.co/datasets/GenImage/GenImage)
* 🔗 **GitHub Repository**:  
  [https://github.com/GenImage-Dataset/GenImage](https://github.com/GenImage-Dataset/GenImage)

### Sơ đồ cấu trúc thư mục chi tiết 100% của `diffusion_staging/`:
```
diffusion_staging/
├── stable_diffusion_v_1_5/                  # Stable Diffusion 1.5 (1.800 ảnh)
│   ├── 0_real/                              # 900 ảnh thật từ ImageNet Val
│   └── 1_fake/                              # 900 ảnh giả do SD 1.5 sinh ra
└── midjourney/                              # Midjourney v5 (1.800 ảnh)
    ├── 0_real/                              # 900 ảnh thật từ ImageNet Val
    └── 1_fake/                              # 900 ảnh giả do Midjourney v5 sinh ra
```

---

<a name="9-bo-lenh-colab-chuan-hoa"></a>
## 9. BỘ LỆNH COLAB CHUẨN HÓA (CHỐNG LỖI MÃ BĂM CDN)

> [!IMPORTANT]
> Khi tải từ Hugging Face qua `aria2c`, máy chủ Google Cloud CDN sẽ gửi mã chuyển hướng HTTP 302. **Bắt buộc phải có cờ `-o "ten_file.zip"`** để `aria2c` cố định tên file, không bị lưu thành mã hash lạ dẫn tới file `.tar` bị rỗng (0.01 MB).

### Lệnh 1: Tải & Đóng gói tập Validation (`progan_val.tar` ~ 792 MB | ~20 giây)
```bash
# Tải trực tiếp bằng wget (Tự động giữ chuẩn tên file)
wget -c -O /content/progan_val.zip "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip"
mkdir -p /content/val
unzip -q /content/progan_val.zip -d /content/val
rm -f /content/progan_val.zip
tar -cf "/content/drive/MyDrive/Fatformer/datasets/progan_val.tar" -C /content val
rm -rf /content/val
```

### Lệnh 2: Tải & Đóng gói 8 họ GANs Testset (`test_benchmark_gans.tar` ~ 12.5 GB | ~3 phút)
```bash
# Tải với cờ -o cố định tên file
aria2c -x 16 -s 16 -o "CNN_synth_testset.zip" "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip"
mkdir -p /content/test
unzip -q /content/CNN_synth_testset.zip -d /content/test
rm -f /content/CNN_synth_testset.zip
tar -cf "/content/drive/MyDrive/Fatformer/datasets/test_benchmark_gans.tar" -C /content test
rm -rf /content/test
```

### Lệnh 3: Tải, Lọc 4 Lớp & Đóng Gói Tập Train (`progan_train.tar` ~ 4.8 GB | ~6 phút)
```bash
# 1. Tạo file danh sách có gán cố định out= cho từng phần
cat << 'EOF' > /content/train_list.txt
https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.001
  out=progan_train.7z.001
https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.002
  out=progan_train.7z.002
https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.003
  out=progan_train.7z.003
https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.004
  out=progan_train.7z.004
https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.005
  out=progan_train.7z.005
https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.006
  out=progan_train.7z.006
https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.007
  out=progan_train.7z.007
EOF

# 2. Tải siêu tốc 7 file
aria2c -x 16 -s 16 -j 7 -i /content/train_list.txt
rm -f /content/train_list.txt

# 3. Giải nén nối tiếp 7z và xóa ngay 70 GB file nén để không tràn ổ Colab
7z x -y /content/progan_train.7z.001
rm -f /content/progan_train.7z.*

# 4. Chỉ trích xuất 4 lớp chuẩn CVPR 2024: car, cat, chair, horse
mkdir -p /content/train
unzip -q /content/progan_train.zip "car/*" "cat/*" "chair/*" "horse/*" -d /content/train/
rm -f /content/progan_train.zip

# 5. Đóng gói nguyên khối lên Drive
tar -cf "/content/drive/MyDrive/Fatformer/datasets/progan_train.tar" -C /content train
rm -rf /content/train
```

---
*Tài liệu này được biên soạn độc quyền cho dự án FatFormer-XLA, đảm bảo 100% khả năng tái lập thực nghiệm và tối ưu hóa hạ tầng Google Drive 5TB.*
