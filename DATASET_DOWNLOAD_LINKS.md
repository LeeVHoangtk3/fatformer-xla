# 📦 DANH SÁCH LIÊN KẾT TẢI DỮ LIỆU & TRỌNG SỐ MÔ HÌNH (DATASET & PRETRAINED WEIGHTS DOWNLOAD LINKS)
# DỰ ÁN: FATFORMER-XLA (FORENSIC AI-IMAGE DETECTION)

> **Mục đích tài liệu**: Cung cấp danh mục liên kết trực tiếp (*Direct Download Links*), gương tốc độ cao (*High-Speed Mirrors*) và các câu lệnh tự động tải (`wget` / `curl`) cho toàn bộ trọng số, dữ liệu huấn luyện và tập benchmark đối chứng phục vụ quá trình triển khai trên Google Colab Pro và máy trạm.

---

## MỤC LỤC NHANH
1. [Trọng Số Tiền Huấn Luyện & Mô Hình Gốc](#1-trong-so-tien-huan-luyen--mo-hinh-goc)
2. [Tập Huấn Luyện ProGAN 4-Class (CNNDetection - Wang et al.)](#2-tap-huan-luyen-progan-4-class)
3. [Tập Xúc Tác GenImage 10% (Zhu et al. - NeurIPS 2023)](#3-tap-xuc-tac-genimage-10)
4. [Tập Kiểm Thử 8 Họ GANs (Official Benchmark - CVPR 2024)](#4-tap-kiem-thu-8-ho-gans)
5. [Tập Kiểm Thử 10 Biến Thể Diffusion (Official DMs Benchmark - CVPR 2024)](#5-tap-kiem-thu-10-bien-the-diffusion)
6. [Script Tự Động Tải Nhanh Trên Colab / Linux Terminal](#6-script-tu-dong-tai-nhanh-tren-colab--linux-terminal)
7. [Quy Cách Đóng Gói File .tar Trước Khi Upload Lên Drive 5TB](#7-quy-cach-dong-goi-file-tar-chuan-io)

---

<a name="1-trong-so-tien-huan-luyen--mo-hinh-goc"></a>
## 1. TRỌNG SỐ TIỀN HUẤN LUYỆN & MÔ HÌNH GỐC

| Thành phần | Nguồn phát hành | Liên kết tải trực tiếp (Direct Link) | Dung lượng | Vị trí lưu chuẩn |
| :--- | :--- | :--- | :---: | :--- |
| **CLIP ViT-L/14 Backbone** | OpenAI Official CDN | [Tải ViT-L-14.pt](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) | **~900 MB** | `pretrained/ViT-L-14.pt` |
| **FatFormer Checkpoint gốc** (4-Class CVPR 2024) | Google Drive Tác giả FatFormer | [Tải từ Google Drive](https://drive.google.com/file/d/1Q_Kgq4ygDf8XEHgAf-SgDN6Ru_IOTLkj/view?usp=sharing) | **~350 MB** | `pretrained/fatformer_4class_ckpt.pth` |

> *Ghi chú*: Hai file này đã được lưu sẵn trong thư mục gốc của repository (`ViT-L-14.pt` và `fatformer_4class_ckpt.pth`).

---

<a name="2-tap-huan-luyen-progan-4-class"></a>
## 2. TẬP HUẤN LUYỆN PROGAN 4-CLASS (CNNDetection - Wang et al.)
Tập dữ liệu dùng để huấn luyện bộ thích ứng FAA và rèn luyện độ bền vi sai SRM.  
* **Cấu trúc 4 lớp đối tượng**: `car`, `cat`, `chair`, `horse` (khớp 100% mục 620 paper CVPR 2024).

### A. Tải trực tiếp tốc độ cao từ Hugging Face (sywang/CNNDetection)
Tác giả chia nhỏ bộ dữ liệu thành 7 phần định dạng 7z:
* 🔗 Part 1: [progan_train.7z.001](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.001) (~1.9 GB)
* 🔗 Part 2: [progan_train.7z.002](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.002) (~1.9 GB)
* 🔗 Part 3: [progan_train.7z.003](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.003) (~1.9 GB)
* 🔗 Part 4: [progan_train.7z.004](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.004) (~1.9 GB)
* 🔗 Part 5: [progan_train.7z.005](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.005) (~1.9 GB)
* 🔗 Part 6: [progan_train.7z.006](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.006) (~1.9 GB)
* 🔗 Part 7: [progan_train.7z.007](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.007) (~1.2 GB)

### B. Tập Validation ProGAN
* 🔗 [progan_val.zip](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip) (~350 MB)

### C. Gương Dự Phòng Tác Giả FatFormer (Baidu Netdisk)
* 🔗 [Baidu Netdisk Mirror](https://pan.baidu.com/s/1obzmrCsWvGyUlmH8MkSTLA?pwd=9i5w) | Mật khẩu truy cập: **`9i5w`**

---

<a name="3-tap-xuc-tac-genimage-10"></a>
## 3. TẬP XÚC TÁC GENIMAGE 10% (Zhu et al. - NeurIPS 2023)
Dùng để trích xuất 3.600 ảnh (Stable Diffusion v1.5 + Midjourney v5) làm chất xúc tác cho **Chiến lược 3** và mở rộng tập kiểm thử in-the-wild.

| Nguồn lưu trữ | Liên kết truy cập | Ghi chú hướng dẫn |
| :--- | :--- | :--- |
| **Google Drive Chính Thức** *(Khuyên dùng)* | 🔗 [Google Drive GenImage Official Folder](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS) | Chỉ cần tải 2 thư mục: `stable_diffusion_v_1_5` và `midjourney`. |
| **Hugging Face Datasets** | 🔗 [Hugging Face: GenImage/GenImage](https://huggingface.co/datasets/GenImage/GenImage) | Tải qua `huggingface_hub` hoặc git lfs. |
| **GitHub Repository** | 🔗 [GitHub GenImage-Dataset](https://github.com/GenImage-Dataset/GenImage) | Chứa thông tin metadata và mã nguồn sinh ảnh. |
| **Baidu Netdisk Mirror** | Có sẵn trong repo GitHub của GenImage | Mật khẩu truy cập: **`ztf1`** |

---

<a name="4-tap-kiem-thu-8-ho-gans"></a>
## 4. TẬP KIỂM THỬ 8 HỌ GANS (OFFICIAL BENCHMARK - CVPR 2024)
Bao gồm 8 họ mô hình: **ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, Deepfake**.

* **Liên kết OneDrive Tác giả FatFormer (Tải tốc độ cao - Khuyên dùng)**:
  * 🔗 [OneDrive Tác giả - GANs Testset](https://1drv.ms/u/s!Aqkrc9gPuk8jqaM2khCvAejz_K4Jow?e=e0yeDQ) *(Dung lượng: ~5.8 GB)*
* **Liên kết Hugging Face (File Zip nguyên gốc của CNNDetection)**:
  * 🔗 [CNN_synth_testset.zip](https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip) *(Dung lượng: ~5.8 GB)*
* **Gương Baidu Netdisk**:
  * 🔗 [Baidu Netdisk Mirror](https://pan.baidu.com/s/1aAiW8oMQcIZIaLYuQIOAjg?pwd=75cz) | Mật khẩu truy cập: **`75cz`**

---

<a name="5-tap-kiem-thu-10-bien-the-diffusion"></a>
## 5. TẬP KIỂM THỬ 10 BIẾN THỂ DIFFUSION (OFFICIAL DMs BENCHMARK - CVPR 2024)
Bao gồm 10 biến thể: **PNDM, Guided Diffusion (tử huyệt 76.1%), DALL-E, VQ-Diffusion, LDM (200, 200 CFG, 100), Glide (100-27, 50-27, 100-10)**.

* **Liên kết OneDrive Tác giả FatFormer (Tải trọn gói tốc độ cao - Khuyên dùng)**:
  * 🔗 [OneDrive Tác giả - Diffusion Testset](https://1drv.ms/u/s!Aqkrc9gPuk8jqaM1LAthli7KdRhr2A?e=ebbG9r) *(Dung lượng: ~6.5 GB)*
* **Gương Baidu Netdisk (DIRE / DiffusionForensics)**:
  * 🔗 [Baidu Netdisk Mirror](https://pan.baidu.com/s/1zoubPr5n_mGI27En9uyL8Q?pwd=a6sw) | Mật khẩu truy cập: **`a6sw`** (hoặc **`dire`**)
* **Kho mã nguồn gốc tham chiếu**:
  * [GitHub DIRE (DiffusionForensics)](https://github.com/ZhendongWang6/DIRE#diffusionforensics-dataset)
  * [GitHub UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect#data)

---

<a name="6-script-tu-dong-tai-nhanh-tren-colab--linux-terminal"></a>
## 6. SCRIPT TỰ ĐỘNG TẢI NHANH TRÊN COLAB / LINUX TERMINAL

### Lệnh 1: Tải trọn bộ 8 họ GANs Testset (~5.8 GB)
```bash
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip
unzip -q CNN_synth_testset.zip -d ./test_gans/
rm CNN_synth_testset.zip
```

### Lệnh 2: Tải tập Validation ProGAN (~350 MB)
```bash
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip
unzip -q progan_val.zip -d ./val_progan/
rm progan_val.zip
```

### Lệnh 3: Tải toàn bộ 7 phần của tập Train ProGAN (Hugging Face)
```bash
# Cài đặt công cụ giải nén 7z nếu chạy trên Linux/Colab
sudo apt-get install -y p7zip-full

# Tải song song 7 files
wget -c https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.001 &
wget -c https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.002 &
wget -c https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.003 &
wget -c https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.004 &
wget -c https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.005 &
wget -c https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.006 &
wget -c https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.007 &
wait

# Giải nén nối tiếp
7z x progan_train.7z.001
rm progan_train.7z.*
unzip -q progan_train.zip
rm progan_train.zip
```

---

<a name="7-quy-cach-dong-goi-file-tar-chuan-io"></a>
## 7. QUY CÁCH ĐÓNG GÓI FILE .TAR TRƯỚC KHI UPLOAD LÊN DRIVE 5TB

Để DataLoader đọc mượt mà và Colab không bị sập phiên, Thành viên A sau khi tải về máy tính cá nhân cần trích xuất đúng các lớp cần thiết và đóng gói thành **file `.tar` nguyên khối** theo đúng bảng tên sau:

| File .tar cần nạp lên Drive | Thành phần bên trong file | Số lượng ảnh | Dung lượng .tar |
| :--- | :--- | :---: | :---: |
| `progan_train.tar` | 4 lớp `car, cat, chair, horse` (mỗi lớp 3k Real / 3k Fake) | 24.000 ảnh | **~4.8 GB** |
| `diffusion_staging.tar` | 1.200 SD v1.5 + 600 Midjourney v5 $	imes$ Real/Fake | 3.600 ảnh | **~720 MB** |
| `test_benchmark_clean.tar` | 18 thư mục test clean chuẩn paper CVPR 2024 | ~72.000 ảnh | **~12.5 GB** |
| `test_degraded.tar` | Sinh từ script `data/make_degraded.py` (JPEG 30/50/70, Blur) | ~36.000 ảnh | **~6.2 GB** |
| `test_extended.tar` | SD v1.4, SD v1.5, Midjourney v5 (Tập mở rộng in-the-wild) | ~4.000 ảnh | **~800 MB** |

> **Lệnh đóng gói trên máy trạm (Linux / Mac / Git Bash Windows)**:
> ```bash
> # Ví dụ đóng gói tập train
> tar -cf progan_train.tar progan_train/
> 
> # Ví dụ đóng gói tập staging
> tar -cf diffusion_staging.tar diffusion_staging/
> ```
> Sau đó upload trực tiếp các file `.tar` vào thư mục: `MyDrive/FatFormer_Hub/datasets/` trên Google Drive 5TB.

---
*Tài liệu này được biên soạn độc quyền cho dự án FatFormer-XLA, đảm bảo tính toàn vẹn khoa học và khả năng tái lập 100% thực nghiệm.*
