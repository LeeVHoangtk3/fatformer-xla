# TASK 1.1: THIẾT LẬP KẾT NỐI DRIVE 5TB, COLAB PRO & DUMMY DATASET

- **Mã nhiệm vụ**: `Task 1.1`
- **Người phụ trách**: **Thành viên A** *(Data & Infra Lead)*
- **Môi trường thực hiện**: 🖥️ **Local** (chuẩn bị dữ liệu mẫu) + ☁️ **Colab** (kiểm thử mount Drive & giải nén SSD)
- **Thời hạn hoàn thành**: Tuần 1 (Ngày 3)
- **Độ ưu tiên**: 🔴 Cực kỳ khẩn cấp (Khởi động hạ tầng cho toàn nhóm)
- **Đầu vào (Input)**:
  - Tài khoản Google AI Pro (GPU A100/L4, Google Drive 5TB trực tiếp).
  - 20 ảnh mẫu (10 ảnh real, 10 ảnh fake) kích thước $\ge 256 \times 256$.
- **Đầu ra (Output)**:
  - Cấu trúc thư mục `Fatformer/` trên Google Drive 5TB của tài khoản Google AI Pro.
  - Các tệp dữ liệu nén `.tar` sẵn sàng trên Drive 5TB (`progan_train.tar`, `progan_val.tar`, `diffusion_staging.tar`, `test_benchmark_gans.tar`, `test_benchmark_diffusion.tar`).
  - Notebook [`notebooks/00_setup_env.ipynb`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/notebooks/00_setup_env.ipynb) chứa hàm giải nén SSD NVMe.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Khởi tạo cây thư mục chuẩn trên Google Drive 5TB trực tiếp của tài khoản Google AI Pro (Mô hình Siêu Nút Hợp Nhất - Unified Super-Node, không cần chia sẻ quyền hay tạo shortcut).
2. Xây dựng hàm `extract_to_local()` giải nén tệp `.tar` sang SSD NVMe `/content/dataset_local/` của Colab (tốc độ đọc > 500 MB/s, triệt tiêu 100% nghẽn cổ chai FUSE).
3. Đóng gói và lưu trữ toàn bộ các bộ dữ liệu dưới dạng tệp `.tar` nguyên khối để Thành viên B và C có thể kiểm thử pipeline DataLoader và forward pass ngay lập tức.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Bước 1: Khởi tạo Thư mục Trên Drive 5TB
1. Đăng nhập vào tài khoản Google AI Pro, truy cập Google Drive, tạo thư mục gốc:
   ```
   Drive của tôi / Fatformer/
   ├── checkpoint/     # Tự động lưu file .pth sau mỗi epoch
   ├── datasets/       # Chứa các file .tar nguyên khối
   ├── log/            # File log, tensorboard, csv metrics
   └── pretrained/     # Chứa fatformer_4class_ckpt.pth và ViT-L-14.pt
   ```
2. Đưa các file checkpoint gốc (`fatformer_4class_ckpt.pth`, `ViT-L-14.pt`) vào `Fatformer/pretrained/`.

### Bước 2: Chuẩn Bị & Đóng Gói Dữ Liệu Dưới Dạng `.tar`
1. Đảm bảo toàn bộ ảnh bên trong mỗi tập đều tuân thủ cấu trúc phân lớp nhị phân:
   ```
   dataset_name/
   ├── 0_real/   # Ảnh thật
   └── 1_fake/   # Ảnh sinh bởi AI
   ```
2. Đóng gói thành file `.tar`:
   ```bash
   tar -cf dataset_name.tar -C dataset_name .
   ```
3. Tải các file `.tar` lên thư mục `Fatformer/datasets/` trên Drive 5TB.

### Bước 3: Viết Notebook `notebooks/00_setup_env.ipynb`
Cài đặt hàm mount và giải nén tiêu chuẩn chống nghẽn I/O:
```python
import os
import shutil
import tarfile
import time
from google.colab import drive

# 1. Mount trực tiếp tài khoản Google AI Pro 5TB
drive.mount('/content/drive')

DRIVE_HUB = "/content/drive/MyDrive/Fatformer"
LOCAL_DIR = "/content/dataset_local"

def extract_to_local(tar_filename: str, target_dir: str = LOCAL_DIR):
    """
    Sao chép tệp .tar từ Drive 5TB sang SSD máy ảo Colab và giải nén.
    Tránh hiện tượng DataLoader đọc từng file qua mạng FUSE gây nghẽn và ngắt phiên.
    """
    start_time = time.time()
    tar_path = os.path.join(DRIVE_HUB, "datasets", tar_filename)
    local_tar = os.path.join("/content", tar_filename)
    
    assert os.path.exists(tar_path), f"Không tìm thấy file tại {tar_path}"
    
    print(f"[*] Đang sao chép {tar_filename} sang SSD Colab...")
    shutil.copy(tar_path, local_tar)
    
    print(f"[*] Đang giải nén vào {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open(local_tar, "r") as tar:
        tar.extractall(target_dir)
        
    os.remove(local_tar) # Giải phóng dung lượng SSD sau khi giải nén
    elapsed = time.time() - start_time
    print(f"[✓] Giải nén thành công sau {elapsed:.2f}s! Dữ liệu sẵn sàng tại {target_dir}.")
```

---

## 3. RỦI RO & PHÒNG NGỪA
* **Rủi ro**: File `.tar` bị lỗi checksum hoặc đường dẫn sai khi mount Drive.
* **Biện pháp**: Luôn kiểm tra `os.path.exists()` trước khi copy; in thông báo dung lượng và thời gian thực hiện rõ ràng.

---

## 4. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [x] Mount Drive 5TB thành công tại `/content/drive/MyDrive/Fatformer/`.
- [x] Tạo đầy đủ 4 thư mục quản trị: `checkpoint/`, `datasets/`, `log/`, `pretrained/`.
- [x] Đóng gói và upload toàn vẹn các file `.tar` vào `Fatformer/datasets/` không có file ảnh lẻ.
- [x] Bàn giao đường dẫn và notebook cho Thành viên B & C test DataLoader.

---

## 5. KẾT QUẢ THỰC HIỆN & BÁO CÁO NGHIỆM THU (TASK COMPLETION REPORT)

- **Trạng thái thực tế**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC (DONE)`
- **Thời gian hoàn thành**: 2026-09-24
- **Đơn vị lưu trữ**: Google Drive 5TB (Google AI Pro trực tiếp)

### Cấu Trúc Thực Tế Đã Triển Khai Trên Drive:
```
Drive của tôi / Fatformer/
├── checkpoint/                             # Sẵn sàng lưu checkpoint qua các epoch
├── datasets/                               # Kho dữ liệu nén nguyên khối (.tar)
│   ├── diffusion_staging.tar       (1.80 GB)  # Dữ liệu xúc tác Diffusion (Chiến lược 3)
│   ├── progan_train.tar           (13.90 GB)  # Dữ liệu huấn luyện chính ProGAN 4-class
│   ├── progan_val.tar             (798.0 MB)  # Dữ liệu validation ProGAN
│   ├── test_benchmark_diffusion.tar(1.22 GB)  # 10 tập test Diffusion paper chuẩn
│   └── test_benchmark_gans.tar    (18.74 GB)  # 8 tập test GANs paper chuẩn
├── log/                                    # Sẵn sàng ghi nhật ký & Tensorboard
└── pretrained/                             # Lưu fatformer_4class_ckpt.pth & ViT-L-14.pt
```

### Đánh Giá Chất Lượng Nghiệm Thu:
1. **100% Tuân thủ cơ chế I/O SSD**: Không có bất kỳ file ảnh rời rạc nào trên Drive, loại bỏ hoàn toàn nguy cơ nghẽn I/O qua giao thức mạng FUSE.
2. **Vượt tiến độ quy hoạch**: Không chỉ chuẩn bị dữ liệu dummy của Tuần 1, Thành viên A đã hoàn thành nạp toàn bộ dữ liệu huấn luyện ProGAN, dữ liệu Staging Diffusion và toàn bộ dữ liệu Benchmark Clean của bài báo CVPR 2024.
3. **Bàn giao**: Toàn bộ dữ liệu tại `/content/drive/MyDrive/Fatformer/datasets/` đã sẵn sàng để Thành viên B & C sử dụng cho Task 1.2, 1.3 và 1.4.
