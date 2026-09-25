# TASK 2.1: XÂY DỰNG BỘ KIỂM THỬ SUY BIẾN VẬT LÝ (DEGRADED TEST SET)

- **Mã nhiệm vụ**: `Task 2.1`
- **Người phụ trách**: **Thành viên A** *(Data & Infra Lead)*
- **Môi trường thực hiện**: 🖥️ **Local** (phát triển script) + ☁️ **Colab CPU/T4** (chạy sinh hàng loạt ảnh tốc độ cao)
- **Thời hạn hoàn thành**: Tuần 2 (Ngày 10)
- **Độ ưu tiên**: 🟠 Cao (Dữ liệu nền tảng để đo độ sụt giảm của mô hình gốc)
- **Đầu vào (Input)**:
  - 18 tập test Clean gốc paper CVPR 2024 (8 GANs + 10 Diffusion).
  - Quy chuẩn tham số suy biến: JPEG ($Q \in \{30, 50, 70\}$), Blur ($\sigma \in \{1.0, 2.0\}$), Down-Up ($224 \rightarrow 112 \rightarrow 224$).
- **Đầu ra (Output)**:
  - Script [`tools/make_degraded.py`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/tools/make_degraded.py).
  - Tệp nén `datasets/test_degraded.tar` lưu trữ trên Google Drive 5TB nhóm.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Hiện thực hóa script tự động hóa sinh 3 họ biến dạng vật lý phổ biến nhất trên các nền tảng mạng xã hội:
   - **Nén lượng tử JPEG**: $Q=30$ (nén rất nặng), $Q=50$ (nén trung bình), $Q=70$ (nén nhẹ).
   - **Làm mờ Gauss (Gaussian Blur)**: $\sigma=1.0$ và $\sigma=2.0$.
   - **Downsample-Upsample**: Thu nhỏ $224 \rightarrow 112$ rồi phóng to lại $224$ bằng nội suy Bicubic.
2. Giữ nguyên 100% cấu trúc phân loại nhị phân (`0_real/`, `1_fake/`) và tên file để phục vụ tính toán chính xác các chỉ số ACC, AP, AUC.
3. Đóng gói toàn bộ tập kiểm thử thành tệp `test_degraded.tar` lưu trữ trên Google Drive 5TB.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Bước 1: Viết script `tools/make_degraded.py`
```python
import os
import io
import argparse
from PIL import Image, ImageFilter
from tqdm import tqdm

def apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Mô phỏng nén lượng tử JPEG qua bộ đệm bộ nhớ."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()

def apply_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Làm mờ ảnh bằng bộ lọc Gaussian Blur."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))

def apply_down_up(image: Image.Image, intermediate_size=(112, 112)) -> Image.Image:
    """Mô phỏng resize mạng xã hội (Downsample -> Upsample Bicubic)."""
    orig_size = image.size
    down = image.resize(intermediate_size, Image.Resampling.BICUBIC)
    up = down.resize(orig_size, Image.Resampling.BICUBIC)
    return up

def process_dataset(input_dir: str, output_root: str):
    degradations = {
        "jpeg_q30": lambda img: apply_jpeg(img, 30),
        "jpeg_q50": lambda img: apply_jpeg(img, 50),
        "jpeg_q70": lambda img: apply_jpeg(img, 70),
        "blur_s1": lambda img: apply_blur(img, 1.0),
        "blur_s2": lambda img: apply_blur(img, 2.0),
        "down_up": lambda img: apply_down_up(img, (112, 112))
    }
    
    for deg_name, deg_func in degradations.items():
        deg_out_dir = os.path.join(output_root, deg_name)
        print(f"[*] Đang xử lý biến thể: {deg_name} -> {deg_out_dir}")
        
        for root, _, files in os.walk(input_dir):
            for file in tqdm(files, desc=deg_name):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rel_path = os.path.relpath(root, input_dir)
                    target_dir = os.path.join(deg_out_dir, rel_path)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(target_dir, os.path.splitext(file)[0] + ".jpg")
                    
                    try:
                        with Image.open(src_file) as img:
                            img = img.convert("RGB")
                            deg_img = deg_func(img)
                            deg_img.save(dst_file, "JPEG", quality=95)
                    except Exception as e:
                        print(f"[!] Lỗi khi xử lý {src_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Thư mục tập test Clean")
    parser.add_argument("--output_root", type=str, required=True, help="Thư mục xuất ảnh suy biến")
    args = parser.parse_args()
    process_dataset(args.input_dir, args.output_root)
```

### Bước 2: Chạy Trên Colab và Đóng Gói `.tar`
Chạy trên môi trường Colab CPU/T4 để xử lý đa luồng tốc độ cao:
```bash
python tools/make_degraded.py \
    --input_dir /content/dataset_local/test_clean \
    --output_root /content/dataset_local/test_degraded

# Đóng gói và lưu trực tiếp lên Drive 5TB
cd /content/dataset_local/
tar -cf test_degraded.tar test_degraded
cp test_degraded.tar /content/drive/MyDrive/FatFormer_Hub/datasets/
```

---

## 3. RỦI RO & PHÒNG NGỪA
* **Rủi ro**: File ảnh gốc bị lỗi định dạng RGBA/Grayscale dẫn đến crash script.
* **Biện pháp**: Ép chuyển đổi `img.convert("RGB")` và bọc khối `try...except` xử lý từng ảnh đơn lẻ.

---

## 4. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Tệp `test_degraded.tar` xuất hiện trên Drive 5TB, kích thước đầy đủ không bị hỏng.
- [ ] Mở ngẫu nhiên ảnh trong `jpeg_q30/`, `blur_s1/`, `down_up/`: ảnh hiển thị rõ đặc trưng biến dạng thị giác, giữ nguyên cấu trúc nhãn `0_real/` và `1_fake/`.
- [ ] Bàn giao cho Thành viên C để chạy benchmark mốc sàn tại Task 2.3.
