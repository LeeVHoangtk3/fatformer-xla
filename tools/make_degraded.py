"""
Task 2.1: Script tự động sinh bộ kiểm thử suy biến vật lý (Degraded Test Set).

Người phụ trách: Thành viên A (Data & Infra Lead)
Mô tả:
    Sinh tự động 6 biến thể suy biến vật lý từ tập test Clean gốc paper CVPR 2024:
    - Nén JPEG: Q=30 (nặng), Q=50 (trung bình), Q=70 (nhẹ)
    - Gaussian Blur: sigma=1.0, sigma=2.0
    - Downsample-Upsample: 224 -> 112 -> 224 (Bicubic)

Sử dụng:
    python tools/make_degraded.py \\
        --input_dir /content/dataset_local/test_clean \\
        --output_root /content/dataset_local/test_degraded

Cấu trúc đầu ra:
    test_degraded/
    ├── jpeg_q30/
    │   ├── <dataset_name>/
    │   │   ├── 0_real/   # Nhãn 0 (Ảnh thật)
    │   │   └── 1_fake/   # Nhãn 1 (Ảnh sinh bởi AI)
    │   └── ...
    ├── jpeg_q50/
    ├── jpeg_q70/
    ├── blur_s1/
    ├── blur_s2/
    └── down_up/
"""

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
    parser = argparse.ArgumentParser(
        description="Task 2.1: Sinh bộ kiểm thử suy biến vật lý (Degraded Test Set)"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Thư mục tập test Clean (chứa các thư mục con dataset)")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Thư mục xuất ảnh suy biến")
    args = parser.parse_args()
    process_dataset(args.input_dir, args.output_root)
