"""
tools/make_degraded.py
======================
Task 2.1: Script tự động sinh bộ kiểm thử suy biến vật lý (Degraded Test Set).

Người phụ trách: Thành viên A (Data & Infra Lead)
Nâng cấp: Tối ưu đa tiến trình (Multiprocessing), cơ chế Resume/Skip an toàn, 
          bộ lọc linh hoạt và báo cáo thống kê chuyên nghiệp.

Mô tả:
    Sinh 6 biến thể suy biến vật lý từ tập test Clean gốc paper CVPR 2024:
    - Nén JPEG: Q=30 (nặng), Q=50 (trung bình), Q=70 (nhẹ)
    - Gaussian Blur: sigma=1.0, sigma=2.0
    - Downsample-Upsample: 224 -> 112 -> 224 (Bicubic)

Cấu trúc đầu ra:
    test_degraded/
    ├── jpeg_q30/
    │   ├── <subset_name>/
    │   │   ├── 0_real/   # Nhãn 0 (Ảnh thật)
    │   │   └── 1_fake/   # Nhãn 1 (Ảnh sinh bởi AI)
    │   └── ...
    ├── jpeg_q50/
    ├── jpeg_q70/
    ├── blur_s1/
    ├── blur_s2/
    └── down_up/

Ví dụ sử dụng:
    # 1. Chạy toàn bộ 6 biến thể với đa tiến trình (Colab CPU đa nhân):
    python tools/make_degraded.py \\
        --input_dir /content/dataset_local/test_clean \\
        --output_root /content/dataset_local/test_degraded \\
        --num_workers 4

    # 2. Chạy nhanh riêng 3 biến thể nén JPEG (để mở khóa ngay Task 2.3):
    python tools/make_degraded.py \\
        --input_dir /content/dataset_local/test_clean \\
        --output_root /content/dataset_local/test_degraded \\
        --degradations jpeg_q30,jpeg_q50,jpeg_q70 \\
        --num_workers 4

    # 3. Đóng gói file .tar sau khi hoàn thành:
    tar -cf /content/drive/MyDrive/Fatformer/datasets/test_degraded.tar -C /content/dataset_local test_degraded
"""

import os
import io
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageFilter
from tqdm import tqdm

# Fix encoding UTF-8 cho terminal Windows / Colab
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

ALL_DEGRADATIONS = [
    "jpeg_q30",
    "jpeg_q50",
    "jpeg_q70",
    "blur_s1",
    "blur_s2",
    "down_up"
]


# ─── 1. CÁC HÀM SUY BIẾN VẬT LÝ NGUYÊN BẢN (CORE OPERATORS) ───────────────

def apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Mô phỏng nén lượng tử JPEG qua bộ đệm byte trong RAM (không tạo file tạm)."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def apply_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Làm mờ ảnh bằng bộ lọc Gaussian Blur."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_down_up(image: Image.Image, intermediate_size=(112, 112)) -> Image.Image:
    """Mô phỏng nén resize mạng xã hội (Downsample -> Upsample Bicubic)."""
    orig_size = image.size
    down = image.resize(intermediate_size, Image.Resampling.BICUBIC)
    up = down.resize(orig_size, Image.Resampling.BICUBIC)
    return up


def apply_degradation(image: Image.Image, deg_name: str) -> Image.Image:
    """Áp dụng phép suy biến tương ứng theo tên biến thể."""
    if deg_name == "jpeg_q30":
        return apply_jpeg(image, 30)
    elif deg_name == "jpeg_q50":
        return apply_jpeg(image, 50)
    elif deg_name == "jpeg_q70":
        return apply_jpeg(image, 70)
    elif deg_name == "blur_s1":
        return apply_blur(image, 1.0)
    elif deg_name == "blur_s2":
        return apply_blur(image, 2.0)
    elif deg_name == "down_up":
        return apply_down_up(image, (112, 112))
    else:
        raise ValueError(f"Biến thể suy biến không hợp lệ: '{deg_name}'")


# ─── 2. HÀM XỬ LÝ ĐƠN ẢNH CHO TIẾN TRÌNH CON (WORKER TASK) ────────────────

def _worker_process_single_image(task):
    """
    Hàm thực thi xử lý một ảnh độc lập (picklable cho ProcessPoolExecutor).
    task = (src_file, dst_file, deg_name, force)
    """
    src_file, dst_file, deg_name, force = task

    # Cơ chế Resume / Skip an toàn: nếu file đã tồn tại và hợp lệ (> 0 bytes)
    if not force and os.path.exists(dst_file) and os.path.getsize(dst_file) > 0:
        return "skipped", None

    try:
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        with Image.open(src_file) as img:
            img = img.convert("RGB")
            deg_img = apply_degradation(img, deg_name)
            # Lưu định dạng PNG không nén lossy để tránh lỗi nén kép (double compression)
            deg_img.save(dst_file, format="PNG")
        return "processed", None
    except Exception as e:
        return "failed", f"{src_file} -> {e}"


# ─── 3. THU THẬP TẤT CẢ FILE VÀ ĐIỀU PHỐI ĐA TIẾN TRÌNH ──────────────────

def collect_image_files(input_dir: str, subsets_filter=None):
    """Quét toàn bộ danh sách tệp ảnh trong thư mục test Clean."""
    image_tasks_rel = []
    input_dir_path = Path(input_dir).resolve()

    if not input_dir_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục input: {input_dir}")

    for root, _, files in os.walk(input_dir_path):
        rel_root = os.path.relpath(root, input_dir_path)
        
        # Nếu có lọc subset, kiểm tra xem đường dẫn tương đối có thuộc subset được chọn không
        if subsets_filter:
            parts = Path(rel_root).parts
            if parts and parts[0] != '.' and not any(sub in parts for sub in subsets_filter):
                continue

        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                rel_path = os.path.relpath(os.path.join(root, file), input_dir_path)
                image_tasks_rel.append(rel_path)

    return image_tasks_rel


def process_dataset(
    input_dir: str,
    output_root: str,
    selected_degradations=None,
    selected_subsets=None,
    num_workers: int = 4,
    force: bool = False
):
    start_total_time = time.time()
    input_dir_path = Path(input_dir).resolve()
    output_root_path = Path(output_root).resolve()

    if selected_degradations is None or "all" in selected_degradations:
        active_degradations = ALL_DEGRADATIONS
    else:
        active_degradations = [d for d in selected_degradations if d in ALL_DEGRADATIONS]

    print("=" * 86)
    print("FATFORMER-XLA: PIPELINE SINH BỘ KIỂM THỬ SUY BIẾN VẬT LÝ (TASK 2.1)")
    print("=" * 86)
    print(f"• Thư mục Clean gốc:        {input_dir_path}")
    print(f"• Thư mục Degraded đầu ra:  {output_root_path}")
    print(f"• Số tiến trình (Workers):   {num_workers}")
    print(f"• Chế độ ghi đè (--force):   {force}")
    print(f"• Danh sách biến thể ({len(active_degradations)}): {active_degradations}")
    if selected_subsets:
        print(f"• Bộ lọc subsets ({len(selected_subsets)}):   {selected_subsets}")
    print("-" * 86)

    print("[*] Đang quét danh sách ảnh trong tập Clean...")
    rel_files = collect_image_files(str(input_dir_path), subsets_filter=selected_subsets)
    total_clean_images = len(rel_files)
    print(f"  -> Tìm thấy tổng cộng {total_clean_images:,} ảnh Clean hợp lệ.")

    if total_clean_images == 0:
        print("[!] CẢNH BÁO: Không tìm thấy ảnh nào trong thư mục input. Vui lòng kiểm tra lại đường dẫn!")
        return

    summary_stats = {}

    for deg_idx, deg_name in enumerate(active_degradations, 1):
        print(f"\n[{deg_idx}/{len(active_degradations)}] Đang xử lý biến thể: '{deg_name}'...")
        deg_out_dir = output_root_path / deg_name
        deg_start_time = time.time()

        tasks = []
        for rel_file in rel_files:
            src_file = str(input_dir_path / rel_file)
            # Giữ nguyên cấu trúc thư mục con và lưu định dạng .png không suy biến thêm
            dst_rel = Path(rel_file).with_suffix(".png")
            dst_file = str(deg_out_dir / dst_rel)
            tasks.append((src_file, dst_file, deg_name, force))

        processed_count = 0
        skipped_count = 0
        failed_count = 0
        error_samples = []

        if num_workers > 1:
            chunksize = max(20, min(100, len(tasks) // (num_workers * 8) + 1))
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for status, err in tqdm(
                    executor.map(_worker_process_single_image, tasks, chunksize=chunksize),
                    total=len(tasks),
                    desc=f"  Progress '{deg_name}'"
                ):
                    if status == "processed":
                        processed_count += 1
                    elif status == "skipped":
                        skipped_count += 1
                    else:
                        failed_count += 1
                        if len(error_samples) < 5 and err:
                            error_samples.append(err)
        else:
            for task in tqdm(tasks, desc=f"  Progress '{deg_name}'"):
                status, err = _worker_process_single_image(task)
                if status == "processed":
                    processed_count += 1
                elif status == "skipped":
                    skipped_count += 1
                else:
                    failed_count += 1
                    if len(error_samples) < 5 and err:
                        error_samples.append(err)

        deg_elapsed = time.time() - deg_start_time
        summary_stats[deg_name] = {
            "processed": processed_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "elapsed": deg_elapsed
        }

        print(f"  -> Hoàn thành '{deg_name}' trong {deg_elapsed:.2f}s | "
              f"Tạo mới: {processed_count:,} | Bỏ qua (đã có): {skipped_count:,} | Lỗi: {failed_count}")

        if error_samples:
            print("  [!] Mẫu lỗi tiêu biểu:")
            for err in error_samples:
                print(f"      - {err}")

    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 86)
    print("BẢNG TỔNG HỢP KẾT QUẢ SINH TẬP SUY BIẾN VẬT LÝ (TASK 2.1)")
    print("=" * 86)
    print(f"{'Biến thể':<15} | {'Tạo mới':<10} | {'Đã có (Skip)':<12} | {'Lỗi':<6} | {'Thời gian (s)':<12}")
    print("-" * 86)
    for deg_name, stats in summary_stats.items():
        print(f"{deg_name:<15} | {stats['processed']:<10,d} | {stats['skipped']:<12,d} | "
              f"{stats['failed']:<6,d} | {stats['elapsed']:<12.2f}")
    print("=" * 86)
    print(f"Tổng thời gian thực thi: {total_elapsed:.2f} giây ({total_elapsed / 60:.2f} phút)")
    print(f"[✓] Toàn bộ dữ liệu suy biến đã sẵn sàng tại: {output_root_path}")
    print("\n[HƯỚNG DẪN ĐÓNG GÓI CHO GOOGLE DRIVE 5TB]:")
    print(f"  tar -cf /content/drive/MyDrive/Fatformer/datasets/test_degraded.tar -C {output_root_path.parent} {output_root_path.name}")
    print("=" * 86)


# ─── 4. ENTRY POINT ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 2.1: Sinh bộ kiểm thử suy biến vật lý đa tiến trình (Degraded Test Set)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Thư mục chứa tập test Clean gốc (ví dụ: /content/dataset_local/test_clean)"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Thư mục lưu các biến thể suy biến (ví dụ: /content/dataset_local/test_degraded)"
    )
    parser.add_argument(
        "--degradations",
        type=str,
        default="all",
        help="Danh sách biến thể suy biến, phân cách bằng dấu phẩy. Mặc định 'all' hoặc chọn: "
             "jpeg_q30,jpeg_q50,jpeg_q70,blur_s1,blur_s2,down_up"
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="all",
        help="Danh sách tập con cần sinh (phân cách bằng dấu phẩy), mặc định 'all'"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help=f"Số tiến trình CPU xử lý song song (Mặc định: {min(4, os.cpu_count() or 1)})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ghi đè lại ảnh nếu tệp đích đã tồn tại"
    )

    args = parser.parse_args()

    deg_list = [d.strip() for d in args.degradations.split(",") if d.strip()]
    subset_list = None if args.subsets == "all" else [s.strip() for s in args.subsets.split(",") if s.strip()]

    process_dataset(
        input_dir=args.input_dir,
        output_root=args.output_root,
        selected_degradations=deg_list,
        selected_subsets=subset_list,
        num_workers=args.num_workers,
        force=args.force
    )
