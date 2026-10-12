"""
FatFormer-XLA: Packaging Script for Kaggle Offline Environment
Tạo gói nén fatformer_code.zip sạch sẽ, loại trừ tệp rác để tải lên Kaggle Dataset.
"""

import os
import sys
import zipfile
from pathlib import Path

# Đảm bảo in unicode an toàn trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".agents",
    "DATASET",
    "checkpoints",
    "pretrained",
    "logs",
    "docs",
    "scratch",
    "FatFormer-main",
    ".venv",
    "venv",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".zip",
    ".tar",
    ".gz.bak",
    ".tmp",
}

EXCLUDE_FILES = {
    ".DS_Store",
    "Thumbs.db",
    "fatformer_code.zip",
}

# Các thư mục và file mục tiêu cần đưa vào gói
TARGET_PATHS = [
    "train.py",
    "configs",
    "src",
    "tools",
]


def should_include(file_path: Path, root_path: Path) -> bool:
    rel_path = file_path.relative_to(root_path)
    
    # Kiểm tra xem có nằm trong thư mục bị loại trừ không
    for part in rel_path.parts[:-1]:
        if part in EXCLUDE_DIRS:
            return False
            
    # Kiểm tra tên file
    if file_path.name in EXCLUDE_FILES:
        return False
        
    # Kiểm tra đuôi mở rộng
    if file_path.suffix.lower() in EXCLUDE_EXTENSIONS and file_path.name != "bpe_simple_vocab_16e6.txt.gz":
        return False
        
    return True


def package_code(root_dir: str = ".", output_zip: str = "fatformer_code.zip"):
    root_path = Path(root_dir).resolve()
    zip_path = root_path / output_zip
    
    print("=" * 70)
    print("FATFORMER-XLA: ĐÓNG GÓI MÃ NGUỒN CHO KAGGLE OFFLINE")
    print("=" * 70)
    print(f"Thư mục gốc:  {root_path}")
    print(f"Gói nén đầu ra: {zip_path}")
    print("-" * 70)
    
    total_files = 0
    total_uncompressed_bytes = 0
    
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for target in TARGET_PATHS:
            target_full = root_path / target
            if not target_full.exists():
                print(f"[!] Cảnh báo: Không tìm thấy mục {target}")
                continue
                
            if target_full.is_file():
                if should_include(target_full, root_path):
                    rel_name = target_full.relative_to(root_path).as_posix()
                    zipf.write(target_full, arcname=rel_name)
                    file_size = target_full.stat().st_size
                    total_uncompressed_bytes += file_size
                    total_files += 1
                    print(f" [+] File: {rel_name:<50} ({file_size:>8,} bytes)")
            elif target_full.is_dir():
                for item in sorted(target_full.rglob("*")):
                    if item.is_file() and should_include(item, root_path):
                        rel_name = item.relative_to(root_path).as_posix()
                        zipf.write(item, arcname=rel_name)
                        file_size = item.stat().st_size
                        total_uncompressed_bytes += file_size
                        total_files += 1
                        
                print(f" [✓] Đã đóng gói thư mục: {target:<20}")

    compressed_size = zip_path.stat().st_size
    ratio = (1.0 - (compressed_size / total_uncompressed_bytes)) * 100 if total_uncompressed_bytes > 0 else 0
    
    print("=" * 70)
    print("KẾT QUẢ ĐÓNG GÓI THÀNH CÔNG:")
    print(f" • Tổng số tệp đã đóng gói:   {total_files:,} files")
    print(f" • Dung lượng trước khi nén:  {total_uncompressed_bytes / (1024 * 1024):.2f} MB ({total_uncompressed_bytes:,} bytes)")
    print(f" • Dung lượng sau khi nén:    {compressed_size / (1024 * 1024):.2f} MB ({compressed_size:,} bytes)")
    print(f" • Tỷ lệ nén giảm:            {ratio:.1f}%")
    print(f" • Vị trí tệp nén:            {zip_path}")
    print("=" * 70)
    print("\n👉 BƯỚC TIẾP THEO TRÊN KAGGLE:")
    print("1. Mở Kaggle Dataset: https://www.kaggle.com/datasets/hoangleev/fatformer-code")
    print("2. Nhấn 'New Version' -> Tải tệp fatformer_code.zip vừa tạo lên.")
    print("3. Khởi động Kaggle Notebook và chạy cell giải nén.")
    print("=" * 70)


if __name__ == "__main__":
    package_code()
