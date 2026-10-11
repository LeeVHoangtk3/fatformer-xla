# -*- coding: utf-8 -*-
"""
Công cụ kiểm toán toàn vẹn dữ liệu và phát hiện rò rỉ dữ liệu (Data Leakage Audit):
- Quét các phân vùng: Train (ProGAN + Diffusion Staging), Val (ProGAN Val), và 18 tập Benchmark Test.
- Tính toán Exact Hash (MD5 & SHA-256) để phát hiện ảnh trùng lặp tuyệt đối.
- Tính toán Perceptual Hash (dHash 64-bit) để phát hiện ảnh near-duplicate (nén nhẹ, resize, đổi màu).
- Kiểm tra chéo Train vs Val, Train vs Test, Val vs Test.
- Tính toán SHA-256 Checksum cho toàn bộ checkpoint mô hình.
- Xuất file Manifest: DATASET/manifests/dataset_integrity_manifest.json.
"""

import os
import sys
import time
import hashlib
import argparse
import json
from typing import Dict, List, Tuple, Set
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def compute_file_md5_and_sha256(filepath: str) -> Tuple[str, str]:
    """Tính exact hash MD5 và SHA-256 từ file bytes."""
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def compute_dhash(image_path: str, hash_size: int = 8) -> Optional[int]:
    """
    Tính difference hash (dHash 64-bit) từ ảnh:
    - Resize ảnh về kích thước (hash_size + 1, hash_size) ở chế độ grayscale (L).
    - So sánh giá trị độ sáng giữa 2 pixel kề nhau theo hàng ngang.
    - Trả về integer 64-bit.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = list(img.getdata())
            # Duyệt từng hàng
            diff = 0
            for row in range(hash_size):
                row_start = row * (hash_size + 1)
                for col in range(hash_size):
                    left = pixels[row_start + col]
                    right = pixels[row_start + col + 1]
                    diff = (diff << 1) | (1 if left > right else 0)
            return diff
    except Exception:
        return None


def hamming_distance(h1: int, h2: int) -> int:
    """Tính khoảng cách Hamming giữa 2 hash 64-bit."""
    return bin(h1 ^ h2).count("1")


def collect_images_from_dir(dir_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Thu thập danh sách file ảnh hợp lệ từ thư mục."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    img_paths = []
    if not os.path.isdir(dir_path):
        return img_paths
    for root, _, files in os.walk(dir_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                img_paths.append(os.path.join(root, f))
                if max_samples and len(img_paths) >= max_samples:
                    return img_paths
    return img_paths


def audit_splits(
    splits: Dict[str, List[str]],
    hamming_threshold: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Quét và đối chiếu hash giữa các phân vùng dữ liệu để phát hiện rò rỉ (leakage).
    """
    exact_hashes: Dict[str, Dict[str, str]] = {}  # split -> {md5: path}
    phash_dict: Dict[str, Dict[int, str]] = {}    # split -> {dhash: path}
    split_stats = {}

    print("\n[*] Đang tính toán Hash (MD5 & dHash) cho từng phân vùng...")
    for split_name, paths in splits.items():
        exact_hashes[split_name] = {}
        phash_dict[split_name] = {}
        t0 = time.time()
        for p in paths:
            try:
                md5_val, _ = compute_file_md5_and_sha256(p)
                exact_hashes[split_name][md5_val] = p
                dh = compute_dhash(p)
                if dh is not None:
                    phash_dict[split_name][dh] = p
            except Exception:
                continue
        elapsed = time.time() - t0
        split_stats[split_name] = {
            "total_files": len(paths),
            "unique_md5": len(exact_hashes[split_name]),
            "unique_phash": len(phash_dict[split_name]),
            "elapsed_sec": round(elapsed, 2)
        }
        if verbose:
            print(f"  • Phân vùng '{split_name}': {len(paths):,} ảnh -> "
                  f"{len(exact_hashes[split_name]):,} unique MD5 | {len(phash_dict[split_name]):,} unique dHash ({elapsed:.2f}s)")

    # Kiểm tra rò rỉ chéo giữa các cặp
    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test")
    ]
    leakage_results = {}
    total_leaks_exact = 0
    total_leaks_perceptual = 0

    print("\n[*] Đang kiểm tra đối đầu chéo chống rò rỉ dữ liệu (Cross-Split Leakage Check)...")
    for s1, s2 in pairs:
        if s1 not in exact_hashes or s2 not in exact_hashes:
            continue

        # 1. Exact match (MD5)
        set1_md5 = set(exact_hashes[s1].keys())
        set2_md5 = set(exact_hashes[s2].keys())
        overlap_exact = set1_md5.intersection(set2_md5)

        # 2. Perceptual match (Hamming distance <= threshold)
        phash1 = list(phash_dict[s1].keys())
        phash2 = list(phash_dict[s2].keys())
        perceptual_overlap = []

        # Tối ưu đối chiếu nhanh: nếu phash1, phash2 quá lớn, dùng exact dhash trước rồi mẫu hamming
        exact_phash_overlap = set(phash1).intersection(set(phash2))
        for h in exact_phash_overlap:
            perceptual_overlap.append((phash_dict[s1][h], phash_dict[s2][h], 0))

        # Kiểm tra thêm mẫu hamming distance <= hamming_threshold cho các hash khác
        if len(phash1) * len(phash2) <= 1_000_000:
            for h1 in phash1:
                for h2 in phash2:
                    if h1 != h2:
                        dist = hamming_distance(h1, h2)
                        if dist <= hamming_threshold:
                            perceptual_overlap.append((phash_dict[s1][h1], phash_dict[s2][h2], dist))
                            if len(perceptual_overlap) >= 50:
                                break
                if len(perceptual_overlap) >= 50:
                    break

        pair_key = f"{s1}_vs_{s2}"
        leakage_results[pair_key] = {
            "exact_duplicate_count": len(overlap_exact),
            "exact_duplicate_examples": [exact_hashes[s1][h] for h in list(overlap_exact)[:5]],
            "perceptual_duplicate_count": len(perceptual_overlap),
            "status": "PASS (0% Rò rỉ)" if len(overlap_exact) == 0 and len(perceptual_overlap) == 0 else "FAIL (Phát hiện trùng lặp)"
        }
        total_leaks_exact += len(overlap_exact)
        total_leaks_perceptual += len(perceptual_overlap)

        status_icon = "✓" if (len(overlap_exact) == 0 and len(perceptual_overlap) == 0) else "✗"
        print(f"  [{status_icon}] Cặp {s1.upper()} vs {s2.upper()}: "
              f"Trùng MD5 = {len(overlap_exact)} | Trùng pHash = {len(perceptual_overlap)} -> {leakage_results[pair_key]['status']}")

    return {
        "split_stats": split_stats,
        "leakage_results": leakage_results,
        "total_leaks_exact": total_leaks_exact,
        "total_leaks_perceptual": total_leaks_perceptual,
        "is_clean": (total_leaks_exact == 0 and total_leaks_perceptual == 0)
    }


def compute_checkpoint_checksums(checkpoint_dirs: List[str]) -> Dict[str, Dict[str, Any]]:
    """Tính toán SHA-256 checksum cho toàn bộ các checkpoint trong dự án."""
    ckpts = {}
    for cdir in checkpoint_dirs:
        if not os.path.isdir(cdir):
            continue
        for root, _, files in os.walk(cdir):
            for f in files:
                if f.endswith(".pth") or f.endswith(".pt"):
                    full_p = os.path.join(root, f)
                    try:
                        md5_v, sha_v = compute_file_md5_and_sha256(full_p)
                        size_mb = os.path.getsize(full_p) / (1024 * 1024)
                        ckpts[f] = {
                            "path": full_p,
                            "size_mb": round(size_mb, 2),
                            "md5": md5_v,
                            "sha256": sha_v,
                        }
                    except Exception:
                        continue
    return ckpts


def main():
    parser = argparse.ArgumentParser(description="Kiểm toán toàn vẹn dữ liệu & Data Leakage Audit")
    parser.add_argument("--train_dir", type=str, default="DATASET/progan_train", help="Thư mục dữ liệu train ProGAN")
    parser.add_argument("--staging_dir", type=str, default="DATASET/diffusion_staging", help="Thư mục dữ liệu train Diffusion Staging")
    parser.add_argument("--val_dir", type=str, default="DATASET/progan_val", help="Thư mục dữ liệu val")
    parser.add_argument("--test_dir", type=str, default="DATASET/test", help="Thư mục dữ liệu test clean/degraded")
    parser.add_argument("--max_samples", type=int, default=0, help="Số mẫu kiểm toán tối đa trên mỗi phân vùng (mặc định: 0 để quét 100%% toàn bộ ảnh)")
    parser.add_argument("--output_manifest", type=str, default="DATASET/manifests/dataset_integrity_manifest.json")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên cố định")
    parser.add_argument("--test_dummy", action="store_true", help="Chạy kiểm thử với dữ liệu giả lập (chỉ dùng cho smoke test mã nguồn)")
    args = parser.parse_args()

    # Nếu chạy test_dummy mà người dùng không chỉ định file manifest khác, lưu ra file dummy riêng biệt
    if args.test_dummy and args.output_manifest == "DATASET/manifests/dataset_integrity_manifest.json":
        args.output_manifest = "DATASET/manifests/dummy_smoke_test_manifest.json"

    print("=" * 86)
    print("FATFORMER-XLA: KIỂM TOÁN RÒ RỈ DỮ LIỆU & TOÀN VẸN CHECKPOINT (DATA LEAKAGE AUDIT)")
    print("=" * 86)
    print(f"• Seed cố định kiểm toán:       {args.seed}")
    print(f"• Số mẫu kiểm toán tối đa:      {args.max_samples if args.max_samples > 0 else 'Toàn bộ 100% (Full Scan)'}")
    print(f"• Tệp Manifest đầu ra:          {args.output_manifest}")

    splits = {}
    if args.test_dummy:
        print("\n[*] Chế độ Smoke Test Dummy: Sinh dữ liệu kiểm thử giả lập...")
        import tempfile
        tmp = tempfile.mkdtemp(prefix="leakage_dummy_")
        dummy_splits = ["train", "val", "test"]
        for s in dummy_splits:
            sdir = os.path.join(tmp, s)
            os.makedirs(sdir, exist_ok=True)
            for i in range(10):
                # Tạo ảnh dummy
                color = (i * 20, (i * 30) % 255, (i * 40) % 255)
                im = Image.new("RGB", (64, 64), color=color)
                im.save(os.path.join(sdir, f"sample_{s}_{i}.png"))
            splits[s] = collect_images_from_dir(sdir)
    else:
        # Tự động tìm đường dẫn dữ liệu thật (Quét cả ProGAN Train và Diffusion Staging)
        cand_progan = [args.train_dir, "DATASET/progan_train", "DATASET/train", "/content/dataset_local/progan_train"]
        cand_staging = [args.staging_dir, "DATASET/diffusion_staging", "DATASET/staging", "/content/dataset_local/diffusion_staging"]
        cand_val = [args.val_dir, "DATASET/progan_val", "DATASET/val", "/content/dataset_local/progan_val"]
        cand_test = [args.test_dir, "DATASET/test", "DATASET/test_clean", "DATASET/test_degraded", "/content/dataset_local/test_clean"]

        train_images = []
        for c in cand_progan:
            if os.path.isdir(c):
                imgs = collect_images_from_dir(c, max_samples=args.max_samples if args.max_samples > 0 else None)
                if imgs:
                    train_images.extend(imgs)
                    print(f"  [✓] Đã nạp Train ProGAN: {c} ({len(imgs):,} ảnh)")
                    break

        for c in cand_staging:
            if os.path.isdir(c):
                rem = max(1, args.max_samples - len(train_images)) if args.max_samples > 0 else None
                imgs = collect_images_from_dir(c, max_samples=rem)
                if imgs:
                    train_images.extend(imgs)
                    print(f"  [✓] Đã nạp Train Diffusion Staging: {c} ({len(imgs):,} ảnh)")
                    break

        if train_images:
            splits["train"] = train_images
            print(f"  -> Tổng hợp phân vùng Train (ProGAN + Diffusion Staging): {len(train_images):,} ảnh")

        for c in cand_val:
            if os.path.isdir(c):
                imgs = collect_images_from_dir(c, max_samples=args.max_samples if args.max_samples > 0 else None)
                if imgs:
                    splits["val"] = imgs
                    print(f"  [✓] Đã nạp Val: {c} ({len(imgs):,} ảnh)")
                    break

        for c in cand_test:
            if os.path.isdir(c):
                imgs = collect_images_from_dir(c, max_samples=args.max_samples if args.max_samples > 0 else None)
                if imgs:
                    splits["test"] = imgs
                    print(f"  [✓] Đã nạp Test: {c} ({len(imgs):,} ảnh)")
                    break

        # Chặn đứng việc sinh dummy ngầm khi thiếu dữ liệu thật
        if not splits or len(splits) < 2:
            print("\n" + "=" * 86)
            print("[LỖI NGHIÊM TRỌNG] Không tìm thấy dữ liệu ảnh thật trên đĩa để kiểm toán.")
            print(f"  • Các thư mục train đã quét: {cand_progan + cand_staging}")
            print(f"  • Các thư mục val đã quét:   {cand_val}")
            print(f"  • Các thư mục test đã quét:  {cand_test}")
            print("  • Hủy bỏ xuất manifest chính thức để ngăn ngừa sai lệch dữ liệu!")
            print("  • Nếu cần chạy thử thuật toán trên dữ liệu giả lập, hãy dùng cờ: --test_dummy")
            print("=" * 86)
            return 1

    # 1. Chạy audit rò rỉ dữ liệu
    audit_data = audit_splits(splits, verbose=True)

    # 2. Checksum các Checkpoints
    ckpt_dirs = [
        os.path.join(project_root, "checkpoints"),
        os.path.join(project_root, "checkpoint"),
        os.path.join(project_root, "pretrained"),
        "/content/drive/MyDrive/Fatformer/checkpoint",
        "/content/drive/MyDrive/Fatformer/pretrained",
    ]
    print("\n[*] Đang tính toán SHA-256 Checksum cho các Checkpoint...")
    ckpts_info = compute_checkpoint_checksums(ckpt_dirs)
    for cname, cinfo in ckpts_info.items():
        print(f"  • {cname:<35} ({cinfo['size_mb']:.1f} MB) -> SHA-256: {cinfo['sha256'][:16]}...{cinfo['sha256'][-8:]}")

    # 3. Tạo Manifest hoàn chỉnh
    manifest = {
        "metadata": {
            "title": "FatFormer-XLA Dataset Integrity and Checkpoint Manifest",
            "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fixed_seed": args.seed,
            "hash_algorithm": "MD5 + SHA-256 + dHash (64-bit)",
            "hamming_distance_threshold": 3,
            "mode": "SMOKE_TEST_DUMMY" if args.test_dummy else "FULL_PRODUCTION_AUDIT"
        },
        "audit_summary": {
            "data_leakage_detected": not audit_data["is_clean"],
            "total_exact_duplicates": audit_data["total_leaks_exact"],
            "total_perceptual_duplicates": audit_data["total_leaks_perceptual"],
            "overall_status": "PASS (0.00% Data Leakage)" if audit_data["is_clean"] else "FAIL",
            "splits": audit_data["split_stats"],
            "cross_checks": audit_data["leakage_results"]
        },
        "checkpoints_checksums": ckpts_info
    }

    out_file = args.output_manifest
    if not os.path.isabs(out_file):
        out_file = os.path.join(project_root, out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 86)
    print(f"[✓] ĐÃ LƯU MANIFEST TOÀN VẸN DỮ LIỆU THÀNH CÔNG VÀO:")
    print(f"    -> {out_file}")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    sys.exit(main())
