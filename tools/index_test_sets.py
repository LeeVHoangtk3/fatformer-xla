"""
FatFormer-XLA: Lap chi muc SHA-256 + pHash cho cac tap test (Task 7.7).

Muc dich:
    Tao "reference hash index" cho:
      - Tap test CommunityForensics (2.000 anh)  -> loc trong luc build manifest v2
      - 18 tap test cua Task 5.1 (110.329 anh)   -> audit cap 2

Dac diem:
    - Doc TRUC TIEP tu thu muc hoac tu file .tar (khong giai nen 20 GB ra dia).
    - Tinh hash song song (process pool) de 110k anh chi mat ~10-15 phut.
    - Tu suy ra label tu duong dan (0_real/1_fake, real/fake) neu co.

Vi du:
    python tools/index_test_sets.py \
        --tar /content/drive/MyDrive/Fatformer/datasets/test_communityforensics.tar \
        --out DATASET/logs/phase7/cfe_test_hashes.json

    python tools/index_test_sets.py \
        --tar /content/drive/MyDrive/Fatformer/datasets/test_benchmark_gans.tar \
              /content/drive/MyDrive/Fatformer/datasets/test_benchmark_diffusion.tar \
        --workers 8 \
        --out DATASET/logs/phase7/test18_hashes.json.gz
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from PIL import Image

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
for p in (PROJECT_ROOT, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from hash_utils import phash_from_pil, sha256_bytes  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
REAL_DIRS = {"0_real", "real", "0", "authentic"}
FAKE_DIRS = {"1_fake", "fake", "1", "synthetic", "generated"}


def infer_label(path: str) -> Optional[int]:
    """Suy ra nhan 0/1 tu cac thanh phan duong dan."""
    parts = [p.lower() for p in path.replace("\\", "/").split("/")]
    for part in reversed(parts[:-1]):
        if part in REAL_DIRS:
            return 0
        if part in FAKE_DIRS:
            return 1
    return None


def _hash_one(payload: Tuple[str, bytes]) -> Optional[Dict]:
    """Tinh hash cho 1 anh. Tra ve None neu khong giai ma duoc."""
    path, data = payload
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            width, height = img.size
            phash = phash_from_pil(img)
    except Exception:
        return None
    return {
        "path": path,
        "sha256": sha256_bytes(data),
        "phash": phash,
        "width": int(width),
        "height": int(height),
        "bytes": len(data),
        "label": infer_label(path),
    }


def _hash_batch(batch: List[Tuple[str, bytes]]) -> List[Dict]:
    out = []
    for item in batch:
        r = _hash_one(item)
        if r is not None:
            out.append(r)
    return out


def iter_dir_images(root: str) -> Iterator[Tuple[str, bytes]]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(IMAGE_EXTS):
                full = os.path.join(dirpath, name)
                try:
                    with open(full, "rb") as f:
                        yield full, f.read()
                except OSError:
                    continue


def iter_tar_images(tar_path: str) -> Iterator[Tuple[str, bytes]]:
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf:
            if not member.isfile() or not member.name.lower().endswith(IMAGE_EXTS):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            data = f.read()
            yield f"{os.path.basename(tar_path)}::{member.name}", data


def _chunks(iterable: Iterable[Tuple[str, bytes]], size: int) -> Iterator[List[Tuple[str, bytes]]]:
    batch: List[Tuple[str, bytes]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def attach_manifest_metadata(records: List[Dict], manifest_path: str) -> None:
    """Gan model_name/architecture/real_source tu cfe_manifest.json (neu co)."""
    if not manifest_path or not os.path.exists(manifest_path):
        return
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("records", data.get("images", [])) if isinstance(data, dict) else data
    by_name: Dict[str, Dict] = {}
    by_rel: Dict[str, Dict] = {}
    for e in entries or []:
        name = str(e.get("image_name") or e.get("file_name") or e.get("path") or "")
        if not name:
            continue
        by_name[os.path.basename(name)] = e
        by_rel[name.replace("\\", "/")] = e

    for r in records:
        key = r["path"].split("::")[-1].replace("\\", "/")
        e = by_rel.get(key) or by_name.get(os.path.basename(key))
        if not e:
            continue
        for field in ("model_name", "architecture", "real_source", "subset"):
            if field in e and e[field] is not None:
                r[field] = e[field]


def main() -> None:
    parser = argparse.ArgumentParser(description="Index SHA-256 + pHash for test sets")
    parser.add_argument("--dirs", nargs="*", default=[], help="Thu muc anh can quet")
    parser.add_argument("--tar", nargs="*", default=[], help="File .tar chua anh can quet")
    parser.add_argument("--manifest", type=str, default="", help="cfe_manifest.json de gan model_name")
    parser.add_argument("--out", type=str, required=True, help="File JSON (ho tro .gz)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=0, help="Gioi han de test nhanh")
    args = parser.parse_args()

    if not args.dirs and not args.tar:
        parser.error("Can it nhat --dirs hoac --tar")

    t0 = time.time()
    streams = []
    for d in args.dirs:
        if os.path.isdir(d):
            streams.append((d, iter_dir_images(d)))
        else:
            print(f"[!] Bo qua thu muc khong ton tai: {d}")
    for t in args.tar:
        if os.path.isfile(t):
            streams.append((t, iter_tar_images(t)))
        else:
            print(f"[!] Bo qua tar khong ton tai: {t}")

    records: List[Dict] = []
    skipped = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for source, stream in streams:
            print(f"[*] Quet: {source}")
            n_source = 0
            for batch in _chunks(stream, args.batch_size):
                results = list(pool.map(_hash_one, batch))
                for r in results:
                    if r is None:
                        skipped += 1
                    else:
                        records.append(r)
                        n_source += 1
                done = len(records)
                if done % (args.batch_size * 10) < args.batch_size:
                    rate = done / max(1e-6, time.time() - t0)
                    print(f"    ... {done:,} anh ({rate:.0f} anh/s, loi={skipped})", flush=True)
                if args.max_images and len(records) >= args.max_images:
                    break
            print(f"    -> {n_source:,} anh hop le tu {os.path.basename(source)}")
            if args.max_images and len(records) >= args.max_images:
                records = records[:args.max_images]
                break

    if args.manifest:
        attach_manifest_metadata(records, args.manifest)

    for r in records:
        if r.get("label") is None:
            r["label"] = 1  # test set CFE: mặc định coi là fake nếu không suy ra được

    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_images": len(records),
            "n_skipped_unreadable": skipped,
            "sources": [os.path.abspath(s) for s, _ in streams],
            "phash_bits": 64,
            "tool": "tools/index_test_sets.py",
        },
        "images": records,
    }

    from hash_utils import save_hash_index

    save_hash_index(payload, args.out)
    elapsed = time.time() - t0
    print("=" * 84)
    print(f"XONG: {len(records):,} anh | bo qua {skipped:,} | {elapsed/60:.1f} phut")
    print(f"-> {args.out}")
    print("=" * 84)


if __name__ == "__main__":
    main()
