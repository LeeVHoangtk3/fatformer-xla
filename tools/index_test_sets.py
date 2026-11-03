"""
FatFormer-XLA: Lap chi muc SHA-256 + pHash cho cac tap test (Task 7.7).

Muc dich:
    Tao "reference hash index" cho:
      - Tap test CommunityForensics (2.000 anh)  -> loc trong luc build manifest v2
      - 18 tap test cua Task 5.1 (82.003 anh) + cac nhom GAN phu trong tar
        (28.326 anh theo bao cao Drive) -> audit toan bo 110.329 anh, cap 2

Dac diem:
    - Doc TRUC TIEP tu thu muc hoac tu file .tar (khong giai nen 20 GB ra dia).
    - Tinh hash song song (process pool) de 110k anh chi mat ~10-15 phut.
    - Tu suy ra label tu duong dan (0_real/1_fake, real/fake) neu co.

Vi du:
    python tools/index_test_sets.py --role cfe \
        --tar /content/drive/MyDrive/Fatformer/datasets/test_communityforensics.tar \
        --out DATASET/logs/phase7/cfe_test_hashes.json

    python tools/index_test_sets.py --role test18 \
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
import posixpath
import re
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from PIL import Image

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
for p in (PROJECT_ROOT, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from hash_utils import phash_from_pil, sha256_bytes  # noqa: E402
from build_generator_denylist import EXPECTED_TASK51_TESTS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".ppm", ".pgm", ".pnm")
REAL_DIRS = {"0_real", "real", "0", "authentic", "nature", "natural", "real_images"}
FAKE_DIRS = {"1_fake", "fake", "1", "synthetic", "generated", "ai", "fake_images"}
# The raw CNNDetection GAN archive has 13 groups; Task 5.1 evaluates eight.
# Keep the five supplementary groups in the isolation reference as well.
# See docs/bao_cao/overall/bao_cao_kiem_toan_toan_bo_datasets_drive_5tb.md III.5.
SUPPLEMENTAL_TEST_SETS = ("crn", "imle", "san", "seeingdark", "whichfaceisreal")


def _folder_key(name):
    return re.sub(r"[\s_-]+", "", name.strip().lower())


def _path_parts(path):
    return [p for p in str(path).split("::")[-1].replace("\\", "/").split("/") if p not in ("", ".")]


def _is_image_path(path):
    parts = _path_parts(path)
    return bool(parts) and "__MACOSX" not in parts and parts[-1].lower().endswith(IMAGE_EXTS)


def infer_test_set(path):
    canonical = {_folder_key(p): p for p in EXPECTED_TASK51_TESTS}
    matches = {canonical[_folder_key(p)] for p in _path_parts(path)[:-1] if _folder_key(p) in canonical}
    return next(iter(matches)) if len(matches) == 1 else None


def infer_supplemental_set(path):
    canonical = {_folder_key(p): p for p in SUPPLEMENTAL_TEST_SETS}
    matches = {canonical[_folder_key(p)] for p in _path_parts(path)[:-1] if _folder_key(p) in canonical}
    return next(iter(matches)) if len(matches) == 1 else None


def infer_label(path: str) -> Optional[int]:
    """Suy ra nhan 0/1 tu cac thanh phan duong dan."""
    parts = _path_parts(path)
    real = {_folder_key(p) for p in REAL_DIRS}
    fake = {_folder_key(p) for p in FAKE_DIRS}
    for part in reversed(parts[:-1]):
        if _folder_key(part) in real:
            return 0
        if _folder_key(part) in fake:
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
        "test_set": infer_test_set(path),
        "supplemental_set": infer_supplemental_set(path),
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
            if _is_image_path(os.path.join(dirpath, name)):
                full = os.path.join(dirpath, name)
                try:
                    with open(full, "rb") as f:
                        yield full, f.read()
                except OSError as exc:
                    raise RuntimeError(f"Cannot read reference image: {full}") from exc


def iter_tar_images(tar_path: str) -> Iterator[Tuple[str, bytes]]:
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf:
            if not member.isfile() or not _is_image_path(member.name):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            data = f.read()
            yield f"{os.path.basename(tar_path)}::{member.name}", data


def inspect_layout(dirs, tars, role):
    """Inspect headers before decoding: report exact paths that cannot be classified."""
    counts, supplemental_counts, examples, n_images = {}, {}, [], 0

    def inspect(path):
        nonlocal n_images
        if not _is_image_path(path):
            return
        n_images += 1
        label, subset = infer_label(path), infer_test_set(path)
        supplemental = infer_supplemental_set(path)
        if role == "test18" and (label is None or (subset is None and supplemental is None)
                                 or (subset is not None and supplemental is not None)):
            if len(examples) < 10:
                examples.append({"path": path, "label": label, "test_set": subset, "supplemental_set": supplemental})
        if subset or supplemental:
            target = counts if subset else supplemental_counts
            bucket = target.setdefault(subset or supplemental, {"real": 0, "fake": 0, "unknown": 0})
            bucket[{0: "real", 1: "fake"}.get(label, "unknown")] += 1

    for root in dirs:
        for dirpath, _, names in os.walk(root):
            for name in names:
                inspect(os.path.join(dirpath, name))
    for path in tars:
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                if member.isfile():
                    inspect(f"{os.path.basename(path)}::{member.name}")
    missing = sorted(set(EXPECTED_TASK51_TESTS) - set(counts)) if role == "test18" else []
    missing_classes = [s for s, c in counts.items() if not c["real"] or not c["fake"]] if role == "test18" else []
    return {"role": role, "n_images": n_images, "by_test_set": counts,
            "by_supplemental_set": supplemental_counts,
            "unknown_examples": examples, "missing_test_sets": missing, "missing_classes": missing_classes,
            "complete": n_images > 0 and not examples and not missing and not missing_classes}


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
    if not manifest_path:
        return
    if not os.path.isfile(manifest_path):
        raise ValueError(f"Reference manifest missing: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    entries = data.get("records", data.get("images", [])) if isinstance(data, dict) else data
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Reference manifest has no records: {manifest_path}")

    def normalize(name):
        return posixpath.normpath(str(name).split("::")[-1].replace("\\", "/")).lstrip("/")

    by_name = {}
    by_rel: Dict[str, Dict] = {}
    for e in entries or []:
        if not isinstance(e, dict):
            raise ValueError("Reference manifest records must be objects")
        name = next((e.get(k) for k in ("rel_path", "relative_path", "path", "file_path", "file_name", "image_name")
                     if e.get(k)), "")
        if not name:
            raise ValueError(f"Reference record lacks an image path; keys={sorted(e)}")
        name = normalize(name)
        if name in by_rel:
            raise ValueError(f"Duplicate reference manifest path: {name}")
        by_name.setdefault(posixpath.basename(name), []).append(e)
        by_rel[name] = e

    matched_entries = set()
    for r in records:
        key = normalize(r["path"])
        # Accept the tar's enclosing directory and old Colab absolute roots in
        # either direction, but never choose arbitrarily between equal names.
        matches = [e for rel, e in by_rel.items()
                   if key == rel or key.endswith("/" + rel) or rel.endswith("/" + key)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous reference manifest path: {key}")
        e = matches[0] if matches else None
        if e is None:
            named = by_name.get(posixpath.basename(key), [])
            if len(named) == 1:
                e = named[0]
            elif len(named) > 1:
                raise ValueError(f"Ambiguous reference filename: {key}")
        if not e:
            raise ValueError(f"Image absent from reference manifest: {key}")
        if id(e) in matched_entries:
            raise ValueError(f"Multiple archived images match one manifest record: {key}")
        matched_entries.add(id(e))
        manifest_label = e.get("label")
        if manifest_label is not None and r.get("label") is not None and int(manifest_label) != r["label"]:
            raise ValueError(f"Reference label conflicts with image directory: {key}")
        for field in ("model_name", "generator", "label", "architecture", "real_source", "subset"):
            if field in e and e[field] is not None:
                r[field] = e[field]
        if r.get("label") is not None:
            r["label"] = int(r["label"])
        # The CFE preparation notebook records the upstream hash, then saves a
        # PNG. Preserve that provenance without overwriting the hash of this PNG.
        if e.get("sha256"):
            r["source_sha256"] = e["sha256"]
    if len(matched_entries) != len(entries):
        raise ValueError(f"Reference manifest coverage mismatch: matched {len(matched_entries)}/{len(entries)} records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index SHA-256 + pHash for test sets")
    parser.add_argument("--dirs", nargs="*", default=[], help="Thu muc anh can quet")
    parser.add_argument("--tar", nargs="*", default=[], help="File .tar chua anh can quet")
    parser.add_argument("--manifest", type=str, default="", help="cfe_manifest.json de gan model_name")
    parser.add_argument("--out", type=str, required=True, help="File JSON (ho tro .gz)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=0, help="Gioi han de test nhanh")
    parser.add_argument("--role", choices=("cfe", "test18"), required=True)
    parser.add_argument("--inspect-only", action="store_true", help="Validate archive layout without hashing images")
    parser.add_argument("--layout-report", default="", help="Save archive layout counts and unrecognized paths")
    args = parser.parse_args()

    if not args.dirs and not args.tar:
        parser.error("Can it nhat --dirs hoac --tar")
    if args.workers < 1 or args.batch_size < 1:
        parser.error("--workers and --batch-size must be positive")
    if args.role == "cfe" and not args.manifest:
        parser.error("--role cfe requires --manifest with per-image labels and generators")
    if args.manifest and not os.path.isfile(args.manifest):
        parser.error(f"Reference manifest missing: {args.manifest}")

    t0 = time.time()
    streams = []
    for d in args.dirs:
        if os.path.isdir(d):
            streams.append((d, iter_dir_images(d)))
        else:
            parser.error(f"Reference directory missing: {d}")
    for t in args.tar:
        if os.path.isfile(t):
            streams.append((t, iter_tar_images(t)))
        else:
            parser.error(f"Reference tar missing: {t}")

    if args.role == "test18" or args.inspect_only:
        layout = inspect_layout(args.dirs, args.tar, args.role)
        from hash_utils import save_hash_index
        layout_path = args.layout_report or args.out + ".layout.json"
        save_hash_index(layout, layout_path)
        print("LAYOUT:", json.dumps(layout, ensure_ascii=False), flush=True)
        if not layout["complete"]:
            parser.error(f"Unsupported/incomplete {args.role} archive layout. See {layout_path}; "
                         f"unknown_paths={layout['unknown_examples'][:3]}, "
                         f"missing_sets={layout['missing_test_sets']}, missing_classes={layout['missing_classes']}. "
                         "No labels or generators were guessed; hashing has not started.")
        if args.inspect_only:
            return

    records: List[Dict] = []
    skipped = 0
    # A genuine serial mode is useful in restricted runtimes and for diagnostics.
    # Worker failures are not silently ignored or converted into missing images.
    manager = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else nullcontext(None)
    with manager as pool:
        for source, stream in streams:
            print(f"[*] Quet: {source}")
            n_source = 0
            for batch in _chunks(stream, args.batch_size):
                results = list(pool.map(_hash_one, batch)) if pool else [_hash_one(item) for item in batch]
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
        if r.get("label") not in (0, 1):
            parser.error(f"Unknown reference label; supply --manifest: {r['path']}")
        if r["label"] == 1 and not r.get("model_name"):
            if r.get("test_set") or r.get("supplemental_set"):
                r["model_name"] = r.get("test_set") or r["supplemental_set"]
            elif not r.get("generator"):
                parser.error(f"Unknown fake generator; supply --manifest: {r['path']}")
    sets = sorted({r["test_set"] for r in records if r.get("test_set")})
    complete = bool(records) and skipped == 0 and not args.max_images
    if args.role == "test18":
        complete = complete and set(sets) == set(EXPECTED_TASK51_TESTS)
    if args.role == "cfe":
        complete = complete and bool(args.manifest) and len(records) == 2000

    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_images": len(records),
            "n_skipped_unreadable": skipped,
            "sources": [os.path.abspath(s) for s, _ in streams],
            "phash_bits": 64,
            "tool": "tools/index_test_sets.py",
            "complete": complete, "role": args.role,
            "test_sets": sets,
            "supplemental_sets": sorted({r["supplemental_set"] for r in records if r.get("supplemental_set")}),
            "n_core_images": sum(r.get("test_set") is not None for r in records),
            "n_supplemental_images": sum(r.get("supplemental_set") is not None for r in records),
        },
        "images": records,
    }

    from hash_utils import save_hash_index

    save_hash_index(payload, args.out)
    elapsed = time.time() - t0
    print("=" * 84)
    if not payload["metadata"]["complete"]:
        raise SystemExit(
            f"Reference index incomplete: role={args.role}, images={len(records)}, "
            f"unreadable={skipped}, max_images={args.max_images}, "
            f"missing_test_sets={sorted(set(EXPECTED_TASK51_TESTS) - set(sets)) if args.role == 'test18' else []}. "
            "CFE requires exactly 2000 readable images matched to its manifest; "
            "test18 requires all 18 sets. Not eligible for production audit."
        )
    print(f"XONG: {len(records):,} anh | bo qua {skipped:,} | {elapsed/60:.1f} phut")
    print(f"-> {args.out}")
    print("=" * 84)


if __name__ == "__main__":
    main()
