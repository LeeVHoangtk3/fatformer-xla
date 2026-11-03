"""
FatFormer-XLA: Hash utilities dung chung cho kho du lieu v2 (Task 7.7).

Muc dich:
    - Chuan hoa DUY NHAT mot cach tinh SHA-256 + pHash cho moi tool
      (build_manifest_v2, audit_manifest_v2, index_test_sets, verify_data_isolation).
    - Cung cap phep so khop pHash vector hoa (numpy uint64) de audit 15k x 110k
      trong vai chuc giay thay vi vai gio.

Ghi chu ky thuat:
    pHash o day PHAI giu nguyen cong thuc cua tools/verify_data_isolation.py
    (DCT 2D 32x32, lay khoi 8x8 tan so thap, median bo goc DC) de hash sinh ra
    tu bat ky tool nao cung so sanh duoc voi nhau.
"""

from __future__ import annotations

import hashlib
import io
import os
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image
from scipy.fftpack import dct

PHASH_BITS = 64

# numpy >= 2.0 co np.bitwise_count; fallback cho numpy cu.
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")


# ---------------------------------------------------------------------------
# SHA-256
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    """Tinh SHA-256 cua mot blob byte (dung cho anh doc tu parquet)."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Tinh SHA-256 cua tep tren dia."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# pHash (giu nguyen cong thuc Task 6.2)
# ---------------------------------------------------------------------------
def phash_from_pil(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Tinh pHash 64-bit tu PIL Image (DCT 2D + median)."""
    img_size = hash_size * highfreq_factor
    img_gray = img.convert("L").resize((img_size, img_size), Image.Resampling.BICUBIC)
    pixels = np.asarray(img_gray, dtype=np.float64)

    dct_data = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    dct_lowfreq = dct_data[:hash_size, :hash_size]

    med = np.median(dct_lowfreq[1:, 1:])
    diff = dct_lowfreq > med

    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
    return hash_int


def phash_from_bytes(data: bytes, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Tinh pHash tu byte anh (khong ghi ra dia)."""
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return phash_from_pil(img, hash_size=hash_size, highfreq_factor=highfreq_factor)


def phash_from_file(path: str, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    with Image.open(path) as img:
        img.load()
        return phash_from_pil(img, hash_size=hash_size, highfreq_factor=highfreq_factor)


# ---------------------------------------------------------------------------
# Hamming distance
# ---------------------------------------------------------------------------
def hamming(a: int, b: int) -> int:
    """Khoang cach Hamming giua 2 hash 64-bit."""
    return int(bin(int(a) ^ int(b)).count("1"))


def _popcount_axis1(x: np.ndarray) -> np.ndarray:
    """Dem so bit 1 cua tung phan tu uint64 (giu nguyen shape dau vao)."""
    if _HAS_BITWISE_COUNT:
        return np.bitwise_count(x).astype(np.int32)
    # Fallback: tach moi uint64 thanh 8 byte roi unpackbits theo truc moi.
    x = np.ascontiguousarray(x)
    x8 = x.view(np.uint8).reshape(x.shape + (8,))
    table = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    return table[x8].sum(axis=-1, dtype=np.int32)


def min_hamming_to_set(
    hashes: np.ndarray,
    reference: np.ndarray,
    chunk_size: int = 1024,
    reference_chunk_size: int = 4096,
) -> np.ndarray:
    """
    Voi moi hash trong `hashes`, tra ve khoang cach Hamming nho nhat toi `reference`.

    Tham so:
        hashes:    np.ndarray dtype uint64, shape (N,)
        reference: np.ndarray dtype uint64, shape (M,)
        chunk_size: so hash xu ly moi lan de gioi han bo nho (N_chunk x M).
    """
    if len(hashes) == 0:
        return np.zeros(0, dtype=np.int32)
    if len(reference) == 0:
        return np.full(len(hashes), PHASH_BITS, dtype=np.int32)

    if chunk_size <= 0 or reference_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    out = np.full(len(hashes), PHASH_BITS, dtype=np.int32)
    for start in range(0, len(hashes), chunk_size):
        block = hashes[start:start + chunk_size].astype(np.uint64, copy=False)
        best = out[start:start + len(block)]
        for rstart in range(0, len(reference), reference_chunk_size):
            ref = reference[rstart:rstart + reference_chunk_size].astype(np.uint64, copy=False)
            xor = np.bitwise_xor(block[:, None], ref[None, :])
            np.minimum(best, _popcount_axis1(xor).min(axis=1), out=best)
    return out


def min_hamming_to_array(h: int, reference: np.ndarray) -> int:
    """Khoang cach Hamming nho nhat cua 1 hash toi ca mang."""
    if len(reference) == 0:
        return PHASH_BITS
    return int(_popcount_axis1(np.bitwise_xor(reference, np.uint64(h))).min())


def pairwise_hamming_matrix(a: np.ndarray, b: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """Ma tran khoang cach Hamming (len(a) x len(b)), tinh theo chunk."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.int32)
    out = np.empty((len(a), len(b)), dtype=np.int32)
    for start in range(0, len(a), chunk_size):
        xor = np.bitwise_xor(
            a[start:start + chunk_size, None].astype(np.uint64, copy=False),
            b[None, :].astype(np.uint64, copy=False),
        )
        out[start:start + chunk_size] = _popcount_axis1(xor)
    return out


# ---------------------------------------------------------------------------
# IO hash index
# ---------------------------------------------------------------------------
def to_uint64(values: Iterable[int]) -> np.ndarray:
    """Chuyen list int (co the > 2**63) sang np.uint64 an toan."""
    return np.fromiter((int(v) & ((1 << 64) - 1) for v in values), dtype=np.uint64)


def load_hash_index(path: str) -> dict:
    """Doc file hash index dang JSON (ho tro .gz)."""
    import gzip
    import json

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_hash_index(payload: dict, path: str) -> None:
    """Ghi file hash index dang JSON (tu dong gzip neu duoi .gz)."""
    import gzip
    import json

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path + ".tmp", "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)
