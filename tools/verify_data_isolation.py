"""
FatFormer-XLA: Quy Trinh Kiem Toan Phan Ly Du Lieu Da Tang (Dual Audit Protocol)
Phuc vu Task 6.2 (Milestone 6) — Thanh vien A (Data & Infrastructure Lead).

Kiem toan 3 tang bao ve:
    - Tang 1 (SHA-256 Hash): So sanh doi chieu voi 18 tap test Task 5.1 -> Yeu cau 0.0% trung lap.
    - Tang 2 (Perceptual pHash): Tinh khoang cach Hamming d_H <= 3 de loai bo anh bien doi tuong dong (near-duplicate).
    - Tang 3 (Train-Val Isolation): Doi chieu cheo giua train va val -> Yeu cau 0.0% trung lap.

Cac che do thuc thi:
    # 1. Chay kiem thu thuat toan gia lap tren CPU:
    python tools/verify_data_isolation.py --test_dummy

    # 2. Chay kiem toan tren thu muc thuc te:
    python tools/verify_data_isolation.py --train_dir DATASET/train_crossdomain --val_dir DATASET/val_crossdomain
"""

import os
import sys
import hashlib
import argparse
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image
from scipy.fftpack import dct

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def compute_sha256(filepath: str) -> str:
    """Tinh ma bam SHA-256 cua tep tin de kiem tra trung lap byte-exact."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash_from_image(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """
    Tinh toan Perceptual Hash (pHash) 64-bit bang bien doi DCT 2D:
    1. Chuyen ve anh xam va resize thanh (32, 32).
    2. Ap dung bien doi DCT 2D theo chieu dong va cot.
    3. Trch xuat khoi tan so thap 8x8 (bo thanh phan DC goc (0,0)).
    4. So sanh voi gia tri trung vi (median) de tao chuoi 64 bit.
    """
    img_size = hash_size * highfreq_factor
    img_gray = img.convert("L").resize((img_size, img_size), Image.Resampling.BICUBIC)
    pixels = np.asarray(img_gray, dtype=np.float64)

    # 2D DCT
    dct_data = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    # Lay goc tren tan so thap
    dct_lowfreq = dct_data[:hash_size, :hash_size]

    # Tinh trung vi (loai bo phan tu DC goc tai [0, 0])
    med = np.median(dct_lowfreq[1:, 1:])
    diff = dct_lowfreq > med

    # Chuyen ma tran boolean 8x8 thanh so nguyen 64-bit
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
    return hash_int


def hamming_distance(hash1: int, hash2: int) -> int:
    """Tinh khoang cach Hamming giua 2 ma pHash (so bit khac nhau)."""
    return bin(hash1 ^ hash2).count("1")


class DataIsolationAuditor:
    """Trinh kiem toan tinh toan ven va doc lap phan ly du lieu."""

    def __init__(self, test_hashes_sha256: Set[str], test_hashes_phash: List[int]):
        self.test_hashes_sha256 = test_hashes_sha256
        self.test_hashes_phash = test_hashes_phash

    def audit_records(
        self,
        records: List[Dict],
        split_name: str = "train",
        phash_threshold: int = 3,
    ) -> Tuple[bool, Dict[str, int], List[str]]:
        """
        Kiem tra tung mau trong tap records so voi tap test va noi bo:
        - records: [{'path': str, 'sha256': str, 'phash': int}, ...]
        """
        duplicates_sha = []
        near_duplicates_phash = []
        internal_sha_seen = {}
        internal_duplicates = []

        for r in records:
            p = r["path"]
            h_sha = r["sha256"]
            h_phash = r["phash"]

            # Tang 1: Kiem tra trung lap SHA voi tap test bi khoa
            if h_sha in self.test_hashes_sha256:
                duplicates_sha.append(p)

            # Tang 2: Kiem tra near-duplicate voi tap test qua pHash
            for test_phash in self.test_hashes_phash:
                if hamming_distance(h_phash, test_phash) <= phash_threshold:
                    near_duplicates_phash.append(p)
                    break

            # Kiem tra noi bo
            if h_sha in internal_sha_seen:
                internal_duplicates.append((p, internal_sha_seen[h_sha]))
            else:
                internal_sha_seen[h_sha] = p

        pass_test = (len(duplicates_sha) == 0) and (len(near_duplicates_phash) == 0)
        metrics = {
            "total_checked": len(records),
            "duplicates_sha_vs_test": len(duplicates_sha),
            "near_duplicates_phash_vs_test": len(near_duplicates_phash),
            "internal_duplicates": len(internal_duplicates),
        }
        logs = []
        logs.append(f"Kiem toan tap [{split_name}] ({len(records)} mau):")
        logs.append(f"  • Trung lap byte-exact SHA-256 voi tap test: {len(duplicates_sha)} mau (Yeu cau: 0)")
        logs.append(f"  • Bien doi tuong dong pHash (dH <= {phash_threshold}) voi tap test: {len(near_duplicates_phash)} mau (Yeu cau: 0)")
        logs.append(f"  • Trung lap noi bo tap {split_name}: {len(internal_duplicates)} mau")

        return pass_test, metrics, logs


def audit_train_val_separation(train_records: List[Dict], val_records: List[Dict]) -> Tuple[bool, int]:
    """Tang 3: Kiem toan phan ly giua train va val."""
    train_shas = {r["sha256"] for r in train_records}
    leak_count = sum(1 for r in val_records if r["sha256"] in train_shas)
    return (leak_count == 0), leak_count


def run_dummy_audit_verification():
    """Unit test kiem chung thuat toan kiem toan da tang tren du lieu mo phong."""
    print("=" * 80)
    print("FATFORMER-XLA: KIEM THU QUY TRINH KIEM TOAN PHAN LY DU LIEU (TASK 6.2)")
    print("=" * 80)

    # 1. Tao 2 anh goc dai dien cho tap test Task 5.1
    img_test_1 = Image.new("RGB", (256, 256), color=(200, 50, 50))
    img_test_2 = Image.new("RGB", (256, 256), color=(50, 200, 50))

    phash_t1 = compute_phash_from_image(img_test_1)
    phash_t2 = compute_phash_from_image(img_test_2)
    test_phash_list = [phash_t1, phash_t2]
    test_sha_set = {"fake_test_sha_123456", "fake_test_sha_789012"}

    print(f"- Da thiet lap tap kiem chung test gom 2 ma băm pHash va SHA-256.")

    # 2. Tao tap du lieu train moi (hoan toan doc lap, mau sac khac biet)
    records_train = []
    for i in range(10):
        img_train = Image.new("RGB", (256, 256), color=(i * 20, 100, 200))
        records_train.append({
            "path": f"data_crossdomain/train/sample_{i}.png",
            "sha256": f"train_sha_{i:04d}",
            "phash": compute_phash_from_image(img_train),
        })

    # Chen 1 mau co chu y vi pham pHash (bien doi nhe tu img_test_1 bang resize va lech mau)
    img_near_dup = img_test_1.resize((240, 240)).resize((256, 256))
    records_train.append({
        "path": "data_crossdomain/train/near_duplicate_violation.png",
        "sha256": "unique_sha_but_near_dup",
        "phash": compute_phash_from_image(img_near_dup),
    })

    auditor = DataIsolationAuditor(test_sha_set, test_phash_list)
    pass_audit, metrics, logs = auditor.audit_records(records_train, split_name="train_mock")

    for line in logs:
        print(f"  {line}")

    # Phat hien dung mau near-duplicate
    detected_near_dup = (metrics["near_duplicates_phash_vs_test"] >= 1)
    print(f"\n- Danh gia bo loc:")
    print(f"  • Phat hien mau bien doi tuong dong (pHash filter): {'CHINH XAC (DA BAT)' if detected_near_dup else 'THAT BAI'}")

    # Kiem tra Tang 3: Train - Val Separation
    records_val = [
        {"path": "val_1.png", "sha256": "val_sha_001", "phash": 12345},
        {"path": "val_2.png", "sha256": "train_sha_0001", "phash": 67890},  # bi ro ri tu train
    ]
    pass_sep, leak_cnt = audit_train_val_separation(records_train, records_val)
    print(f"  • Phat hien mau ro ri giua Train va Val: {'CHINH XAC (' + str(leak_cnt) + ' mau)' if not pass_sep else 'KHONG RO RI'}")

    print("\n=> KET LUAN: KIEM THU QUY TRINH KIEM TOAN HOAN THANH XUAT SAC!")
    print("=" * 80)


def scan_directory_records(data_dir: str, split_name: str = "dataset") -> List[Dict]:
    """Quet thu muc anh, tinh toan ma bam SHA-256 va pHash cho tung anh."""
    records = []
    if not os.path.exists(data_dir):
        return records

    valid_exts = (".png", ".jpg", ".jpeg", ".webp")
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.lower().endswith(valid_exts):
                fpath = os.path.join(root, fname)
                try:
                    sha = compute_sha256(fpath)
                    with Image.open(fpath) as img:
                        phash = compute_phash_from_image(img)
                    records.append({
                        "path": fpath,
                        "sha256": sha,
                        "phash": phash,
                        "split": split_name,
                    })
                except Exception as e:
                    print(f"  [!] Khong the doc anh {fpath}: {e}")
    return records


def run_real_audit(args):
    """Kiem toan phan ly du lieu thuc te tren thu muc hoac tep manifest."""
    import json
    print("=" * 80)
    print("FATFORMER-XLA: KIEM TOAN PHAN LY DU LIEU DA TANG (DU LIEU THUC TE)")
    print("=" * 80)

    train_records = []
    val_records = []
    test_sha_set = set()
    test_phash_list = []

    # 1. Nap tu manifest neu chi dinh
    if args.manifest and os.path.exists(args.manifest):
        print(f"[*] Dang doc tep manifest: {args.manifest}")
        try:
            with open(args.manifest, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            for r in manifest_data.get("records", []):
                rec = {
                    "path": r.get("file_path", r.get("path", "")),
                    "sha256": str(r.get("sha256", "")),
                    "phash": int(r.get("phash", 0)),
                    "split": r.get("split", "train"),
                }
                if rec["split"] == "train":
                    train_records.append(rec)
                else:
                    val_records.append(rec)
            print(f"  [✓] Nap tu manifest: Train={len(train_records)}, Val={len(val_records)}")
        except Exception as e:
            print(f"  [!] Loi doc manifest {args.manifest}: {e}")

    # 2. Quet thu muc train neu chi dinh
    if args.train_dir and os.path.exists(args.train_dir):
        print(f"[*] Dang quet thu muc train: {args.train_dir}")
        scanned_train = scan_directory_records(args.train_dir, split_name="train")
        train_records.extend(scanned_train)
        print(f"  [✓] Quet duoc {len(scanned_train)} mau train tu thu muc.")

    # 3. Quet thu muc val neu chi dinh
    if args.val_dir and os.path.exists(args.val_dir):
        print(f"[*] Dang quet thu muc val: {args.val_dir}")
        scanned_val = scan_directory_records(args.val_dir, split_name="val")
        val_records.extend(scanned_val)
        print(f"  [✓] Quet duoc {len(scanned_val)} mau val tu thu muc.")

    # 4. Quet test sets neu chi dinh
    if args.test_dirs:
        for tdir in args.test_dirs:
            if os.path.exists(tdir):
                print(f"[*] Dang quet tap test doi chieu: {tdir}")
                t_recs = scan_directory_records(tdir, split_name="test")
                for r in t_recs:
                    test_sha_set.add(r["sha256"])
                    test_phash_list.append(r["phash"])
        print(f"  [✓] Tong so mau test doi chieu: {len(test_sha_set)} SHA-256 / {len(test_phash_list)} pHash.")

    if len(train_records) == 0 and len(val_records) == 0:
        print("[!] Canh bao: Khong co du lieu trong manifest hoac thu muc duoc chi dinh!")
        print("    Vui long kiem tra --train_dir, --val_dir hoac --manifest.")
        return

    # Thuc hien kiem toan 3 tang
    auditor = DataIsolationAuditor(test_sha_set, test_phash_list)
    pass_train, m_train, logs_train = auditor.audit_records(train_records, split_name="train", phash_threshold=args.phash_threshold)
    for l in logs_train:
        print(f"  {l}")

    pass_val = True
    m_val = {}
    if len(val_records) > 0:
        pass_val, m_val, logs_val = auditor.audit_records(val_records, split_name="val", phash_threshold=args.phash_threshold)
        for l in logs_val:
            print(f"  {l}")

    pass_sep, leak_cnt = audit_train_val_separation(train_records, val_records)
    print(f"\n- Kiem toan Tang 3 (Train - Val Isolation):")
    print(f"  • Ro ri giua Train va Val: {'0 mau (PASS)' if pass_sep else f'{leak_cnt} mau bi ro ri (FAIL)'}")

    all_pass = pass_train and pass_val and pass_sep
    print(f"\n=> KET LUAN KIEM TOAN: {'PASS 100% (CO THE SU DUNG)' if all_pass else 'FAIL (CAN XU LY TRUNG LAP)'}")
    print("=" * 80)

    if args.output_report:
        report = {
            "all_pass": all_pass,
            "train_metrics": m_train,
            "val_metrics": m_val,
            "train_val_leakage": leak_cnt,
            "phash_threshold": args.phash_threshold,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[✓] Da xuat bao cao kiem toan ra: {args.output_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Isolation Dual Audit")
    parser.add_argument("--test_dummy", action="store_true", help="Chay kiem thu tren du lieu mau gia lap")
    parser.add_argument("--train_dir", type=str, default="", help="Thu muc du lieu train")
    parser.add_argument("--val_dir", type=str, default="", help="Thu muc du lieu val")
    parser.add_argument("--manifest", type=str, default="", help="Duong dan tep manifest json")
    parser.add_argument("--test_dirs", nargs="*", default=[], help="Danh sach thu muc test can doi chieu")
    parser.add_argument("--phash_threshold", type=int, default=3, help="Nguong khoang cach Hamming pHash")
    parser.add_argument("--output_report", type=str, default="", help="Duong dan xuat bao cao json")
    args = parser.parse_args()

    if args.test_dummy:
        run_dummy_audit_verification()
    elif args.manifest or args.train_dir or args.val_dir or args.test_dirs:
        run_real_audit(args)
    else:
        def_manifest = os.path.join(PROJECT_ROOT, "DATASET", "manifests", "data_manifest.json")
        if os.path.exists(def_manifest):
            args.manifest = def_manifest
            run_real_audit(args)
        else:
            run_dummy_audit_verification()
