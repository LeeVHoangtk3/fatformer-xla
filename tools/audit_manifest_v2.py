"""
FatFormer-XLA - Task 7.7: Kiem toan co lap cua manifest v2 (3 cap + generator).

Day la cong cu *doc lap*: no khong tin vao builder, ma tinh lai tu manifest
va tu cac hash index cua tap test.

3 cap kiem toan (dung theo Task 7.7 muc 5 buoc 6):
    Cap 1: manifest v2  vs  tap test CommunityForensics (2.000 anh)
    Cap 2: manifest v2  vs  18 tap test Task 5.1
    Cap 3: noi bo  train vs val_calib vs val_select

Hai lop kiem tra song song:
    - Hash  : SHA-256 (byte-exact) + pHash Hamming (threshold 8 voi test,
              3 voi noi bo) -> bat ca anh bi crop/resize/nén lại.
    - Ten   : danh sach generator cua tap test (lay tu chinh hash index, khong
              lay tu deny-list) duoc so khop *instance-disjoint* voi generator
              trong manifest. Trung instance -> violation; cung ho khac phien
              ban -> `related` (bao cao rieng, khong tinh violation).

Dau ra (O2): DATASET/logs/phase7/data_isolation_report.json
    - `violations` = 0 la dieu kien bat buoc de mo Task 7.11.

Vi du:
    python tools/index_test_sets.py --dirs DATASET/test_communityforensics \\
        --manifest DATASET/test_communityforensics/cfe_manifest.json \\
        --out DATASET/logs/phase7/hashes_cfe_test.json

    python tools/index_test_sets.py --tar <18 tar> --workers 8 \\
        --out DATASET/logs/phase7/hashes_18sets.json.gz

    python tools/audit_manifest_v2.py \\
        --manifest DATASET/manifests/data_manifest_v2.json \\
        --ref cfe=DATASET/logs/phase7/hashes_cfe_test.json \\
        --ref test18=DATASET/logs/phase7/hashes_18sets.json.gz \\
        --out DATASET/logs/phase7/data_isolation_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
for p in (PROJECT_ROOT, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from hash_utils import (  # noqa: E402
    load_hash_index,
    min_hamming_to_set,
    sha256_file,
    to_uint64,
)
from build_generator_denylist import DenyMatcher  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REAL_DIRS = {"0_real", "real", "0", "authentic"}
FAKE_DIRS = {"1_fake", "fake", "1", "synthetic", "generated"}
REQUIRED_FIELDS = ("path", "sha256", "phash", "generator", "family", "split", "label",
                   "source", "width", "height")


# ---------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def infer_generator(path: str) -> Optional[str]:
    """Suy ra ten generator tu duong dan trong index tap test."""
    tail = path.split("::")[-1].replace("\\", "/")
    parts = [p for p in tail.split("/") if p and p not in (".", "..")]
    for i, part in enumerate(parts):
        if part.lower() in REAL_DIRS | FAKE_DIRS:
            return parts[i - 1] if i > 0 else None
    return parts[0] if len(parts) > 1 else None


class RefIndex:
    """Hash index cua mot tap test."""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        payload = load_hash_index(path) or {}
        self.images: List[Dict] = payload.get("images", payload.get("records", [])) or []
        self.sha: set = {str(i["sha256"]) for i in self.images if i.get("sha256")}
        self.phash = to_uint64([i["phash"] for i in self.images if i.get("phash") is not None])
        self.generators: Dict[str, int] = {}
        for i in self.images:
            gen = i.get("model_name") or i.get("generator") or infer_generator(str(i.get("path", "")))
            if not gen:
                continue
            gen = str(gen).strip()
            self.generators[gen] = self.generators.get(gen, 0) + 1
        self.meta = payload.get("metadata", {})

    def __repr__(self) -> str:
        return (f"RefIndex({self.name}: {len(self.images):,} anh, "
                f"{len(self.sha):,} sha, {len(self.generators)} generator)")


# ---------------------------------------------------------------------------
def load_manifest(path: str) -> Tuple[Dict, List[Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"] if isinstance(data, dict) else data
    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    return meta, records


def check_pair(records: List[Dict], ref: RefIndex, threshold: int,
               max_examples: int = 20) -> Dict:
    """So khop SHA-256 + pHash cua `records` voi mot tap test."""
    sha_hits: List[Dict] = []
    for r in records:
        if r["sha256"] in ref.sha:
            sha_hits.append({"path": r["path"], "ref": "sha256", "generator": r.get("generator")})
    phash_hits: List[Dict] = []
    if len(ref.phash) and len(records):
        arr = to_uint64([r["phash"] for r in records])
        d = min_hamming_to_set(arr, ref.phash, chunk_size=512)
        for r, dist in zip(records, d):
            if int(dist) <= threshold:
                phash_hits.append({"path": r["path"], "generator": r.get("generator"),
                                   "distance": int(dist)})
    n = len(sha_hits) + len(phash_hits)
    return {
        "reference": ref.name,
        "reference_path": os.path.abspath(ref.path),
        "n_reference_images": len(ref.images),
        "n_reference_generators": len(ref.generators),
        "threshold": threshold,
        "n_checked": len(records),
        "sha256_matches": len(sha_hits),
        "phash_matches": len(phash_hits),
        "violations": n,
        "examples": (sha_hits + phash_hits)[:max_examples],
        "phash_distance_histogram": _hist(d, threshold) if len(ref.phash) and len(records) else {},
    }


def _hist(distances: np.ndarray, threshold: int) -> Dict[str, int]:
    """Bang tan suat khoang cach pHash nho (de chung minh nguong 8 la hop ly)."""
    if not len(distances):
        return {}
    bins = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64]
    out: Dict[str, int] = {}
    prev = -1
    for b in bins:
        cnt = int(((distances > prev) & (distances <= b)).sum()) if prev >= 0 else int((distances <= b).sum())
        out[f"{max(0, prev) + 1 if prev >= 0 else 0}-{b}"] = cnt
        prev = b
    out[f">{bins[-1]}"] = int((distances > bins[-1]).sum())
    return out


def check_internal(records: List[Dict], threshold: int) -> Dict:
    """Cap 3: train vs val_calib, train vs val_select, val_calib vs val_select."""
    by_split: Dict[str, List[Dict]] = {}
    for r in records:
        by_split.setdefault(r["split"], []).append(r)
    splits = sorted(by_split)
    pairs: Dict[str, Dict] = {}
    total = 0
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            ra, rb = by_split[a], by_split[b]
            sha_b = {r["sha256"] for r in rb}
            sha_hits = [r["path"] for r in ra if r["sha256"] in sha_b]
            arr = to_uint64([r["phash"] for r in ra])
            ref = to_uint64([r["phash"] for r in rb])
            near = []
            if len(arr) and len(ref):
                d = min_hamming_to_set(arr, ref, chunk_size=512)
                near = [{"path": r["path"], "distance": int(dd)}
                        for r, dd in zip(ra, d) if int(dd) <= threshold]
            n = len(sha_hits) + len(near)
            total += n
            pairs[f"{a}__vs__{b}"] = {
                "n_a": len(ra), "n_b": len(rb), "threshold": threshold,
                "sha256_matches": len(sha_hits), "phash_matches": len(near),
                "violations": n, "examples": ([{"path": p, "ref": "sha256"} for p in sha_hits[:10]]
                                              + near[:10]),
            }
    # Mot anh khong duoc nam o 2 split (kiem tra lai tu du lieu, khong tin metadata)
    seen: Dict[str, str] = {}
    multi = 0
    for r in records:
        if r["sha256"] in seen and seen[r["sha256"]] != r["split"]:
            multi += 1
        seen[r["sha256"]] = r["split"]
    pairs["sha_in_multiple_splits"] = {"count": multi, "violations": multi}
    total += multi
    return {"pairs": pairs, "violations": total}


def check_generators(records: List[Dict], refs: Sequence[RefIndex],
                     denylist: Optional[Dict]) -> Dict:
    """
    Kiem tra generator-disjoint tren *ten that* cua tap test (khong dung deny-list).

    `blocked`  = cung instance => violation, phai bang 0.
    `related`  = cung ho, khac phien ban da xac dinh => bao cao rieng.
    """
    test_gens: Dict[str, List[str]] = {}
    for ref in refs:
        for g in ref.generators:
            test_gens.setdefault(g, []).append(ref.name)
    if not test_gens:
        return {"n_test_generators": 0, "blocked": [], "related": [], "violations": 0,
                "note": "khong co hash index tap test -> bo qua kiem tra ten"}

    matcher = DenyMatcher(sorted(test_gens))
    train_gens: Dict[str, int] = {}
    for r in records:
        g = str(r.get("generator") or "")
        train_gens[g] = train_gens.get(g, 0) + 1

    dmatcher = DenyMatcher.from_payload(denylist) if denylist else None
    blocked, related, allowed = [], [], []
    for g, n in sorted(train_gens.items()):
        v = matcher.classify(g)
        item = {"generator": g, "n": n, "matched_test_generator": v.get("matched"),
                "refs": test_gens.get(v.get("matched") or "", [])}
        if v["status"] == "blocked":
            blocked.append(item)
        elif v["status"] == "related":
            related.append(item)
        else:
            allowed.append(item)
    return {
        "n_test_generators": len(test_gens),
        "test_generators": dict(sorted(test_gens.items())),
        "n_train_generators": len(train_gens),
        "blocked": blocked,
        "related": related,
        "allowed": allowed,
        "violations": len(blocked),
        "denylist_status": ({g: (dmatcher.classify(g)["status"] if dmatcher else None)
                             for g in sorted(train_gens)} if dmatcher else None),
    }


def check_integrity(records: List[Dict], manifest_meta: Dict, verify_files: bool,
                    phash_internal: int) -> Dict:
    """Kiem tra cau truc + tai lap lai cac chi so DoD tu chinh manifest."""
    problems: List[Dict] = []
    missing = [r["path"] for r in records
               if any(r.get(f) in (None, "") for f in REQUIRED_FIELDS)]
    if missing:
        problems.append({"check": "required_fields", "n": len(missing), "examples": missing[:5]})

    shas = [r["sha256"] for r in records]
    if len(set(shas)) != len(shas):
        problems.append({"check": "unique_sha256", "n": len(shas) - len(set(shas))})

    bad_files, bad_sha = [], []
    total_bytes = 0
    if verify_files:
        for r in records:
            p = r["path"]
            if not os.path.isabs(p):
                p = os.path.join(PROJECT_ROOT, p)
            if not os.path.isfile(p):
                bad_files.append(r["path"])
                continue
            total_bytes += os.path.getsize(p)
            if sha256_file(p) != r["sha256"]:
                bad_sha.append(r["path"])
        if bad_files:
            problems.append({"check": "file_missing", "n": len(bad_files), "examples": bad_files[:5]})
        if bad_sha:
            problems.append({"check": "sha256_mismatch", "n": len(bad_sha), "examples": bad_sha[:5]})

    # Cac chi so DoD, tinh lai doc lap
    n = len(records)
    splits = sorted({r["split"] for r in records})
    by_split = {sp: [r for r in records if r["split"] == sp] for sp in splits}
    fam = {}
    for r in records:
        fam[r["family"]] = fam.get(r["family"], 0) + 1
    fam_new = {}
    for r in records:
        if not r.get("is_replay"):
            fam_new[r["family"]] = fam_new.get(r["family"], 0) + 1
    replay = sum(1 for r in records if r.get("is_replay"))
    real = sum(1 for r in records if int(r["label"]) == 0)
    real_frac = {sp: sum(1 for r in v if int(r["label"]) == 0) / max(1, len(v))
                 for sp, v in by_split.items()}
    fv = [v for k, v in fam.items() if k != "real" and v > 0]
    summary = {
        "n_records": n,
        "n_bytes_images": total_bytes,
        "by_split": {k: len(v) for k, v in by_split.items()},
        "split_ratio": {k: round(len(v) / max(1, n), 5) for k, v in by_split.items()},
        "family_totals": dict(sorted(fam.items())),
        "family_new_only": dict(sorted(fam_new.items())),
        "family_imbalance": round(max(fv) / min(fv), 4) if fv else 0,
        "n_families": len(fam),
        "real_frac_by_split": {k: round(v, 5) for k, v in real_frac.items()},
        "real_frac_total": round(real / max(1, n), 5),
        "replay": replay,
        "replay_frac": round(replay / max(1, n), 5),
        "n_generators": len({r["generator"] for r in records}),
        "n_sources": len({r["source"] for r in records}),
    }
    # Ghi lai de doi chieu voi metadata builder (khong tin metadata)
    declared = (manifest_meta.get("counts") or {})
    mismatched = []
    for key, value in (("by_split", summary["by_split"]), ("by_family", summary["family_totals"])):
        decl = declared.get(key)
        if decl:
            decl_int = {k: int(v) for k, v in decl.items()}
            if decl_int != value:
                mismatched.append({"key": key, "declared": decl_int, "recomputed": value})
    if mismatched:
        problems.append({"check": "metadata_mismatch", "detail": mismatched})
    return {"summary": summary, "problems": problems, "failures": len(problems)}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Audit isolation for data_manifest_v2.json")
    ap.add_argument("--manifest", default="DATASET/manifests/data_manifest_v2.json")
    ap.add_argument("--ref", action="append", default=[], metavar="NAME=PATH",
                    help="Hash index tap test, vd: cfe=DATASET/logs/phase7/hashes_cfe_test.json")
    ap.add_argument("--ref18", nargs="*", default=[],
                    help="Hash index cua 18 tap (duoc gop thanh mot cap)")
    ap.add_argument("--pair1_names", nargs="*", default=["cfe"],
                    help="Ten cac --ref thuoc cap 1 (mac dinh: cfe)")
    ap.add_argument("--pair2_names", nargs="*", default=["test18", "18sets", "gans", "diffusion"],
                    help="Ten cac --ref thuoc cap 2")
    ap.add_argument("--denylist", default="DATASET/logs/phase7/generator_denylist.json")
    ap.add_argument("--threshold_external", type=int, default=8)
    ap.add_argument("--threshold_internal", type=int, default=3)
    ap.add_argument("--no-verify-files", action="store_true")
    ap.add_argument("--out", default="DATASET/logs/phase7/data_isolation_report.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    manifest_path = args.manifest if os.path.isabs(args.manifest) else os.path.join(PROJECT_ROOT, args.manifest)
    if not os.path.exists(manifest_path):
        raise SystemExit(f"[LOI] khong thay manifest: {manifest_path}")
    t0 = datetime.now(timezone.utc).timestamp()
    meta, records = load_manifest(manifest_path)
    print("=" * 100)
    print("FATFORMER-XLA - KIEM TOAN CO LAP KHO DU LIEU V2 (TASK 7.7)")
    print("=" * 100)
    print(f"- Manifest : {manifest_path}")
    print(f"- So ban ghi: {len(records):,}")
    if not args.quiet:
        print(f"- Split    : { {sp: sum(1 for r in records if r['split'] == sp) for sp in sorted({r['split'] for r in records})} }")

    refs: Dict[str, RefIndex] = {}
    for spec in args.ref:
        if "=" not in spec:
            raise SystemExit(f"--ref phai co dang NAME=PATH, nhan duoc: {spec}")
        name, path = spec.split("=", 1)
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        if not os.path.exists(path):
            print(f"[!] bo qua ref khong ton tai: {name} -> {path}")
            continue
        refs[name] = RefIndex(name, path)
        print(f"- Ref      : {refs[name]}")

    for path in args.ref18:
        p = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
        if not os.path.exists(p):
            print(f"[!] bo qua ref18 khong ton tai: {p}")
            continue
        name = os.path.basename(p).split(".")[0]
        refs[name] = RefIndex(name, p)
        print(f"- Ref18    : {refs[name]}")

    pair1 = [refs[n] for n in args.pair1_names if n in refs]
    pair2 = [refs[n] for n in refs if n not in args.pair1_names and n not in
             {x for x in args.pair1_names if x in refs}]

    report: Dict = {
        "metadata": {
            "created_at": now_iso(),
            "task": "Task 7.7 - audit co lap kho du lieu v2",
            "tool": "tools/audit_manifest_v2.py",
            "manifest": manifest_path,
            "manifest_version": meta.get("version"),
            "n_records": len(records),
            "thresholds": {"external": args.threshold_external, "internal": args.threshold_internal},
            "hash_algorithms": {"sha256": "byte-exact", "phash": "DCT 8x8 64-bit (tools/hash_utils.py)"},
            "references": {r.name: {"path": r.path, "n_images": len(r.images),
                                    "n_generators": len(r.generators)} for r in refs.values()},
        },
        "pairs": {},
    }

    # --- Cap 1: vs CFE test ---
    if pair1:
        merged = {
            "reference": "+".join(r.name for r in pair1),
            "n_reference_images": sum(len(r.images) for r in pair1),
            "n_reference_generators": sum(len(r.generators) for r in pair1),
            "threshold": args.threshold_external,
            "n_checked": len(records),
            "sha256_matches": 0, "phash_matches": 0, "violations": 0, "examples": [],
            "per_reference": [],
        }
        for ref in pair1:
            res = check_pair(records, ref, args.threshold_external)
            merged["per_reference"].append(res)
            merged["sha256_matches"] += res["sha256_matches"]
            merged["phash_matches"] += res["phash_matches"]
            merged["violations"] += res["violations"]
            merged["examples"].extend(res["examples"])
        merged["examples"] = merged["examples"][:20]
        report["pairs"]["v2_vs_cfe_test"] = merged

    # --- Cap 2: vs 18 tap Task 5.1 ---
    if pair2:
        merged = {
            "reference": "+".join(r.name for r in pair2),
            "n_reference_images": sum(len(r.images) for r in pair2),
            "n_reference_generators": sum(len(r.generators) for r in pair2),
            "threshold": args.threshold_external,
            "n_checked": len(records),
            "sha256_matches": 0, "phash_matches": 0, "violations": 0, "examples": [],
            "per_reference": [],
        }
        for ref in pair2:
            res = check_pair(records, ref, args.threshold_external)
            merged["per_reference"].append(res)
            merged["sha256_matches"] += res["sha256_matches"]
            merged["phash_matches"] += res["phash_matches"]
            merged["violations"] += res["violations"]
            merged["examples"].extend(res["examples"])
        merged["examples"] = merged["examples"][:20]
        report["pairs"]["v2_vs_task51_18sets"] = merged

    # --- Cap 3: noi bo ---
    internal = check_internal(records, args.threshold_internal)
    report["pairs"]["internal_splits"] = internal

    # --- Generator-disjoint ---
    denylist = None
    dl_path = args.denylist if os.path.isabs(args.denylist) else os.path.join(PROJECT_ROOT, args.denylist)
    if os.path.exists(dl_path):
        with open(dl_path, "r", encoding="utf-8") as f:
            denylist = json.load(f)
    gen_check = check_generators(records, list(refs.values()), denylist)
    report["generator_check"] = gen_check

    # --- Toan ven ---
    integrity = check_integrity(records, meta, not args.no_verify_files, args.threshold_internal)
    report["integrity"] = integrity

    hash_violations = sum(p.get("violations", 0) for p in report["pairs"].values())
    gen_violations = int(gen_check.get("violations", 0))
    report["violations"] = int(hash_violations + gen_violations)
    report["violations_breakdown"] = {
        "hash_pairs": int(hash_violations),
        "generator_instance": gen_violations,
        "integrity_problems": int(integrity["failures"]),
    }
    report["pass"] = bool(report["violations"] == 0 and integrity["failures"] == 0)
    report["elapsed_s"] = round(datetime.now(timezone.utc).timestamp() - t0, 1)

    out = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    print()
    print("-" * 100)
    for name, pair in report["pairs"].items():
        if "violations" in pair:
            print(f"  {name:<28} vi pham = {pair['violations']:>4}  "
                  f"(sha={pair.get('sha256_matches', 0)}, phash={pair.get('phash_matches', 0)}, "
                  f"thr={pair.get('threshold')})")
    print(f"  {'generator_instance':<28} vi pham = {gen_violations:>4}  "
          f"({len(gen_check.get('related', []))} related duoc khai bao)")
    print(f"  {'integrity':<28} van de   = {integrity['failures']:>4}")
    s = integrity["summary"]
    print("-" * 100)
    print(f"  Tong anh {s['n_records']:,} | real {s['real_frac_total']:.2%} | "
          f"replay {s['replay_frac']:.2%} | chenh lech ho {s['family_imbalance']:.3f}x")
    print(f"  Split: {s['split_ratio']}")
    print("-" * 100)
    print(f"  => violations = {report['violations']} | "
          f"{'PASS (duoc phep mo Task 7.11)' if report['pass'] else 'FAIL (chan Task 7.11)'}")
    print(f"  -> {out}")
    print("=" * 100)


if __name__ == "__main__":
    main()
