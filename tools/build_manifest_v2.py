"""
FatFormer-XLA - Task 7.7: Xay kho du lieu da bo sinh v2 (manifest v2).

Cong cu nay thay the y tuong "stream toan bo Parquet qua `datasets`" (bat kha
thi: 186 shard x ~1.2 GB = ~229 GB) bang chien luoc **metadata-first**:

    1. INDEX  - chi doc *metadata* cua tung shard qua DuckDB/httpfs (~350 s cho
                186 shard), khong doc cot `image_data`.
    2. PLAN   - tu chon mot so shard toi thieu (greedy co-location) du cho
                quota tung generator, tinh truoc dung luong tai va chan neu
                vuot ngan sach 60 GB.
    3. FETCH  - tai nguyen shard da chon ve cache (resume duoc), roi doc tung
                batch bang pyarrow, giai ma + do chat luong + SHA-256 + pHash
                va luu nguyen byte goc.
    4. REPLAY - tai su dung 3.600 mau `diffusion_staging` cu, danh dau
                `is_replay: true`.
    5. SELECT - khu trung lap (SHA-256 tuyet doi + pHash <= 3), ap quota, bu
                thieu trong noi bo ho.
    6. SPLIT  - chia phan tang 85/7.5/7.5, chia het (largest remainder), moi
                o (family x label x generator) deu co mat trong ca 3 split.
    7. FINAL  - ghi manifest v2 + bao cao rang buoc + hinh phan bo.

Vi du:
    # 0) Sinh deny-list truoc (bat buoc)
    python tools/build_generator_denylist.py \
        --cfe-manifest /content/dataset_local/test_communityforensics/cfe_manifest.json \
        --eval-metadata /content/dataset_local/communityforensics/eval_set_metadata.csv \
        --out DATASET/logs/phase7/generator_denylist.json

    # 1) Kiem tra ke hoach (khong tai gi)
    python tools/build_manifest_v2.py --config configs/phase7_data_v2.yaml --dry-run

    # 2) Chay that (co the dat --resume nhieu lan)
    python tools/build_manifest_v2.py --config configs/phase7_data_v2.yaml \
        --cache_dir /content/cache_v2 --resume --workers 8

    # 3) Chi chay lai tu mot buoc
    python tools/build_manifest_v2.py --config ... --stage split --resume
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
for p in (PROJECT_ROOT, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from hash_utils import (  # noqa: E402
    min_hamming_to_set,
    pairwise_hamming_matrix,
    phash_from_pil,
    sha256_bytes,
    to_uint64,
    sha256_file,
    load_hash_index,
)
from build_generator_denylist import DenyMatcher, EXPECTED_TASK51_TESTS  # noqa: E402
from data_v2_contract import contract_problems

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

T0 = time.time()
STAGES = ("index", "plan", "fetch", "replay", "select", "split", "finalize")
SPLITS = ("train", "val_calib", "val_select")
_DIGITS = re.compile(r"[^A-Za-z0-9]+")


# ---------------------------------------------------------------------------
# Tien ich chung
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"\n[LOI] {msg}\n", flush=True)
    raise SystemExit(2)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_rank(seed: int, *parts: object) -> str:
    """Khoa sap xep tat dinh (khong phu thuoc thu tu duyet / phien ban)."""
    return hashlib.sha256(("|".join([str(seed)] + [str(p) for p in parts])).encode()).hexdigest()


def slug(name: str, max_len: int = 60) -> str:
    s = _DIGITS.sub("_", str(name)).strip("_").lower()
    return (s or "unknown")[:max_len]


def rel(path: str) -> str:
    """Duong dan tuong doi so voi goc repo, dang posix."""
    try:
        out = os.path.relpath(os.path.abspath(path), PROJECT_ROOT)
    except ValueError:  # khac o dia tren Windows
        out = os.path.abspath(path)
    return out.replace("\\", "/")


def abspath(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def read_json(path: str, default=None):
    if not path or not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def read_jsonl(path: str, recover_tail: bool = False) -> List[Dict]:
    out: List[Dict] = []
    if not os.path.exists(path):
        return out
    with open(path, "rb") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if not recover_tail or i != len(lines) - 1 or line.endswith(b"\n"):
                raise
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                for row in out:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(path + ".tmp", path)
    if recover_tail and lines and not lines[-1].endswith(b"\n"):
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
    return out


def append_jsonl(path: str, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


def largest_remainder(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Chia `total` theo trong so, dam bao tong dung bang `total`."""
    keys = list(weights)
    s = sum(weights[k] for k in keys)
    if total <= 0 or s <= 0:
        return {k: 0 for k in keys}
    raw = {k: total * weights[k] / s for k in keys}
    base = {k: int(math.floor(v)) for k, v in raw.items()}
    rest = total - sum(base.values())
    order = sorted(keys, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:rest]:
        base[k] += 1
    return base


# ---------------------------------------------------------------------------
# Cau hinh
# ---------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    name: str
    kind: str
    url_template: str
    n_shards: int
    family_of_architecture: Dict[str, str] = field(default_factory=dict)
    architecture_by_model: Dict[str, str] = field(default_factory=dict)
    family: Optional[str] = None
    model_name: Optional[str] = None
    license: str = ""
    citation: str = ""

    def url(self, shard: int) -> str:
        fmt = self.url_template
        # 'train-{:05d}-of-00014' -> 00000; 'HFCF_small_{}' -> 0
        try:
            return fmt.format(shard)
        except (IndexError, KeyError):
            return fmt.replace("{}", str(shard))

    def shard_key(self, shard: int) -> str:
        return f"{self.name}:{shard}"


class Config:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f) or {}
        if not isinstance(self.data, dict) or "targets" not in self.data:
            die(f"Config khong hop le: {self.path}")
        self.sha256 = sha256_file(self.path)
        self.p = self.data["paths"]
        self.t = self.data["targets"]
        self.datasets = {k: DatasetSpec(name=k, **v) for k, v in self.data["datasets"].items()}
        self.quality = self.data.get("quality", {})
        self.replay_cfg = self.data.get("replay", {})
        self.seed = int(self.data.get("seed", 42))

    def path_of(self, key: str) -> str:
        return abspath(self.p[key])

    def state(self, name: str) -> str:
        return os.path.join(self.path_of("state_dir"), name)


def ensure_state_context(cfg: Config):
    """Resume only against the same config, denylist and replay manifest."""
    context = {"schema": 2, "config": cfg.sha256}
    for key, path in (("denylist", cfg.path_of("denylist")),
                      ("replay", abspath(cfg.replay_cfg.get("manifest", "")))):
        context[key] = sha256_file(path) if os.path.isfile(path) else None
    old = read_json(cfg.state("context.json"))
    if old is not None and old != context:
        die("Build state context changed (config/denylist/replay). Use a new state/raw directory or --force.")
    if old is None:
        if any(os.path.exists(cfg.state(n)) for n in ("candidates.jsonl", "replay.jsonl", "selected.jsonl")):
            die("Legacy state without fingerprints. Use a new state directory or --force.")
        write_json(cfg.state("context.json"), context)


def check_replay_generators(cfg: Config, matcher=None):
    matcher = matcher or DenyMatcher.from_payload(read_json(cfg.path_of("denylist")))
    payload = read_json(abspath(cfg.replay_cfg.get("manifest", "")), {})
    records = payload.get("records", [])
    allowed_gens = cfg.replay_cfg.get("allowed_generators")
    if allowed_gens is not None:
        allowed_set = set(allowed_gens)
        records = [r for r in records if r.get("generator") in allowed_set]
    if len(records) != int(cfg.t["replay_total"]):
        die(f"Replay manifest count ({len(records)}) does not match the contract ({cfg.t['replay_total']}).")
    display = cfg.replay_cfg.get("generator_display", {})
    for row in records:
        if int(row.get("label", -1)) != 1:
            continue
        raw = str(row.get("generator") or "")
        name = display.get(raw, raw)
        verdict = matcher.classify(name)
        if verdict["status"] != "allowed":
            die(f"Replay generator {name!r} is {verdict['status']} by {verdict.get('matched')!r}. "
                "Resolve generator identity/replay policy before building; no automatic exemption.")


# ---------------------------------------------------------------------------
# INDEX: metadata-only scan
# ---------------------------------------------------------------------------
CFE_INDEX_SQL = """
SELECT {cols}, count(*) n, sum(CASE WHEN nsfw_flag THEN 1 ELSE 0 END) nsfw
FROM read_parquet('{url}') GROUP BY {groups} ORDER BY {ncol} DESC
"""
PLAIN_INDEX_SQL = "SELECT count(*) n FROM read_parquet('{url}')"

_COL_CACHE: Dict[str, List[str]] = {}


def _duckdb():
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_progress_bar=false")
    con.execute("SET threads=4")
    return con


def _cfe_columns(con, url: str, ds_name: str) -> List[str]:
    """Liet ke cot cua shard (footer-only, rat nhanh). Cache theo dataset."""
    if ds_name in _COL_CACHE:
        return _COL_CACHE[ds_name]
    desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{url}')").fetchall()
    cols = [d[0] for d in desc]
    _COL_CACHE[ds_name] = cols
    return cols


def stage_index(cfg: Config, force: bool = False, only: Optional[Sequence[str]] = None) -> List[Dict]:
    """Quet metadata cua moi shard cua moi dataset. Cache ra state/index_<ds>.json."""
    os.makedirs(os.path.dirname(cfg.state("x")), exist_ok=True)
    rows: List[Dict] = []
    con = _duckdb()
    for name, spec in cfg.datasets.items():
        revision = spec.url_template.split("/resolve/", 1)[-1].split("/", 1)[0]
        if "/resolve/" in spec.url_template and not re.fullmatch(r"[0-9a-f]{40}", revision):
            die(f"Pin a dataset commit revision before indexing: {name}. See the Colab runbook.")
        if only and name not in only:
            existing = read_json(cfg.state(f"index_{name}.json"))
            if existing:
                rows.extend(existing)
            continue
        cache = cfg.state(f"index_{name}.json")
        signature = sha256_text(json.dumps(spec.__dict__, sort_keys=True))
        signature_path = cfg.state(f"index_{name}_context.json")
        if existing := (None if force else read_json(cache)):
            if read_json(signature_path) != {"dataset_sha256": signature}:
                die(f"Stale/unversioned metadata index for {name}. Reindex with --force or use a new state directory.")
            log(f"INDEX {name}: dung cache ({len(existing)} dong)")
            rows.extend(existing)
            continue
        log(f"INDEX {name}: quet {spec.n_shards} shard (metadata only)")
        ds_rows: List[Dict] = []
        failed_shards = []
        t0 = time.time()
        for i in range(spec.n_shards):
            url = spec.url(i)
            try:
                if spec.kind == "cfe":
                    cols = _cfe_columns(con, url, name)
                    wanted = [c for c in ("label", "architecture", "subset", "model_name",
                                          "real_source") if c in cols]
                    ncol = str(len(wanted) + 1)
                    sql = CFE_INDEX_SQL.format(
                        cols=", ".join(wanted),
                        url=url,
                        groups=", ".join(str(j + 1) for j in range(len(wanted))),
                        ncol=ncol,
                    )
                    res = con.execute(sql).fetchall()
                    for item in res:
                        rec = dict(zip(wanted, item[:-2]))
                        rec["n"] = int(item[-2])
                        rec["nsfw"] = int(item[-1] or 0)
                        arch = rec.get("architecture") or spec.family_of_architecture.get(
                            str(rec.get("model_name")), None)
                        if arch is None and spec.architecture_by_model:
                            arch = spec.architecture_by_model.get(str(rec.get("model_name")))
                        ds_rows.append({
                            "dataset": name, "shard": i,
                            "label": int(rec.get("label", 1)),
                            "architecture": arch,
                            "subset": rec.get("subset"),
                            "model_name": rec.get("model_name"),
                            "real_source": rec.get("real_source"),
                            "n": rec["n"], "nsfw": rec["nsfw"],
                        })
                else:
                    n = con.execute(PLAIN_INDEX_SQL.format(url=url)).fetchone()[0]
                    ds_rows.append({
                        "dataset": name, "shard": i, "label": 1,
                        "architecture": "External", "subset": "plain",
                        "model_name": spec.model_name or name,
                        "real_source": None, "n": int(n), "nsfw": 0,
                    })
            except Exception as exc:  # noqa: BLE001
                log(f"  [!] shard {i} loi: {type(exc).__name__}: {str(exc)[:120]}")
                failed_shards.append(i)
            if (i + 1) % 20 == 0 or i + 1 == spec.n_shards:
                log(f"  {name}: {i + 1}/{spec.n_shards} shard ({time.time() - t0:.0f}s)")
        if failed_shards:
            die(f"INDEX {name}: incomplete metadata coverage; failed shards {failed_shards}. Retry without publishing this index.")
        if not ds_rows:
            die(f"INDEX {name}: khong doc duoc shard nao. Kiem tra mang / URL template.")
        write_json(cache, ds_rows)
        write_json(signature_path, {"dataset_sha256": signature})
        log(f"INDEX {name}: {len(ds_rows)} dong -> {rel(cache)}")
        rows.extend(ds_rows)
    return rows


# ---------------------------------------------------------------------------
# Kich thuoc shard (HEAD) - de tinh ngan sach byte truoc khi tai
# ---------------------------------------------------------------------------
def http_head_size(url: str, token: Optional[str], cache: Dict[str, int]) -> int:
    """Lay Content-Length cua shard (HEAD, fallback GET Range 0-0)."""
    if url in cache:
        return cache[url]
    main_url = re.sub(r"/resolve/[0-9a-f]{40}/", "/resolve/main/", url)
    if main_url in cache:
        cache[url] = cache[main_url]
        return cache[url]
    if os.path.exists(url):  # duong dan cuc bo (dung cho kiem thu offline)
        cache[url] = os.path.getsize(url)
        return cache[url]
    import requests

    headers = {"User-Agent": "fatformer-xla/7.7"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    size = 0
    try:
        r = requests.head(url, headers=headers, allow_redirects=True, timeout=45)
        size = int(r.headers.get("Content-Length") or 0)
    except Exception:  # noqa: BLE001
        size = 0
    if not size:
        try:
            h = dict(headers, Range="bytes=0-0")
            r = requests.get(url, headers=h, stream=True, timeout=45, allow_redirects=True)
            cr = r.headers.get("Content-Range") or ""
            if "/" in cr:
                size = int(cr.rsplit("/", 1)[1])
            r.close()
        except Exception:  # noqa: BLE001
            size = 0
    cache[url] = size
    return size


# ---------------------------------------------------------------------------
# PLAN: chon shard toi thieu (greedy co-location)
# ---------------------------------------------------------------------------
def _index_stats(index_rows: Sequence[Dict], dataset: str) -> Dict[Tuple, Dict[int, Dict]]:
    """(label, model_name) -> {shard: {n, nsfw, real_sources}}.

    Mot shard co the chua CUNG model voi NHIEU real_source (vi du StyleSwin
    trong shard 75 vua celeba vua ffhq). Phai CONG don, khong duoc ghi de,
    neu khong so ung vien se bi ha xuong con mot nua.
    """
    out: Dict[Tuple, Dict[int, Dict]] = {}
    for r in index_rows:
        if r["dataset"] != dataset:
            continue
        key = (r["label"], r["model_name"] or "")
        slot = out.setdefault(key, {}).setdefault(
            r["shard"], {"n": 0, "nsfw": 0, "real_sources": set()})
        slot["n"] += int(r["n"])
        slot["nsfw"] += int(r["nsfw"])
        for part in str(r.get("real_source") or "").split(","):
            if part.strip():
                slot["real_sources"].add(part.strip())
    return out


def _family_of_row(cfg: Config, r: Dict) -> Optional[str]:
    spec = cfg.datasets.get(r["dataset"])
    if spec is None:
        return None
    fam = spec.family_of_architecture.get(r.get("architecture") or "")
    if fam is None and spec.kind == "plain":
        fam = spec.family
    return fam


def _allowed_alternatives(cfg: Config, index_rows: Sequence[Dict], family: str,
                          matcher: DenyMatcher, limit: int = 10) -> List[str]:
    agg: Dict[str, Dict] = {}
    for r in index_rows:
        if r["label"] != 1 or _family_of_row(cfg, r) != family:
            continue
        name = r["model_name"] or ""
        if matcher.classify(name)["status"] != "allowed":
            continue
        a = agg.setdefault(name, {"n": 0, "dataset": r["dataset"]})
        a["n"] += r["n"]
    top = sorted(agg.items(), key=lambda kv: (-kv[1]["n"], kv[0]))[:limit]
    return [f"{k} ({v['n']:,} anh, {v['dataset']})" for k, v in top]


def build_plan(
    cfg: Config,
    index_rows: Sequence[Dict],
    matcher: DenyMatcher,
    limit_per_family: Optional[int] = None,
    surplus: Optional[float] = None,
) -> Dict:
    check_replay_generators(cfg, matcher)
    surplus = float(surplus if surplus is not None else cfg.t.get("surplus", 1.15))
    nsfw_on = bool(cfg.quality.get("reject_nsfw", True))
    allow_related = bool(cfg.data.get("allow_related_generators", False))

    entries: List[Dict] = []
    for fam_cfg, kind in ((cfg.data.get("fakes", []), "fake"), (cfg.data.get("reals", []), "real")):
        for e in fam_cfg:
            item = dict(e)
            item["kind"] = kind
            if kind == "real":
                item.setdefault("family", "real")
            entries.append(item)

    if limit_per_family:
        # Gia lap nhanh: cat moi ho xuong <= limit (dung cho smoke test).
        for it in entries:
            if it["kind"] == "fake":
                it["target"] = min(int(it["target"]), int(limit_per_family))

    # ---- thong ke ung vien / bi chan theo ho (bang bang chung cua Buoc 1) ----
    fam_stats: Dict[str, Dict[str, int]] = {}
    for r in index_rows:
        if r["label"] != 1:
            continue
        fam = _family_of_row(cfg, r)
        if not fam:
            continue
        st = fam_stats.setdefault(fam, {"n_candidates": 0, "n_blocked": 0, "n_related": 0})
        st["n_candidates"] += r["n"]
        status = matcher.classify(r["model_name"] or "")["status"]
        if status == "blocked":
            st["n_blocked"] += r["n"]
        elif status == "related":
            st["n_related"] += r["n"]
    for fam in {it["family"] for it in entries if it["kind"] == "fake"}:
        fam_stats.setdefault(fam, {"n_candidates": 0, "n_blocked": 0, "n_related": 0})

    # ---- chon shard ----
    stats_by_ds = {ds: _index_stats(index_rows, ds) for ds in cfg.datasets}
    sizes: Dict[str, int] = read_json(cfg.state("shard_sizes.json"), {}) or {}
    token = os.environ.get("HF_TOKEN") or None

    # Uu tien generator hiem truoc de greedy co-location dat hieu qua cao.
    order = sorted(
        [it for it in entries if it["dataset"] in stats_by_ds],
        key=lambda it: (sum(s["n"] for s in stats_by_ds[it["dataset"]].get((1 if it["kind"] == "fake" else 0, it["model_name"]), {}).values()), it["id"]),
    )

    chosen: Dict[str, set] = {}
    plan_entries: List[Dict] = []
    for it in order:
        ds = it["dataset"]
        spec = cfg.datasets[ds]
        label = 1 if it["kind"] == "fake" else 0
        avail = stats_by_ds[ds].get((label, it["model_name"]))
        if not avail:
            die(f"Khong tim thay du lieu cho {it['id']} ({ds} / label={label} / model={it['model_name']})")
        if it["kind"] == "fake" and matcher.classify(it["model_name"])["status"] != "allowed":
            verdict = matcher.classify(it["model_name"])
            if verdict["status"] == "blocked" or not allow_related:
                alts = _allowed_alternatives(cfg, index_rows, it["family"], matcher)
                die(
                    f"{it['id']}: generator '{it['model_name']}' bi deny-list chan "
                    f"({verdict['status']}, khop '{verdict['matched']}').\n"
                    "    KHONG duoc noi deny-list. Hai lua chon hop le:\n"
                    "      1) doi generator khac trong config (sua `fakes`),\n"
                    "      2) neu day la generator cung ho nhung khac phien ban, dat\n"
                    "         `allow_related_generators: true` trong config va khai bao\n"
                    "         trong bao cao (se duoc ghi vao `related_generators`).\n"
                    f"    Generator cung ho '{it['family']}' dang duoc phep: "
                    + ("; ".join(alts) if alts else "(khong con)"),
                )
            log(f"  [i] {it['id']}: '{it['model_name']}' la '{verdict['status']}' "
                f"(khop '{verdict['matched']}') - chap nhan vi allow_related_generators=true")
        if it["kind"] == "real" and it.get("real_source"):
            want_src = str(it["real_source"])
            avail = {s: v for s, v in avail.items()
                     if want_src in (v.get("real_sources") or set())}

        need = int(math.ceil(int(it["target"]) * surplus))
        usable = {s: v for s, v in avail.items() if (v["n"] - (v["nsfw"] if nsfw_on else 0)) > 0}
        total_usable = sum(v["n"] - (v["nsfw"] if nsfw_on else 0) for v in usable.values())
        if total_usable < int(it["target"]):
            die(
                f"{it['id']}: chi co {total_usable} anh kha dung < target {it['target']}. "
                "Tang nguon hoac giam target trong config."
            )
        already = chosen.setdefault(ds, set())
        picked: List[int] = []
        got = 0
        # 1) uu tien shard da chon (giam so shard phai tai)
        for shard in sorted(usable, key=lambda s: (-usable[s]["n"], s)):
            if got >= need:
                break
            if shard in already:
                picked.append(shard)
                got += usable[shard]["n"] - (usable[shard]["nsfw"] if nsfw_on else 0)
        # 2) neu chua du, uu tien shard du can ma dung luong byte nho nhat (tiet kiem bang thong)
        def _shard_priority(s: int):
            valid_n = usable[s]["n"] - (usable[s]["nsfw"] if nsfw_on else 0)
            if valid_n >= need:
                u = spec.url(s)
                sz = http_head_size(u, token, sizes)
                if sz > 0:
                    return (0, sz, -valid_n, s)
            return (1, -valid_n, 0, s)

        for shard in sorted(usable, key=_shard_priority):
            if got >= need:
                break
            if shard in picked:
                continue
            picked.append(shard)
            got += usable[shard]["n"] - (usable[shard]["nsfw"] if nsfw_on else 0)
        already.update(picked)

        shards = []
        for shard in sorted(picked):
            url = spec.url(shard)
            size = http_head_size(url, token, sizes)
            shards.append({"shard": shard, "url": url, "n": usable[shard]["n"], "size_bytes": size})
        plan_entries.append({
            "id": it["id"], "dataset": ds, "kind": it["kind"], "family": it["family"],
            "model_name": it["model_name"], "target": int(it["target"]),
            "need": need, "available": total_usable,
            "denylist_status": (matcher.classify(it["model_name"])["status"]
                                if it["kind"] == "fake" else "n/a"),
            "shards": shards,
            "predicted_bytes": sum(s["size_bytes"] for s in shards),
        })

    write_json(cfg.state("shard_sizes.json"), sizes)

    # ---- ngan sach ----
    shard_union: Dict[str, Dict[int, Dict]] = {}
    for e in plan_entries:
        for s in e["shards"]:
            shard_union.setdefault(e["dataset"], {})[s["shard"]] = s
    predicted = sum(s["size_bytes"] for ds in shard_union.values() for s in ds.values())
    budget = float(cfg.t.get("byte_budget_gb", 60)) * 1e9
    if predicted > budget:
        die(f"Shard plan exceeds byte budget: {predicted} > {budget}.")

    per_family: Dict[str, Dict[str, int]] = {}
    for e in plan_entries:
        if e["kind"] != "fake":
            continue
        st = per_family.setdefault(e["family"], {"n_candidates": 0, "n_blocked": 0,
                                                 "n_related": 0, "n_kept": 0})
        base = fam_stats.get(e["family"], {})
        st["n_candidates"] = base.get("n_candidates", 0)
        st["n_blocked"] = base.get("n_blocked", 0)
        st["n_related"] = base.get("n_related", 0)
        st["n_kept"] += e["target"]

    plan = {
        "created_at": now_iso(),
        "config_sha256": cfg.sha256,
        "seed": cfg.seed,
        "surplus": surplus,
        "limit_per_family": limit_per_family,
        "entries": plan_entries,
        "reals_target": sum(e["target"] for e in plan_entries if e["kind"] == "real"),
        "fakes_new_target": sum(e["target"] for e in plan_entries if e["kind"] == "fake"),
        "replay_composition": replay_composition(cfg),
        "predicted_bytes": predicted,
        "byte_budget": budget,
        "n_shards": sum(len(v) for v in shard_union.values()),
        "shards": {
            ds: [{"shard": s, "url": v["url"], "size_bytes": v["size_bytes"]}
                 for s, v in sorted(vv.items())]
            for ds, vv in shard_union.items()
        },
        "family_table": per_family,
    }
    return plan


def replay_composition(cfg: Config) -> Dict:
    """Doc kho replay cu de biet thanh phan that (khong doan)."""
    data = read_json(abspath(cfg.replay_cfg.get("manifest") or "DATASET/manifests/data_manifest.json"))
    if not data or "records" not in data:
        return {"n_real": 0, "n_fake": 0, "by_generator": {}}
    by_gen: Dict[str, int] = {}
    n_real = n_fake = 0
    allowed_gens = cfg.replay_cfg.get("allowed_generators")
    allowed_set = set(allowed_gens) if allowed_gens is not None else None
    for r in data["records"]:
        gen = str(r.get("generator") or "unknown")
        if allowed_set is not None and gen not in allowed_set:
            continue
        by_gen[gen] = by_gen.get(gen, 0) + 1
        if int(r.get("label", 0)) == 0:
            n_real += 1
        else:
            n_fake += 1
    return {"n_real": n_real, "n_fake": n_fake, "by_generator": dict(sorted(by_gen.items()))}


def print_plan(plan: Dict, cfg: Config) -> None:
    print()
    print("=" * 108)
    print("BANG CHUNG BUOC 1 - family | n_candidates | n_blocked | n_related | n_kept")
    print("=" * 108)
    print(f"{'family':<20}{'n_candidates':>16}{'n_blocked':>12}{'n_blocked_%':>13}"
          f"{'n_related':>12}{'n_kept(new)':>14}")
    print("-" * 108)
    for fam, st in sorted(plan["family_table"].items()):
        pct = 100.0 * st["n_blocked"] / max(1, st["n_candidates"])
        print(f"{fam:<20}{st['n_candidates']:>16,}{st['n_blocked']:>12,}{pct:>12.1f}%"
              f"{st['n_related']:>12,}{st['n_kept']:>14,}")
    print("-" * 108)
    print()
    print("=" * 100)
    print("KE HOACH TAI DU LIEU")
    print("=" * 100)
    print(f"{'id':<32}{'dataset':<17}{'family':<19}{'target':>8}{'shards':>8}")
    print("-" * 100)
    for e in plan["entries"]:
        print(f"{e['id']:<32}{e['dataset']:<17}{e['family']:<19}{e['target']:>8,}{len(e['shards']):>8}")
    print("-" * 100)
    new_f = plan["fakes_new_target"]
    new_r = plan["reals_target"]
    comp = plan.get("replay_composition", {})
    replay = int(comp.get("n_real", 0)) + int(comp.get("n_fake", 0))
    total = new_f + new_r + replay
    n_real_total = new_r + int(comp.get("n_real", 0))
    print(f"Fake moi        : {new_f:,}")
    print(f"Real moi        : {new_r:,}")
    print(f"Replay          : {replay:,}  ({comp.get('n_real', 0):,} real + {comp.get('n_fake', 0):,} fake)")
    print(f"TONG            : {total:,}  (real {100.0 * n_real_total / max(1, total):.2f}%, "
          f"replay {100.0 * replay / max(1, total):.2f}%)")
    print(f"Shard phai tai  : {plan['n_shards']}  |  du kien {plan['predicted_bytes'] / 1e9:.2f} GB"
          f"  (ngan sach {plan['byte_budget'] / 1e9:.0f} GB)")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Tai file (resume duoc) + doc shard
# ---------------------------------------------------------------------------
def download_file(url: str, dest: str, token: Optional[str], retries: int = 3,
                  on_bytes=None, check_deadline=None) -> Tuple[int, bool]:
    """Tai `url` ve `dest` voi Range-resume. Tra ve (bytes_moi, da_tai_lai)."""
    import requests

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    headers = {"User-Agent": "fatformer-xla/7.7"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    part = dest + ".part"
    have = os.path.getsize(dest) if os.path.exists(dest) else (
        os.path.getsize(part) if os.path.exists(part) else 0)
    if os.path.exists(dest) and not os.path.exists(part) and have:
        return 0, False

    session = requests.Session()
    transferred = 0
    for attempt in range(1, retries + 1):
        try:
            if check_deadline:
                check_deadline()
            h = dict(headers)
            if have:
                h["Range"] = f"bytes={have}-"
            with session.get(url, headers=h, stream=True, timeout=(30, 600)) as r:
                if r.status_code == 416:
                    expected = int(r.headers.get("Content-Range", "0/0").rsplit("/", 1)[-1])
                    if os.path.exists(part) and expected > 0 and os.path.getsize(part) == expected:
                        os.replace(part, dest)
                        return transferred, False
                    raise IOError("Invalid Range response; partial file is not verified complete")
                if r.status_code == 200 and have:
                    have = 0  # server khong ho tro Range -> tai lai tu dau
                    if os.path.exists(part):
                        os.remove(part)
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0) + have
                written = 0
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 22):
                        if chunk:
                            if check_deadline:
                                check_deadline()
                            transferred += len(chunk)
                            if on_bytes:
                                on_bytes(len(chunk))
                            f.write(chunk)
                            written += len(chunk)
                            if written % (1 << 28) < (1 << 22):
                                log(f"    ... {os.path.basename(dest)}: "
                                    f"{100.0 * (have + written) / max(1, total):.0f}%")
                size = os.path.getsize(part)
                if total and size < total:
                    raise IOError(f"tai thieu: {size}/{total}")
                os.replace(part, dest)
                return transferred, True
        except Exception as exc:  # noqa: BLE001
            have = os.path.getsize(part) if os.path.exists(part) else 0
            log(f"    [!] tai loi (lan {attempt}/{retries}): {type(exc).__name__}: {str(exc)[:100]}")
            time.sleep(3 * attempt)
    die(f"Khong tai duoc {url}")


def _iter_local_rows(path: str, columns: Optional[List[str]], batch_size: int = 64) -> Iterator[List[Dict]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pylist()


def _shard_row_count(path: str) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _quality_check(data: bytes, cfg: Config) -> Tuple[Optional[Dict], str]:
    """Giai ma + kiem tra chat luong. Tra ve (info, ly_do_tu_choi)."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            width, height = img.size
            mode = img.mode
            fmt = (img.format or "").lower()
            if min(width, height) < int(cfg.t.get("min_side", 200)):
                return None, "too_small"
            allowed = [f.lower() for f in cfg.quality.get("allowed_formats", [])]
            if allowed and fmt not in allowed:
                return None, f"format_{fmt or 'unknown'}"
            if cfg.quality.get("reject_fully_transparent", True) and mode in ("RGBA", "LA"):
                alpha = img.getchannel("A")
                lo, hi = alpha.getextrema()
                if hi == 0:
                    return None, "fully_transparent"
            if cfg.quality.get("reject_flat_images", True):
                lo, hi = img.convert("L").getextrema()
                if int(hi) - int(lo) <= 2:
                    return None, "flat_image"
            phash = phash_from_pil(img)
    except Exception:  # noqa: BLE001
        return None, "decode_error"
    return {"width": int(width), "height": int(height), "format": fmt or "png",
            "phash": int(phash), "mode": mode}, ""


def _extract_blob(row: Dict, kind: str) -> Optional[bytes]:
    if kind == "cfe":
        return row.get("image_data")
    for key in ("image", "img", "jpeg", "png"):
        v = row.get(key)
        if isinstance(v, dict):
            b = v.get("bytes")
            if b:
                return b
        elif isinstance(v, (bytes, bytearray)):
            return bytes(v)
    return None


def _select_rows_for_shard(
    cfg: Config,
    spec: DatasetSpec,
    shard: int,
    local: str,
    entries: List[Dict],
    per_entry: Dict[str, int],
    excluded=None,
) -> Tuple[Dict[str, str], Optional[Dict[str, Dict]], Dict[str, int]]:
    """Chon image_id cho tung entry trong 1 shard. Tra ve (entry_of_name, row_meta, can)."""
    nsfw_on = bool(cfg.quality.get("reject_nsfw", True))
    excluded = excluded or set()
    can: Dict[str, int] = {}
    if spec.kind == "plain":
        n = _shard_row_count(local)
        pool = [f"{shard}:{i}" for i in range(n) if f"{shard}:{i}" not in excluded]
        row_meta = {k: {"model_name": spec.model_name or spec.name, "label": 1,
                        "nsfw_flag": False, "real_source": None} for k in pool}
    else:
        row_meta = {}
        for batch in _iter_local_rows(
                local, ["image_name", "model_name", "label", "nsfw_flag", "real_source"], 256):
            for row in batch:
                row_meta[row["image_name"]] = row
        pool = None

    entry_of_name: Dict[str, str] = {}
    for e in entries:
        need = int(e["need"])
        have = int(per_entry.get(e["id"], 0))
        if have >= need:
            continue
        want = need - have
        if spec.kind == "plain":
            names = pool
            # Neu nhu cau < so anh trong shard, uu tien phan bo deu (buoc nhay)
            if want < len(names):
                step = len(names) / float(want)
                names = [names[int(i * step)] for i in range(want)]
        else:
            label = 1 if e["kind"] == "fake" else 0
            names = [n for n, m in row_meta.items()
                     if n not in excluded and m.get("model_name") == e["model_name"]
                     and int(m.get("label", 1)) == label
                     and not (nsfw_on and m.get("nsfw_flag"))]
            if e["kind"] == "real" and e.get("real_source"):
                names = [n for n in names
                         if (row_meta[n].get("real_source") or "") == e["real_source"]]
            names.sort(key=lambda n: stable_rank(cfg.seed, e["id"], spec.name, shard, n))
            names = names[:want]
        for n in names:
            if n in entry_of_name:
                continue
            entry_of_name[n] = e["id"]
        can[e["id"]] = len(names)
    return entry_of_name, row_meta, can


def stage_fetch(
    cfg: Config,
    plan: Dict,
    cache_dir: str,
    resume: bool,
    token: Optional[str],
) -> Dict:
    ensure_state_context(cfg)
    cand_path = cfg.state("candidates.jsonl")
    done_path = cfg.state("fetch_done.json")
    bytes_path = cfg.state("bytes_downloaded.json")

    if not resume and os.path.exists(cand_path):
        os.remove(cand_path)
    existing = read_jsonl(cand_path, recover_tail=True) if resume else []
    done = (read_json(done_path, {}) or {}) if resume else {}
    bytes_dl = read_json(bytes_path, {"bytes": 0}) or {"bytes": 0}
    # Recover images lost/corrupted after a runtime reset before honoring done markers.
    valid = []
    broken_shards = set()
    for row in existing:
        path = abspath(row["path"])
        if os.path.isfile(path) and sha256_file(path) == row["sha256"]:
            valid.append(row)
        else:
            broken_shards.add(cfg.datasets[row["dataset"]].shard_key(row["shard"]))
    if len(valid) != len(existing):
        tmp = cand_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for row in valid:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, cand_path)
        existing = valid
    for key in broken_shards:
        done.pop(key, None)
    if resume and done and not existing:
        done = {}
    write_json(done_path, done)
    started = time.monotonic()
    prior_seconds = float(bytes_dl.get("fetch_seconds", 0))

    def checkpoint_budget():
        bytes_dl["fetch_seconds"] = prior_seconds + time.monotonic() - started
        write_json(bytes_path, bytes_dl)
        if bytes_dl["fetch_seconds"] >= float(cfg.t.get("wall_clock_budget_h", 4)) * 3600:
            die("Fetch wall-clock budget reached. Checkpoint saved; stop and review the plan.")
        if bytes_dl.get("bytes", 0) >= float(cfg.t.get("byte_budget_gb", 60)) * 1e9:
            die("Download byte budget reached. Checkpoint saved; stop and review the plan.")

    def account_bytes(n):
        bytes_dl["bytes"] = bytes_dl.get("bytes", 0) + n
        checkpoint_budget()
    seen = {(r["dataset"], r["shard"], r["image_id"]) for r in existing}
    per_entry: Dict[str, int] = {}
    for r in existing:
        per_entry[r["entry_id"]] = per_entry.get(r["entry_id"], 0) + 1
    if existing:
        log(f"FETCH: resume voi {len(existing)} anh da co")

    raw_root = cfg.path_of("raw_root")
    entry_by_id = {e["id"]: e for e in plan["entries"]}

    total_rows = 0
    for ds_name, shards in plan["shards"].items():
        spec = cfg.datasets[ds_name]
        for s in shards:
            checkpoint_budget()
            shard = s["shard"]
            key = spec.shard_key(shard)
            if key in done:
                log(f"FETCH {key}: da xong ({done[key]} anh), bo qua")
                continue
            local = os.path.join(cache_dir, ds_name, f"shard_{shard:05d}.parquet")
            if os.path.exists(local) and os.path.getsize(local) > 0:
                if os.path.getsize(local) != s["size_bytes"]:
                    die(f"Cached shard size mismatch: {local}. Remove the invalid cache file and resume.")
                log(f"FETCH {key}: cache co san ({os.path.getsize(local) / 1e6:.0f} MB)")
            else:
                log(f"FETCH {key}: tai {s['url']}")
                download_file(s["url"], local, token, on_bytes=account_bytes,
                              check_deadline=checkpoint_budget)

            entries = [entry_by_id[eid] for eid, e in entry_by_id.items()
                       if e["dataset"] == ds_name and any(x["shard"] == shard for x in e["shards"])]
            if not entries:
                done[key] = 0
                write_json(done_path, done)
                continue

            entry_of_name, row_meta, can = _select_rows_for_shard(
                cfg, spec, shard, local, entries, per_entry,
                {name for ds, sh, name in seen if ds == ds_name and sh == shard})
            if not entry_of_name:
                done[key] = 0
                write_json(done_path, done)
                continue
            log(f"FETCH {key}: can {len(entry_of_name)} anh / {len(entries)} generator "
                f"({sum(can.values())} ung vien)")

            cols = ["image_name", "image_data"] if spec.kind == "cfe" else ["image"]
            kept = 0
            rejected: Dict[str, int] = {}
            buffer: List[Dict] = []
            idx = 0
            for batch in _iter_local_rows(local, cols, batch_size=64):
                checkpoint_budget()
                for row in batch:
                    name = row.get("image_name") if spec.kind == "cfe" else f"{shard}:{idx}"
                    idx += 1
                    eid = entry_of_name.get(name)
                    if eid is None or (ds_name, shard, name) in seen:
                        continue
                    blob = _extract_blob(row, spec.kind)
                    if not blob:
                        rejected["no_bytes"] = rejected.get("no_bytes", 0) + 1
                        continue
                    info, why = _quality_check(blob, cfg)
                    if info is None:
                        rejected[why] = rejected.get(why, 0) + 1
                        continue
                    e = entry_by_id[eid]
                    sha = sha256_bytes(blob)
                    ext = "jpg" if info["format"] in ("jpeg", "jpg") else info["format"]
                    out_dir = os.path.join(raw_root, e["family"], slug(e["model_name"]))
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"{sha[:20]}.{ext}")
                    if not os.path.exists(out_path) or sha256_file(out_path) != sha:
                        with open(out_path + ".tmp", "wb") as f:
                            f.write(blob)
                        os.replace(out_path + ".tmp", out_path)
                    buffer.append({
                        "entry_id": eid, "dataset": ds_name, "shard": shard, "image_id": name,
                        "path": rel(out_path), "sha256": sha, "phash": info["phash"],
                        "width": info["width"], "height": info["height"],
                        "format": ext, "bytes": len(blob),
                        "label": 1 if e["kind"] == "fake" else 0,
                        "family": e["family"], "generator": e["model_name"],
                        "source": e["model_name"] if e["kind"] == "real" else ds_name,
                        "real_source": e.get("real_source") or (
                            e["model_name"] if e["kind"] == "real" else None),
                        "is_replay": False,
                    })
                    per_entry[eid] = per_entry.get(eid, 0) + 1
                    seen.add((ds_name, shard, name))
                    kept += 1
                if len(buffer) >= 128:
                    append_jsonl(cand_path, buffer)
                    total_rows += len(buffer)
                    buffer = []
            if buffer:
                append_jsonl(cand_path, buffer)
                total_rows += len(buffer)
            done[key] = kept
            write_json(done_path, done)
            rej = ", ".join(f"{k}={v}" for k, v in sorted(rejected.items())) or "khong loai"
            log(f"FETCH {key}: +{kept} anh ({rej})")
    checkpoint_budget()

    log(f"FETCH: tong {total_rows} anh moi, da tai {bytes_dl['bytes'] / 1e9:.2f} GB")
    return {"n_new": total_rows, "bytes_downloaded": bytes_dl["bytes"], "done": done}


# ---------------------------------------------------------------------------
# REPLAY
# ---------------------------------------------------------------------------
def stage_replay(cfg: Config, replay_root: str, copy_mode: str = "copy") -> Dict:
    ensure_state_context(cfg)
    check_replay_generators(cfg)
    src_manifest = abspath(cfg.replay_cfg.get("manifest") or "DATASET/manifests/data_manifest.json")
    data = read_json(src_manifest)
    if not data or "records" not in data:
        die(f"Khong doc duoc replay manifest: {src_manifest}")
    records = data["records"]
    allowed_gens = cfg.replay_cfg.get("allowed_generators")
    if allowed_gens is not None:
        allowed_set = set(allowed_gens)
        records = [r for r in records if r.get("generator") in allowed_set]
    want = int(cfg.replay_cfg.get("total") or cfg.t.get("replay_total", 3000))
    if len(records) != want:
        die(f"Replay manifest sau khi loc chi co {len(records)} mau != {want}")

    fam_map = cfg.replay_cfg.get("family_of_generator", {})
    disp = cfg.replay_cfg.get("generator_display", {})
    real_src = cfg.replay_cfg.get("real_source_of_generator", {})
    markers = cfg.replay_cfg.get("path_markers", ["diffusion_staging/"])
    raw_root = cfg.path_of("raw_root")
    out_path = cfg.state("replay.jsonl")
    cached = read_jsonl(out_path)
    if len(cached) == want and all(os.path.isfile(abspath(r["path"]))
                                  and sha256_file(abspath(r["path"])) == r["sha256"] for r in cached):
        log("REPLAY: da co du, bo qua")
        return {"n": want, "cached": True}

    rows: List[Dict] = []
    missing = 0
    for r in records:
        old = str(r.get("file_path") or r.get("path") or "").replace("\\", "/")
        # Manifest cu ghi duong dan /kaggle/input/.../diffusion_staging/diffusion_staging/<...>.
        # Thu moi vi tri cua marker (uu tien vi tri CUOI = thu muc goc that su) va lay
        # ung vien dau tien ton tai tren dia - ho tro ca 2 cach giai nen tar.
        cands: List[str] = []
        for m in markers:
            positions, idx = [], old.find(m)
            while idx >= 0:
                positions.append(idx)
                idx = old.find(m, idx + 1)
            for idx in reversed(positions):
                cand = old[idx + len(m):]
                if cand and cand not in cands:
                    cands.append(cand)
        src = tail = None
        for cand in cands:
            p = os.path.join(replay_root, cand.replace("/", os.sep))
            if os.path.isfile(p):
                src, tail = p, cand
                break
        if src is None:
            missing += 1
            continue
        with open(src, "rb") as f:
            blob = f.read()
        info, why = _quality_check(blob, cfg)
        if info is None:
            log(f"  [!] replay bi loai ({why}): {tail}")
            continue
        gen = str(r.get("generator") or "unknown")
        fam = fam_map.get(gen, "real" if int(r.get("label", 0)) == 0 else "commercial")
        gen_name = disp.get(gen, gen)
        sha = sha256_bytes(blob)
        ext = "jpg" if info["format"] in ("jpeg", "jpg") else info["format"]
        out_dir = os.path.join(raw_root, fam, slug(gen_name))
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, f"{sha[:20]}.{ext}")
        if os.path.abspath(dst) != os.path.abspath(src):
            if not os.path.exists(dst) or sha256_file(dst) != sha:
                if copy_mode == "hardlink":
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                elif copy_mode == "copy":
                    shutil.copy2(src, dst)
                else:
                    dst = src
        rows.append({
            "entry_id": f"replay_{gen}", "dataset": "diffusion_staging", "shard": -1,
            "image_id": tail, "path": rel(dst), "sha256": sha, "phash": info["phash"],
            "width": info["width"], "height": info["height"], "format": ext,
            "bytes": len(blob), "label": int(r.get("label", 0)), "family": fam,
            "generator": gen_name, "source": "diffusion_staging",
            "real_source": real_src.get(gen), "is_replay": True,
        })
    if missing:
        die(f"REPLAY: {missing} tep khong tim thay duoi {replay_root}. "
            "Giai nen diffusion_staging.tar truoc, hoac truyen --replay_root dung.")
    if len(rows) != want:
        die(f"REPLAY: chi nap duoc {len(rows)}/{want} mau.")
    with open(out_path + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(out_path + ".tmp", out_path)
    log(f"REPLAY: {len(rows)} mau ({sum(r['label'] == 0 for r in rows)} real)")
    return {"n": len(rows), "cached": False}


# ---------------------------------------------------------------------------
# SELECT: dedupe + quota
# ---------------------------------------------------------------------------
def _dedupe(records: List[Dict], threshold: int) -> Tuple[List[Dict], Dict[str, int]]:
    """Khu trung lap SHA-256 + pHash <= threshold. Giu ban dau tien theo thu tu.

    So sanh ca TRONG cung mot lo (khong chi voi lo truoc): neu chi so voi
    `kept_phash` thi hai anh gan trung nam canh nhau se lot qua. Vi vay moi lo
    512 anh duoc doi chieu ca voi nhau (ma tran 512x512) va voi phan da giu.
    """
    kept: List[Dict] = []
    seen_sha: set = set()
    kept_phash = np.zeros(0, dtype=np.uint64)
    stats = {"sha_dup": 0, "phash_dup": 0}

    def flush(chunk: List[Dict]) -> None:
        nonlocal kept_phash
        if not chunk:
            return
        arr = to_uint64([c["phash"] for c in chunk])
        if len(kept_phash):
            mask = min_hamming_to_set(arr, kept_phash) > threshold
        else:
            mask = np.ones(len(arr), dtype=bool)
        intra = pairwise_hamming_matrix(arr, arr, chunk_size=512) <= threshold
        for i in range(len(arr)):
            if mask[i]:
                mask[i + 1:] &= ~intra[i, i + 1:]
        sel = [c for c, keep in zip(chunk, mask) if keep]
        stats["phash_dup"] += len(chunk) - len(sel)
        kept.extend(sel)
        if sel:
            kept_phash = np.concatenate([kept_phash, to_uint64([c["phash"] for c in sel])])

    chunk: List[Dict] = []
    for r in records:
        if r["sha256"] in seen_sha:
            stats["sha_dup"] += 1
            continue
        seen_sha.add(r["sha256"])
        chunk.append(r)
        if len(chunk) >= 512:
            flush(chunk)
            chunk = []
    flush(chunk)
    return kept, stats


def _filter_against_refs(records: List[Dict], refs: Dict[str, Dict], threshold: int) -> Tuple[List[Dict], Dict[str, int]]:
    """Loai anh trung/tuong dong voi tap test (theo SHA-256 + pHash).

    `stats` chi ghi cac muc CO loai thuc su (khac 0) de `if stats:` la dieu kien
    "co vi pham" chuan xac, khong bi danh lua boi cac muc dem bang 0.
    """
    if not refs:
        return records, {}
    stats: Dict[str, int] = {}
    for name, payload in refs.items():
        sha_set = set(payload.get("sha256", []))
        ph = to_uint64(payload.get("phash", []))
        keep = []
        for r in records:
            if r["sha256"] in sha_set:
                stats[f"{name}_sha"] = stats.get(f"{name}_sha", 0) + 1
                continue
            keep.append(r)
        records = keep
        if len(ph) and len(records):
            arr = to_uint64([r["phash"] for r in records])
            d = min_hamming_to_set(arr, ph)
            keep = [r for r, dd in zip(records, d) if dd > threshold]
            dropped = len(records) - len(keep)
            if dropped:
                stats[f"{name}_phash"] = dropped
            records = keep
    return records, stats


def filter_reference_generators(records: List[Dict], refs: Dict[str, Dict]):
    """Enforce generator isolation for every indexed group, including GAN extras."""
    names = {str(row.get("model_name") or row.get("generator") or "")
             for ref in refs.values() for row in ref.get("images", []) if row.get("label") == 1}
    matcher = DenyMatcher(name for name in names if name)
    kept, blocked = [], []
    for row in records:
        if row["label"] == 1 and matcher.classify(row["generator"])["status"] != "allowed":
            blocked.append(row)
        else:
            kept.append(row)
    return kept, ({"reference_generator_overlap": len(blocked)} if blocked else {})


def stage_select(cfg: Config, plan: Dict, refs: Dict[str, Dict], strict: bool) -> Dict:
    ensure_state_context(cfg)
    check_replay_generators(cfg)
    if not refs:
        die("Selection requires complete CFE and Task 5.1 reference indexes.")
    audit_cfg = cfg.data.get("audit", {})
    groups = {role: [v for v in refs.values() if v.get("metadata", {}).get("role") == role]
              for role in ("cfe", "test18")}
    if sum(len(v["images"]) for v in groups["cfe"]) != int(audit_cfg.get("cfe_images", 2000)):
        die("CFE reference coverage does not match expected image count.")
    sets = {i.get("test_set") for v in groups["test18"] for i in v["images"] if i.get("test_set")}
    if sets != set(audit_cfg.get("task51_tests", EXPECTED_TASK51_TESTS)):
        die("Task 5.1 reference coverage must include all expected test sets.")
    write_json(cfg.state("references.json"), {v["path"]: v["fingerprint"] for v in refs.values()})
    cands = read_jsonl(cfg.state("candidates.jsonl"))
    replay = read_jsonl(cfg.state("replay.jsonl"))
    log(f"SELECT: {len(cands)} ung vien + {len(replay)} replay")

    # Replay uu tien (khong bi dedupe loai) - no la mau bat buoc.
    replay_sorted = sorted(replay, key=lambda r: stable_rank(cfg.seed, "replay", r["sha256"]))
    replay_sorted, replay_gen_stats = filter_reference_generators(replay_sorted, refs)
    if replay_gen_stats:
        die(f"Replay generator overlaps an indexed test group: {replay_gen_stats}")
    external_thr = int(cfg.t.get("phash_external", 8))
    replay, ref_stats_r = _filter_against_refs(replay_sorted, refs, external_thr)
    if ref_stats_r:
        log(f"SELECT: replay bi loai do trung test: {ref_stats_r}")
        die(
            "Replay co mau trung/tuong dong voi tap test. Day la mau bat buoc (3.600) "
            "va khong the thay the tu dong. Xem chi tiet trong bao cao va cap nhat "
            "replay manifest (hoac dung --replay_root khac)."
        )

    by_entry: Dict[str, List[Dict]] = {}
    for r in cands:
        by_entry.setdefault(r["entry_id"], []).append(r)
    for eid in by_entry:
        by_entry[eid].sort(key=lambda r: stable_rank(cfg.seed, "sel", eid, r["sha256"]))

    # Khu trung lap trong noi bo phan moi (replay da giu cho).
    all_new = [r for eid in sorted(by_entry) for r in by_entry[eid]]
    all_new, generator_stats = filter_reference_generators(all_new, refs)
    all_new, ref_stats_new = _filter_against_refs(all_new, refs, external_thr)
    ref_stats_new.update(generator_stats)
    if ref_stats_new:
        log(f"SELECT: loai do trung test: {ref_stats_new}")
    deduped, dstats = _dedupe(replay + all_new, int(cfg.t.get("phash_internal", 3)))
    log(f"SELECT: dedupe sha={dstats['sha_dup']}, phash={dstats['phash_dup']}")

    pool: Dict[str, List[Dict]] = {}
    for r in deduped:
        pool.setdefault(r["entry_id"], []).append(r)

    targets = {e["id"]: int(e["target"]) for e in plan["entries"]}
    replay_pool: Dict[str, List[Dict]] = {}
    for r in replay:
        replay_pool.setdefault(r["entry_id"], []).append(r)
    targets.update({eid: len(rows) for eid, rows in replay_pool.items()})

    families: Dict[str, List[str]] = {}
    for e in plan["entries"]:
        families.setdefault(e["family"], []).append(e["id"])
    families["replay"] = sorted(replay_pool)

    chosen: Dict[str, List[Dict]] = {}
    for eid, tgt in targets.items():
        chosen[eid] = pool.get(eid, [])[:tgt]

    # Bu thieu trong noi bo ho. Muc tieu cua DoD la TONG cua ho (family), khong
    # phai tung generator: khi mot generator bi khu trung lap/test-filter lam hut,
    # ta giu nguyen tong cua ho va lay bu tu cac generator con du trong CUNG ho.
    # Viec phan bo lai duoc ghi vao `generator_reallocation` de bao cao minh bach.
    realloc: Dict[str, Dict[str, int]] = {}
    for fam, eids in families.items():
        if fam == "replay":
            continue
        fam_target = sum(targets[e] for e in eids)
        for _ in range(16):
            need = fam_target - sum(len(chosen.get(e, [])) for e in eids)
            if need <= 0:
                break
            spare = {e: len(pool.get(e, [])) - len(chosen.get(e, [])) for e in eids
                     if len(pool.get(e, [])) > len(chosen.get(e, []))}
            if not spare:
                break
            add = largest_remainder(need, spare)
            for e, k in add.items():
                if k <= 0:
                    continue
                cur = len(chosen.get(e, []))
                chosen[e] = pool.get(e, [])[:cur + k]
                realloc.setdefault(fam, {})[e] = realloc.get(fam, {}).get(e, 0) + k
        missing = fam_target - sum(len(chosen.get(e, [])) for e in eids)
        if missing:
            detail = ", ".join(f"{e}: co {len(pool.get(e, []))}/{targets[e]}" for e in eids)
            msg = (f"Ho '{fam}' thieu {missing} anh sau khi khu trung lap ({detail}).")
            if strict:
                die(msg + " Tang `surplus` hoac bo sung generator trong config.")
            log(f"[!] {msg} (che do khong strict)")
        elif realloc.get(fam):
            log(f"  [i] Ho '{fam}': phan bo lai trong ho {realloc[fam]} (tong ho khong doi)")

    final: List[Dict] = []
    for eid in sorted(chosen):
        final.extend(chosen[eid])
    final.sort(key=lambda r: (r["family"], r["label"], r["generator"], r["sha256"]))
    with open(cfg.state("selected.jsonl") + ".tmp", "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(cfg.state("selected.jsonl") + ".tmp", cfg.state("selected.jsonl"))
    log(f"SELECT: {len(final)} anh cuoi (ref_stats={ {**ref_stats_r, **ref_stats_new} })")
    summary = {"n": len(final), "dedupe": dstats,
               "ref_stats": {**ref_stats_r, **ref_stats_new},
               "generator_reallocation": realloc}
    write_json(cfg.state("select_report.json"), summary)
    return summary


# ---------------------------------------------------------------------------
# SPLIT
# ---------------------------------------------------------------------------
def generator_partition(rows, counts):
    """Exact group assignment when generator sizes permit both validation quotas."""
    groups = {}
    for r in rows:
        groups.setdefault(r["generator"], []).append(r)
    a, b = counts["val_calib"], counts["val_select"]
    states = {(0, 0): ()}
    for name in sorted(groups):
        size = len(groups[name])
        nxt = {}
        for (x, y), choices in states.items():
            nxt.setdefault((x, y), choices + ("train",))
            if x + size <= a:
                nxt.setdefault((x + size, y), choices + ("val_calib",))
            if y + size <= b:
                nxt.setdefault((x, y + size), choices + ("val_select",))
        states = nxt
        if len(states) > 200000:
            return None, "group_search_limit; manual group allocation required"
    if (a, b) not in states:
        return None, "generator_sizes_cannot_match_validation_quotas"
    assignment = dict(zip(sorted(groups), states[(a, b)]))
    return assignment, None


def stage_split(cfg: Config) -> Dict:
    ensure_state_context(cfg)
    records = read_jsonl(cfg.state("selected.jsonl"))
    ratios = {k: float(v) for k, v in cfg.t["split_ratios"].items()}
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        die(f"Invalid split_ratios: {ratios}")
    cells = {}
    for r in records:
        key = (r["family"], r["label"], bool(r["is_replay"]),
               r.get("real_source", "") if r["label"] == 0 else "")
        cells.setdefault(key, []).append(r)
    allocations = {k: largest_remainder(len(v), ratios) for k, v in cells.items()}
    desired = largest_remainder(len(records), ratios)
    totals = {s: sum(a[s] for a in allocations.values()) for s in SPLITS}
    # Correct rounding across strata so total split counts are exact as well.
    while totals != desired:
        donor = next(s for s in SPLITS if totals[s] > desired[s])
        receiver = next(s for s in SPLITS if totals[s] < desired[s])
        candidates = [k for k, a in allocations.items() if a[donor] > 0]
        key = min(candidates, key=lambda k: (
            abs(allocations[k][donor] - 1 - len(cells[k]) * ratios[donor])
            + abs(allocations[k][receiver] + 1 - len(cells[k]) * ratios[receiver]), str(k)))
        allocations[key][donor] -= 1
        allocations[key][receiver] += 1
        totals[donor] -= 1
        totals[receiver] += 1
    policy = []
    for key, rows in cells.items():
        rows.sort(key=lambda r: stable_rank(cfg.seed, "split", r["sha256"]))
        counts = allocations[key]
        assignment, reason = (None, "real_source_stratification" if key[1] == 0 else "fixed_replay")
        if key[1] == 1 and not key[2]:
            assignment, reason = generator_partition(rows, counts)
            if reason and reason.startswith("group_search_limit"):
                die(reason)
        if assignment is not None:
            for r in rows:
                r["split"] = assignment[r["generator"]]
        else:
            i = 0
            for sp in SPLITS:
                for r in rows[i:i + counts[sp]]:
                    r["split"] = sp
                i += counts[sp]
        policy.append({"family": key[0], "label": key[1], "is_replay": key[2],
                       "real_source": key[3], "mode": "generator" if assignment else "image",
                       "reason": reason, "generators": sorted({r["generator"] for r in rows}),
                       "counts": counts})
    with open(cfg.state("split.jsonl") + ".tmp", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(cfg.state("split.jsonl") + ".tmp", cfg.state("split.jsonl"))
    write_json(cfg.state("split_policy.json"), policy)
    log(f"SPLIT: {totals}")
    return {"counts": totals, "policy": policy}


# ---------------------------------------------------------------------------
# FINALIZE
# ---------------------------------------------------------------------------
def _counts(records: Sequence[Dict]) -> Dict:
    def tally(keys, pred=lambda r: True):
        out: Dict[str, int] = {}
        for r in records:
            if not pred(r):
                continue
            k = "|".join(str(r[x]) for x in keys)
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    return {
        "by_split": tally(["split"]),
        "by_family": tally(["family"]),
        "by_family_split": tally(["family", "split"]),
        "by_label": tally(["label"]),
        "by_label_split": tally(["label", "split"]),
        "by_source": tally(["source"]),
        "by_real_source": tally(["real_source"], lambda r: r["label"] == 0),
        "by_generator": tally(["family", "generator"]),
        "replay_by_family": tally(["family"], lambda r: r["is_replay"]),
    }


def evaluate_constraints(cfg: Config, records: Sequence[Dict]) -> Dict:
    t = cfg.t
    n = len(records)
    splits = list(t["split_ratios"])
    replay = [r for r in records if r["is_replay"]]
    by_split = {sp: [r for r in records if r["split"] == sp] for sp in splits}
    fam_tot = {f: sum(1 for r in records if r["family"] == f) for f in {r["family"] for r in records}}
    fam_new = {f: sum(1 for r in records if r["family"] == f and not r["is_replay"])
               for f in fam_tot}
    fam_fake_tot = {f: v for f, v in fam_tot.items() if f != "real"}
    fam_fake_new = {f: v for f, v in fam_new.items() if f != "real"}
    real_frac = {sp: (sum(1 for r in by_split[sp] if r["label"] == 0) / max(1, len(by_split[sp])))
                 for sp in splits}
    fam_vals = [v for v in fam_fake_tot.values() if v > 0]
    imbalance = (max(fam_vals) / min(fam_vals)) if fam_vals else 0.0

    checks: Dict[str, Dict] = {}

    def check(name, ok, value, requirement):
        checks[name] = {"pass": bool(ok), "value": value, "requirement": requirement}

    check("total_in_range", int(t["min_total"]) <= n <= int(t["max_total"]), n,
          f"{t['min_total']}..{t['max_total']}")
    check("five_families_present", all(fam_tot.get(f, 0) > 0
                                       for f in ("commercial", "latent_diffusion", "pixel_diffusion",
                                                 "modern_gans", "autoregressive")),
          sorted(fam_tot), "5 ho deu co mat")
    check("new_per_family_le_max", all(v <= int(t["max_new_per_family"]) for v in fam_fake_new.values()),
          fam_fake_new, f"<= {t['max_new_per_family']} anh moi/ho")
    check("family_imbalance_le", imbalance <= float(t["max_family_imbalance"]) + 1e-9,
          round(imbalance, 4), f"<= {t['max_family_imbalance']} (tinh tren tong ho fake, gom replay)")
    check("replay_total", len(replay) == int(t["replay_total"]), len(replay), f"= {t['replay_total']}")
    replay_frac = len(replay) / max(1, n)
    lo, hi = t["replay_frac"]
    check("replay_frac_in_range", lo - 1e-9 <= replay_frac <= hi + 1e-9,
          round(replay_frac, 5), f"{lo}..{hi}")
    rlo, rhi = t["real_frac"]
    check("real_frac_per_split", all(rlo - 1e-9 <= v <= rhi + 1e-9 for v in real_frac.values()),
          {k: round(v, 5) for k, v in real_frac.items()}, f"{rlo}..{rhi} moi split")
    split_actual = {sp: len(by_split[sp]) / max(1, n) for sp in splits}
    tol = float(t.get("split_ratio_tolerance", 0.005))
    check("split_ratio", all(abs(split_actual[sp] - float(t["split_ratios"][sp])) <= tol
                             for sp in splits),
          {k: round(v, 5) for k, v in split_actual.items()}, f"85/7.5/7.5 (+-{tol*100:.1f}pp)")
    split_fam = {sp: {f: sum(1 for r in by_split[sp] if r["family"] == f) for f in fam_fake_tot}
                 for sp in splits}
    split_imbal = {sp: (max(v.values()) / max(1, min(vv for vv in v.values() if vv > 0))
                        if any(v.values()) else 0.0) for sp, v in split_fam.items()}
    check("family_imbalance_per_split",
          all(v <= float(t["max_family_imbalance"]) + 1e-6 for v in split_imbal.values()),
          {k: round(v, 4) for k, v in split_imbal.items()}, f"<= {t['max_family_imbalance']}")
    check("required_fields",
          all(all(r.get(f) not in (None, "") for f in
                  ("path", "sha256", "phash", "generator", "family", "split", "label", "source",
                   "width", "height")) for r in records),
          sum(1 for r in records if any(r.get(f) in (None, "") for f in
                                        ("path", "sha256", "phash", "generator", "family", "split",
                                         "label", "source", "width", "height"))),
          "0 ban ghi thieu truong")
    check("unique_sha256", len({r["sha256"] for r in records}) == n,
          n - len({r["sha256"] for r in records}), "0 trung SHA-256")
    check("splits_disjoint_by_sha",
          len({(r["sha256"], r["split"]) for r in records}) == n, True, "moi anh chi o 1 split")
    check("min_side", all(min(r["width"], r["height"]) >= int(t["min_side"]) for r in records),
          min(min(r["width"], r["height"]) for r in records), f">= {t['min_side']}")
    check("replay_has_flag", all("is_replay" in r for r in records), True, "co truong is_replay")
    additional = contract_problems(records, cfg.t)
    check("data_contract", not additional, additional, "full Task 7.7 contract")

    all_pass = all(c["pass"] for c in checks.values())
    return {
        "all_pass": all_pass,
        "n_failed": sum(1 for c in checks.values() if not c["pass"]),
        "checks": checks,
        "family_totals": fam_tot,
        "family_new_only": fam_new,
        "real_frac_by_split": real_frac,
        "replay_frac": replay_frac,
        "family_imbalance": imbalance,
    }


def _distribution_png(records: Sequence[Dict], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = ["commercial", "latent_diffusion", "pixel_diffusion", "modern_gans", "autoregressive", "real"]
    splits = list(SPLITS)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=150)
    width = 0.26
    x = np.arange(len(fams))
    for i, sp in enumerate(splits):
        vals = [sum(1 for r in records if r["split"] == sp and r["family"] == f) for f in fams]
        axes[0].bar(x + (i - 1) * width, vals, width, label=sp)
    axes[0].set_xticks(x, [f.replace("_", "\n") for f in fams])
    axes[0].set_title("Phan bo family x split (kho v2)")
    axes[0].set_ylabel("so anh")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    for i, sp in enumerate(splits):
        sub = [r for r in records if r["split"] == sp]
        fake = sum(1 for r in sub if r["label"] == 1)
        real = len(sub) - fake
        axes[1].bar([i - 0.2], [real], 0.4, label="real" if i == 0 else None, color="tab:blue")
        axes[1].bar([i + 0.2], [fake], 0.4, label="fake" if i == 0 else None, color="tab:orange")
        axes[1].text(i, max(real, fake) * 1.01, f"{100.0 * real / max(1, len(sub)):.1f}% real",
                     ha="center", fontsize=9)
    axes[1].set_xticks(range(len(splits)), splits)
    axes[1].set_title("Real / Fake trong tung split")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _constraints_md(report: Dict, cfg: Config) -> str:
    lines = ["# Rang buoc kho du lieu v2 (Task 7.7)", ""]
    lines.append(f"- Tong anh: **{report['counts']['by_split'] and sum(report['counts']['by_split'].values()):,}**")
    lines.append(f"- Ket qua: **{'PASS' if report['constraints']['all_pass'] else 'FAIL'}**"
                 f" ({report['constraints']['n_failed']} muc khong dat)")
    lines.append("")
    lines.append("| Rang buoc | Gia tri | Yeu cau | Ket qua |")
    lines.append("| :--- | ---: | :--- | :---: |")
    for name, c in report["constraints"]["checks"].items():
        lines.append(f"| `{name}` | {c['value']} | {c['requirement']} | {'PASS' if c['pass'] else 'FAIL'} |")
    lines.append("")
    lines.append("## Phan bo theo ho (gom replay)")
    lines.append("")
    lines.append("| Ho | Tong | Moi (khong replay) | Replay |")
    lines.append("| :--- | ---: | ---: | ---: |")
    for fam, n in sorted(report["constraints"]["family_totals"].items()):
        new = report["constraints"]["family_new_only"].get(fam, 0)
        lines.append(f"| {fam} | {n:,} | {new:,} | {n - new:,} |")
    return "\n".join(lines) + "\n"


def stage_finalize(cfg: Config, plan: Dict, extra: Dict) -> Dict:
    ensure_state_context(cfg)
    references = read_json(cfg.state("references.json"), {})
    if not references or any(not os.path.isfile(p) or sha256_file(p) != digest for p, digest in references.items()):
        die("Reference indexes missing or changed since selection; rerun select/split/finalize.")
    records = read_jsonl(cfg.state("split.jsonl"))
    if not records:
        die("Khong co ban ghi nao trong split.jsonl")
    records.sort(key=lambda r: (r["split"], r["family"], str(r["label"]), r["generator"], r["sha256"]))
    counts = _counts(records)
    constraints = evaluate_constraints(cfg, records)
    if not constraints["all_pass"]:
        write_json(cfg.path_of("constraint_report"), {"constraints": constraints})
        failed = [k for k, v in constraints["checks"].items() if not v["pass"]]
        die(f"Refusing to publish invalid manifest: {failed}")
    missing = [r["path"] for r in records if not os.path.isfile(abspath(r["path"]))
               or sha256_file(abspath(r["path"])) != r["sha256"]]
    if missing:
        die(f"Selected image missing or corrupt before finalize: {missing[:5]}")

    body = json.dumps(records, sort_keys=True, ensure_ascii=False)
    records_sha = sha256_bytes(body.encode("utf-8"))
    denylist_path = cfg.path_of("denylist")
    denylist = read_json(denylist_path, {}) or {}

    provenance = []
    for ds, shards in plan["shards"].items():
        spec = cfg.datasets[ds]
        provenance.append({
            "dataset": ds,
            "url_template": spec.url_template,
            "n_shards_used": len(shards),
            "bytes_downloaded": sum(s["size_bytes"] for s in shards),
            "license": spec.license,
            "citation": spec.citation,
            "shards": [s["shard"] for s in shards],
        })

    metadata = {
        "version": "2.0",
        "task": "Task 7.7 - kho du lieu da bo sinh v2",
        "created_at": now_iso(),
        "generated_by": "tools/build_manifest_v2.py",
        "config": rel(cfg.path),
        "config_sha256": cfg.sha256,
        "seed": cfg.seed,
        "n_records": len(records),
        "counts": counts,
        "ratios": {
            "real_frac_by_split": constraints["real_frac_by_split"],
            "replay_frac": constraints["replay_frac"],
            "family_imbalance": constraints["family_imbalance"],
        },
        "constraints": constraints,
        "constraints_enforced": bool(extra.get("strict", True)),
        "generator_denylist": {
            "path": rel(denylist_path),
            "sha256": sha256_file(denylist_path) if os.path.exists(denylist_path) else None,
            "n_generators": denylist.get("n_generators"),
            "match_semantics": denylist.get("match_semantics"),
        },
        "split_policy": {
            "mode": "generator_when_feasible",
            "cells": "family x label x is_replay x real_source",
            "ratios": cfg.t["split_ratios"],
            "declaration": "Group by generator when quota-feasible; image exceptions listed per stratum.",
            "strata": read_json(cfg.state("split_policy.json"), []),
            "generator_coverage": {
                sp: sorted({f"{r['family']}/{r['generator']}" for r in records if r["split"] == sp})
                for sp in SPLITS
            },
        },
        "quality": cfg.quality,
        "provenance": provenance,
        "replay": {
            "source_manifest": cfg.replay_cfg.get("manifest"),
            "total": sum(1 for r in records if r["is_replay"]),
        },
        "records_sha256": records_sha,
        "reference_fingerprints": references,
    }

    payload = {"metadata": metadata, "records": records}
    out = cfg.path_of("manifest")
    write_json(out, payload)
    manifest_dir = os.path.dirname(os.path.abspath(out))
    for sp in SPLITS:
        sp_records = [r for r in records if r.get("split") == sp]
        sp_meta = dict(metadata)
        sp_meta["split"] = sp
        sp_meta["n_records"] = len(sp_records)
        sp_meta["counts"] = _counts(sp_records)
        sp_out = os.path.join(manifest_dir, f"data_manifest_v2_{sp}.json")
        write_json(sp_out, {"metadata": sp_meta, "records": sp_records})
        log(f"FINALIZE: [{sp}] {len(sp_records)} ban ghi -> {rel(sp_out)}")
    train_select = [r for r in records if r["split"] in ("train", "val_select")]
    write_json(os.path.join(manifest_dir, "data_manifest_v2_train_select.json"), {
        "metadata": {"parent_records_sha256": records_sha, "purpose": "training_and_checkpoint_selection"},
        "records": train_select})
    write_json(cfg.path_of("constraint_report"), {"metadata": metadata, "constraints": constraints})
    with open(cfg.path_of("constraint_report_md"), "w", encoding="utf-8") as f:
        f.write(_constraints_md({"counts": counts, "constraints": constraints}, cfg))
    try:
        _distribution_png(records, cfg.path_of("distribution_png"))
    except Exception as exc:  # noqa: BLE001
        log(f"[!] khong ve duoc hinh phan bo: {type(exc).__name__}: {exc}")
    log(f"FINALIZE: {len(records)} ban ghi -> {rel(out)} (sha256 records={records_sha[:16]}...)")
    return {"manifest": out, "constraints": constraints, "records_sha256": records_sha}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_refs(paths: Sequence[str], max_phash: int) -> Dict[str, Dict]:
    refs: Dict[str, Dict] = {}
    for p in paths or []:
        data = load_hash_index(p)
        if not data or not data.get("metadata", {}).get("complete"):
            die(f"Reference index missing or incomplete: {p}")
        imgs = data.get("images", data.get("records", []))
        name = os.path.basename(p).replace(".json.gz", "").replace(".json", "")
        phash = [int(i["phash"]) for i in imgs if i.get("phash") is not None]
        if max_phash:  # 0 = dung tat ca
            die("Partial reference indexes are not permitted for data v2")
        if not imgs or len(phash) != len(imgs) or any(not i.get("sha256") for i in imgs):
            die(f"Reference hash coverage incomplete: {p}")
        refs[name] = {
            "sha256": [i["sha256"] for i in imgs if i.get("sha256")],
            "phash": phash,
            "path": os.path.abspath(p), "fingerprint": sha256_file(p),
            "images": imgs, "metadata": data.get("metadata", {}),
        }
        log(f"REF {name}: {len(refs[name]['sha256'])} sha, {len(refs[name]['phash'])} phash")
    return refs


def main() -> None:
    ap = argparse.ArgumentParser(description="Build data_manifest_v2.json (Task 7.7)")
    ap.add_argument("--config", default="configs/phase7_data_v2.yaml")
    ap.add_argument("--stage", default="all", choices=("all",) + STAGES)
    ap.add_argument("--dry-run", action="store_true", help="Chi index + plan, khong tai anh")
    ap.add_argument("--limit_per_family", type=int, default=0, help="Gioi han so anh moi/ho (smoke test)")
    ap.add_argument("--cache_dir", default="", help="Thu muc cache shard Parquet")
    ap.add_argument("--resume", action="store_true", help="Tiep tuc tu trang thai da luu")
    ap.add_argument("--force", action="store_true", help="Xoa cache trang thai va lam lai")
    ap.add_argument("--replay_root", default="", help="Thu muc goc cua diffusion_staging da giai nen")
    ap.add_argument("--replay_copy_mode", default="", choices=("", "copy", "hardlink", "reference"))
    ap.add_argument("--ref_index", nargs="*", default=[],
                    help="Hash index tap test (index_test_sets.py) de loc truoc khi chon")
    ap.add_argument("--denylist", default="", help="generator_denylist.json")
    ap.add_argument("--surplus", type=float, default=0.0, help="He so du khi lay ung vien (mac dinh tu config)")
    ap.add_argument("--strict", action="store_true", help="Bat buoc moi rang buoc phai dat")
    ap.add_argument("--workers", type=int, default=0, help="(du tru) so luong tai song song")
    args = ap.parse_args()

    cfg = Config(abspath(args.config))
    cache_dir = abspath(args.cache_dir or os.path.join("DATASET", "cache_v2"))
    denylist_path = abspath(args.denylist or cfg.path_of("denylist"))
    if not os.path.exists(denylist_path):
        die(f"Chua co deny-list: {denylist_path}. Chay tools/build_generator_denylist.py truoc.")
    matcher = DenyMatcher.from_payload(read_json(denylist_path))
    cfg.p["denylist"] = denylist_path
    strict = bool(args.strict) or (not args.limit_per_family and args.stage in ("all", "finalize"))
    check_replay_generators(cfg, matcher)

    if args.force:
        for f in ("context.json", "references.json", "split_policy.json", "candidates.jsonl", "fetch_done.json", "replay.jsonl", "selected.jsonl", "split.jsonl"):
            p = cfg.state(f)
            if os.path.exists(p):
                os.remove(p)
    ensure_state_context(cfg)

    stage = args.stage
    index_rows: List[Dict] = []
    if stage in ("all", "index", "plan") or args.dry_run:
        index_rows = stage_index(cfg, force=args.force)
    plan_path = cfg.state("plan.json")
    plan = None if args.force else read_json(plan_path)
    if plan and stage in ("fetch", "replay", "select", "split", "finalize") and not args.dry_run:
        if plan.get("config_sha256") != cfg.sha256:
            die("plan.json duoc tao tu config khac. Them --force hoac chay lai --stage plan.")
    if plan is None or stage in ("all", "index", "plan") or args.dry_run:
        plan = build_plan(cfg, index_rows, matcher, args.limit_per_family or None,
                          args.surplus or None)
        write_json(plan_path, plan)
    print_plan(plan, cfg)
    if args.dry_run:
        log("DRY-RUN: dung o buoc plan (khong tai anh).")
        return

    if stage in ("all", "fetch"):
        stage_fetch(cfg, plan, cache_dir, args.resume, os.environ.get("HF_TOKEN"))
    if stage in ("all", "replay"):
        if not args.replay_root:
            die("Can --replay_root (thu muc giai nen diffusion_staging.tar).")
        stage_replay(cfg, abspath(args.replay_root),
                     args.replay_copy_mode or cfg.replay_cfg.get("copy_mode", "copy"))
    if stage in ("select", "split", "finalize") and not os.path.exists(cfg.state("candidates.jsonl")):
        die("Chua co candidates.jsonl - chay --stage fetch truoc.")
    if stage in ("all", "select"):
        refs = load_refs(args.ref_index, 0)
        stage_select(cfg, plan, refs, strict)
    if stage in ("all", "split"):
        stage_split(cfg)
    if stage in ("all", "finalize"):
        res = stage_finalize(cfg, plan, {"strict": strict})
        c = res["constraints"]
        print()
        print("=" * 100)
        print(f"KET QUA RANG BUOC: {'PASS' if c['all_pass'] else 'FAIL'} ({c['n_failed']} muc)")
        for name, chk in c["checks"].items():
            if not chk["pass"]:
                print(f"  [X] {name}: {chk['value']} (yeu cau {chk['requirement']})")
        print("=" * 100)


if __name__ == "__main__":
    main()
