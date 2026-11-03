"""
FatFormer-XLA: Sinh generator deny-list TU DONG cho kho du lieu v2 (Task 7.7).

Nguyen tac (bat di bat dich cua Task 7.7):
    - Danh sach generator bi cam phai duoc tinh tu dong tu manifest tap test,
      KHONG duoc go tay.
    - Khong noi rong deny-list de du so luong; neu thieu anh thi mo rong nguon.
    - So khop theo *instance* (giu nguyen version): MidjourneyV5_2 KHONG block
      MidjourneyV4, nhung block MidjourneyV5 / V5_2 / V5_2-hd (version khong xac
      dinh hoac trung tien to). Day la ngu nghia "generator-disjoint" dung dan:
      cac version khac nhau la cac generator khac nhau, nhung khong duoc phep
      dung mot ten chung chung de lach luat.

Nguon dau vao (tat ca deu tu dong):
    1. cfe_manifest.json cua tap test CommunityForensics (2.000 anh) - moi ban ghi
       co `model_name` + `architecture`.
    2. eval_set_metadata.csv cua CommunityForensics (21 model cua CFE-Eval) -
       danh sach day du hon tap 2.000 anh, tranh truong hop tap con chi lay 5/21
       model roi dung 16 model con lai de train (gian lan benchmark).
    3. Bang 18 tap test cua Task 5.1 trong bao cao
       docs/bao_cao/C/bao_cao_task_5.1_full_benchmark_18_datasets.md

Dau ra:
    DATASET/logs/phase7/generator_denylist.json

Vi du:
    python tools/build_generator_denylist.py \
        --cfe-manifest /content/dataset_local/test_communityforensics/cfe_manifest.json \
        --eval-metadata /content/dataset_local/communityforensics/eval_set_metadata.csv \
        --task51-report docs/bao_cao/C/bao_cao_task_5.1_full_benchmark_18_datasets.md \
        --out DATASET/logs/phase7/generator_denylist.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_TASK51_REPORT = os.path.join(
    PROJECT_ROOT, "docs", "bao_cao", "C", "bao_cao_task_5.1_full_benchmark_18_datasets.md"
)

# Alias chi cho cac ten CHI MOT generator (khong map theo ho kien truc).
# Vi du: "dalle" (DALL-E 1) va "dall-e 3" la cung mot dong model nhung khac
# version; alias giup so khop dung token.
ALIAS_TABLE: Dict[str, List[str]] = {
    "dalle": ["dall e", "dall-e"],
    "vqdiffusion": ["vq diffusion", "vq-diffusion"],
    "pndm": ["pseudo numerical methods", "pseudo numerical"],
    "deepfake": ["deep fake", "faceforensics", "face forensics"],
    "gaugan": ["gau gan"],
}

# 18 tap test cua Task 5.1 (fallback khi khong parse duoc bao cao).
EXPECTED_TASK51_TESTS = [
    "progan", "stargan", "stylegan", "stylegan2", "deepfake", "biggan", "cyclegan", "gaugan",
    "vqdiffusion", "ldm_200_cfg", "ldm_200", "ldm_100", "glide_100_10", "glide_50_27",
    "glide_100_27", "guided", "pndm", "dalle",
]

# Token danh dau version/bien the. Tach khoi "base" de so sanh instance.
_VERSION_TOKEN_RE = re.compile(
    r"^(?:v\d*|\d+|xl|xxl|turbo|lite|hd|ada|deep|base|large|small|tiny|mini|"
    r"fp16|fp32|bf16|ema|final|latest|cfg)$"
)

# Hau to dinh lien vao ten (StyleGANXL -> StyleGAN XL).
_GLUED_SUFFIX_RE = re.compile(r"(?<=[a-z]{3})(xl|xxl|ada|turbo|lite|hd)$")


def _camel_split(s: str) -> str:
    """Tach camelCase nhung GIU cac chuoi viet hoa lien (StyleGAN, BigGAN)."""
    # lower -> Upper (StyleGAN -> Style GAN, GuidedDiffusion -> Guided Diffusion)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    # acronym -> Capitalized (HTMLParser -> HTML Parser)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return s


def _tokenize(s: str) -> List[str]:
    s = s.replace("\\", "/")
    s = re.sub(r"\.(safetensors|ckpt|pt|pth|bin|gguf)$", "", s, flags=re.IGNORECASE).lower()
    # Tach chuoi-so o MOI vi tri (StyleGAN2 -> StyleGAN 2, khong chi stylegan2)
    s = re.sub(r"(?<=[a-z])(?=[0-9])", " ", s)
    s = re.sub(r"(?<=[0-9])(?=[a-z])", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    tokens: List[str] = []
    for raw in s.split():
        raw = _GLUED_SUFFIX_RE.sub(r" \1", raw)
        tokens.extend(t for t in raw.split() if t)
    return [t.lower() for t in tokens]


def name_forms(name: str) -> List[List[str]]:
    """
    Tra ve cac cach token hoa cua mot ten.

    Vi sao can 2 form:
        - "StyleGAN2" -> ["stylegan", "2"] (khong duoc tach Style/GAN)
        - "GuidedDiffusion" -> ["guided", "diffusion"] (can tach camel)
    Dung ca hai de tranh ca false negative (GuidedDiffusion) va false positive
    (Style GAN).
    """
    if not name:
        return []
    raw = str(name).strip()
    if "/" in raw.replace("\\", "/"):
        raw = raw.replace("\\", "/").split("/")[-1]
    forms: List[List[str]] = []
    for candidate in (raw, _camel_split(raw)):
        toks = _tokenize(candidate)
        if toks and toks not in forms:
            forms.append(toks)
    return forms


def split_base_version(tokens: Sequence[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Tach base tokens va version tokens (giu thu tu)."""
    base: List[str] = []
    version: List[str] = []
    for t in tokens:
        if _VERSION_TOKEN_RE.match(t):
            version.append(t)
        else:
            base.append(t)
    return tuple(base), tuple(version)


def _starts_with(seq: Sequence[str], prefix: Sequence[str]) -> bool:
    return len(seq) >= len(prefix) and tuple(seq[: len(prefix)]) == tuple(prefix)


def _alias_variants(tokens: Sequence[str]) -> List[List[str]]:
    """Sinh cac bien the alias o muc token (chi ap dung cho cum token lien tiep)."""
    text = " ".join(tokens)
    variants: List[List[str]] = []
    for key, aliases in ALIAS_TABLE.items():
        key_tokens = key.split()
        alias_token_lists = [a.split() for a in aliases]
        for kt, at in zip([key_tokens] * len(alias_token_lists), alias_token_lists):
            if _contains_subsequence(tokens, kt):
                replaced = _replace_subsequence(tokens, kt, at)
                if replaced and replaced not in variants:
                    variants.append(replaced)
            if _contains_subsequence(tokens, at):
                replaced = _replace_subsequence(tokens, at, kt)
                if replaced and replaced not in variants:
                    variants.append(replaced)
    # alias cho ca chuoi (vi du "vqdiffusion" <-> "vq diffusion")
    for key, aliases in ALIAS_TABLE.items():
        if text == key:
            for a in aliases:
                toks = a.split()
                if toks not in variants:
                    variants.append(toks)
        for a in aliases:
            if text == a:
                toks = key.split()
                if toks not in variants:
                    variants.append(toks)
    return variants


def _contains_subsequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    for i in range(len(haystack) - len(needle) + 1):
        if tuple(haystack[i:i + len(needle)]) == tuple(needle):
            return True
    return False


def _replace_subsequence(haystack: Sequence[str], needle: Sequence[str], repl: Sequence[str]) -> List[str]:
    for i in range(len(haystack) - len(needle) + 1):
        if tuple(haystack[i:i + len(needle)]) == tuple(needle):
            return list(haystack[:i]) + list(repl) + list(haystack[i + len(needle):])
    return []


def _all_forms(name: str) -> List[Tuple[str, ...]]:
    """Tat ca form token (goc + camel + alias), da loai trung."""
    out: List[Tuple[str, ...]] = []
    for form in name_forms(name):
        for variant in [form] + _alias_variants(form):
            t = tuple(variant)
            if t and t not in out:
                out.append(t)
    return out


class DenyMatcher:
    """
    So khop instance-disjoint.

    Trang thai:
        blocked  : cung instance, hoac version khong xac dinh (an toan)
        related  : cung base nhung version khac nhau da xac dinh (bao cao rieng)
        allowed  : khong lien quan
    """

    def __init__(self, denied_names: Iterable[str]):
        self.entries: List[Dict] = []
        seen: Set[Tuple[str, ...]] = set()
        for raw in denied_names:
            forms = _all_forms(raw)
            if not forms:
                continue
            # Chon form dai nhat lam canonical (giu nhieu thong tin nhat)
            canonical = max(forms, key=len)
            base, version = split_base_version(canonical)
            if not base:
                continue
            key = (base, version)
            if key in seen:
                continue
            seen.add(key)
            self.entries.append({
                "raw": str(raw),
                "normalized": " ".join(canonical),
                "base": list(base),
                "version": list(version),
                "forms": [list(f) for f in forms],
            })
        # Alias/derived forms cua tung entry
        for e in self.entries:
            extra_forms: List[Tuple[str, ...]] = []
            for form in e["forms"]:
                for variant in _alias_variants(form):
                    t = tuple(variant)
                    if t not in extra_forms:
                        extra_forms.append(t)
            # Them cac form da tach base/version (vi du alias "dall e 3")
            e["match_forms"] = [tuple(f) for f in e["forms"]] + extra_forms
            e["base_version_pairs"] = []
            for f in e["match_forms"]:
                b, v = split_base_version(f)
                if b and (b, v) not in e["base_version_pairs"]:
                    e["base_version_pairs"].append((b, v))

    @classmethod
    def from_payload(cls, payload: Dict) -> "DenyMatcher":
        entries = payload.get("entries")
        if entries:
            m = cls([])
            m.entries = entries
            for e in m.entries:
                if "match_forms" not in e:
                    e["match_forms"] = [tuple(f) for f in e.get("forms", [])]
                if "base_version_pairs" not in e:
                    e["base_version_pairs"] = []
                    for f in e["match_forms"]:
                        b, v = split_base_version(f)
                        if b and (b, v) not in e["base_version_pairs"]:
                            e["base_version_pairs"].append((b, v))
            return m
        return cls(payload.get("generators", []))

    def _classify_pair(
        self,
        cand_base: Sequence[str],
        cand_version: Sequence[str],
        deny_base: Sequence[str],
        deny_version: Sequence[str],
    ) -> Optional[str]:
        if not cand_base or not deny_base:
            return None
        cb, db = tuple(cand_base), tuple(deny_base)
        cv, dv = tuple(cand_version), tuple(deny_version)

        if cb == db:
            if not cv and not dv:
                return "blocked"
            if not cv or not dv:
                # Candidate/deny thieu version -> khong the chung minh khac instance.
                return "blocked"
            if cv == dv:
                return "blocked"
            # Version la *tien to* cua nhau (v5 vs v5_1) => cung instance mo ho.
            if _starts_with(cv, dv) or _starts_with(dv, cv):
                return "blocked"
            return "related"

        if _starts_with(cb, db):
            # Candidate = deny + ten phu (vd kandinsky-2-2-decoder).
            extra = cb[len(db):]
            return "related" if all(_VERSION_TOKEN_RE.match(t) for t in extra) else "blocked"

        if _starts_with(db, cb):
            # Candidate la ten ngan hon (vd "sdxl" vs "lcm lora sdxl").
            # Khong block tu dong: day la base model, khong phai cung instance.
            # Nhung neu candidate chi la mot phan cua ten deny thi van co the
            # la cung generator bi viet tat -> chi bao "related".
            return "related"

        return None

    def classify(self, model_name: str) -> Dict:
        """
        Tra ve {"status": "blocked"|"related"|"allowed", "matched": raw, "reason": str}.
        """
        forms = _all_forms(model_name)
        if not forms:
            return {"status": "allowed", "matched": None, "reason": "empty_name"}

        best: Optional[Dict] = None
        rank = {"allowed": 0, "related": 1, "blocked": 2}
        for cand_form in forms:
            cbase, cver = split_base_version(cand_form)
            if not cbase:
                continue
            for entry in self.entries:
                for db, dv in entry["base_version_pairs"]:
                    status = self._classify_pair(cbase, cver, db, dv)
                    if status is None:
                        continue
                    if best is None or rank[status] > rank[best["status"]]:
                        reasons = {
                            "blocked": "same_instance_or_unknown_version",
                            "related": "same_base_different_version",
                        }
                        best = {
                            "status": status,
                            "matched": entry["raw"],
                            "reason": reasons[status],
                        }
                    if best and best["status"] == "blocked":
                        return best
        return best or {"status": "allowed", "matched": None, "reason": "no_match"}

    def is_blocked(self, model_name: str) -> bool:
        return self.classify(model_name)["status"] == "blocked"

    def is_denied(self, model_name: str) -> bool:
        """Tuong thich API cu: blocked hoac related deu la 'bi cam'."""
        return self.classify(model_name)["status"] in ("blocked", "related")


def parse_task51_report(report_path: str) -> List[str]:
    """Parse ten 18 tap test tu bang markdown cua bao cao Task 5.1."""
    if not report_path or not os.path.exists(report_path):
        return []
    pattern = re.compile(r"^\|\s*(?:\*\*[^|]*\*\*)?\s*\|\s*`([^`]+)`")
    found: List[str] = []
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                name = m.group(1).strip()
                if name and name not in found:
                    found.append(name)
    return found


def load_cfe_manifest(path: str) -> List[Dict]:
    """Doc cfe_manifest.json va tra ve danh sach ban ghi."""
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("records", data.get("images", [])) or []
    return data or []


def load_eval_metadata(path: str) -> List[Dict]:
    """Doc eval_set_metadata.csv (Model, Architecture, License, RealSource...)."""
    if not path or not os.path.exists(path):
        return []
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = (row.get("Model") or row.get("model") or "").strip()
            if not model:
                continue
            rows.append({
                "model_name": model,
                "architecture": (row.get("Architecture") or "").strip(),
                "license": (row.get("License") or "").strip(),
                "real_source": (row.get("RealSource") or "").strip(),
                "origin": "eval_set_metadata",
            })
    return rows


def build_denylist(
    cfe_manifest_path: str,
    task51_report: str,
    eval_metadata_path: str = "",
    extra: Sequence[str] = (),
    allow_empty_cfe: bool = False,
) -> Dict:
    cfe_records = load_cfe_manifest(cfe_manifest_path)
    eval_rows = load_eval_metadata(eval_metadata_path)
    if not cfe_records and not eval_rows and not allow_empty_cfe:
        raise SystemExit(
            "[!] Khong doc duoc cfe_manifest.json lan eval_set_metadata.csv. Hay truyen "
            "--cfe-manifest hop le (nam trong test_communityforensics.tar) hoac "
            "--allow-empty-cfe de chi dung 18 tap."
        )

    # --- Thu thap ten + metadata nguon -------------------------------------
    cfe_models: List[str] = []
    cfe_by_arch: Dict[str, Set[str]] = {}
    for r in cfe_records:
        if int(r.get("label", 1)) != 1:
            continue
        model = str(r.get("model_name") or r.get("generator") or "").strip()
        if not model:
            continue
        if model not in cfe_models:
            cfe_models.append(model)
        cfe_by_arch.setdefault(str(r.get("architecture", "unknown")), set()).add(model)

    eval_models: List[str] = []
    eval_by_arch: Dict[str, Set[str]] = {}
    for r in eval_rows:
        model = r["model_name"]
        if model not in eval_models:
            eval_models.append(model)
        eval_by_arch.setdefault(r["architecture"] or "unknown", set()).add(model)

    task51_names = parse_task51_report(task51_report)
    parsed_from_report = bool(task51_names)
    if not task51_names:
        task51_names = list(EXPECTED_TASK51_TESTS)

    # Gop ten tho, giu thu tu uu tien: test manifest -> eval metadata -> 18 tap.
    raw_names: List[str] = []
    for name in cfe_models + eval_models + task51_names + list(extra):
        if name and name not in raw_names:
            raw_names.append(name)

    matcher = DenyMatcher(raw_names)

    # Kiem tra cheo: chinh cac ten deny phai tu block duoc no.
    self_block_failures = []
    for e in matcher.entries:
        verdict = matcher.classify(e["raw"])
        if verdict["status"] != "blocked":
            self_block_failures.append({"raw": e["raw"], "status": verdict["status"],
                                        "matched": verdict["matched"]})

    # Gan architecture/license cho tung entry de bao cao.
    arch_by_norm: Dict[str, Set[str]] = {}
    for arch, models in list(cfe_by_arch.items()) + list(eval_by_arch.items()):
        for m in models:
            for f in _all_forms(m):
                arch_by_norm.setdefault(" ".join(f), set()).add(arch)

    for e in matcher.entries:
        arches: Set[str] = set()
        for f in e["match_forms"]:
            arches |= arch_by_norm.get(" ".join(f), set())
        e["architectures"] = sorted(arches) if arches else ["unknown"]

    return {
        "source": [
            "cfe_test_manifest" if cfe_models else None,
            "cfe_eval_metadata" if eval_models else None,
            "task_5.1_18_datasets",
        ] + (["manual_extra"] if extra else []),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_generators": len(matcher.entries),
        "n_raw_names": len(raw_names),
        "generators": [e["raw"] for e in matcher.entries],
        "entries": matcher.entries,
        "normalization_rules": [
            "lowercase",
            "strip_author_prefix",
            "strip_weight_extension",
            "alpha_digit_boundary_split",
            "camel_case_split_variant",
            "keep_version_tokens",
            "alias_table",
            "base_version_instance_match",
        ],
        "match_semantics": {
            "blocked": "cung base + cung version, hoac thieu version (khong chung minh duoc khac instance)",
            "related": "cung base nhung version khac nhau da xac dinh",
            "allowed": "khac base hoac khong lien quan",
        },
        "details": {
            "cfe_manifest": os.path.abspath(cfe_manifest_path) if cfe_manifest_path else None,
            "cfe_records": len(cfe_records),
            "cfe_models": sorted(cfe_models),
            "cfe_models_by_architecture": {k: sorted(v) for k, v in cfe_by_arch.items()},
            "eval_metadata": os.path.abspath(eval_metadata_path) if eval_metadata_path else None,
            "eval_models": sorted(eval_models),
            "eval_models_by_architecture": {k: sorted(v) for k, v in eval_by_arch.items()},
            "task51_report": os.path.abspath(task51_report) if task51_report else None,
            "task51_names": task51_names,
            "task51_parsed_from_report": parsed_from_report,
            "extra": list(extra),
            "self_block_failures": self_block_failures,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build generator deny-list for Task 7.7")
    parser.add_argument("--cfe-manifest", type=str, default="",
                        help="Duong dan cfe_manifest.json cua tap test CFE (2.000 anh)")
    parser.add_argument("--eval-metadata", type=str, default="",
                        help="eval_set_metadata.csv cua CFE (21 model) - nen dung")
    parser.add_argument("--task51-report", type=str, default=DEFAULT_TASK51_REPORT)
    parser.add_argument("--extra", nargs="*", default=[], help="Ten generator bo sung (chi khi that su can)")
    parser.add_argument("--allow-empty-cfe", action="store_true")
    parser.add_argument("--out", type=str, default="DATASET/logs/phase7/generator_denylist.json")
    parser.add_argument("--verify-model", type=str, default="", help="Kiem tra 1 model_name")
    args = parser.parse_args()

    payload = build_denylist(
        cfe_manifest_path=args.cfe_manifest,
        task51_report=args.task51_report,
        eval_metadata_path=args.eval_metadata,
        extra=args.extra,
        allow_empty_cfe=args.allow_empty_cfe,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("=" * 84)
    print("FATFORMER-XLA: GENERATOR DENY-LIST (TASK 7.7)")
    print("=" * 84)
    print(f"- CFE test models : {len(payload['details']['cfe_models'])}")
    print(f"- CFE eval models : {len(payload['details']['eval_models'])}")
    print(f"- Task 5.1 tests  : {len(payload['details']['task51_names'])} "
          f"({'parse tu bao cao' if payload['details']['task51_parsed_from_report'] else 'fallback'})")
    print(f"- Tong generator bi cam (sau chuan hoa): {payload['n_generators']}")
    if payload["details"]["self_block_failures"]:
        print(f"  [!] {len(payload['details']['self_block_failures'])} ten khong tu block duoc:")
        for item in payload["details"]["self_block_failures"][:10]:
            print(f"      - {item}")
    print(f"- Da ghi: {args.out}")

    if args.verify_model:
        matcher = DenyMatcher.from_payload(payload)
        verdict = matcher.classify(args.verify_model)
        print(f"- Kiem tra '{args.verify_model}': {verdict}")
    print("=" * 84)


if __name__ == "__main__":
    main()
