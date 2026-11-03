"""Task 7.7 data contract. Validate records, never trust manifest summaries."""
from collections import Counter
import re

SPLITS = ("train", "val_calib", "val_select")
FAMILIES = ("commercial", "latent_diffusion", "pixel_diffusion", "modern_gans", "autoregressive")
FIELDS = ("path", "sha256", "phash", "generator", "family", "split", "label",
          "source", "width", "height", "is_replay")
DEFAULT_TARGETS = {
    "total": 15000, "min_total": 14400, "max_total": 16000,
    "split_ratios": dict(zip(SPLITS, (.85, .075, .075))),
    "split_ratio_tolerance": .005, "max_family_imbalance": 1.5,
    "real_frac": [.45, .55], "replay_total": 3600, "replay_frac": [.20, .25],
    "max_new_per_family": 1200, "min_side": 200, "min_real_sources": 3,
}


def record_errors(row):
    if not isinstance(row, dict):
        return ["record_not_object"]
    errors = ["missing_" + k for k in FIELDS if row.get(k) is None or row.get(k) == ""]
    if errors:
        return errors
    if not re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])):
        errors.append("invalid_sha256")
    if type(row["phash"]) is not int or not 0 <= row["phash"] < 2**64:
        errors.append("invalid_phash")
    if type(row["label"]) is not int or row["label"] not in (0, 1):
        errors.append("invalid_label")
    if type(row["is_replay"]) is not bool:
        errors.append("invalid_is_replay")
    if row["split"] not in SPLITS:
        errors.append("invalid_split")
    if row["family"] not in FAMILIES + ("real",):
        errors.append("invalid_family")
    if (row["label"] == 0) != (row["family"] == "real"):
        errors.append("label_family_mismatch")
    if any(type(row[k]) is not int or row[k] <= 0 for k in ("width", "height")):
        errors.append("invalid_dimensions")
    return errors


def contract_problems(records, targets=None):
    t = {**DEFAULT_TARGETS, **(targets or {})}
    problems = []

    def require(ok, name):
        if not ok:
            problems.append({"check": name})

    malformed = [{"index": i, "errors": e} for i, r in enumerate(records)
                 if (e := record_errors(r))]
    if malformed:
        return [{"check": "record_schema", "n": len(malformed), "examples": malformed[:5]}]
    n = len(records)
    require(n > 0 and t["min_total"] <= n <= t["max_total"], "total_in_range")
    require(n == t["total"], "target_total")
    require(len({r["sha256"] for r in records}) == n, "unique_sha256")
    require(len({r["path"] for r in records}) == n, "unique_paths")
    require({r["split"] for r in records} == set(SPLITS), "three_splits")
    replay = sum(r["is_replay"] for r in records)
    replay_tol = int(t.get("replay_tolerance", 50))
    require(abs(replay - t["replay_total"]) <= replay_tol, "replay_total")
    require(t["replay_frac"][0] <= replay / max(1, n) <= t["replay_frac"][1], "replay_fraction")
    real_sources = {str(r.get("real_source") or "").strip().lower()
                    for r in records if r["label"] == 0 and not r["is_replay"]}
    real_sources -= {"", "unknown", "n/a", "real"}
    require(len(real_sources) >= t["min_real_sources"], "real_sources")
    new = Counter(r["family"] for r in records if r["label"] == 1 and not r["is_replay"])
    require(all(v <= t["max_new_per_family"] for v in new.values()), "new_family_cap")
    require(all(min(r["width"], r["height"]) >= t["min_side"] for r in records if not r.get("is_replay")), "min_side")
    for scope, rows in [("all", records)] + [(s, [r for r in records if r["split"] == s]) for s in SPLITS]:
        fam = Counter(r["family"] for r in rows if r["label"] == 1)
        require(all(fam[f] > 0 for f in FAMILIES), "five_families_" + scope)
        require(bool(fam) and all(fam[f] > 0 for f in FAMILIES)
                and max(fam.values()) / max(1, min(fam[f] for f in FAMILIES)) <= t["max_family_imbalance"] + 1e-9,
                "family_balance_" + scope)
        frac = sum(r["label"] == 0 for r in rows) / max(1, len(rows))
        require(t["real_frac"][0] <= frac <= t["real_frac"][1], "real_fraction_" + scope)
        if scope != "all":
            require(abs(len(rows) / max(1, n) - t["split_ratios"][scope])
                    <= t["split_ratio_tolerance"], "split_ratio_" + scope)
    return problems
