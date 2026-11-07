"""Evaluate fixed/router/condition-oracle using held-out measured image predictions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.degradation_router import DegradationRouter
from src.models.unified_fatformer import validate_router_report
from src.training.runtime_state import sha256_file
from tools.fit_router import fpr_threshold, probabilities
from tools.router_data import SUBSETS, load_cache, params_digest


def fake90(p, y):
    # Diagnostic ROC operating point on held-out scores, never used for deployment.
    threshold = fpr_threshold(p, y, .10)
    return float((p[y == 1] >= threshold).mean()*100)


def compare_subset(data, scores, subset, n_boot, rng):
    selected = data["subsets"] == subset
    y, condition = data["targets"][selected], data["is_degraded"][selected]
    ids = data["sample_ids"][selected]
    p = {name: value[selected] for name, value in scores.items()}
    # Resample source images within Real/Fake strata, retaining both views together.
    unique = np.unique(ids)
    row_map = {sid: np.flatnonzero(ids == sid) for sid in unique}
    real = [sid for sid in unique if y[row_map[sid][0]] == 0]
    fake = [sid for sid in unique if y[row_map[sid][0]] == 1]
    if not real or not fake:
        raise ValueError(f"Subset {subset} must contain both classes")
    draws = {name: [] for name in scores}
    for _ in range(n_boot):
        sampled = np.concatenate([rng.choice(real, len(real)), rng.choice(fake, len(fake))])
        ix = np.concatenate([row_map[sid] for sid in sampled])
        for name in scores:
            draws[name].append(fake90(p[name][ix], y[ix]))
    result = dict(n_source_images=len(unique), n_views=len(y),
                  n_real_source_images=len(real), n_fake_source_images=len(fake))
    for name in scores:
        result[name] = dict(fake_at_real90=fake90(p[name], y),
                            ci95=np.percentile(draws[name], [2.5, 97.5]).tolist(),
                            by_condition={tag: fake90(p[name][condition == c], y[condition == c])
                                          for tag, c in (("clean", 0), ("q30", 1))})
    result["delta_pp"] = result["router"]["fake_at_real90"]-result["fixed"]["fake_at_real90"]
    result["delta_ci95"] = np.percentile(np.asarray(draws["router"])-draws["fixed"], [2.5, 97.5]).tolist()
    result["win"] = result["delta_pp"] >= 1.
    result["router_ci_above_fixed"] = result["router"]["ci95"][0] > result["fixed"]["ci95"][1]
    return result


def generate_router_ablation(cache_path, params_path, output_dir, n_boot=1000, seed=42):
    if n_boot < 1000:
        raise ValueError("Task 7.13 requires at least 1000 bootstrap resamples")
    data, meta = load_cache(cache_path, "evaluation")
    params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    router = DegradationRouter.from_params(params)
    if (meta["checkpoint_sha256"] != params["source"]["checkpoint_sha256"] or
            meta["preprocess"] != params["preprocess"]):
        raise ValueError("Evaluation model/preprocessing differs from calibration")
    if set(data["sample_ids"].tolist()) & set(params["source"]["calibration_image_sha256"]):
        raise ValueError("Calibration/evaluation source image hashes overlap")
    if set(data["subsets"].tolist()) != set(SUBSETS):
        raise ValueError("Evaluation must cover exactly the five CFE subsets")
    pc = probabilities(data["logits"], params["calibration"]["clean"]["T"])
    pf = probabilities(data["logits"], params["calibration"]["forensic"]["T"])
    routed, labels, _, thresholds = router.route_features(torch.as_tensor(data["features"]).float(),
                                                        torch.from_numpy(pc), torch.from_numpy(pf))
    fixed_scores = pc if params["fixed"]["branch"] == "clean" else pf
    oracle_scores = np.where(data["is_degraded"] == 0, pc, pf)
    scores = dict(fixed=fixed_scores, router=routed.numpy(), oracle=oracle_scores)
    rng = np.random.default_rng(seed)
    subsets = {subset: compare_subset(data, scores, subset, n_boot, rng) for subset in SUBSETS}
    wins = sum(row["win"] for row in subsets.values())
    separated = sum(row["router_ci_above_fixed"] for row in subsets.values())
    won = wins >= 3 and separated >= 2
    # Report deployed decisions separately from the ROC diagnostic Fake@Real90.
    fixed_labels = fixed_scores >= params["fixed"]["threshold"]
    oracle_thresholds = np.where(data["is_degraded"] == 0, params["calibration"]["clean"]["condition_threshold"],
                                  params["calibration"]["forensic"]["condition_threshold"])
    deployed = {}
    for name, prediction in dict(fixed=fixed_labels, router=labels.numpy(), oracle=oracle_scores >= oracle_thresholds).items():
        deployed[name] = {}
        for tag, c in (("clean", 0), ("q30", 1)):
            mask = data["is_degraded"] == c
            real, fake = mask & (data["targets"] == 0), mask & (data["targets"] == 1)
            deployed[name][tag] = dict(real_acc=float((~prediction[real].astype(bool)).mean()*100),
                                        fake_acc=float(prediction[fake].mean()*100))
    report = dict(schema="router_ablation/v2", evidence_level="measured", status="evaluated",
                  params_digest=params_digest(params), params_file_sha256=sha256_file(params_path),
                  evaluation_cache_sha256=sha256_file(cache_path), checkpoint_sha256=meta["checkpoint_sha256"],
                  calibration_cache_sha256=params["source"]["cache_sha256"],
                  evaluation_split_sha256=meta["source"]["evaluation_split_sha256"],
                  preprocess=meta["preprocess"], branch_design=meta["branch_design"],
                  protocol="Primary: pooled equal clean/Q30 views per subset; paired source-image bootstrap within labels",
                  oracle_note="Condition oracle (clean->clean calibration, Q30->forensic); not deployable, not a guaranteed mathematical upper bound",
                  metric_note="Fake@Real90 uses held-out ROC scores; deployed decisions use only val_calib thresholds",
                  bootstrap=dict(n=n_boot, seed=seed, unit="source image, both views retained", paired=True,
                                 strata="real_fake_source_images", confidence_level=.95),
                  subsets=subsets, deployed=deployed,
                  verdict=dict(wins=wins, total_subsets=5, ci_separated=separated,
                               criteria_met=won, fallback_required=not won),
                  gate_eligible=False, gate_note="Router comparison only; G7.3 packaging and latency remain separate")
    validate_router_report(params, report)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/"router_ablation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# Task 7.13 — measured held-out ablation", "", report["protocol"], report["metric_note"],
             report["oracle_note"], "", f"Wins >=1 pp: {wins}/5; router CI above fixed CI: {separated}/5.",
             "Verdict: "+("THẮNG" if won else "KHÔNG THẮNG — dùng calibrated fixed fallback"), ""]
    for subset, row in subsets.items():
        lines.append(f"## {subset} ({row['n_source_images']} source images)")
        for name in scores:
            value = row[name]
            lines.append(f"- {name}: Fake@Real90={value['fake_at_real90']:.2f}%; CI95={value['ci95']}; clean/Q30={value['by_condition']}")
        lines.append(f"- Delta={row['delta_pp']:.2f} pp; paired CI95={row['delta_ci95']}")
    (output/"router_ablation.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--params", default="DATASET/logs/phase7/router_params.json")
    parser.add_argument("--output-dir", default="DATASET/logs/phase7")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = generate_router_ablation(args.cache, args.params, args.output_dir, args.n_boot, args.seed)
    print(report["verdict"])
