"""Fit Task 7.13 only from a measured val_calib cache (no fabricated signals)."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.degradation_router import DegradationRouter
from src.training.runtime_state import sha256_file
from tools.router_data import FEATURES, PREPROCESS, load_cache


def fpr_threshold(probabilities, targets, fpr=.05):
    """Smallest float32 threshold above the excluded order statistic; ties are conservative.

    Threshold may exceed 1 by one float32 ULP to reject even saturated scores of 1.
    This uses the same >= comparator and float32 precision as inference.
    """
    if not 0 <= fpr < 1:
        raise ValueError("Invalid target FPR")
    real = np.sort(np.asarray(probabilities, dtype=np.float32)[np.asarray(targets) == 0])
    if not len(real) or not np.isfinite(real).all():
        raise ValueError("Finite Real calibration samples are required")
    allowed = int(np.floor(fpr*len(real)))
    return float(np.nextafter(real[len(real)-allowed-1], np.float32(np.inf)))


def probabilities(logits, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid calibration temperature")
    return torch.softmax(torch.as_tensor(logits).float()/temperature, -1)[:, 1].numpy()


def fit_table(w, p, targets, min_real=50):
    if min_real < 50 or (targets == 0).sum() < min_real:
        raise ValueError("Need at least 50 Real calibration views; cannot fit a supported bin")
    edges = [0.] + [float(v) for v in np.unique(np.quantile(w, [1/3, 2/3]).astype(np.float32)) if 0 < v < 1] + [1.]
    merges = []
    while len(edges) > 2:
        bins = np.searchsorted(edges[1:-1], w, side="right")
        counts = [int(((bins == i) & (targets == 0)).sum()) for i in range(len(edges)-1)]
        sparse = next((i for i, n in enumerate(counts) if n < min_real), None)
        if sparse is None:
            break
        boundary = sparse if sparse == len(counts)-1 else sparse+1
        merges.append({"removed_boundary": edges[boundary], "real_views_before": counts})
        edges.pop(boundary)
    bins = np.searchsorted(edges[1:-1], w, side="right")
    table = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = bins == i
        threshold = fpr_threshold(p[mask], targets[mask])
        real = targets[mask] == 0
        table.append(dict(w_min=lo, w_max=hi, threshold=threshold, n_real=int(real.sum()),
                          fpr_measured=float((p[mask][real] >= threshold).mean())))
    return table, merges


def fit_router_calibration(cache_path, output_path):
    data, meta = load_cache(cache_path, "val_calib")
    x, y, groups = data["features"], data["is_degraded"], data["sample_ids"]
    if len(set(groups)) < 5:
        raise ValueError("5-fold grouped CV requires at least five independent source images")
    aucs = []
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    for train, validation in splitter.split(x, y, groups):
        scaler_cv = StandardScaler().fit(x[train])
        clf_cv = LogisticRegression(C=1., max_iter=1000, class_weight="balanced", random_state=42)
        clf_cv.fit(scaler_cv.transform(x[train]), y[train])
        aucs.append(float(roc_auc_score(y[validation], clf_cv.predict_proba(scaler_cv.transform(x[validation]))[:, 1])))
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=1., max_iter=1000, class_weight="balanced", random_state=42).fit(scaler.transform(x), y)
    # Same backbone/logits, two condition-specific temperature calibrations, as allowed by 7.15.
    from tools.calibrate_and_select import ModelWithTemperature
    branches = {}
    for name, condition in (("clean", 0), ("forensic", 1)):
        mask = y == condition
        temp, _ = ModelWithTemperature().fit(torch.as_tensor(data["logits"][mask]).float(),
                                            torch.as_tensor(data["targets"][mask]).long())
        scores = probabilities(data["logits"], temp)
        threshold = fpr_threshold(scores, data["targets"])
        fake = data["targets"] == 1
        real = ~fake
        branches[name] = dict(T=float(temp), threshold=threshold,
                              condition_threshold=fpr_threshold(scores[mask], data["targets"][mask]),
                              calibration_fake_recall=float((scores[fake] >= threshold).mean()),
                              calibration_fpr=float((scores[real] >= threshold).mean()))
    fixed_name = max(branches, key=lambda name: branches[name]["calibration_fake_recall"])
    fixed = dict(branch=fixed_name, threshold=branches[fixed_name]["threshold"], selection="val_calib_fake_recall_at_fpr05")
    params = dict(schema="degradation_router/v2", status="fitted", evidence_level="measured",
                  features=FEATURES, preprocess=PREPROCESS, branch_design="single_logits_condition_temperature",
                  scaler=dict(mean=scaler.mean_.tolist(), std=scaler.scale_.tolist()),
                  coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]),
                  n_calib=len(set(groups.tolist())), n_views=len(y), calib_auc=float(np.mean(aucs)),
                  cv=dict(method="StratifiedGroupKFold", n_splits=5, group="source_image_sha256", aucs=aucs),
                  calibration=branches, fixed=fixed, fpr_target=.05,
                  source=dict(cache_sha256=sha256_file(cache_path), checkpoint_sha256=meta["checkpoint_sha256"],
                              calibration_image_sha256=sorted(set(groups.tolist())), metadata=meta),
                  deployment_enabled=False)
    candidate = DegradationRouter(params["scaler"]["mean"], params["scaler"]["std"], params["coef"], params["intercept"])
    w = candidate.predict_features(torch.as_tensor(x).float()).numpy()
    pc = probabilities(data["logits"], branches["clean"]["T"])
    pf = probabilities(data["logits"], branches["forensic"]["T"])
    params["threshold_table"], params["bin_merges"] = fit_table(w, (1-w)*pc+w*pf, data["targets"])
    edges = np.linspace(0, 1, 21)
    params["w_histogram"] = dict(edges=edges.tolist(), clean=np.histogram(w[y == 0], edges)[0].tolist(),
                                 q30=np.histogram(w[y == 1], edges)[0].tolist())
    params["signal_auc"] = {name: float(roc_auc_score(y, x[:, i])) for i, name in enumerate(FEATURES)}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(params, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="Measured val_calib NPZ from tools/router_data.py")
    parser.add_argument("--output", default="DATASET/logs/phase7/router_params.json")
    args = parser.parse_args()
    params = fit_router_calibration(args.cache, args.output)
    print(f"Fitted {params['n_calib']} source images; grouped CV AUC={params['calib_auc']:.4f}; deployment remains disabled")
