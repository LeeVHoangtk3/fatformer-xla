"""Summarize verified ablations; missing evidence is an error, never a zero score."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.ablation_v2_common import NAMES, WEIGHTS, read_json, validate_history
from src.training.runtime_state import sha256_file


def percent(value, key):
    if isinstance(value, bool) or value is None or not np.isfinite(float(value)) or not 0 <= float(value) <= 100:
        raise ValueError(f"Invalid/missing percentage: {key}")
    return float(value)


def interval(value, key):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Missing CI: {key}")
    lo, hi = (percent(x, key) for x in value)
    if lo > hi:
        raise ValueError(f"Reversed CI: {key}")
    return [lo, hi]


def parse_cfe_eval_json(clean_json, q30_json):
    result = {}
    partitions = []
    for condition, path in (("clean", clean_json), ("q30", q30_json)):
        data = read_json(path)
        if (data.get("report_schema") != "v3" or data.get("evidence_level") != "measured"
                or data.get("gate_eligible") is not True or data.get("git_dirty") is not False):
            raise ValueError(f"Not eligible measured evaluation: {path}")
        if data["dataset"]["condition"] != condition or data.get("degrade_order") != "post_resize":
            raise ValueError("Wrong evaluation condition/order")
        partition = data.get("partition")
        if not partition or partition.get("split") != "evaluation" or partition["n"] != data["dataset"]["n"]:
            raise ValueError("Missing held-out partition evidence")
        partitions.append(partition)
        metrics = data["metrics"]
        result[f"auc_{condition}"] = percent(metrics["auc"], "auc")
        result[f"auc_{condition}_ci95"] = interval(metrics["ci95"]["auc"], "auc")
        if condition == "q30":
            result["fake_at_real90_q30"] = percent(metrics["fake_at_real90"], "fake_at_real90")
            result["fake_at_real90_q30_ci95"] = interval(metrics["ci95"]["fake_at_real90"], "fake_at_real90")
            # Temperature was frozen from the baseline, not fit for each new model.
            result["ece_q30_frozen_temperature_pct"] = percent(metrics["ece"], "ece")
    if partitions[0] != partitions[1]:
        raise ValueError("Clean and Q30 partitions differ")
    result["partition"] = partitions[0]
    return result


def parse_val_groups_csv(csv_path, selected_epoch=None):
    if selected_epoch is None:
        raise ValueError("Require selected epoch from verified checkpoint, not CSV re-selection")
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [r for r in rows if int(r["epoch"]) == selected_epoch]
    if len(matches) != 1:
        raise ValueError("Selected checkpoint epoch missing/duplicated in validation log")
    row = matches[0]
    result = {k: percent(float(row[k]) * 100., k) for k in ("clean_r", "clean_f", "q30_r", "q30_f")}
    result["worst_group"] = percent(float(row["worst"]) * 100., "worst")
    if result["clean_r"] < 90 or not np.isclose(result["worst_group"], min(result[k] for k in ("clean_r", "clean_f", "q30_r", "q30_f"))):
        raise ValueError("Validation guard/worst-group inconsistent")
    return {"selected_epoch": selected_epoch, **result}


def evaluate_delta_verdict(delta_pp, ci95=None):
    if ci95 is None:
        return "Chưa đủ bằng chứng: delta mô tả, chưa có CI của chênh lệch."
    lo, hi = ci95
    if not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError("Invalid delta CI")
    if delta_pp >= 2 and lo > 0:
        return "Có bằng chứng cải thiện trên tập evaluation và seed đã đo."
    if delta_pp <= -1 and hi < 0:
        return "Có bằng chứng giảm trên tập evaluation và seed đã đo."
    return "Chưa đủ bằng chứng kết luận cải thiện hoặc giảm."


def contrasts(values):
    a1, a2, a3, a4 = values
    return {"guard": a2-a1, "feature": a3-a1,
            "interaction_2x2": a4-a2-a3+a1, "full_over_best_single": a4-np.maximum(a2, a3)}


def paired_cfe_bootstrap(predictions, n_boot=1000, seed=42):
    """Same stratified image resamples across models; conditional on one training seed."""
    from sklearn.metrics import roc_auc_score
    if n_boot < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    rng = np.random.default_rng(seed)
    output = {}
    for condition in ("clean", "q30"):
        y = predictions[NAMES[0]][condition]["targets"]
        real, fake = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
        aucs, fake90s = [], []
        for _ in range(n_boot):
            ix = np.concatenate([rng.choice(real, len(real)), rng.choice(fake, len(fake))])
            by, av, fv = y[ix], [], []
            for name in NAMES:
                p = predictions[name][condition]["probabilities"][ix]
                av.append(roc_auc_score(by, p)*100.)
                fv.append((p[by == 1] >= np.quantile(p[by == 0], .9)).mean()*100.)
            aucs.append(av); fake90s.append(fv)
        for metric, samples in ((f"auc_{condition}", aucs), (f"fake_at_real90_{condition}", fake90s)):
            draws = np.asarray(samples).T
            output[metric] = {key: np.percentile(vals, [2.5, 97.5]).tolist()
                              for key, vals in contrasts(draws).items()}
    return output


def load_verified_run(folder, name):
    folder = Path(folder)
    proof = read_json(folder / "run_verification.json")
    if proof.get("schema") != "ablation-run/v1" or proof.get("evidence_level") != "measured":
        raise ValueError("Missing measured run verification")
    for file, digest in proof["files"].items():
        if Path(file).name != file or sha256_file(folder/file) != digest:
            raise ValueError(f"Run artifact changed: {file}")
    with (folder / "train_history.csv").open(encoding="utf-8", newline="") as handle:
        steps = validate_history(list(csv.DictReader(handle)), proof["epochs"])
    if steps != proof["steps_per_epoch"]:
        raise ValueError("Step count changed")
    result = {"epochs": proof["epochs"], **parse_val_groups_csv(folder/"val_groups.csv", proof["selected_epoch"]),
              **parse_cfe_eval_json(folder/f"eval_v3_{name}_clean.json", folder/f"eval_v3_{name}_q30.json")}
    predictions = {}
    for condition in ("clean", "q30"):
        evaluation = read_json(folder/f"eval_v3_{name}_{condition}.json")
        load = evaluation["checkpoints"][0]["load_report"]
        layout = load.get("layout", {})
        if (load.get("sha256") != proof["files"][proof["best_name"]] or load.get("missing") != 0
                or load.get("unexpected") != 0 or layout.get("n") != 3 or layout.get("dropped") != []):
            raise ValueError("Evaluation checkpoint does not match verified best/canonical layout")
        if evaluation["git_commit"] != proof["source_commit"]:
            raise ValueError("Evaluation and training source commits differ")
        if evaluation["partition"]["calibration_sha256"] != proof["context"]["calibration_file_sha256"]:
            raise ValueError("Calibration changed since training")
        prediction_path = folder/f"logits_{name}_{condition}.npz"
        if evaluation.get("predictions") != {"file": prediction_path.name, "sha256": sha256_file(prediction_path)}:
            raise ValueError("Prediction NPZ is not bound to evaluation JSON")
        with np.load(prediction_path, allow_pickle=False) as data:
            ids, y, logits = data["sample_ids"].astype(str), data["targets"], data["logits"]
            temperature = float(data["temperature"])
        if (len(ids) != len(set(ids)) or y.shape != ids.shape or logits.shape != (len(ids), 2)
                or set(y.tolist()) != {0, 1} or not np.isfinite(logits).all() or temperature <= 0
                or not np.isfinite(temperature) or len(y) != evaluation["dataset"]["n"]):
            raise ValueError("Invalid evaluation predictions")
        if temperature != evaluation["calibration"]["T"]:
            raise ValueError("NPZ and evaluation temperatures differ")
        from tools.eval_communityforensics import calculate_metrics_v3
        import torch
        p = torch.softmax(torch.from_numpy(logits).float()/temperature, dim=-1)[:, 1].numpy()
        recomputed = calculate_metrics_v3(p, y, threshold=evaluation["calibration"]["T_star"])
        for key in ("auc", "fake_at_real90", "ece"):
            if not np.isclose(recomputed[key], evaluation["metrics"][key], atol=.02):
                raise ValueError("NPZ predictions and evaluation metrics differ")
        order = np.argsort(ids)
        predictions[condition] = {"ids": ids[order], "targets": y[order], "probabilities": p[order]}
    if (not np.array_equal(predictions["clean"]["ids"], predictions["q30"]["ids"])
            or not np.array_equal(predictions["clean"]["targets"], predictions["q30"]["targets"])):
        raise ValueError("Clean/Q30 sample IDs differ")
    return result, predictions, proof


def build_summary(experiment_dir, n_boot=1000):
    data, predictions, proofs = {}, {}, []
    for name in NAMES:
        data[name], predictions[name], proof = load_verified_run(Path(experiment_dir)/name, name)
        if (proof["context"].get("loss_mode") != "v2" or
                (proof["context"]["weight_feat"], proof["context"]["weight_guard"]) != WEIGHTS[NAMES.index(name)]):
            raise ValueError("Incorrect v2 ablation weights")
        proofs.append(proof)
    contexts = []
    for proof in proofs:
        ctx = dict(proof["context"])
        ctx.pop("weight_feat"); ctx.pop("weight_guard")
        contexts.append(ctx)
    if any(c != contexts[0] for c in contexts) or len({p["source_commit"] for p in proofs}) != 1:
        raise ValueError("Training contexts differ outside the two ablated weights")
    for name in NAMES:
        if data[name]["partition"] != data[NAMES[0]]["partition"]:
            raise ValueError("Evaluation partitions differ between models")
        for condition in ("clean", "q30"):
            ref, current = predictions[NAMES[0]][condition], predictions[name][condition]
            if not np.array_equal(ref["ids"], current["ids"]) or not np.array_equal(ref["targets"], current["targets"]):
                raise ValueError("Models do not share evaluation sample IDs and labels")
    descriptive = {key: {"delta_pp": float(value), "ci95": None,
                         "verdict": evaluate_delta_verdict(value)}
                   for key, value in contrasts([data[n]["worst_group"] for n in NAMES]).items()}
    return {"report_schema": "ablation-v2/v1", "evidence_level": "measured", "gate_eligible": False,
            "gate_note": "Ablation evidence only; this report does not evaluate every G7.2 requirement.",
            "units": "percent; deltas in percentage points", "configurations": data,
            "worst_group_descriptive_contrasts": descriptive,
            "paired_cfe_contrast_ci95": paired_cfe_bootstrap(predictions, n_boot),
            "bootstrap": {"n": n_boot, "seed": 42, "unit": "stratified images, paired across models",
                          "limitation": "Conditional on training seed 42; excludes variation across training seeds."},
            "primary_driver": None,
            "summary": "Chưa kết luận loss đóng góp nhiều nhất trên worst_group: thiếu CI độc lập của chỉ số selection. Cả bốn run dùng dữ liệu v2 nên không tách được đóng góp riêng của dữ liệu."}


def write_outputs(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"ablation_mini_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    columns = ("epochs", "selected_epoch", "worst_group", "clean_r", "q30_r", "q30_f",
               "fake_at_real90_q30", "auc_clean", "auc_q30", "ece_q30_frozen_temperature_pct")
    with (output/"ablation_mini_v2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("config", *columns))
        writer.writeheader()
        for name in NAMES:
            writer.writerow({"config": name, **{key: report["configurations"][name][key] for key in columns}})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, keys in zip(axes, (("worst_group", "clean_r"), ("fake_at_real90_q30", "auc_q30"))):
        for i, key in enumerate(keys):
            values = [report["configurations"][n][key] for n in NAMES]
            intervals = [report["configurations"][n].get(key+"_ci95") for n in NAMES]
            errors = np.asarray([[max(0., v-ci[0]), max(0., ci[1]-v)] for v, ci in zip(values, intervals)]).T if all(intervals) else None
            ax.bar(np.arange(4)+(i-.5)*.35, values, width=.35, label=key, yerr=errors, capsize=3)
        ax.set_xticks(range(4), ["A1", "A2", "A3", "A4"]); ax.set_ylim(0, 110); ax.set_ylabel("Percent"); ax.legend()
    axes[0].set_title("Selection validation: descriptive, no CI")
    axes[1].set_title("Held-out CFE Q30: image bootstrap CI")
    fig.tight_layout(); fig.savefig(output/"m7_ablation.png", dpi=180); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    # Validate everything before writing any report. No dummy mode or secondary official path.
    report = build_summary(args.experiment_dir, args.bootstrap)
    write_outputs(report, args.output_dir)
    print(report["summary"])


if __name__ == "__main__":
    main()
