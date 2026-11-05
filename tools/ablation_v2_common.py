"""Validation shared by the ablation runner, evaluator and summarizer."""
import json
from pathlib import Path

import numpy as np

from src.training.runtime_state import sha256_file

NAMES = ("a1_data", "a2_data_guard", "a3_data_featcons", "a4_full")
WEIGHTS = ((0., 0.), (0., .5), (.3, 0.), (.3, .5))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_partition(evaluation, calibration_split, calibration):
    """Use the original calibration IDs, never refit/re-split after seeing results."""
    artifact = read_json(calibration)
    if artifact.get("schema") != "calibration/v1" or artifact.get("evidence_level") != "measured":
        raise ValueError("Require measured calibration/v1")
    if artifact.get("calibration_source_sha256") != sha256_file(calibration_split):
        raise ValueError("Calibration split is not the source bound to calibration.json")
    partitions = []
    for path, tag in ((calibration_split, "calibration"), (evaluation, "evaluation")):
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["split"]) != tag:
                raise ValueError(f"Require split={tag}: {path}")
            ids = payload["sample_ids"].astype(str)
            targets = payload["targets"].copy()
        if (ids.ndim != 1 or targets.shape != ids.shape or len(set(ids)) != len(ids)
                or set(targets.tolist()) != {0, 1} or any(not x or x.startswith("row:") for x in ids)):
            raise ValueError("Invalid/duplicate sample IDs or missing class coverage")
        partitions.append(dict(zip(ids.tolist(), targets.tolist())))
    if set(partitions[0]) & set(partitions[1]):
        raise ValueError("Calibration/evaluation sample IDs overlap")
    if len(partitions[0]) != artifact["n_calib"]:
        raise ValueError("Calibration count mismatch")
    return partitions[1], {
        "split": "evaluation", "evaluation_sha256": sha256_file(evaluation),
        "calibration_split_sha256": sha256_file(calibration_split),
        "calibration_sha256": sha256_file(calibration), "n": len(partitions[1]),
    }


def filter_records(records_by_subset, expected):
    selected, found = {}, {}
    for subset, records in records_by_subset.items():
        selected[subset] = []
        for record in records:
            key = f"{record['subset']}/{record['label']}/{record['fname']}"
            if key in expected:
                if key in found or record["label"] != expected[key]:
                    raise ValueError("Duplicate ID or target mismatch in CFE")
                found[key] = record["label"]
                selected[subset].append(record)
    if found != expected:
        raise ValueError(f"CFE evaluation IDs missing/mismatched: {len(found)}/{len(expected)}")
    return selected


def validate_history(history, epochs):
    """A best file alone is never evidence that a run finished."""
    if [int(row["epoch"]) for row in history] != list(range(1, epochs + 1)):
        raise ValueError("Incomplete or duplicate training epochs")
    steps = [int(row["train_steps"]) for row in history]
    if not steps or min(steps) <= 0 or len(set(steps)) != 1:
        raise ValueError("Inconsistent steps_per_epoch")
    for row in history:
        for key in ("loss_total", "grad_norm", "clean_r", "clean_f", "q30_r", "q30_f"):
            if not np.isfinite(float(row[key])):
                raise ValueError(f"Non-finite {key} at epoch {row['epoch']}")
    return steps[0]
