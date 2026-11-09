"""Strict identities and frozen partition checks for calibration artifacts.

Identity is subset/class/filename, not image-content identity. Content duplicates
across differently named files require a separate data audit.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id(value):
    parts = str(value).replace("\\", "/").split("/")
    if ".." in parts or len(parts) < 3:
        raise ValueError(f"Expected subset/class/filename identity: {value}")
    classes = {"0": 0, "0_real": 0, "real": 0,
               "1": 1, "1_fake": 1, "fake": 1}
    if parts[-2] not in classes or not parts[-3] or not parts[-1]:
        raise ValueError(f"Invalid class/path in sample identity: {value}")
    return f"{parts[-3]}/{classes[parts[-2]]}/{parts[-1]}"


def identities(values, targets):
    targets = np.asarray(targets)
    if targets.ndim != 1 or targets.dtype.kind not in "iu" or not np.isin(targets, [0, 1]).all():
        raise ValueError("Targets must be a one-dimensional integer 0/1 array")
    if len(values) != len(targets) or not len(targets):
        raise ValueError("Empty or inconsistent sample identities/targets")
    ids = [sample_id(value) for value in values]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate canonical sample identity")
    if any(int(sid.split("/")[1]) != int(y) for sid, y in zip(ids, targets)):
        raise ValueError("Sample identity class disagrees with target")
    return ids


def read_partition(path, expected_split):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["split"]) != expected_split:
                raise ValueError(f"Require split={expected_split}: {path}")
            targets = payload["targets"].copy()
            ids = identities(payload["sample_ids"], targets)
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if payload.get("split") != expected_split:
                raise ValueError(f"Require split={expected_split}: {path}")
            payload = payload["samples"]
        if not isinstance(payload, list):
            raise ValueError("JSON partition must contain a list of samples")
        targets = np.asarray([item.get("target", item.get("label")) for item in payload])
        ids = identities([item.get("sample_id", item.get("path")) for item in payload], targets)
    else:
        raise ValueError("Partition must be NPZ or JSON")
    return ids, targets


def validate_partitions(values, targets, calibration_split, evaluation_split):
    actual = identities(values, targets)
    calib, labels = read_partition(calibration_split, "calibration")
    evaluation, _ = read_partition(evaluation_split, "evaluation")
    if set(calib) & set(evaluation):
        raise ValueError("Calibration/evaluation partition overlap")
    if set(actual) != set(calib):
        raise ValueError("Calibration samples do not exactly match the frozen split")
    actual_labels = dict(zip(actual, map(int, targets)))
    if any(actual_labels[sid] != int(y) for sid, y in zip(calib, labels)):
        raise ValueError("Frozen calibration labels disagree")
    return {
        "status": "verified_disjoint_ids",
        "identity_unit": "subset/class/filename; content duplicates not audited",
        "n_calibration": len(calib), "n_evaluation": len(evaluation),
        "overlap": 0,
        "calibration_split_sha256": sha256_file(calibration_split),
        "evaluation_split_sha256": sha256_file(evaluation_split),
        "sample_ids_sha256": hashlib.sha256("\n".join(sorted(actual)).encode()).hexdigest(),
    }
