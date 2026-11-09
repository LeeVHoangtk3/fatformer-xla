"""Recover only frozen calibration rows from a measured pool; never refit on evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.calibration_data import identities, read_partition, sha256_file, validate_partitions


def recover(source, calibration_split, evaluation_split, output, audit_out):
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError("Recovery must preserve the original measured pool")
    with np.load(source, allow_pickle=False) as payload:
        if "evidence_level" in payload and str(payload["evidence_level"]) != "measured":
            raise ValueError("Cannot recover synthetic/unverified logits")
        logits, targets = payload["logits"].copy(), payload["targets"].copy()
        paths = payload["paths"].copy()
        ids = identities(paths, targets)
        checkpoint_sha = str(payload["checkpoint_sha256"])
        preprocessing = str(payload["preprocessing"])
    if logits.shape != (len(targets), 2) or not np.isfinite(logits).all():
        raise ValueError("Invalid finite [N,2] logits")
    if len(checkpoint_sha) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha):
        raise ValueError("Missing measured source checkpoint SHA-256")
    calib_ids, calib_labels = read_partition(calibration_split, "calibration")
    eval_ids, _ = read_partition(evaluation_split, "evaluation")
    index = dict(zip(ids, range(len(ids))))
    missing = set(calib_ids) - set(index)
    if missing:
        raise ValueError(f"Pool missing {len(missing)} calibration samples")
    rows = [index[sid] for sid in calib_ids]
    if not np.array_equal(targets[rows], calib_labels):
        raise ValueError("Pool labels disagree with the frozen calibration split")
    partition = validate_partitions(calib_ids, targets[rows], calibration_split, evaluation_split)
    audit = {
        "schema": "calibration_recovery/v1", "status": "recovered",
        "method": "Select frozen calibration IDs from saved logits; no model forward",
        "scientific_status": "post_hoc_recovery; evaluation previously inspected",
        "source_pool": str(source), "source_pool_sha256": sha256_file(source),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_pool_n": len(ids), "source_pool_evaluation_overlap": len(set(ids) & set(eval_ids)),
        "partition_validation": partition,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, logits=logits[rows], targets=targets[rows],
                        sample_ids=np.array(calib_ids), paths=paths[rows], split="calibration",
                        checkpoint_sha256=checkpoint_sha, preprocessing=preprocessing,
                        evidence_level="measured", manifest_sha256=sha256_file(calibration_split),
                        partition_validation=json.dumps(partition, sort_keys=True),
                        recovery=json.dumps(audit, sort_keys=True))
    audit["output_npz"] = str(output)
    audit["output_npz_sha256"] = sha256_file(output)
    Path(audit_out).parent.mkdir(parents=True, exist_ok=True)
    Path(audit_out).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "calibration_split", "evaluation_split", "output", "audit_out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
