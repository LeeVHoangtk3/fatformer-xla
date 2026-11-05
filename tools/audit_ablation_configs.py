"""Audit the four explicit v2 loss configs; this is not a GPU readiness gate."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.ablation_v2_common import NAMES, WEIGHTS


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def audit_configs(directory=ROOT/"configs/ablation_v2"):
    configs, hashes = {}, []
    for name, weights in zip(NAMES, WEIGHTS):
        config = yaml.safe_load((Path(directory)/f"{name}.yaml").read_text(encoding="utf-8-sig"))
        if "inherits" in config:
            raise ValueError("Training CLI does not implement inherits; use full explicit configs")
        loss = config["loss"]
        if loss.get("mode") != "v2" or (loss["weight_feat"], loss["weight_guard"]) != weights:
            raise ValueError(f"Wrong loss mode/ablation weights: {name}")
        if config["experiment"]["seed"] != 42 or config["training"]["epochs"] < 1:
            raise ValueError("Require seed 42 and a positive common epoch horizon")
        if config["dataset"].get("val_split") != "val_select":
            raise ValueError("Require isolated val_select")
        stripped = copy.deepcopy(config)
        for key in ("weight_feat", "weight_guard"):
            del stripped["loss"][key]
        hashes.append(canonical_hash(stripped)); configs[name] = config
    if len(set(hashes)) != 1:
        raise ValueError("Configs differ outside the two ablated loss weights")
    return configs


def run_audit(directory=ROOT/"configs/ablation_v2"):
    try:
        audit_configs(directory)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"FAIL: {exc}"); return False
    print("PASS: four explicit v2 configs differ only in Guard/Feature weights. GPU/data/run readiness not assessed.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs_dir", type=Path, default=ROOT/"configs/ablation_v2")
    raise SystemExit(0 if run_audit(parser.parse_args().configs_dir) else 1)
