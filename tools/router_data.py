"""Collect and validate measured Task 7.13 image/gating/logit caches."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.degradation_router import DegradationRouter, forward_with_gating
from src.training.runtime_state import sha256_file

FEATURES = ["blockiness", "log1p_lapvar", "lambda_bar"]
SUBSETS = ["1_commercial", "2_latent_diffusion", "3_pixel_diffusion",
           "4_modern_gans", "5_other_autoregressive"]
PREPROCESS = "resize224_bicubic_centercrop224_jpeg30_post_resize_clip_normalize/v1"


def params_digest(params):
    import hashlib
    return hashlib.sha256(json.dumps(params, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_cache(path, split):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files if k != "metadata"}
        meta = json.loads(str(archive["metadata"]))
    if (meta.get("schema") != "router-data/v2" or meta.get("evidence_level") != "measured"
            or meta.get("split") != split or meta.get("features") != FEATURES
            or meta.get("preprocess") != PREPROCESS):
        raise ValueError("Require measured router-data/v2 with correct split/features/preprocess")
    n = len(data["targets"])
    if n == 0 or data["features"].shape != (n, 3) or data["logits"].shape != (n, 2):
        raise ValueError("Invalid feature/logit shapes")
    for key in ("sample_ids", "targets", "is_degraded", "subsets"):
        if data[key].shape != (n,):
            raise ValueError(f"Invalid {key} shape")
    if (not np.isfinite(data["features"]).all() or not np.isfinite(data["logits"]).all()
            or (data["features"][:, 1] < 0).any()
            or set(data["targets"].tolist()) != {0, 1}
            or set(data["is_degraded"].tolist()) != {0, 1}
            or ((data["features"][:, 2] < 0) | (data["features"][:, 2] > 2)).any()):
        raise ValueError("Nonfinite signals/logits or invalid class/condition/gating labels")
    groups = {}
    for sid, y, condition, subset in zip(data["sample_ids"], data["targets"], data["is_degraded"], data["subsets"]):
        if len(sid) != 64 or any(c not in "0123456789abcdef" for c in sid):
            raise ValueError("Sample IDs must be source-image SHA-256 hashes")
        group = groups.setdefault(str(sid), [])
        group.append((int(y), int(condition), str(subset)))
    if any(len(rows) != 2 or {r[1] for r in rows} != {0, 1}
           or len({(r[0], r[2]) for r in rows}) != 1 for rows in groups.values()):
        raise ValueError("Require one clean and one Q30 view per unique source image")
    digest = meta.get("checkpoint_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Missing checkpoint SHA-256")
    return data, meta


def collect(args):
    from tools.eval_communityforensics import build_cfe_model_v3, compress_jpeg_q30, get_git_info, scan_cfe_dataset
    if args.manifest:
        payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        records = payload["records"]
        if not records or any(r.get("split") != "val_calib" for r in records):
            raise ValueError("Calibration collector only accepts val_calib records")
        split, source = "val_calib", {"manifest_sha256": sha256_file(args.manifest)}
    else:
        from tools.ablation_v2_common import filter_records
        if not args.evaluation_split:
            raise ValueError("CFE evaluation requires the frozen --evaluation-split NPZ")
        with np.load(args.evaluation_split, allow_pickle=False) as reference:
            if str(reference["split"]) != "evaluation":
                raise ValueError("Require split=evaluation")
            ids, targets = reference["sample_ids"], reference["targets"]
        if len(set(ids.tolist())) != len(ids) or targets.shape != ids.shape:
            raise ValueError("Invalid frozen evaluation IDs")
        selected = filter_records(scan_cfe_dataset(args.cfe_dir), dict(zip(ids.tolist(), targets.tolist())))
        records = [r for rows in selected.values() for r in rows]
        split, source = "evaluation", {"evaluation_split_sha256": sha256_file(args.evaluation_split)}
    # Fail before loading a large model if any source image is missing or changed.
    for record in records:
        digest = sha256_file(record["path"])
        if record.get("sha256", digest) != digest:
            raise ValueError(f"Image hash changed: {record['path']}")
        record["content_sha256"] = digest
    if len({r["content_sha256"] for r in records}) != len(records):
        raise ValueError("Duplicate source images in collection manifest")
    device = torch.device(args.device)
    model, report = build_cfe_model_v3("anchored", args.checkpoint, args.clip_path, device)
    crop = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.CenterCrop(224)])
    normalize = transforms.Normalize([.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711])
    router = DegradationRouter().to(device)
    values = {key: [] for key in ("features", "logits", "targets", "sample_ids", "is_degraded", "subsets")}
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start+args.batch_size]
        originals = []
        for record in batch:
            with Image.open(record["path"]) as image:
                originals.append(crop(image.convert("RGB")))
        for condition in (0, 1):
            images = [compress_jpeg_q30(im) if condition else im for im in originals]
            raw = torch.stack([transforms.functional.to_tensor(im) for im in images]).to(device)
            logits, lam = forward_with_gating(model, normalize(raw))
            values["features"].extend(router.extract_signals(raw, lam).cpu().numpy())
            values["logits"].extend(logits.cpu().numpy())
            values["targets"].extend(r["label"] for r in batch)
            values["sample_ids"].extend(r["content_sha256"] for r in batch)
            values["is_degraded"].extend([condition]*len(batch))
            values["subsets"].extend(r.get("subset", r.get("family", "")) for r in batch)
        print(f"Collected {min(start+len(batch), len(records))}/{len(records)} source images", flush=True)
    commit, dirty = get_git_info()
    meta = dict(schema="router-data/v2", evidence_level="measured", split=split, features=FEATURES,
                preprocess=PREPROCESS, checkpoint_sha256=report["sha256"], source=source,
                git_commit=commit, git_dirty=dirty, branch_design="single_logits_condition_temperature")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{k: np.asarray(v) for k, v in values.items()}, metadata=json.dumps(meta))
    load_cache(output, split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--cfe-dir")
    parser.add_argument("--evaluation-split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clip-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    collect(args)
