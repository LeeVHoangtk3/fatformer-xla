"""Epoch-boundary resume state and immutable input fingerprints."""
import hashlib
import random
from pathlib import Path

import numpy as np
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_path(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.is_absolute():
        return path.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Image path escapes data root: {path}")
    return resolved


def source_sha256(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def capture_rng(loader=None):
    generator = getattr(loader, "generator", None)
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": generator.get_state() if generator is not None else None}


def restore_rng(state, loader=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the original CUDA device count")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    generator = getattr(loader, "generator", None)
    if state["loader"] is not None:
        if generator is None:
            raise ValueError("Checkpoint requires a DataLoader generator")
        generator.set_state(state["loader"].cpu())


def training_context(args, root):
    names = ("branch", "epochs", "batch_size", "grad_accum_steps", "num_workers", "use_amp",
             "learning_rate", "anchor_learning_rate", "grad_clip_norm", "temperature", "label_smoothing",
             "weight_ce", "weight_ce_deg", "alpha_kd", "beta_cons", "weight_feat", "weight_guard",
             "fake_index", "real_guard_threshold", "selection_threshold", "q30_max_samples",
             "img_resolution", "crop_resolution", "val_split", "dry_run")
    context = {name: getattr(args, name, None) for name in names}
    context["device_type"] = torch.device(args.device).type
    context["torch_version"] = torch.__version__
    context["cuda_runtime"] = torch.version.cuda
    cached = {}
    assets = getattr(args, "_resolved_model_assets", {})
    for name in ("train_manifest", "degradation_config", "calibration_file", "checkpoint_init", "teacher_checkpoint", "clip_path"):
        path = assets.get(name, getattr(args, name, None))
        if path and Path(path).is_file():
            key = str(Path(path).resolve())
            if key not in cached:
                cached[key] = sha256_file(path)
            context[name + "_sha256"] = cached[key]
        else:
            context[name + "_sha256"] = None
    context["source_sha256"] = source_sha256(root)
    return context
