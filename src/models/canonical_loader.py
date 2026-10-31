"""Strict checkpoint loading with an explicit FatFormer adapter-block index.

Integration contract: ``load_canonical`` receives a checkpoint path (or a
synthetic state-dict object in tests) and the same ``SimpleNamespace`` model
arguments accepted by :func:`src.models.build_model`. The adapter index is
derived from recognized checkpoint key prefixes and copied onto a cloned
namespace before model construction. The ``build_model(**model_args)`` form
shown in an earlier task sketch is illustrative; the evaluator passes a
``SimpleNamespace`` and the repository factory accepts that object.

``adapter_index`` maps each logical ``block_<i>`` to the explicit checkpoint
prefixes under ``clip_model.visual`` and/or ``image_encoder``. No checkpoint
key is silently renamed. ``image_encoder.*`` is an alias of
``clip_model.visual.*`` in this model: it is retained when the target
state_dict exposes that alias, and discarded only when the corresponding
canonical key is present and the target has no alias key. Top-level ``srm.*``
and ``gating.*`` keys are discarded only when the target has no such module.

Raises:
    FileNotFoundError: a requested checkpoint path does not exist.
    TypeError / ValueError: invalid input, malformed state_dict, duplicate
        prefixes, unsupported adapter indices, or invalid model arguments.
    IncompleteLayoutError: checkpoint adapter blocks do not exactly match a
        supported architecture layout (including the legacy Gen 1 mixture).
    ShapeMismatchError: a shared checkpoint/model key has incompatible shape.
    CheckpointCompatibilityError: strict loading finds missing or unexpected
        keys, or torch reports another state_dict incompatibility.
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch import nn
from . import build_model


LAYOUT_COUNTS: Tuple[int, ...] = (24, 12, 8, 6, 4, 3)
_ADAPTER_KEY_RE = re.compile(
    r"^(?P<root>clip_model\.visual|image_encoder)\.transformer\.resblocks\."
    r"(?P<block>\d+)\.(?:forgery_aware_adapter|attn\.adapter)(?:\.|$)",
    re.IGNORECASE,
)
class CheckpointCompatibilityError(RuntimeError):
    """Checkpoint and constructed model have incompatible state_dict keys."""

    def __init__(self, message: str, *, report: Optional[dict] = None):
        super().__init__(message)
        self.report = report or {}


class IncompleteLayoutError(CheckpointCompatibilityError):
    """Adapter blocks cannot be represented by one supported model layout."""


class ShapeMismatchError(CheckpointCompatibilityError):
    """A shared state_dict key has a different tensor shape."""


def _supported_indices(n: int, layers: int) -> Set[int]:
    if n <= 0 or layers % n:
        return set()
    stride = layers // n
    return {(i + 1) * stride - 1 for i in range(n)}


def _is_tensor_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) for key in value
    ) and any(torch.is_tensor(item) for item in value.values())


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Extract a tensor state_dict from a path, mapping, or ``state_dict`` object.

    Recognizes nested ``model``, ``state_dict`` and ``model_state`` mappings
    and strips repeated leading ``module.`` prefixes. Prefix collisions raise
    ``ValueError`` rather than overwriting a tensor.
    """
    if isinstance(ckpt, (str, os.PathLike, Path)):
        path = Path(ckpt)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint không tồn tại: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # compatibility with older PyTorch
            payload = torch.load(path, map_location="cpu")
    else:
        payload = ckpt

    if not isinstance(payload, Mapping):
        state_method = getattr(payload, "state_dict", None)
        if not callable(state_method):
            raise TypeError("Checkpoint phải là path, mapping hoặc object có state_dict().")
        payload = state_method()

    for container_key in ("model", "state_dict", "model_state"):
        nested = payload.get(container_key)
        if _is_tensor_mapping(nested):
            payload = nested
            break

    if not _is_tensor_mapping(payload):
        raise ValueError("Không tìm thấy state_dict chứa tensor trong checkpoint.")

    result: Dict[str, torch.Tensor] = {}
    for raw_key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        key = raw_key
        while key.startswith("module."):
            key = key[len("module."):]
        if key in result:
            raise ValueError(f"Trùng key sau khi bỏ prefix module.: {key}")
        result[key] = value
    return result


def infer_adapter_blocks(sd: Mapping[str, torch.Tensor]) -> Set[int]:
    """Return block indices found under recognized FatFormer adapter prefixes.

    Recognized roots are exactly ``clip_model.visual`` and ``image_encoder``;
    adapter modules are exactly ``forgery_aware_adapter`` or the legacy
    ``attn.adapter`` key spelling.
    """
    blocks: Set[int] = set()
    for key in sd:
        match = _ADAPTER_KEY_RE.match(key)
        if match:
            blocks.add(int(match.group("block")))
    return blocks


def resolve_layout(blocks: Set[int], layers: int = 24) -> dict:
    """Resolve an exact supported adapter layout, or describe nearest partial.

    Exact equality is required: merely being a subset of the 24-block layout
    does not make a checkpoint compatible. Among exact matches, the smallest
    ``n`` is selected. If none matches, the candidate with the smallest
    symmetric difference is reported (ties use the smaller ``n``); callers
    official callers must reject that partial layout; forensic callers may
    inspect it only through ``load_canonical(..., allow_partial=True)``.
    """
    if layers <= 0:
        raise ValueError("layers phải lớn hơn 0.")
    normalized = {int(block) for block in blocks}
    if any(block < 0 or block >= layers for block in normalized):
        raise ValueError(f"Adapter index ngoài [0, {layers}): {sorted(normalized)}")
    if not normalized:
        raise IncompleteLayoutError("Không tìm thấy adapter block nào trong checkpoint.")

    candidates = []
    for n in LAYOUT_COUNTS:
        expected = _supported_indices(n, layers)
        if expected:
            dropped = normalized - expected
            missing = expected - normalized
            candidates.append((n, expected, dropped, missing))

    exact = [candidate for candidate in candidates if not candidate[2] and not candidate[3]]
    if exact:
        n, indices, _, _ = min(exact, key=lambda item: item[0])
        return {
            "n": n, "indices": sorted(indices), "dropped": [], "missing": [],
            "partial": False, "expressible": True,
        }

    n, indices, dropped, missing = min(
        candidates,
        key=lambda item: (len(item[2]) + len(item[3]), item[0]),
    )
    return {
        "n": n, "indices": sorted(indices), "dropped": sorted(dropped),
        "missing": sorted(missing), "partial": True, "expressible": False,
    }


def _adapter_index(sd: Mapping[str, torch.Tensor], blocks: Iterable[int]) -> Dict[str, List[str]]:
    """Map logical block names to the recognized key prefixes actually present."""
    result: Dict[str, List[str]] = {}
    for block in sorted(blocks):
        prefixes = set()
        for key in sd:
            match = _ADAPTER_KEY_RE.match(key)
            if match and int(match.group("block")) == block:
                prefix = key[:match.start("block")] + str(block) + "."
                prefixes.add(prefix)
        result[f"block_{block}"] = sorted(prefixes)
    return result


def canonicalize(
    sd: Mapping[str, torch.Tensor], model_sd: Mapping[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Prepare keys for the model and report key/shape compatibility.

    No key is renamed. Duplicate ``image_encoder`` aliases are omitted only
    when the model does not expose that alias and its ``clip_model.visual``
    counterpart exists in both checkpoint and target. Shape mismatches raise
    ``ShapeMismatchError`` with the complete mismatch list.
    """
    clean: Dict[str, torch.Tensor] = {}
    dropped_aliases: List[str] = []
    dropped_top_level: List[str] = []
    for key, tensor in sd.items():
        if key.startswith(("srm.", "gating.")) and key not in model_sd:
            dropped_top_level.append(key)
            continue
        if key.startswith("image_encoder.") and key not in model_sd:
            counterpart = "clip_model.visual." + key[len("image_encoder."):]
            if counterpart in sd and counterpart in model_sd:
                dropped_aliases.append(key)
                continue
        clean[key] = tensor

    shape_mismatch = [
        {"key": key, "checkpoint_shape": tuple(tensor.shape),
         "model_shape": tuple(model_sd[key].shape)}
        for key, tensor in clean.items()
        if key in model_sd and tuple(tensor.shape) != tuple(model_sd[key].shape)
    ]
    report = {
        "missing_keys": sorted(set(model_sd) - set(clean)),
        "unexpected_keys": sorted(set(clean) - set(model_sd)),
        "shape_mismatch": shape_mismatch,
        "dropped_alias_keys": sorted(dropped_aliases),
        "dropped_top_level_keys": sorted(dropped_top_level),
    }
    if shape_mismatch:
        raise ShapeMismatchError(
            "Shape mismatch trong checkpoint: " + ", ".join(item["key"] for item in shape_mismatch),
            report=report,
        )
    return clean, report


def _clone_model_args(model_args: Any, adapter_indices: List[int]) -> Any:
    """Clone the evaluator's namespace and set explicit adapter indices."""
    if not hasattr(model_args, "__dict__"):
        raise TypeError("model_args phải là SimpleNamespace hoặc object dạng namespace.")
    cloned = copy.copy(model_args)
    setattr(cloned, "adapter_block_indices", list(adapter_indices))
    return cloned


def _sha256(path: Any) -> Optional[str]:
    if not isinstance(path, (str, os.PathLike, Path)):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_canonical(
    ckpt_path: Any,
    model_args: SimpleNamespace,
    allow_partial: bool = False,
) -> Tuple[nn.Module, dict]:
    """Build and strictly load a model using its explicit checkpoint layout.

    ``model_args`` is the ``SimpleNamespace`` used by the evaluator. Its
    original object is never mutated. By default an unrepresentable layout is
    rejected. ``allow_partial=True`` is an explicit forensic escape hatch: it
    builds the nearest supported layout, loads compatible tensors with
    ``strict=False`` and returns the complete dropped/missing/unexpected report.
    This mode must never be used for an official benchmark.
    """
    state_dict = extract_state_dict(ckpt_path)
    layers = int(getattr(model_args, "layers", 24))
    blocks = infer_adapter_blocks(state_dict)
    requested_indices = getattr(model_args, "adapter_block_indices", None)
    if requested_indices is not None:
        if not isinstance(requested_indices, (list, tuple)) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in requested_indices
        ):
            raise TypeError("model_args.adapter_block_indices phải là list/tuple số nguyên.")
        if len(set(requested_indices)) != len(requested_indices):
            raise ValueError("model_args.adapter_block_indices chứa index trùng lặp.")
        if set(requested_indices) != blocks:
            raise IncompleteLayoutError(
                "Adapter-index caller không khớp các prefix adapter trong checkpoint: "
                f"caller={sorted(requested_indices)}, checkpoint={sorted(blocks)}."
            )
    layout = resolve_layout(blocks, layers=layers)
    adapter_index = _adapter_index(state_dict, blocks)
    if layout["partial"] and not allow_partial:
        raise IncompleteLayoutError(
            "Checkpoint adapter layout không biểu diễn được chính xác: "
            f"blocks={sorted(blocks)}, nearest_n={layout['n']}, "
            f"dropped={layout['dropped']}, missing={layout['missing']}. "
            "Checkpoint bị từ chối; không ánh xạ ngầm sang n=3/n=8.",
            report={"layout": layout, "adapter_blocks": sorted(blocks), "adapter_index": adapter_index},
        )

    model = build_model(_clone_model_args(model_args, layout["indices"]))
    clean_sd, report = canonicalize(state_dict, model.state_dict())
    missing_keys = report["missing_keys"]
    unexpected_keys = report["unexpected_keys"]
    report.update({
        "missing": len(missing_keys),
        "unexpected": len(unexpected_keys),
        "adapter_blocks": sorted(blocks),
        "adapter_index": adapter_index,
        "layout": layout,
        "n_keys": len(state_dict),
        "path": str(ckpt_path) if isinstance(ckpt_path, (str, os.PathLike, Path)) else None,
        "sha256": _sha256(ckpt_path),
    })
    if (missing_keys or unexpected_keys) and not allow_partial:
        raise CheckpointCompatibilityError(
            f"Checkpoint không tương thích: missing={missing_keys}; unexpected={unexpected_keys}",
            report=report,
        )

    try:
        incompatibility = model.load_state_dict(clean_sd, strict=not allow_partial)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"PyTorch từ chối state_dict: {exc}", report=report
        ) from exc

    # Keep numeric fields for evaluator compatibility and full lists for audit.
    report["missing"] = len(incompatibility.missing_keys)
    report["unexpected"] = len(incompatibility.unexpected_keys)
    report["missing_keys"] = list(incompatibility.missing_keys)
    report["unexpected_keys"] = list(incompatibility.unexpected_keys)
    return model, report
