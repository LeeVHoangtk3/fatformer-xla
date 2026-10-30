from types import SimpleNamespace

import pytest
import torch
from torch import nn

import src.models.canonical_loader as canonical_loader
from src.models.clip import model as clip_model_module


class _Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))


class _Block(nn.Module):
    def __init__(self, enabled):
        super().__init__()
        if enabled:
            self.forgery_aware_adapter = _Adapter()


class _ToyModel(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.adapter_block_indices = tuple(indices)
        self.clip_model = nn.Module()
        self.clip_model.visual = nn.Module()
        self.clip_model.visual.transformer = nn.Module()
        self.clip_model.visual.transformer.resblocks = nn.ModuleList(
            [_Block(i in set(indices)) for i in range(24)]
        )
        self.head = nn.Linear(2, 2, bias=False)


def _install_toy_builder(monkeypatch):
    def build_model(args):
        return _ToyModel(args.adapter_block_indices)

    monkeypatch.setattr(canonical_loader, "build_model", build_model, raising=False)
    return build_model


@pytest.mark.parametrize(
    "indices",
    [
        [7, 15, 23],
        [2, 5, 8, 11, 14, 17, 20, 23],
    ],
)
def test_loads_exact_explicit_layout(monkeypatch, indices):
    builder = _install_toy_builder(monkeypatch)
    source = _ToyModel(indices).state_dict()
    model, report = canonical_loader.load_canonical(source, SimpleNamespace(), allow_partial=False)

    assert model.adapter_block_indices == tuple(indices)
    assert report["missing"] == report["unexpected"] == 0
    assert report["layout"]["partial"] is False
    assert builder is not None


def test_constructor_falls_back_to_legacy_adapter_count(monkeypatch):
    captured = {}

    class CaptureTransformer(nn.Module):
        def __init__(self, width, layers, heads, attn_mask=None, add_adapter=None, args=None):
            super().__init__()
            captured["mask"] = list(add_adapter)

    monkeypatch.setattr(clip_model_module, "Transformer", CaptureTransformer)
    clip_model_module.VisionTransformer(224, 14, 16, 24, 4, 8, args=SimpleNamespace(num_vit_adapter=3))
    assert [i for i, enabled in enumerate(captured["mask"]) if enabled] == [7, 15, 23]


def test_constructor_uses_explicit_adapter_indices(monkeypatch):
    captured = {}

    class CaptureTransformer(nn.Module):
        def __init__(self, width, layers, heads, attn_mask=None, add_adapter=None, args=None):
            super().__init__()
            captured["mask"] = list(add_adapter)

    monkeypatch.setattr(clip_model_module, "Transformer", CaptureTransformer)
    indices = [2, 5, 8, 11, 14, 17, 20, 23]
    clip_model_module.VisionTransformer(
        224, 14, 16, 24, 4, 8, args=SimpleNamespace(adapter_block_indices=indices)
    )
    assert [i for i, enabled in enumerate(captured["mask"]) if enabled] == indices


def test_constructor_rejects_conflicting_explicit_indices_and_count(monkeypatch):
    class CaptureTransformer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(clip_model_module, "Transformer", CaptureTransformer)
    with pytest.raises(ValueError, match="mâu thuẫn"):
        clip_model_module.VisionTransformer(
            224, 14, 16, 24, 4, 8,
            args=SimpleNamespace(num_vit_adapter=3, adapter_block_indices=[2, 5, 8, 11, 14, 17, 20, 23]),
        )


@pytest.mark.parametrize("mutation,expected_key", [("missing", "head.weight"), ("unexpected", "extra.weight")])
def test_strict_loading_reports_key_mismatch(monkeypatch, mutation, expected_key):
    _install_toy_builder(monkeypatch)
    source = dict(_ToyModel([7, 15, 23]).state_dict())
    if mutation == "missing":
        source.pop(expected_key)
    else:
        source[expected_key] = torch.zeros(1)

    with pytest.raises(canonical_loader.CheckpointCompatibilityError) as exc:
        canonical_loader.load_canonical(source, SimpleNamespace())
    assert expected_key in str(exc.value)
    assert (exc.value.report["missing_keys"] if mutation == "missing" else exc.value.report["unexpected_keys"])
    assert exc.value.report["missing"] == (1 if mutation == "missing" else 0)
    assert exc.value.report["unexpected"] == (1 if mutation == "unexpected" else 0)


def test_gen1_layout_is_rejected_without_remapping(monkeypatch):
    calls = []

    def build_model(args):
        calls.append(args)
        return _ToyModel(args.adapter_block_indices)

    monkeypatch.setattr(canonical_loader, "build_model", build_model)
    indices = [2, 5, 7, 8, 11, 14, 15, 17, 20, 23]
    source = _ToyModel(indices).state_dict()
    with pytest.raises(canonical_loader.IncompleteLayoutError, match="không ánh xạ ngầm") as exc:
        canonical_loader.load_canonical(source, SimpleNamespace())
    assert exc.value.report["layout"]["partial"] is True
    assert exc.value.report["layout"]["dropped"] == [7, 15]
    # An incompatible layout must be rejected before constructing a model.
    assert calls == []


def test_original_namespace_is_not_mutated(monkeypatch):
    _install_toy_builder(monkeypatch)
    original = SimpleNamespace(num_classes=2)
    source = _ToyModel([7, 15, 23]).state_dict()
    canonical_loader.load_canonical(source, original)
    assert not hasattr(original, "adapter_block_indices")


def test_caller_adapter_index_must_match_checkpoint(monkeypatch):
    _install_toy_builder(monkeypatch)
    source = _ToyModel([7, 15, 23]).state_dict()
    args = SimpleNamespace(adapter_block_indices=[2, 5, 8, 11, 14, 17, 20, 23])
    with pytest.raises(canonical_loader.IncompleteLayoutError, match="không khớp"):
        canonical_loader.load_canonical(source, args)


def test_wrong_adapter_prefix_does_not_guess_layout():
    state = {
        "somewhere.resblocks.7.adapter.weight": torch.zeros(1),
        "clip_model.visual.transformer.resblocks.7.custom_adapter.weight": torch.zeros(1),
    }
    assert canonical_loader.infer_adapter_blocks(state) == set()
    with pytest.raises(canonical_loader.IncompleteLayoutError, match="Không tìm thấy adapter"):
        canonical_loader.resolve_layout(set())


def test_shape_mismatch_is_reported():
    source = {"head.weight": torch.zeros(3, 2)}
    target = {"head.weight": torch.zeros(2, 2)}
    with pytest.raises(canonical_loader.ShapeMismatchError) as exc:
        canonical_loader.canonicalize(source, target)
    assert exc.value.report["shape_mismatch"][0]["key"] == "head.weight"
