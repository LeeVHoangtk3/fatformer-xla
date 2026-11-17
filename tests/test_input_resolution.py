from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from src.models.clip.model import ForgeryAwareAdapterLayer, VisionTransformer
from src.models.srm import SpatialResidualBlock
from tools.infer_unified import get_default_transform
from tools.phase7_corrected import StrictImageDataset


def visual():
    return VisionTransformer(224, 14, 32, 1, 4, 16,
                             adapter_block_indices=[]).eval()


def test_336_visual_forward_preserves_checkpoint_layout_and_gradients():
    model = visual()
    before = {k: v.shape for k, v in model.state_dict().items()}
    image = torch.randn(1, 3, 336, 336, requires_grad=True)
    output = model(image, return_full=True)
    assert output.shape == (1, 577, 16)
    output.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert model.positional_embedding.grad is not None
    assert {k: v.shape for k, v in model.state_dict().items()} == before


def test_224_visual_matches_original_forward_exactly():
    model = visual()
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        tokens = model.conv1(image).flatten(2).permute(0, 2, 1)
        cls = model.class_embedding + torch.zeros(1, 1, 32)
        tokens = model.ln_pre(torch.cat([cls, tokens], dim=1) + model.positional_embedding)
        _, tokens = model.transformer(tokens.permute(1, 0, 2), x_raw_img=image)
        expected = model.ln_post(tokens.permute(1, 0, 2)) @ model.proj
        actual = model(image, return_full=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_non_patch_aligned_input_is_rejected_instead_of_losing_edges():
    with pytest.raises(ValueError, match="patch"):
        visual()(torch.randn(1, 3, 512, 512))


def test_srm_matches_336_patch_grid_and_keeps_224_default():
    model = SpatialResidualBlock(out_dim=128).eval()
    image = torch.randn(1, 3, 336, 336)
    assert model(image, grid_size=(24, 24)).shape == (576, 1, 128)
    assert model(image).shape == (256, 1, 128)


def test_336_adapter_combines_srm_wavelets_and_gating_with_backward():
    args = SimpleNamespace(use_srm=True, use_gating=True)
    model = ForgeryAwareAdapterLayer(d_model=128, args=args).eval()
    model.freq_scale.data.fill_(0.1)
    tokens = torch.randn(577, 1, 128, requires_grad=True)
    output = model(tokens, x_raw_img=torch.randn(1, 3, 336, 336))
    assert output.shape == tokens.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.srm_block.proj[-1].weight.grad is not None
    assert model.freq_scale.grad is not None and torch.isfinite(tokens.grad).all()


def test_inference_transform_accepts_336():
    image = Image.new("RGB", (640, 480))
    assert get_default_transform(336)(image).shape == (3, 336, 336)
    assert get_default_transform()(image).shape == (3, 224, 224)


def test_strict_dataset_uses_declared_resolution(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (640, 480)).save(path)
    records = [dict(path=str(path), label=0)]
    dataset = StrictImageDataset(records, "q30", input_resolution=336)
    image, label, index = dataset[0]
    assert image.shape == (3, 336, 336) and label == index == 0
