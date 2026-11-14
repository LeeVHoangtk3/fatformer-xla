import io

import numpy as np
import pytest
import torch
from PIL import Image

import demo


@pytest.mark.parametrize("size,mode", [((113, 370), "RGB"), ((401, 95), "RGBA"), ((40, 35), "L")])
def test_uploaded_image_matches_canonical_transform(tmp_path, size, mode):
    path = tmp_path / "input.png"
    Image.new(mode, size, 80).save(path)
    image = demo.read_image(path)
    assert image.mode == "RGB"
    assert torch.equal(demo.NORMALIZE(demo.SPATIAL(image)), demo.TRANSFORM(image))


def test_q30_applies_after_crop(tmp_path):
    rng = np.random.default_rng(10)
    image = Image.fromarray(rng.integers(0, 256, (330, 490, 3), dtype=np.uint8))
    crop = demo.SPATIAL(image)
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=30)
    buffer.seek(0)
    expected = np.asarray(Image.open(buffer).convert("RGB"))
    assert np.array_equal(np.asarray(demo.jpeg_q30(crop)), expected)
    assert not np.array_equal(expected, np.asarray(demo.SPATIAL(demo.jpeg_q30(image))))


def test_reject_invalid_and_oversized_input(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="chọn một ảnh"):
        demo.read_image(None)
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Không giải mã"):
        demo.read_image(path)
    Image.new("RGB", (20, 20)).save(path)
    monkeypatch.setattr(demo, "MAX_PIXELS", 399)
    with pytest.raises(ValueError, match="pixel"):
        demo.read_image(path)
    monkeypatch.setattr(demo, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="32 MiB"):
        demo.read_image(path)


def test_exif_geometry_is_preserved_and_animation_rejected(tmp_path):
    path = tmp_path / "exif.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (80, 40)).save(path, exif=exif)
    assert demo.read_image(path).size == (80, 40)
    path = tmp_path / "animated.webp"
    Image.new("RGB", (20, 20), "red").save(
        path, save_all=True, append_images=[Image.new("RGB", (20, 20), "blue")], duration=100)
    with pytest.raises(ValueError, match="ảnh tĩnh"):
        demo.read_image(path)


def test_threshold_is_not_half_and_bad_outputs_are_rejected(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (240, 240)).save(path)

    class FixedModel:
        label = 0

        def predict(self, x):
            return dict(prob_fake=torch.tensor([0.7]), label=torch.tensor([self.label]),
                        threshold=torch.tensor([demo.THRESHOLD]))

    model = FixedModel()
    _, _, result = demo.analyze(model, path)
    assert result["clean"]["label"] == "REAL"
    assert result["source_label"] is None
    assert "Dự đoán REAL" in demo.result_html(result)
    model.label = 1
    with pytest.raises(ValueError, match="Output model"):
        demo.analyze(model, path)


def test_wrong_checkpoint_fails_before_loading(tmp_path):
    path = tmp_path / "unified_fatformer_v2.pth"
    path.write_bytes(b"wrong model with same name")
    with pytest.raises(ValueError, match="SHA-256"):
        demo.load_demo_model(path)


@pytest.mark.parametrize("probability", [0.7, demo.THRESHOLD, 0.95])
def test_decision_explains_the_selected_threshold(probability):
    label = "FAKE" if probability >= demo.THRESHOLD else "REAL"
    result = dict(clean=dict(prob_fake=probability, label=label, threshold=demo.THRESHOLD))
    content = demo.result_html(result)
    assert f"Dự đoán {label}" in content
    assert "Vì sao có quyết định này?" in content
    assert "79.185086%" in content
    assert f'{abs(probability-demo.THRESHOLD)*100:.6f} điểm phần trăm' in content
    assert "không phải mức độ chắc chắn" in content


def test_comparison_explains_label_flip_without_changing_decision():
    result = dict(clean=dict(prob_fake=0.7, label="REAL", threshold=demo.THRESHOLD),
                  q30=dict(prob_fake=0.9, label="FAKE", threshold=demo.THRESHOLD))
    content = demo.result_html(result)
    assert "REAL → FAKE" in content and "+20.0000 điểm phần trăm" in content
    assert "không xác nhận ảnh đã bị làm giả" in content


def test_occlusion_measures_signed_changes_in_the_correct_crop_regions():
    pixels = np.full((224, 224, 3), 128, dtype=np.uint8)
    pixels[:56, :56] = 255
    pixels[56:112, 56:112] = 0
    crop = Image.fromarray(pixels)
    original = demo.NORMALIZE(crop).unsqueeze(0)

    class TwoRegionModel:
        def predict(self, x):
            assert not torch.is_grad_enabled()
            return dict(prob_fake=torch.sigmoid(x[:, 0, :56, :56].mean((1, 2))
                                                + x[:, 0, 56:112, 56:112].mean((1, 2))))

    model = TwoRegionModel()
    with torch.inference_mode():
        baseline = float(model.predict(original)["prob_fake"][0])
    overlay, report = demo.occlusion_explanation(model, crop, baseline)
    assert report["n_forwards"] == len(report["regions"]) == 16
    assert report["localization_validated"] is False
    assert overlay.size == (224, 224)
    for i, region in enumerate(report["regions"]):
        assert region["box_xyxy"] == [(i % 4)*56, (i // 4)*56, (i % 4+1)*56, (i // 4+1)*56]
        if i not in (0, 5):
            assert region["delta_fake_pp"] == pytest.approx(0)
    assert report["regions"][0]["delta_fake_pp"] > 0
    assert report["regions"][5]["delta_fake_pp"] < 0
    assert report["regions"][0]["masked_prob_fake"] == pytest.approx(
        float(torch.sigmoid(original[:, 0, 56:112, 56:112].mean())))
    assert torch.equal(demo.NORMALIZE(crop).unsqueeze(0), original)
    assert "không phải xác suất vùng giả" in demo.explanation_markdown(report)
    assert "ô **1**" in demo.explanation_markdown(report)


def test_flat_explanation_does_not_invent_a_suspicious_region():
    class ConstantModel:
        def predict(self, x):
            return dict(prob_fake=torch.tensor([0.5]))

    _, report = demo.occlusion_explanation(ConstantModel(), Image.new("RGB", (224, 224)), 0.5)
    assert all(region["delta_fake_pp"] == 0 for region in report["regions"])
    assert "Không ô nào" in demo.explanation_markdown(report)
