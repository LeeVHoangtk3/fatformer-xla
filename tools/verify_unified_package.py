"""Check tensor preservation and prediction equivalence; never substitutes for G7.3 quality/T4 checks."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.models.canonical_loader import extract_state_dict
from tools.calibration_data import sha256_file
from tools.infer_unified import load_unified_model, collect_image_paths, get_default_transform


def verify(checkpoint, reference, output, images=None, device="cpu", synthetic_smoke=False, precision="fp32"):
    torch.set_num_threads(2)
    if precision=='fp16' and torch.device(device).type!='cuda':
        raise ValueError('FP16 verification requires CUDA')
    if torch.device(device).type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    a = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    b = torch.load(reference, map_location="cpu", weights_only=True, mmap=True)
    actual, expected = extract_state_dict(a), extract_state_dict(b)
    if set(actual) != set(expected):
        raise ValueError("Reference/package tensor keys differ")
    for key in actual:
        if actual[key].dtype != expected[key].dtype or not torch.equal(actual[key], expected[key]):
            raise ValueError(f"Tensor changed: {key}")
    report = dict(schema="unified_package_verification/v1", checkpoint_sha256=sha256_file(checkpoint),
                  reference_sha256=sha256_file(reference), n_tensors=len(actual), tensors_equal=True,
                  prediction_status="pending", real_image_acceptance=False,
                  g73_status="pending_condition_comparison_and_GPU_latency")
    # Persist structural evidence even if a later real-model smoke test fails.
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if synthetic_smoke or images:
        paths = collect_image_paths(images or [])
        if not synthetic_smoke and len(paths) < 20:
            raise ValueError("Real prediction acceptance requires at least 20 images")
        del a, b, actual, expected
        model, _ = load_unified_model(checkpoint, clip_path="NON_EXISTENT_CLIP.pt", device=device)
        transform = get_default_transform()
        inputs = paths if paths else [None]
        max_error = 0.0
        with torch.inference_mode():
            for path in inputs:
                if path:
                    with Image.open(path) as image:
                        x = transform(image.convert("RGB")).unsqueeze(0).to(device)
                else:
                    x = torch.zeros(1, 3, 224, 224, device=device)
                with torch.amp.autocast(torch.device(device).type,dtype=torch.float16,
                                        enabled=torch.device(device).type=='cuda' and precision=='fp16'):
                    logits = model.backbone(x).float()
                    outputs = model.predict(x)
                if model.router_enabled:
                    raise ValueError('Fixed calibration equivalence does not validate an active router')
                temperature = model.clean_temperature if model.fixed_branch == 'clean' else model.forensic_temperature
                expected_p = torch.softmax(logits / temperature, dim=-1)[:, 1]
                torch.testing.assert_close(outputs["prob_fake"], expected_p, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(outputs["label"], (expected_p >= model.fixed_threshold).long())
                max_error = max(max_error, float((outputs["prob_fake"] - expected_p).abs().max()))
        report.update(prediction_status="passed_synthetic_smoke" if not paths else "passed_real_images",
                      n_images=len(inputs), input_kind="synthetic" if not paths else "real",
                      max_probability_error=max_error, device=device,
                      real_image_acceptance=bool(paths and len(paths) >= 20),
                      precision=precision, standalone_clip_path="NON_EXISTENT_CLIP.pt")
        Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("checkpoint", "reference", "output"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--images", nargs="*")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic_smoke", action="store_true")
    parser.add_argument("--precision", choices=['fp32','fp16'], default='fp32')
    print(json.dumps(verify(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
