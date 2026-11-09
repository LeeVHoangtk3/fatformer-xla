"""Fail-closed inputs, bundle verification and datasets for the corrected Colab run."""
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, features
from torch.utils.data import Dataset
from torchvision import transforms

from tools.calibration_data import identities, read_partition, sample_id, sha256_file, validate_partitions
from tools.extract_calibration_logits import find_calib_samples
from tools.package_unified_fatformer import validate_calibration_against_checkpoint
from tools.calibrate_and_select import calibration_probabilities
from tools.eval_locked_benchmark import ALL_18_SUBSETS, locate_subset_dir, scan_subset_images

CONTRACT = "phase7-corrected-20261109-v2"
TRAINING_SHA = "5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c"
CONDITIONS = ("clean", "q30", "q50", "q75", "webp40", "blur15", "downup05")
INDIST_COUNTS = dict(zip(ALL_18_SUBSETS, [8000,11982,15976,4000,2642,3998,10000,5405] + [2000]*10))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != CONTRACT:
        raise ValueError("Old or unsupported Colab bundle")
    required = {"tools/run_phase7_corrected.py", "tools/phase7_corrected.py",
                "tools/calibration_data.py", "tools/infer_unified.py",
                "tools/package_unified_fatformer.py", "src/models/unified_fatformer.py",
                "inputs/calibration_400.npz", "inputs/calibration_split.npz",
                "inputs/evaluation_split.npz", "inputs/anchored_v2.json"}
    if not required.issubset(manifest.get("files", {})):
        raise ValueError("Incomplete code/input manifest")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Bundle file missing/changed: {name}")
    return {"contract": CONTRACT, "sha256": sha256_file(root / "BUNDLE_MANIFEST.json"),
            "n_verified_files": len(manifest["files"]), "source": manifest.get("source")}


def validate_inputs(checkpoint, calibration, logits, calibration_split, evaluation_split,
                    expected_checkpoint=TRAINING_SHA, expected_counts=(400,1600)):
    source_sha = sha256_file(checkpoint)
    if source_sha != expected_checkpoint:
        raise ValueError("Wrong training checkpoint; no recursive best.pth fallback is allowed")
    cal = json.loads(Path(calibration).read_text(encoding="utf-8"))
    validate_calibration_against_checkpoint(cal, source_sha)
    with np.load(logits, allow_pickle=False) as z:
        if str(z["split"]) != "calibration" or str(z["evidence_level"]) != "measured":
            raise ValueError("Require measured calibration logits")
        if str(z["checkpoint_sha256"]) != source_sha:
            raise ValueError("Calibration logits belong to a different checkpoint")
        labels = z["targets"].copy()
        values = z["logits"].copy()
        precision = str(z['precision']) if 'precision' in z else 'unrecorded'
        batch_size = int(z['inference_batch_size']) if 'inference_batch_size' in z else None
        partition = validate_partitions(z["sample_ids"], labels, calibration_split, evaluation_split)
    if (partition["n_calibration"], partition["n_evaluation"]) != tuple(expected_counts):
        raise ValueError("Expected exactly 400 calibration / 1600 evaluation samples")
    _, evaluation_labels = read_partition(evaluation_split,'evaluation')
    if (np.bincount(labels,minlength=2).tolist() != [expected_counts[0]//2]*2
            or np.bincount(evaluation_labels,minlength=2).tolist() != [expected_counts[1]//2]*2):
        raise ValueError('Frozen partitions must have the declared balanced class counts')
    if partition != cal["partition_validation"] or sha256_file(logits) != cal["calibration_source_sha256"]:
        raise ValueError("Calibration source/partition fingerprint mismatch")
    if values.shape != (len(labels),2) or not np.isfinite(values).all():
        raise ValueError("Invalid calibration logits")
    scores = calibration_probabilities(torch.tensor(values,dtype=torch.float32),float(cal['T']),
                                       cal.get('probability_device','cpu'),cal.get('probability_batch_size',32))
    actual_fpr = float((scores[labels == 0] >= float(cal["T_star_default"])).mean())
    if actual_fpr > float(cal["fpr_target"]) + 1e-12:
        raise ValueError("Serialized calibration exceeds its declared FPR target")
    if abs(actual_fpr-float(cal['fpr_measured'][f"{cal['fpr_target']:.2f}"])) > 1e-8:
        raise ValueError('Stored calibration FPR disagrees with measured scores')
    return {"checkpoint_sha256": source_sha, "calibration_sha256": sha256_file(calibration),
            "logits_sha256": sha256_file(logits), "partition": partition,
            "fpr_calibration": actual_fpr, "T": cal["T"], "threshold": cal["T_star_default"],
            "inference_precision": precision, "extraction_batch_size": batch_size,
            "probability_device": cal.get('probability_device','cpu'),
            "probability_batch_size": cal.get('probability_batch_size',32)}


def resolve_cfe(root, calibration_split, evaluation_split):
    # Resolve the union strictly, including the calibration files used in the 20-image check.
    calib = find_calib_samples(str(root), str(calibration_split))
    evaluation, labels = read_partition(evaluation_split, "evaluation")
    if set(s[2] for s in calib) & set(evaluation):
        raise ValueError("Calibration/evaluation overlap")
    index = {}
    for path in Path(root).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        try:
            sid = sample_id(path)
        except ValueError:
            continue
        if sid in index:
            raise ValueError(f"Ambiguous CFE identity: {sid}")
        index[sid] = str(path.resolve())
    if set(evaluation) - set(index):
        raise ValueError("Frozen evaluation images are missing")
    records = [{"path": index[sid], "label": int(y), "sample_id": sid,
                "subset": sid.split('/')[0]} for sid,y in zip(evaluation, labels)]
    return calib, records


def resolve_indist(root, expected_counts=None):
    expected_counts = INDIST_COUNTS if expected_counts is None else expected_counts
    records, paths = [], set()
    for subset, expected in expected_counts.items():
        folder = locate_subset_dir(str(root), subset)
        folder = str(Path(folder).resolve()) if folder else None
        items = scan_subset_images(folder) if folder else []
        if len(items) != expected or {int(i['label']) for i in items} != {0,1}:
            raise ValueError(f"Wrong/missing in-dist subset {subset}: {len(items)}/{expected}")
        for item in sorted(items, key=lambda i:i['file_path']):
            path = Path(item['file_path']).resolve()
            if path in paths:
                raise ValueError("Duplicate path across in-dist subsets")
            paths.add(path)
            records.append({"path": str(path), "label": int(item['label']), "subset": subset,
                            "sample_id": subset + '/' + path.relative_to(folder).as_posix()})
    return records


def degrade(image, condition):
    if condition == "clean":
        return image
    if condition.startswith('q') or condition == 'webp40':
        webp = condition == 'webp40'
        if webp and not features.check('webp'):
            raise RuntimeError("Pillow WebP support is required; no clean-image fallback")
        buffer = io.BytesIO()
        image.save(buffer, format='WEBP' if webp else 'JPEG', quality=40 if webp else int(condition[1:]))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert('RGB')
    if condition == 'blur15':
        return image.filter(ImageFilter.GaussianBlur(1.5))
    if condition == 'downup05':
        width,height = image.size
        return image.resize((max(16,width//2),max(16,height//2)),Image.Resampling.BILINEAR).resize(image.size,Image.Resampling.BILINEAR)
    raise ValueError(f"Unsupported condition: {condition}")


class StrictImageDataset(Dataset):
    def __init__(self, records, condition='clean'):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.records, self.condition = records,condition
        self.spatial = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                           transforms.CenterCrop(224)])
        self.normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            [0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711])])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        # Corrupt images must fail; no grey/zero/synthetic replacement.
        with Image.open(record['path']) as image:
            image = self.spatial(image.convert('RGB'))
            tensor = self.normalize(degrade(image, self.condition))
        return tensor, int(record['label']), index
