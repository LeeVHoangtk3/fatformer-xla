# CELL 4 THAY THE: chay sau cell 1-3 trong phien Colab hien tai.
# Copy toan bo tep nay vao MOT code cell. Khong chay lai cell 4 cu.
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(REPO_DIR).resolve()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
from tools.calibration_data import read_partition, validate_partitions


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


LOG_DIR = REPO_DIR / "DATASET/logs/phase7"


def frozen_partition(previous, filename, split):
    candidates = ([Path(previous)] if previous else []) + [
        LOG_DIR / "wise_ft_sweep" / filename,
        LOG_DIR / filename,
        LOG_DIR / "calibration" / filename,
    ]
    for path in candidates:
        if path.is_file():
            # Fail on invalid existing input; do not silently choose another split.
            ids, labels = read_partition(path, split)
            print(f"[{split}] {path} | N={len(ids)}")
            return path.resolve(), ids, labels
    raise FileNotFoundError(f"Missing frozen {split} partition. Checked: {candidates}")


calib_split_file, calib_ids, calib_labels = frozen_partition(
    globals().get("calib_split_file"),
    "logits_baseline_clean_calib_split.npz", "calibration",
)
eval_split_file, eval_ids, eval_labels = frozen_partition(
    globals().get("eval_split_file"),
    "logits_baseline_clean_eval_split.npz", "evaluation",
)
assert len(calib_ids) == 400 and np.bincount(calib_labels, minlength=2).tolist() == [200, 200]
assert len(eval_ids) == 1600 and np.bincount(eval_labels, minlength=2).tolist() == [800, 800]
partition_check = validate_partitions(calib_ids, calib_labels, calib_split_file, eval_split_file)
print("Partition check:", partition_check)

assert calib_data_dir and Path(calib_data_dir).is_dir(), "Missing calibration image folder; check cell 3."
assert target_ckpt and Path(target_ckpt).is_file(), "Missing checkpoint; check cell 3."
assert torch.cuda.is_available(), "Select a Colab GPU runtime, then rerun cells 1-3."
ckpt_hash = sha256_file(target_ckpt)
assert ckpt_hash == "5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c", "Wrong Anchored v2 checkpoint."
print("GPU:", torch.cuda.get_device_name(0))
print("Images:", calib_data_dir)

# Reduce to 8 or 4 if this GPU runs out of VRAM; the measured batch is recorded.
EXTRACT_BATCH_SIZE = 32
OUT_LOGITS_NPZ = LOG_DIR / "logits_anchored_v2_clean_calib.npz"
OUT_LOGITS_NPZ.parent.mkdir(parents=True, exist_ok=True)
# Validate a fresh temporary output before replacing any earlier output.
pending_logits = OUT_LOGITS_NPZ.with_name(OUT_LOGITS_NPZ.stem + ".pending.npz")
cmd_extract = [
    sys.executable, "-u", "-m", "tools.extract_calibration_logits",
    "--checkpoint", str(target_ckpt), "--output_npz", str(pending_logits),
    "--data_dir", str(calib_data_dir), "--split_file", str(calib_split_file),
    "--evaluation_split", str(eval_split_file),
    "--device", "cuda", "--batch_size", str(EXTRACT_BATCH_SIZE), "--precision", "fp32",
]
if globals().get("clip_path") and Path(clip_path).is_file():
    cmd_extract.extend(["--clip_path", str(clip_path)])
subprocess.run(cmd_extract, cwd=REPO_DIR, check=True)

with np.load(pending_logits, allow_pickle=False) as payload:
    assert payload["logits"].shape == (400, 2) and np.isfinite(payload["logits"]).all()
    assert str(payload["split"]) == "calibration"
    assert str(payload["evidence_level"]) == "measured"
    assert str(payload["checkpoint_sha256"]) == ckpt_hash
    assert str(payload["manifest_sha256"]) == sha256_file(calib_split_file)
    validate_partitions(payload["sample_ids"], payload["targets"], calib_split_file, eval_split_file)
pending_logits.replace(OUT_LOGITS_NPZ)
print("SUCCESS: 400 measured logits (200 real + 200 fake):", OUT_LOGITS_NPZ)
print("SHA256:", sha256_file(OUT_LOGITS_NPZ))
print("Next: run cell_5_calibrate_fixed.py; do not run the old cell 5 unchanged.")
