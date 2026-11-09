# CELL 5 THAY THE: chay sau cell_4_extract_fixed.py trong CUNG phien Colab.
import json
import subprocess
import sys
from pathlib import Path

CALIB_OUT_DIR = REPO_DIR / "DATASET/logs/phase7/calibration"
CALIB_OUT_DIR.mkdir(parents=True, exist_ok=True)
cmd_calib = [
    sys.executable, "-u", "-m", "tools.calibrate_and_select",
    "--logits_npz", str(OUT_LOGITS_NPZ),
    "--model_name", "anchored_v2", "--checkpoint", str(target_ckpt),
    "--output_dir", str(CALIB_OUT_DIR), "--fpr_target", "0.05",
    "--calibration_split", str(calib_split_file),
    "--evaluation_split", str(eval_split_file),
    "--probability_device", "cpu",
]
subprocess.run(cmd_calib, cwd=REPO_DIR, check=True)
anchored_calib_json = CALIB_OUT_DIR / "anchored_v2.json"
calib_res = json.loads(anchored_calib_json.read_text(encoding="utf-8"))
assert calib_res["evidence_level"] == "measured"
assert calib_res["checkpoint"]["sha256"] == sha256_file(target_ckpt)
assert calib_res["calibration_source_sha256"] == sha256_file(OUT_LOGITS_NPZ)
assert calib_res["partition_validation"] == partition_check
print("SUCCESS:", anchored_calib_json)
print("T:", calib_res["T"], "T*:", calib_res["T_star_default"])
print("FPR on calibration:", calib_res["fpr_measured"])
print("Calibration FPR is not a guarantee of FPR on independent evaluation data.")
print("Next: continue with notebook cell 6 (packaging).")
