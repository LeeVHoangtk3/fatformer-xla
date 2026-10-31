import json
import os
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.rebaseline_cfe_matrix import load_measured_predictions
from tools.calibrate_and_select import run_measured_calibration
from tools.run_phase7_acceptance import (
    split_calibration_and_eval_logits,
    write_candidate_eval_sidecar_from_splits,
)
from tools.sweep_wise_ft_candidates import evaluate_mock_or_real_candidate


class Phase7AcceptanceTests(unittest.TestCase):
    def _write_logits(self, path: str, *, split: str = "") -> None:
        targets = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)
        logits = np.stack([1.0 - targets, targets], axis=1).astype(np.float32)
        payload = {
            "logits": logits,
            "targets": targets,
            "sample_ids": np.asarray([f"sample-{idx:02d}" for idx in range(20)]),
        }
        if split:
            payload["split"] = split
        np.savez_compressed(path, **payload)

    def test_stratified_split_is_disjoint_and_preserves_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "logits_model_clean.npz")
            self._write_logits(source)
            calibration, evaluation = split_calibration_and_eval_logits(source, calib_ratio=0.2, seed=42)

            with np.load(calibration, allow_pickle=False) as calib, np.load(evaluation, allow_pickle=False) as eval_data:
                self.assertEqual(set(calib["targets"].tolist()), {0, 1})
                self.assertEqual(set(eval_data["targets"].tolist()), {0, 1})
                self.assertTrue(set(calib["sample_ids"]).isdisjoint(set(eval_data["sample_ids"])))
                self.assertEqual(len(calib["sample_ids"]) + len(eval_data["sample_ids"]), 20)
                self.assertEqual(str(eval_data["split"]), "evaluation")

    def test_eval_split_never_falls_back_to_full_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_logits(os.path.join(tmp, "logits_baseline_clean.npz"))
            with self.assertRaises(FileNotFoundError):
                load_measured_predictions(tmp, "baseline", "clean", temperature=1.0, use_eval_split=True)

    def test_calibration_rejects_non_calibration_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluation = os.path.join(tmp, "eval_split.npz")
            self._write_logits(evaluation, split="evaluation")
            with self.assertRaisesRegex(ValueError, "split=calibration"):
                run_measured_calibration(evaluation, tmp, "baseline")

    def test_candidate_sidecar_is_evaluation_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean = os.path.join(tmp, "clean_eval.npz")
            q30 = os.path.join(tmp, "q30_eval.npz")
            self._write_logits(clean, split="evaluation")
            self._write_logits(q30, split="evaluation")
            calibration = os.path.join(tmp, "baseline.json")
            with open(calibration, "w", encoding="utf-8") as handle:
                json.dump({
                    "evidence_level": "measured",
                    "model": "baseline",
                    "T": 1.0,
                    "T_star_default": 0.5,
                }, handle)
            checkpoint = os.path.join(tmp, "wise_ft_v2_a35_b25.pth")
            sidecar = checkpoint.replace(".pth", "_eval.json")
            write_candidate_eval_sidecar_from_splits(
                clean, q30, calibration, sidecar, checkpoint, reference_eval_npz=clean
            )

            with open(sidecar, "r", encoding="utf-8") as handle:
                result = json.load(handle)
            self.assertEqual(result["schema"], "candidate_eval/v2")
            self.assertEqual(result["split"], "evaluation")
            self.assertEqual(result["n_samples"], 20)
            measured = evaluate_mock_or_real_candidate(checkpoint, is_dummy=False)
            self.assertEqual(set(measured), {
                "real_acc_clean", "fake_at_real90_clean", "auc_clean",
                "real_acc_q30", "fake_at_real90_q30", "auc_q30",
            })


if __name__ == "__main__":
    unittest.main()
