import os
import sys
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.wise_ft_merge import verify_checkpoint


class WiseFtVerifyTests(unittest.TestCase):
    def test_default_namespace_and_integer_counts_are_supported(self):
        def fake_load(_checkpoint, model_args):
            self.assertIsNotNone(model_args)
            self.assertEqual(model_args.backbone, "CLIP:ViT-L/14")
            return object(), {
                "missing": 0,
                "unexpected": 0,
                "missing_keys": [],
                "unexpected_keys": [],
                "layout": {"n": 3, "indices": [7, 15, 23], "partial": False},
            }

        with patch("src.models.canonical_loader.load_canonical", side_effect=fake_load):
            report = verify_checkpoint("candidate.pth")

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["missing"], 0)
        self.assertEqual(report["unexpected"], 0)

    def test_verify_failure_raises_runtime_error_not_process_exit(self):
        report = {
            "missing": 1,
            "unexpected": 0,
            "missing_keys": ["head.weight"],
            "unexpected_keys": [],
            "layout": {"n": 3, "indices": [7, 15, 23], "partial": False},
        }
        with patch("src.models.canonical_loader.load_canonical", return_value=(object(), report)):
            with self.assertRaisesRegex(RuntimeError, "missing=1"):
                verify_checkpoint("candidate.pth")


if __name__ == "__main__":
    unittest.main()
