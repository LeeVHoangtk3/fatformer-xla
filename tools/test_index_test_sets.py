"""Regression tests for the actual rel_path schema of the CFE preparation notebook."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from PIL import Image

from index_test_sets import attach_manifest_metadata, infer_label, infer_test_set, SUPPLEMENTAL_TEST_SETS
from build_generator_denylist import EXPECTED_TASK51_TESTS
from hash_utils import load_hash_index
from phase7_reference_io import run_logged, index_reference

SCRIPT = Path(__file__).with_name("index_test_sets.py")


class ManifestMappingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cfe_schema_")
        self.path = Path(self.tmp.name) / "cfe_manifest.json"
        self.addCleanup(self.tmp.cleanup)

    def write(self, rows):
        self.path.write_text(json.dumps(rows), encoding="utf-8")

    def test_native_rel_path_and_archive_prefix_preserve_png_hash(self):
        self.write([{"rel_path": "1_commercial/1_fake/fake_0000_MidjourneyV5_2.png",
                     "label": 1, "model_name": "MidjourneyV5_2", "architecture": "Commercial",
                     "sha256": "a" * 64}])
        rows = [{"path": "test.tar::./test_communityforensics/1_commercial/1_fake/fake_0000_MidjourneyV5_2.png",
                 "label": 1, "sha256": "b" * 64, "phash": 42}]
        attach_manifest_metadata(rows, str(self.path))
        self.assertEqual(rows[0]["model_name"], "MidjourneyV5_2")
        self.assertEqual(rows[0]["sha256"], "b" * 64)
        self.assertEqual(rows[0]["source_sha256"], "a" * 64)
        self.assertEqual(rows[0]["phash"], 42)

    def test_duplicate_basenames_resolve_by_group_and_windows_separators(self):
        self.write({"records": [{"rel_path": r"group_a\1_fake\same.png", "model_name": "a", "label": 1},
                                {"rel_path": "group_b/1_fake/same.png", "model_name": "b", "label": 1}]})
        rows = [{"path": f"archive::root/{g}/1_fake/same.png", "label": 1} for g in ("group_a", "group_b")]
        attach_manifest_metadata(rows, str(self.path))
        self.assertEqual([r["model_name"] for r in rows], ["a", "b"])

    def test_label_conflict_is_rejected(self):
        self.write([{"rel_path": "group/1_fake/x.png", "label": 0}])
        with self.assertRaisesRegex(ValueError, "label conflicts"):
            attach_manifest_metadata([{"path": "tar::group/1_fake/x.png", "label": 1}], str(self.path))

    def test_unused_or_duplicate_manifest_records_are_rejected(self):
        self.write([{"rel_path": "g/1_fake/x.png"}, {"rel_path": "g/1_fake/y.png"}])
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            attach_manifest_metadata([{"path": "tar::g/1_fake/x.png"}], str(self.path))
        self.write([{"rel_path": "g/1_fake/x.png"}, {"path": "g/1_fake/x.png"}])
        with self.assertRaisesRegex(ValueError, "Duplicate reference"):
            attach_manifest_metadata([], str(self.path))


class CFECommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="cfe_cli_")
        root = Path(cls.tmp)
        cls.tar = root / "test_communityforensics.tar"
        cls.manifest = root / "cfe_test_manifest.json"
        rows = []
        quotas = [("1_commercial", 300), ("2_latent_diffusion", 300),
                  ("3_pixel_diffusion", 200), ("4_modern_gans", 100), ("5_other_autoregressive", 100)]
        with tarfile.open(cls.tar, "w") as archive:
            for group, quota in quotas:
                for label in (0, 1):
                    for i in range(quota):
                        path = f"{group}/{label}_{'fake' if label else 'real'}/image_{i:04d}.png"
                        buf = io.BytesIO()
                        seed = len(rows)
                        Image.new("RGB", (16, 16), (seed % 256, seed // 256, 7)).save(buf, format="PNG")
                        data = buf.getvalue()
                        member = tarfile.TarInfo("test_communityforensics/" + path)
                        member.size = len(data)
                        archive.addfile(member, io.BytesIO(data))
                        rows.append({"rel_path": path, "group": group, "label": label,
                                     "model_name": "TestGenerator" if label else "COCO",
                                     "architecture": group, "real_source": "coco", "sha256": "a" * 64})
            data = json.dumps(rows).encode("utf-8")
            member = tarfile.TarInfo("test_communityforensics/cfe_manifest.json")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        cls.manifest.write_bytes(data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def run_cli(self, workers):
        out = Path(self.tmp) / f"hashes_{workers}.json"
        result = subprocess.run([sys.executable, str(SCRIPT), "--role", "cfe", "--tar", str(self.tar),
                                 "--manifest", str(self.manifest), "--workers", str(workers), "--out", str(out)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(payload["metadata"]["complete"])
        self.assertEqual(len(payload["images"]), 2000)
        self.assertEqual(sum(r["label"] == 1 for r in payload["images"]), 1000)
        self.assertTrue(all(r["model_name"] == "TestGenerator" for r in payload["images"] if r["label"] == 1))
        self.assertTrue(all(r["sha256"] != r["source_sha256"] for r in payload["images"]))
        return payload

    def test_cli_serial_complete_cfe(self):
        self.run_cli(1)

    def test_cli_process_pool_complete_cfe(self):
        self.run_cli(2)


class Test18CommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="test18_cli_")
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def make_archives(self, layout="standard", missing=None, bad_label=False, unknown_set=False, corrupt=False, supplementary=False):
        paths = []
        gan_sets = EXPECTED_TASK51_TESTS[:8] + (list(SUPPLEMENTAL_TEST_SETS) if supplementary else [])
        for filename, sets in (("test_benchmark_gans.tar", gan_sets),
                               ("test_benchmark_diffusion.tar", EXPECTED_TASK51_TESTS[8:])):
            path = self.root / filename
            paths.append(path)
            with tarfile.open(path, "w") as archive:
                for subset in sets:
                    if subset == missing:
                        continue
                    folder = subset
                    labels = ("0_real", "1_fake")
                    if layout == "aliases":
                        folder = "DALL-E" if subset == "dalle" else subset.upper().replace("_", "-")
                        labels = ("nature", "ai") if subset in EXPECTED_TASK51_TESTS[8:] else ("real_images", "fake_images")
                    for label, name in enumerate(labels):
                        group = "unidentified_generator" if unknown_set and subset == "progan" else folder
                        name = "unidentified_label" if bad_label and subset == "progan" and label == 0 else name
                        member_name = f"./{group}/{'airplane/' if subset == 'progan' else ''}{name}/image.png"
                        blob = io.BytesIO()
                        Image.new("RGB", (16, 16), (len(folder), label, 37)).save(blob, format="PNG")
                        data = b"corrupt image" if corrupt and subset == "progan" and label == 0 else blob.getvalue()
                        member = tarfile.TarInfo(member_name)
                        member.size = len(data)
                        archive.addfile(member, io.BytesIO(data))
                # OS metadata is not a test image, even if its filename ends in PNG.
                member = tarfile.TarInfo("__MACOSX/._image.png")
                member.size = 8
                archive.addfile(member, io.BytesIO(b"metadata"))
        return paths

    def run_cli(self, paths, workers=1, extra=()):
        out = self.root / "hashes_18sets.json.gz"
        result = subprocess.run([sys.executable, str(SCRIPT), "--role", "test18", "--tar", *map(str, paths),
                                 "--workers", str(workers), "--out", str(out), *extra],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        return result, out

    def test_standard_multiclass_layout_and_gzip_consumer(self):
        result, out = self.run_cli(self.make_archives())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = load_hash_index(str(out))
        self.assertTrue(payload["metadata"]["complete"])
        self.assertEqual(len(payload["images"]), 36)
        self.assertEqual(set(payload["metadata"]["test_sets"]), set(EXPECTED_TASK51_TESTS))
        from audit_manifest_v2 import RefIndex
        self.assertEqual(RefIndex("test18", str(out)).problems(), [])

    def test_ai_nature_and_format_aliases_with_four_workers(self):
        result, out = self.run_cli(self.make_archives("aliases"), workers=4)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = load_hash_index(str(out))["images"]
        self.assertEqual(sum(r["label"] == 0 for r in rows), 18)
        self.assertEqual(sum(r["label"] == 1 for r in rows), 18)
        self.assertEqual({r["model_name"] for r in rows if r["label"] == 1}, set(EXPECTED_TASK51_TESTS))

    def test_raw_archive_with_thirteen_gan_groups_preserves_all_images(self):
        result, out = self.run_cli(self.make_archives(supplementary=True), workers=4)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = load_hash_index(str(out))
        self.assertTrue(payload["metadata"]["complete"])
        self.assertEqual(len(payload["images"]), 46)
        self.assertEqual(payload["metadata"]["n_core_images"], 36)
        self.assertEqual(payload["metadata"]["n_supplemental_images"], 10)
        self.assertEqual(set(payload["metadata"]["test_sets"]), set(EXPECTED_TASK51_TESTS))
        self.assertEqual(set(payload["metadata"]["supplemental_sets"]), set(SUPPLEMENTAL_TEST_SETS))
        from audit_manifest_v2 import RefIndex, check_generators
        ref = RefIndex("test18", str(out))
        self.assertEqual(ref.problems(), [])
        for name in SUPPLEMENTAL_TEST_SETS:
            self.assertIn(name, ref.generators)
        overlap = check_generators([{"generator": "crn", "label": 1}], [ref], None)
        self.assertGreater(overlap["violations"], 0)
        from build_manifest_v2 import filter_reference_generators
        rows = [{"generator": "crn", "label": 1}, {"generator": "crn", "label": 0},
                {"generator": "GPT-4o", "label": 1}]
        kept, stats = filter_reference_generators(rows, {"test18": payload})
        self.assertEqual(kept, rows[1:])
        self.assertEqual(stats["reference_generator_overlap"], 1)

    def test_unknown_label_fails_before_hashing_and_reports_the_path(self):
        result, out = self.run_cli(self.make_archives(bad_label=True))
        self.assertEqual(result.returncode, 2)
        self.assertIn("hashing has not started", result.stderr)
        layout = load_hash_index(str(out) + ".layout.json")
        self.assertIn("unidentified_label", layout["unknown_examples"][0]["path"])
        self.assertFalse(out.exists())

    def test_unknown_generator_and_missing_set_do_not_pass(self):
        result, out = self.run_cli(self.make_archives(unknown_set=True, missing="dalle"))
        self.assertEqual(result.returncode, 2)
        layout = load_hash_index(str(out) + ".layout.json")
        self.assertIn("dalle", layout["missing_test_sets"])
        self.assertTrue(layout["unknown_examples"])
        self.assertFalse(out.exists())

    def test_inspection_only_does_not_hash_or_write_index(self):
        result, out = self.run_cli(self.make_archives(), extra=("--inspect-only",))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(out.exists())
        self.assertTrue(load_hash_index(str(out) + ".layout.json")["complete"])

    def test_unreadable_image_is_not_silently_dropped(self):
        result, out = self.run_cli(self.make_archives(corrupt=True))
        self.assertEqual(result.returncode, 1)
        self.assertFalse(load_hash_index(str(out))["metadata"]["complete"])
        self.assertEqual(load_hash_index(str(out))["metadata"]["n_skipped_unreadable"], 1)

    def test_logged_failure_includes_original_stderr(self):
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "specific_unknown_label"):
                run_logged([sys.executable, "-c", "import sys; print('specific_unknown_label', file=sys.stderr); sys.exit(2)"], self.root)
        logs = list((self.root / "DATASET/logs/phase7/commands").glob("*.log"))
        self.assertEqual(len(logs), 1)
        self.assertIn("specific_unknown_label", logs[0].read_text(encoding="utf-8"))

    def test_index_cache_reuses_only_matching_complete_evidence(self):
        (self.root / "tools").mkdir()
        for name in ("index_test_sets.py", "hash_utils.py", "build_generator_denylist.py"):
            shutil.copy2(SCRIPT.parent / name, self.root / "tools" / name)
        paths = self.make_archives()
        out = self.root / "cached.json.gz"
        with redirect_stdout(io.StringIO()):
            index_reference(self.root, "test18", paths, out, workers=1)
        with patch("phase7_reference_io.run_logged", side_effect=AssertionError("must reuse cache")):
            index_reference(self.root, "test18", paths, out, workers=1)
        marker = Path(str(out) + ".inputs.json")
        cached = json.loads(marker.read_text())
        cached["index_sha256"] = "changed"
        marker.write_text(json.dumps(cached))
        with patch("phase7_reference_io.run_logged", side_effect=RuntimeError("must rebuild")):
            with self.assertRaisesRegex(RuntimeError, "must rebuild"):
                index_reference(self.root, "test18", paths, out, workers=1)

    def test_unrecognized_words_do_not_become_generators_or_labels(self):
        self.assertIsNone(infer_test_set("tar::not_progan/1_fake/image.png"))
        self.assertIsNone(infer_label("tar::progan/not_fake/image.png"))

    def test_actual_notebook_step4_and_retry_with_all_three_archives(self):
        import ast
        import yaml
        repo = SCRIPT.parent.parent
        project = self.root / "project"
        (project / "tools").mkdir(parents=True)
        for name in ("index_test_sets.py", "hash_utils.py", "build_generator_denylist.py"):
            shutil.copy2(repo / "tools" / name, project / "tools" / name)
        report_rel = Path("docs/bao_cao/C/bao_cao_task_5.1_full_benchmark_18_datasets.md")
        (project / report_rel).parent.mkdir(parents=True)
        shutil.copy2(repo / report_rel, project / report_rel)
        cfg = yaml.safe_load((repo / "configs/phase7_data_v2.yaml").read_text(encoding="utf-8"))
        cfg["paths"]["denylist"] = str(project / "DATASET/logs/phase7/generator_denylist.json")
        cfg["replay"]["manifest"] = str(repo / "DATASET/manifests/data_manifest.json")
        config_path = project / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        self.make_archives("aliases", supplementary=True)
        CFECommandTests.setUpClass()
        try:
            shutil.copy2(CFECommandTests.tar, self.root / "test_communityforensics.tar")
        finally:
            CFECommandTests.tearDownClass()
        with tarfile.open(self.root / "diffusion_staging.tar", "w"):
            pass
        notebook = json.loads((repo / "notebooks/task/task_7.7/task_7.7_build_data_v2_colab.ipynb").read_text(encoding="utf-8"))
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if c["cell_type"] == "code" and "TAR_CFE_TEST =" in "".join(c["source"]))

        class LocalFixturePaths(ast.NodeTransformer):
            def visit_Constant(inner, node):
                if node.value == "/content/drive/MyDrive/Fatformer/datasets":
                    return ast.copy_location(ast.Constant(str(self.root)), node)
                return node

        tree = LocalFixturePaths().visit(ast.parse(source))
        code = compile(ast.fix_missing_locations(tree), "notebook_step4", "exec")
        env = {"PROJECT_ROOT": project, "CONFIG": str(config_path), "Path": Path, "sys": sys, "json": json}
        previous = Path.cwd()
        output = io.StringIO()
        try:
            os.chdir(project)
            with redirect_stdout(output):
                exec(code, env)
                exec(code, env)  # a retry must reuse both complete indexes
        finally:
            os.chdir(previous)
        log_root = project / "DATASET/logs/phase7"
        self.assertEqual(len(load_hash_index(str(log_root / "hashes_cfe_test.json"))["images"]), 2000)
        self.assertEqual(len(load_hash_index(str(log_root / "hashes_18sets.json.gz"))["images"]), 46)
        self.assertIn("Reusing complete cfe index", output.getvalue())
        self.assertIn("Reusing complete test18 index", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
