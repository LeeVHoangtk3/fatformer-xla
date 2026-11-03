"""
Kiem thu tu dong cho kho du lieu v2 (Task 7.7) - KHONG can mang.

Chay:
    python tools/test_data_v2.py -v

No dung toan bo pipeline (plan -> fetch -> replay -> select -> split -> finalize
-> audit) tren mot kho *gia lap* nho, trong do:
    - Co san mot cap anh trung byte-exact giua 2 shard  -> phai bi khu trung lap.
    - Co san anh trung byte-exact + anh "gan trung" (nen lai JPEG) voi "tap test"
      -> phai bi loc truoc khi chon.
    - Co san replay 40 mau                              -> phai duoc giu du.
    - Chay lai fetch voi --resume                       -> khong duoc nhan doi.
    - Audit duoc chay lai tren manifest bi lam ban (am tinh) -> phai FAIL.

Luu y: config trong test co dung sai RONG hon config that (real_frac, replay_frac,
chenh lech ho). Voi N=208 thi ty le tung split bi chi phoi boi lam tron, nen test
nay chi chung minh *co che*; cac nguong DoD that (N=15.000) duoc kiem o
configs/phase7_data_v2.yaml va duoc audit_manifest_v2 tinh lai doc lap.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
for p in (PROJECT_ROOT, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_manifest_v2 as B  # noqa: E402
import audit_manifest_v2 as A  # noqa: E402
from build_generator_denylist import DenyMatcher  # noqa: E402
from hash_utils import phash_from_pil, sha256_bytes  # noqa: E402

REQUIRED_FIELDS = ("path", "sha256", "phash", "generator", "family", "split", "label",
                   "source", "width", "height")


# ---------------------------------------------------------------------------
def make_png(seed: int, size: int = 256) -> bytes:
    """Anh tong hop co cau truc tan thap (on dinh voi JPEG) va khac nhau theo seed."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    arr = np.kron(blocks, np.ones((size // 8, size // 8, 1), dtype=np.uint8))
    noise = rng.integers(0, 6, size=(size, size, 3), dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    arr[: size // 4, : size // 4] = (seed * 37) % 256
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def recompress_jpeg(blob: bytes, quality: int = 88) -> bytes:
    out = io.BytesIO()
    with Image.open(io.BytesIO(blob)) as im:
        im.convert("RGB").save(out, format="JPEG", quality=quality)
    return out.getvalue()


def write_cfe_shard(path: str, rows) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "image_name": pa.array([r["image_name"] for r in rows], pa.string()),
        "format": pa.array([r.get("format", "png") for r in rows], pa.string()),
        "resolution": pa.array([[r.get("height", 256), r.get("width", 256)] for r in rows],
                               pa.list_(pa.int64())),
        "mode": pa.array(["RGB"] * len(rows), pa.string()),
        "image_data": pa.array([r["blob"] for r in rows], pa.binary()),
        "model_name": pa.array([r["model_name"] for r in rows], pa.string()),
        "nsfw_flag": pa.array([bool(r.get("nsfw", False)) for r in rows], pa.bool_()),
        "prompt": pa.array(["p"] * len(rows), pa.string()),
        "real_source": pa.array([r.get("real_source", "N/A") for r in rows], pa.string()),
        "subset": pa.array(["Manual"] * len(rows), pa.string()),
        "split": pa.array(["train"] * len(rows), pa.string()),
        "label": pa.array([int(r["label"]) for r in rows], pa.int64()),
        "architecture": pa.array([r["architecture"] for r in rows], pa.string()),
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)


def write_plain_shard(path: str, blobs) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "text": pa.array([f"prompt {i}" for i in range(len(blobs))], pa.string()),
        "image": pa.array([{"bytes": b, "path": None} for b in blobs],
                          pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
class DataV2PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="fatformer_v2_")
        cls.cache = os.path.join(cls.tmp, "cache")
        cls.replay_root = os.path.join(cls.tmp, "replay_src")
        os.makedirs(cls.replay_root, exist_ok=True)

        # --- sinh anh -----------------------------------------------------
        cls.blob = {}
        seed = 1000
        for name, n in [("taming", 26), ("gigagan", 8), ("styleswin", 8), ("cips", 5),
                        ("gansformer", 8), ("projectedgan", 8), ("deepfloyd", 20),
                        ("dit", 12), ("sdxl_one", 12), ("gpt4o", 26),
                        ("ffhq", 22), ("coco", 22), ("land", 22), ("vision", 22)]:
            cls.blob[name] = [make_png(seed + i) for i in range(n)]
            seed += 100

        def row(model, label, blob, i, arch, rsrc="N/A", name_suffix=None):
            return {"image_name": f"{model.replace('/', '_')}_{name_suffix or i:05d}.png",
                    "blob": blob, "model_name": model, "label": label,
                    "architecture": arch, "real_source": rsrc}

        # --- shard cfe_small ---------------------------------------------
        shard0 = [row("tamingTransformers", 1, b, i, "Other") for i, b in enumerate(cls.blob["taming"])]
        shard0 += [row("GigaGAN", 1, b, i, "GAN", "coco,imagenet") for i, b in enumerate(cls.blob["gigagan"])]
        shard0 += [row("DeepFloyd", 1, b, i, "PixDiff", "coco") for i, b in enumerate(cls.blob["deepfloyd"])]
        shard0 += [row("DiT", 1, b, i, "LatDiff", "imagenet") for i, b in enumerate(cls.blob["dit"])]
        shard0 += [row("FFHQ", 0, b, i, "Real", "N/A") for i, b in enumerate(cls.blob["ffhq"])]

        shard1 = [row("stablediffusionapi/xl-one", 1, b, i, "LatDiff", "LAION")
                  for i, b in enumerate(cls.blob["sdxl_one"])]
        shard1 += [row("StyleSwin", 1, b, i, "GAN", "celeba") for i, b in enumerate(cls.blob["styleswin"][:4])]
        # Cung model nhung real_source khac -> phai CONG don, khong duoc ghi de
        shard1 += [row("StyleSwin", 1, b, i + 4, "GAN", "ffhq")
                   for i, b in enumerate(cls.blob["styleswin"][4:])]
        shard1 += [row("CIPS", 1, b, i, "GAN", "landscapesHQ") for i, b in enumerate(cls.blob["cips"])]
        shard1 += [row("Gansformer", 1, b, i, "GAN", "ffhq") for i, b in enumerate(cls.blob["gansformer"])]
        shard1 += [row("ProjectedGAN", 1, b, i, "GAN", "ffhq") for i, b in enumerate(cls.blob["projectedgan"])]
        shard1 += [row("COCO", 0, b, i, "Real", "N/A") for i, b in enumerate(cls.blob["coco"])]
        # 1 anh DeepFloyd trung byte-exact voi shard0 -> phai bi dedupe loai
        shard1 += [row("DeepFloyd", 1, cls.blob["deepfloyd"][0], 0, "PixDiff", "coco",
                       name_suffix=999)]

        shard2 = [row("LandscapesHQ", 0, b, i, "Real", "N/A") for i, b in enumerate(cls.blob["land"])]
        shard2 += [row("VISION", 0, b, i, "Real", "N/A") for i, b in enumerate(cls.blob["vision"])]
        # 2 anh NSFW -> phai bi loai ngay o buoc fetch
        shard2 += [row("FFHQ", 0, make_png(9999 + i), 90 + i, "Real", "N/A") for i in range(2)]
        for r in shard2[-2:]:
            r["nsfw"] = True

        for i, rows in enumerate([shard0, shard1, shard2]):
            write_cfe_shard(os.path.join(cls.cache, "cfe_small", f"shard_{i:05d}.parquet"), rows)
        write_plain_shard(os.path.join(cls.cache, "gpt4o", "shard_00000.parquet"), cls.blob["gpt4o"])

        # --- index rows (tuong duong stage_index) ------------------------
        def index_rows(rows, shard, dataset="cfe_small"):
            agg = {}
            for r in rows:
                k = (r["label"], r["model_name"], r["architecture"], r.get("real_source"))
                a = agg.setdefault(k, {"n": 0, "nsfw": 0})
                a["n"] += 1
                a["nsfw"] += 1 if r.get("nsfw") else 0
            return [{"dataset": dataset, "shard": shard, "label": k[0], "architecture": k[2],
                     "subset": "Manual", "model_name": k[1], "real_source": k[3],
                     "n": v["n"], "nsfw": v["nsfw"]} for k, v in agg.items()]

        cls.index = (index_rows(shard0, 0) + index_rows(shard1, 1) + index_rows(shard2, 2)
                     + [{"dataset": "gpt4o", "shard": 0, "label": 1, "architecture": "External",
                         "subset": "plain", "model_name": "GPT-4o", "real_source": None,
                         "n": len(cls.blob["gpt4o"]), "nsfw": 0}])

        # --- replay ------------------------------------------------------
        cls.replay_records = []
        for gen, label, n, sub in [("real", 0, 30, "stable_diffusion_v_1_5"),
                                   ("stable_diffusion", 1, 6, "stable_diffusion_v_1_5"),
                                   ("midjourney", 1, 4, "midjourney")]:
            d = os.path.join(cls.replay_root, sub, "0_real" if label == 0 else "1_fake")
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                blob = make_png(50000 + len(cls.replay_records))
                p = os.path.join(d, f"{i:05d}.png")
                with open(p, "wb") as f:
                    f.write(blob)
                rel_old = ("/kaggle/input/x/diffusion_staging/diffusion_staging/"
                           + os.path.relpath(p, cls.replay_root).replace("\\", "/"))
                cls.replay_records.append({"file_path": rel_old, "label": label,
                                           "generator": gen, "source": "diffusion_staging",
                                           "split": "train"})
        os.makedirs(os.path.join(cls.tmp, "DATASET", "manifests"), exist_ok=True)
        with open(os.path.join(cls.tmp, "DATASET", "manifests", "data_manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"metadata": {"total_samples": len(cls.replay_records)},
                       "records": cls.replay_records}, f)

        # --- config ------------------------------------------------------
        cfg = {
            "version": 2,
            "seed": 42,
            "allow_related_generators": False,
            "paths": {
                "raw_root": os.path.join(cls.tmp, "raw_v2"),
                "manifest": os.path.join(cls.tmp, "manifests", "data_manifest_v2.json"),
                "state_dir": os.path.join(cls.tmp, "state"),
                "denylist": os.path.join(cls.tmp, "denylist.json"),
                "constraint_report": os.path.join(cls.tmp, "constraints.json"),
                "constraint_report_md": os.path.join(cls.tmp, "constraints.md"),
                "distribution_png": os.path.join(cls.tmp, "dist.png"),
                "source_log": os.path.join(cls.tmp, "sources.json"),
            },
            "targets": {
                "total": 208, "min_total": 100, "max_total": 400,
                "split_ratios": {"train": 0.85, "val_calib": 0.075, "val_select": 0.075},
                "max_family_imbalance": 5.0, "real_frac": [0.25, 0.65], "split_ratio_tolerance": 0.05,
                "replay_total": 40, "replay_frac": [0.12, 0.30],
                "max_new_per_family": 40, "min_side": 200,
                "min_images_total_after_denylist": 100,
                "phash_internal": 3, "phash_external": 8,
                "byte_budget_gb": 1, "surplus": 1.25,
            },
            "datasets": {
                "cfe_small": {"kind": "cfe",
                              "family_of_architecture": {"Commercial": "commercial",
                                                         "LatDiff": "latent_diffusion",
                                                         "PixDiff": "pixel_diffusion",
                                                         "GAN": "modern_gans",
                                                         "Other": "autoregressive"},
                              "url_template": os.path.join(cls.cache, "cfe_small", "shard_{:05d}.parquet"),
                              "n_shards": 3, "license": "test"},
                "gpt4o": {"kind": "plain", "family": "commercial", "model_name": "GPT-4o",
                          "url_template": os.path.join(cls.cache, "gpt4o", "shard_{:05d}.parquet"),
                          "n_shards": 1, "license": "test"},
            },
            "fakes": [
                {"id": "ar_taming", "dataset": "cfe_small", "family": "autoregressive",
                 "model_name": "tamingTransformers", "target": 20},
                {"id": "gan_giga", "dataset": "cfe_small", "family": "modern_gans",
                 "model_name": "GigaGAN", "target": 5},
                {"id": "gan_swin", "dataset": "cfe_small", "family": "modern_gans",
                 "model_name": "StyleSwin", "target": 5},
                {"id": "gan_cips", "dataset": "cfe_small", "family": "modern_gans",
                 "model_name": "CIPS", "target": 5},
                {"id": "gan_gans", "dataset": "cfe_small", "family": "modern_gans",
                 "model_name": "Gansformer", "target": 5},
                {"id": "gan_proj", "dataset": "cfe_small", "family": "modern_gans",
                 "model_name": "ProjectedGAN", "target": 5},
                {"id": "pix_df", "dataset": "cfe_small", "family": "pixel_diffusion",
                 "model_name": "DeepFloyd", "target": 20},
                {"id": "lat_dit", "dataset": "cfe_small", "family": "latent_diffusion",
                 "model_name": "DiT", "target": 8},
                {"id": "lat_xl", "dataset": "cfe_small", "family": "latent_diffusion",
                 "model_name": "stablediffusionapi/xl-one", "target": 7},
                {"id": "com_gpt", "dataset": "gpt4o", "family": "commercial",
                 "model_name": "GPT-4o", "target": 20},
            ],
            "reals": [
                {"id": "real_ffhq", "dataset": "cfe_small", "model_name": "FFHQ", "target": 17},
                {"id": "real_coco", "dataset": "cfe_small", "model_name": "COCO", "target": 17},
                {"id": "real_land", "dataset": "cfe_small", "model_name": "LandscapesHQ", "target": 17},
                {"id": "real_vision", "dataset": "cfe_small", "model_name": "VISION", "target": 17},
            ],
            "replay": {
                "manifest": os.path.join(cls.tmp, "DATASET", "manifests", "data_manifest.json"),
                "total": 40, "copy_mode": "copy", "path_markers": ["diffusion_staging/"],
                "family_of_generator": {"stable_diffusion": "latent_diffusion",
                                        "midjourney": "commercial", "real": "real"},
                "generator_display": {"stable_diffusion": "stable_diffusion_v1_5",
                                      "midjourney": "midjourney_v5", "real": "imagenet_replay"},
                "real_source_of_generator": {"real": "ImageNet"},
            },
            "quality": {"reject_flat_images": True, "reject_fully_transparent": True,
                        "reject_nsfw": True, "allowed_formats": ["png", "jpeg", "jpg", "webp"],
                        "max_side": 0},
        }
        cfg_path = os.path.join(cls.tmp, "config.yaml")
        import yaml
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
        cls.cfg = B.Config(cfg_path)

        # denylist gia lap
        m = DenyMatcher(["progan", "stylegan2", "glide"])
        payload = {"n_generators": len(m.entries), "generators": [e["raw"] for e in m.entries],
                   "entries": m.entries,
                   "match_semantics": {"blocked": "same instance",
                                       "related": "same base new version"}}
        with open(cls.cfg.path_of("denylist"), "w", encoding="utf-8") as f:
            json.dump(payload, f)
        cls.matcher = DenyMatcher.from_payload(payload)

        # --- "tap test": 2 anh lay tu CIPS (1 byte-exact, 1 nen JPEG) + 1 anh la
        cls.test_imgs = [cls.blob["cips"][0], recompress_jpeg(cls.blob["cips"][1]),
                         make_png(70001)]
        with Image.open(io.BytesIO(cls.blob["cips"][1])) as im_a, \
                Image.open(io.BytesIO(cls.test_imgs[1])) as im_b:
            cls.jpeg_phash_distance = bin(phash_from_pil(im_a) ^ phash_from_pil(im_b)).count("1")
        cls.ref_path = os.path.join(cls.tmp, "hashes_cfe.json")
        payload = {"metadata": {"n_images": len(cls.test_imgs)}, "images": []}
        names = ["TestOnlyGAN_A", "TestOnlyGAN_B", "ProGAN"]
        for i, blob in enumerate(cls.test_imgs):
            with Image.open(io.BytesIO(blob)) as im:
                payload["images"].append({
                    "path": f"test_communityforensics/1_fake/{i:05d}.png",
                    "sha256": sha256_bytes(blob), "phash": int(phash_from_pil(im)),
                    "label": 1, "model_name": names[i],
                })
        with open(cls.ref_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    def test_01_plan(self):
        plan = B.build_plan(self.cfg, self.index, self.matcher)
        self.assertEqual(plan["fakes_new_target"], 100)
        self.assertEqual(plan["reals_target"], 68)
        self.assertEqual(plan["replay_composition"]["n_real"], 30)
        self.assertEqual(plan["replay_composition"]["n_fake"], 10)
        # ho 'commercial' (dataset plain) van phai co mat trong bang chung
        self.assertIn("commercial", plan["family_table"])
        self.assertEqual(plan["family_table"]["modern_gans"]["n_kept"], 25)
        # StyleSwin co 2 real_source trong cung shard -> phai dem du 8, khong phai 4
        self.assertEqual(plan["family_table"]["modern_gans"]["n_candidates"], 37)
        self.assertLessEqual(plan["predicted_bytes"], plan["byte_budget"])
        B.print_plan(plan, self.cfg)
        with open(self.cfg.state("plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan, f)

    def test_02_fetch_replay_select_split_finalize(self):
        plan = B.build_plan(self.cfg, self.index, self.matcher)
        r1 = B.stage_fetch(self.cfg, plan, self.cache, resume=False, token=None)
        # 211 = 123 fake moi (gom ban trung byte-exact cua DeepFloyd) + 88 real moi
        self.assertEqual(r1["n_new"], 211)
        # resume lan 2: khong duoc nhan doi
        n_before = len(B.read_jsonl(self.cfg.state("candidates.jsonl")))
        r2 = B.stage_fetch(self.cfg, plan, self.cache, resume=True, token=None)
        self.assertEqual(r2["n_new"], 0)
        self.assertEqual(len(B.read_jsonl(self.cfg.state("candidates.jsonl"))), n_before)

        rep = B.stage_replay(self.cfg, self.replay_root, "copy")
        self.assertEqual(rep["n"], 40)

        refs = B.load_refs([self.ref_path], 0)
        self.assertEqual(len(refs["hashes_cfe"]["phash"]), len(self.test_imgs),
                         "load_refs phai giu toan bo pHash khi max_phash=0")
        self.assertLessEqual(self.jpeg_phash_distance, 8,
                             "anh nen lai phai duoc coi la 'gan trung' (d<=8)")
        sel = B.stage_select(self.cfg, plan, refs, strict=True)
        ref_stats = sel["ref_stats"]
        self.assertGreaterEqual(ref_stats.get("hashes_cfe_sha", 0), 1,
                                "phai loc duoc anh trung byte-exact voi tap test")
        self.assertGreaterEqual(sel["dedupe"]["sha_dup"], 1,
                                "phai khu duoc anh trung byte-exact giua 2 shard")
        self.assertEqual(ref_stats.get("hashes_cfe_phash", 0) + ref_stats.get("hashes_cfe_sha", 0),
                         2, "phai loai dung 2/5 anh CIPS da nam trong tap test")

        records = B.read_jsonl(self.cfg.state("selected.jsonl"))
        self.assertEqual(len(records), 208)
        self.assertEqual(len({r["sha256"] for r in records}), len(records))

        B.stage_split(self.cfg)
        res = B.stage_finalize(self.cfg, plan, {"strict": True})
        c = res["constraints"]
        for name, chk in c["checks"].items():
            self.assertTrue(chk["pass"], f"rang buoc '{name}' khong dat: {chk}")
        self.assertTrue(c["all_pass"])
        self.assertEqual(c["family_totals"],
                         {"autoregressive": 20, "commercial": 24, "latent_diffusion": 21,
                          "modern_gans": 25, "pixel_diffusion": 20, "real": 98})

        man = json.load(open(self.cfg.path_of("manifest"), encoding="utf-8"))
        self.assertEqual(man["metadata"]["n_records"], 208)
        for r in man["records"]:
            for f in REQUIRED_FIELDS:
                self.assertIn(f, r)
            self.assertEqual(len(r["sha256"]), 64)
        self.assertEqual(sum(1 for r in man["records"] if r["is_replay"]), 40)
        # moi o (family,label,generator,is_replay) phai co mat o ca 3 split
        cells = {}
        for r in man["records"]:
            cells.setdefault((r["family"], r["label"], r["generator"]), set()).add(r["split"])
        for key, splits in cells.items():
            if key[2] in ("CIPS",):  # CIPS chi con 3 anh sau khi loc test
                continue
            self.assertEqual(len(splits), 3, f"o {key} khong phu du 3 split")

        # --- audit doc lap ---
        report = self._run_audit()
        self.assertEqual(report["violations"], 0)
        self.assertTrue(report["pass"])
        self.assertEqual(report["generator_check"]["violations"], 0)
        self.assertEqual(report["integrity"]["failures"], 0)

        # --- audit am tinh 1: dua 1 anh cua tap test vao manifest -> FAIL ---
        man2 = json.loads(json.dumps(man))
        with open(self.ref_path, encoding="utf-8") as f:
            ref_payload = json.load(f)
        victim = man2["records"][0]
        victim["sha256"] = ref_payload["images"][0]["sha256"]
        victim["phash"] = ref_payload["images"][0]["phash"]
        rep2 = self._run_audit(manifest=self._dump(man2, "bad_manifest.json"),
                               out=os.path.join(self.tmp, "bad_report.json"))
        self.assertGreater(rep2["violations"], 0)
        self.assertFalse(rep2["pass"])

        # --- audit am tinh 2: generator trung instance -> FAIL ---
        man3 = json.loads(json.dumps(man))
        man3["records"][0]["generator"] = "ProGAN"
        rep3 = self._run_audit(manifest=self._dump(man3, "bad_manifest3.json"),
                               out=os.path.join(self.tmp, "bad3.json"))
        self.assertGreater(rep3["generator_check"]["violations"], 0)
        self.assertFalse(rep3["pass"])

        # --- audit am tinh 3: 1 anh nam o 2 split -> FAIL ---
        man4 = json.loads(json.dumps(man))
        dup = json.loads(json.dumps(man4["records"][0]))
        dup["split"] = "val_calib" if dup["split"] != "val_calib" else "train"
        man4["records"].append(dup)
        rep4 = self._run_audit(manifest=self._dump(man4, "bad_manifest4.json"),
                               out=os.path.join(self.tmp, "bad4.json"))
        self.assertGreater(rep4["pairs"]["internal_splits"]["pairs"]["sha_in_multiple_splits"]["count"], 0)
        self.assertGreater(rep4["violations"], 0)

    # ------------------------------------------------------------------
    def _dump(self, payload, name):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return path

    def _run_audit(self, manifest: str = "", out: str = "") -> dict:
        argv = sys.argv
        sys.argv = ["audit", "--manifest", manifest or self.cfg.path_of("manifest"),
                    "--ref", f"cfe={self.ref_path}",
                    "--denylist", self.cfg.path_of("denylist"),
                    "--out", out or os.path.join(self.tmp, "report.json"), "--quiet"]
        try:
            A.main()
        finally:
            sys.argv = argv
        with open(out or os.path.join(self.tmp, "report.json"), encoding="utf-8") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
class UnitTests(unittest.TestCase):
    def test_largest_remainder_exact(self):
        for total in (1, 2, 3, 7, 40, 195, 1500, 15000):
            out = B.largest_remainder(total, {"a": 0.85, "b": 0.075, "c": 0.075})
            self.assertEqual(sum(out.values()), total)

    def test_split_15000_is_exact(self):
        out = B.largest_remainder(15000, {"train": 0.85, "val_calib": 0.075, "val_select": 0.075})
        self.assertEqual(out, {"train": 12750, "val_calib": 1125, "val_select": 1125})

    def test_denylist_instance_semantics(self):
        m = DenyMatcher(["MidjourneyV5_1", "stylegan", "glide_100_10"])
        self.assertEqual(m.classify("MidjourneyV5_1")["status"], "blocked")
        self.assertEqual(m.classify("MidjourneyV5")["status"], "blocked")     # version la tien to
        self.assertEqual(m.classify("MidjourneyV6_1")["status"], "related")  # khac version
        self.assertEqual(m.classify("StyleGAN2")["status"], "blocked")
        self.assertEqual(m.classify("StyleSANXL")["status"], "allowed")
        self.assertEqual(m.classify("glide")["status"], "blocked")
        self.assertEqual(m.classify("GPT-4o")["status"], "allowed")

    def test_denylist_from_config_is_all_allowed(self):
        import yaml
        cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "configs", "phase7_data_v2.yaml"),
                                  encoding="utf-8"))
        with open(os.path.join(PROJECT_ROOT, "DATASET", "logs", "phase7",
                               "generator_denylist.json"), encoding="utf-8") as f:
            payload = json.load(f)
        m = DenyMatcher.from_payload(payload)
        for e in cfg["fakes"]:
            verdict = m.classify(e["model_name"])
            self.assertEqual(verdict["status"], "allowed",
                             f"{e['id']} ({e['model_name']}) bi chan: {verdict}")
        # va khong duoc co generator test nao lot vao config real/fake
        for e in cfg["fakes"]:
            self.assertNotIn(m.classify(e["model_name"])["status"], ("blocked", "related"))

    def test_dedupe_threshold(self):
        # Kiem tra nguong pHash mot cach tat dinh (khong phu thuoc chat luong JPEG)
        base = 0b1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010
        rows = [
            {"sha256": "a", "phash": base},
            {"sha256": "b", "phash": base ^ 0b111},          # d = 3 -> trung
            {"sha256": "c", "phash": base ^ 0b1111111},      # d = 7 -> giu
            {"sha256": "a", "phash": base},                  # trung SHA-256
        ]
        kept, stats = B._dedupe(rows, 3)
        self.assertEqual([r["sha256"] for r in kept], ["a", "c"])
        self.assertEqual(stats["sha_dup"], 1)
        self.assertEqual(stats["phash_dup"], 1)

    def test_quality_rejects(self):
        cfg = B.Config.__new__(B.Config)
        cfg.t = {"min_side": 200}
        cfg.quality = {"reject_flat_images": True, "reject_fully_transparent": True,
                       "allowed_formats": ["png"]}
        flat = io.BytesIO()
        Image.new("RGB", (256, 256), (10, 10, 10)).save(flat, format="PNG")
        self.assertIsNone(B._quality_check(flat.getvalue(), cfg)[0])
        small = io.BytesIO()
        Image.new("RGB", (64, 64), (1, 2, 3)).save(small, format="PNG")
        self.assertEqual(B._quality_check(small.getvalue(), cfg)[1], "too_small")
        self.assertIsNotNone(B._quality_check(make_png(1), cfg)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
