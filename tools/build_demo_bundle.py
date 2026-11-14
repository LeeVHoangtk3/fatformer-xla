"""Build a minimal Colab source bundle, including 20 examples but no weights."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_bundle(output):
    files = [ROOT / name for name in (
        "demo.py", "requirements-demo.txt", "src/__init__.py",
        "tools/infer_unified.py", "tools/package_unified_fatformer.py",
        "docs/demo/README.md", "docs/demo/validation.md",
        "tools/build_demo_bundle.py", "tests/test_demo.py",
        "notebooks/demo_fatformer_xla_colab.ipynb",
        "DATASET/logs/phase7/task_7.15/current/analysis/verification_images.csv",
    )]
    files += sorted((ROOT / "src/models").rglob("*.py"))
    files += sorted((ROOT / "src/models").rglob("*.gz"))
    files += sorted((ROOT / "DATASET/logs/phase7/task_7.15/current/raw/verification_images").glob("*.png"))
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("demo_bundle_manifest.json", json.dumps(manifest, indent=2))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(json.dumps(dict(path=str(output.resolve()), files=len(files), bytes=output.stat().st_size,
                          sha256=digest), indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/demo/fatformer_xla_demo.zip")
    build_bundle(parser.parse_args().output)
