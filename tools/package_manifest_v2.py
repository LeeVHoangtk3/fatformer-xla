"""Package only audited records; refuse stale/incomplete audit evidence."""
import argparse
import io
import json
import os
from pathlib import Path
import tarfile

from hash_utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def validate_handoff(manifest, report, config):
    with open(report, encoding="utf-8") as f:
        audit = json.load(f)
    if (audit.get("pass") is not True or audit.get("violations") != 0
            or audit.get("integrity", {}).get("failures") != 0
            or audit.get("coverage", {}).get("complete") is not True):
        raise ValueError("Audit is not complete and passing")
    meta = audit.get("metadata", {})
    if meta.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("Manifest changed after audit; rerun audit")
    if meta.get("config_sha256") != sha256_file(config):
        raise ValueError("Config changed after audit; rerun audit")
    required = {"v2_vs_cfe_test", "v2_vs_task51_18sets", "internal_splits"}
    if not required <= set(audit.get("pairs", {})):
        raise ValueError("Audit lacks mandatory pairs")
    internal = audit["pairs"]["internal_splits"].get("pairs", {})
    if len([k for k in internal if "__vs__" in k]) != 3:
        raise ValueError("Audit lacks three internal split pairs")
    references = meta.get("references", {})
    if not references:
        raise ValueError("Audit lacks reference fingerprints")
    for ref in references.values():
        if ref.get("sha256") != sha256_file(ref["path"]):
            raise ValueError("Reference changed after audit")
    return audit


def package(manifest, report, config, out, root=ROOT):
    validate_handoff(manifest, report, config)
    manifest_digest = sha256_file(manifest)
    report_digest = sha256_file(report)
    with open(manifest, encoding="utf-8") as f:
        payload = json.load(f)
    root = Path(root).resolve()
    files = []
    for row in payload["records"]:
        path = Path(row["path"])
        if path.is_absolute():
            raise ValueError("Portable archive requires paths relative to its root")
        path = (path if path.is_absolute() else root / path).resolve()
        relative = path.relative_to(root)  # reject paths outside the portable archive root
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Image changed after audit: {path}")
        files.append((path, relative.as_posix()))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    with tarfile.open(partial, "w") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
        archive.add(manifest, arcname="DATASET/manifests/data_manifest_v2.json")
        archive.add(report, arcname="DATASET/logs/phase7/data_isolation_report.json")
        archive.add(config, arcname="configs/phase7_data_v2.yaml")
        for split in ("train", "val_calib", "val_select", "train_select"):
            subset = {"metadata": {"split": split, "parent_manifest_sha256": manifest_digest},
                      "records": [r for r in payload["records"] if r["split"] == split
                                  or (split == "train_select" and r["split"] in ("train", "val_select"))]}
            body = json.dumps(subset, ensure_ascii=False).encode("utf-8")
            info = tarfile.TarInfo(f"DATASET/manifests/data_manifest_v2_{split}.json")
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    if sha256_file(manifest) != manifest_digest or sha256_file(report) != report_digest:
        raise ValueError("Handoff evidence changed during packaging")
    validate_handoff(manifest, report, config)
    # Detect files changed between the initial check and tar.add.
    with tarfile.open(partial, "r:") as archive:
        import hashlib
        for row, (_, name) in zip(payload["records"], files):
            stream = archive.extractfile(name)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"Archived image checksum mismatch: {name}")
    os.replace(partial, out)
    digest = sha256_file(out)
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    return digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="DATASET/manifests/data_manifest_v2.json")
    ap.add_argument("--report", default="DATASET/logs/phase7/data_isolation_report.json")
    ap.add_argument("--config", default="configs/phase7_data_v2.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(package(args.manifest, args.report, args.config, args.out))


if __name__ == "__main__":
    main()
