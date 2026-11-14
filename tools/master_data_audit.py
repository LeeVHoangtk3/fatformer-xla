"""Measured full-catalog audit. No training, calibration refit or cached metrics."""
import csv
import hashlib
import io
import json
import shutil
import sqlite3
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from tools.calibration_data import read_partition, sample_id, sha256_file
from tools.eval_locked_benchmark import ALL_DEGRADED_VARIANTS, SUBSET_ALIASES
from tools.phase7_corrected import CONDITIONS, INDIST_COUNTS, write_json
from tools.run_phase7_corrected import metrics, predict_records


PACKAGE_SHA = "248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0"
SOURCE_SHA = "5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c"
SUPPLEMENTAL = ("crn", "imle", "san", "seeingdark", "whichfaceisreal")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def catalog(drive_root):
    rows = [
        ("progan_train", "datasets/progan_train.tar", 144024, "train"),
        ("progan_val", "datasets/progan_val.tar", 8000, "validation"),
        ("diffusion_staging", "datasets/diffusion_staging.tar", 3600, "legacy_manifest"),
        ("data_v2", "phase7/train_crossdomain_v2.tar", 14400, "v2_manifest"),
        ("cfe", "datasets/test_communityforensics.tar", 2000, "cfe"),
        ("test_gans", "datasets/test_benchmark_gans.tar", 90329, "test"),
        ("test_diffusion", "datasets/test_benchmark_diffusion.tar", 20000, "test"),
        ("test_degraded", "datasets/test_degraded.tar", 661974, "stored_degraded"),
    ]
    label_counts = dict(progan_train=[72012,72012], progan_val=[4000,4000], diffusion_staging=[1800,1800],
        data_v2=[7863,6537], cfe=[1000,1000], test_gans=[45169,45160], test_diffusion=[10000,10000],
        test_degraded=[331014,330960])
    return [dict(name=name, path=str(Path(drive_root)/path), expected_images=count, kind=kind,
                 expected_labels=label_counts[name])
            for name, path, count, kind in rows]


def open_inventory(path):
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE images (archive TEXT, member TEXT, sha256 TEXT, label INTEGER, "
                       "split TEXT, subset TEXT, sample_id TEXT, bytes INTEGER, PRIMARY KEY(archive,member))")
    connection.execute("CREATE INDEX image_hash ON images(sha256)")
    return connection


def manifest_maps(root):
    root = Path(root)
    v2 = json.loads((root/"DATASET/manifests/data_manifest_v2.json").read_text(encoding="utf-8"))["records"]
    legacy = json.loads((root/"DATASET/manifests/data_manifest.json").read_text(encoding="utf-8"))["records"]
    if Counter(row["split"] for row in v2) != dict(train=12240, val_select=1080, val_calib=1080):
        raise ValueError("Wrong v2 manifest split counts")
    by_hash = {row["sha256"]: row for row in v2}
    by_id = {sample_id(row["file_path"]): row for row in legacy}
    if len(by_hash) != len(v2) or len(by_id) != len(legacy):
        raise ValueError("Duplicate identities/hashes in source manifests")
    if Counter(row["split"] for row in legacy) != dict(train=3060, val=540):
        raise ValueError("Wrong legacy manifest split counts")
    calib, cy = read_partition(root/"configs/master_audit/calibration_split.json", "calibration")
    evaluation, ey = read_partition(root/"configs/master_audit/evaluation_split.json", "evaluation")
    if len(calib) != 400 or len(evaluation) != 1600 or set(calib) & set(evaluation):
        raise ValueError("Require the frozen disjoint CFE 400/1600 split")
    partitions = {sid: ("calibration", int(y)) for sid, y in zip(calib, cy)}
    partitions.update({sid: ("evaluation", int(y)) for sid, y in zip(evaluation, ey)})
    return dict(v2=by_hash, legacy=by_id, cfe=partitions)


def safe_member(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return path


def class_label(parts):
    classes = {"0": 0, "0_real": 0, "real": 0, "1": 1, "1_fake": 1, "fake": 1}
    for part in reversed(parts[:-1]):
        if part.lower() in classes:
            return classes[part.lower()]
    raise ValueError(f"No explicit real/fake class directory: {'/'.join(parts)}")


def identify(spec, member, digest, maps):
    parts = safe_member(member).parts
    kind = spec["kind"]
    if kind == "v2_manifest":
        row = maps["v2"].get(digest)
        if row is None:
            raise ValueError(f"Image absent from frozen v2 content hashes: {member}")
        return int(row["label"]), row["split"], "data_v2", row["path"]
    label = class_label(parts)
    if kind == "legacy_manifest":
        sid = sample_id(member)
        row = maps["legacy"].get(sid)
        if row is None or int(row["label"]) != label:
            raise ValueError(f"Missing/mismatched legacy manifest identity: {sid}")
        return label, row["split"], "diffusion_staging", sid
    if kind == "cfe":
        sid = sample_id(member)
        if sid not in maps["cfe"] or maps["cfe"][sid][1] != label:
            raise ValueError(f"Image absent from frozen CFE partitions: {sid}")
        return label, maps["cfe"][sid][0], sid.split('/')[0], sid
    aliases = {alias.lower(): subset for subset, names in SUBSET_ALIASES.items() for alias in names}
    aliases.update({subset: subset for subset in SUPPLEMENTAL})
    subsets = {aliases[p.lower()] for p in parts[:-1] if p.lower() in aliases}
    if kind in {"test", "stored_degraded"}:
        if len(subsets) != 1:
            raise ValueError(f"Missing/ambiguous test subset: {member}")
        split = "test"
        if kind == "stored_degraded":
            variants = set(parts) & set(ALL_DEGRADED_VARIANTS)
            if len(variants) != 1:
                raise ValueError(f"Missing/ambiguous stored degradation: {member}")
            split = variants.pop()
        return label, split, subsets.pop(), spec["name"]+"/"+member
    return label, kind, spec["name"], spec["name"]+"/"+member


def audit_archive(spec, connection, maps, output):
    """Stream/decode every image, retaining only inventory on disk; never replace bad images."""
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    report = dict(name=spec["name"], path=str(path), archive_sha256=sha256_file(path),
                  status="running", images=0, image_bytes=0, ignored_nonimages=0)
    write_json(output, report)
    counts, groups, labels = Counter(), Counter(), Counter()
    with tarfile.open(path, "r|*") as archive:
        for member in archive:
            member_name = safe_member(member.name).as_posix()
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Archive link/device not allowed: {member.name}")
            if PurePosixPath(member.name).suffix.lower() not in IMAGE_EXTENSIONS:
                report["ignored_nonimages"] += 1
                continue
            with archive.extractfile(member) as stream:
                data = stream.read()
            if len(data) != member.size:
                raise ValueError(f"Truncated image member: {member.name}")
            digest = hashlib.sha256(data).hexdigest()
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    if getattr(image, "is_animated", False):
                        raise ValueError(f"Animated image not supported: {member.name}")
            except (OSError, Image.DecompressionBombError) as error:
                raise ValueError(f'Cannot decode {spec["name"]}:{member.name}') from error
            label, split, subset, sid = identify(spec, member_name, digest, maps)
            connection.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?,?)",
                (spec["name"], member_name, digest, label, split, subset, sid, len(data)))
            counts[split] += 1
            groups[subset] += 1
            labels[label] += 1
            report["images"] += 1
            report["image_bytes"] += len(data)
            if report["images"] % 1000 == 0:
                connection.commit()
                write_json(output, dict(report, by_split=dict(counts), by_subset=dict(groups)))
                print(f'{spec["name"]}: decoded/hashed {report["images"]}', flush=True)
    connection.commit()
    if report["images"] != spec["expected_images"]:
        raise ValueError(f'{spec["name"]}: {report["images"]}/{spec["expected_images"]} images')
    if "expected_labels" in spec and [labels[0], labels[1]] != spec["expected_labels"]:
        raise ValueError(f'Wrong Real/Fake counts: {spec["name"]}: {dict(labels)}')
    duplicates = connection.execute("SELECT sample_id FROM images WHERE archive=? GROUP BY sample_id HAVING count(*)>1",
                                   (spec["name"],)).fetchall()
    if duplicates:
        raise ValueError(f"Repeated image identity: {duplicates[:5]}")
    if spec["kind"] in {"v2_manifest", "cfe"}:
        key = "v2" if spec["kind"] == "v2_manifest" else "cfe"
        if report["images"] != len(maps[key]):
            raise ValueError("Manifest coverage is incomplete")
    if spec["kind"] == "v2_manifest":
        distinct = connection.execute("SELECT count(DISTINCT sha256) FROM images WHERE archive=?", (spec["name"],)).fetchone()[0]
        if distinct != len(maps["v2"]):
            raise ValueError("Duplicate/missing v2 content")
    if spec["kind"] == "cfe" and counts != dict(calibration=400, evaluation=1600):
        raise ValueError("Frozen CFE split count mismatch")
    if spec["name"] == "test_gans":
        core = {k: v for k, v in INDIST_COUNTS.items() if k not in ("guided", "ldm_200", "ldm_200_cfg", "ldm_100", "glide_50_27", "glide_100_10", "glide_100_27", "dalle", "pndm", "vqdiffusion")}
        if any(groups[k] != n for k, n in core.items()) or not set(SUPPLEMENTAL).issubset(groups):
            raise ValueError("Incomplete GAN core/supplemental coverage")
    if spec["name"] == "test_diffusion" and any(groups[k] != n for k, n in INDIST_COUNTS.items() if k not in {"progan", "stylegan", "stylegan2", "biggan", "cyclegan", "stargan", "gaugan", "deepfake"}):
        raise ValueError("Incomplete diffusion coverage")
    if spec["kind"] == "stored_degraded" and (set(counts) != set(ALL_DEGRADED_VARIANTS) or any(n != 110329 for n in counts.values())):
        raise ValueError("Incomplete six stored degradation variants")
    if spec["kind"] == "stored_degraded":
        for variant in ALL_DEGRADED_VARIANTS:
            actual = dict(connection.execute("SELECT subset,count(*) FROM images WHERE archive=? AND split=? GROUP BY subset", (spec["name"], variant)))
            if any(actual.get(k) != n for k,n in INDIST_COUNTS.items()) or not set(SUPPLEMENTAL).issubset(actual):
                raise ValueError(f"Incomplete stored-degraded subsets: {variant}")
    report.update(status="completed", by_split=dict(counts), by_subset=dict(groups), by_label=dict(labels))
    write_json(output, report)
    return report


def check_overlap(connection, output):
    """Exact byte duplicates, not an unmeasured pHash or generator-disjoint claim."""
    collisions = connection.execute("SELECT a.archive,a.member,a.split,b.archive,b.member,b.split "
        "FROM images a JOIN images b ON a.sha256=b.sha256 WHERE a.archive='data_v2' "
        "AND b.archive IN ('cfe','test_gans','test_diffusion')").fetchall()
    split_duplicates = connection.execute("SELECT sha256 FROM images WHERE archive='data_v2' GROUP BY sha256 HAVING count(DISTINCT split)>1").fetchall()
    cfe_duplicates = connection.execute("SELECT sha256 FROM images WHERE archive='cfe' GROUP BY sha256 HAVING count(DISTINCT split)>1").fetchall()
    conflicts = connection.execute("SELECT sha256 FROM images GROUP BY sha256 HAVING count(DISTINCT label)>1").fetchall()
    report = dict(exact_v2_vs_test_collisions=collisions, v2_cross_split_hashes=split_duplicates,
                  cfe_cross_split_hashes=cfe_duplicates, perceptual_audit="not_run",
                  conflicting_label_hashes=conflicts,
                  note="Legacy/replay/test archives can share content; counts are records, not independent images.")
    write_json(output, report)
    if collisions or split_duplicates or cfe_duplicates or conflicts:
        raise ValueError(f"Exact data leakage found; see {output}")
    return report


def extract_audited(spec, connection, destination):
    destination = Path(destination).resolve()
    rows = connection.execute("SELECT member,sha256,label,split,subset,sample_id,bytes FROM images WHERE archive=? ORDER BY member", (spec["name"],)).fetchall()
    required = {row[0]: row for row in rows}
    total_bytes = sum(row[-1] for row in rows)
    if shutil.disk_usage(destination.parent).free < total_bytes + 2*1024**3:
        raise OSError("Not enough SSD space for this archive plus 2 GiB reserve")
    seen = set()
    with tarfile.open(spec["path"], "r|*") as archive:
        for member in archive:
            member_name = safe_member(member.name).as_posix()
            if not (member.isdir() or member.isfile()):
                raise ValueError("Archive links/devices are forbidden")
            if member_name not in required:
                continue
            if member_name in seen:
                raise ValueError("Duplicate archive member during extraction")
            seen.add(member_name)
            with archive.extractfile(member) as stream:
                data = stream.read()
            if hashlib.sha256(data).hexdigest() != required[member_name][1]:
                raise ValueError("Archive image changed after audit")
            target = (destination/member_name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Extraction escapes staging directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if seen != set(required):
        raise ValueError("Archive images missing after audit")
    return [dict(path=str(destination/member), member=member, label=label, split=split, subset=subset,
                 sample_id=sid) for member, _, label, split, subset, sid, _ in rows]


def jobs_for(spec, records):
    for split in sorted({r["split"] for r in records}):
        selected = [r for r in records if r["split"] == split]
        conditions = CONDITIONS if spec["kind"] == "test" or (spec["kind"] == "cfe" and split == "evaluation") else ("clean",)
        if spec["kind"] == "v2_manifest" and split != "train":
            conditions = ("clean", "q30")
        for condition in conditions:
            yield split, condition, selected


def summarize_predictions(predictions, threshold, bootstrap=0):
    scores, labels = predictions["prob_fake"], predictions["targets"]
    pooled = metrics(scores, labels, threshold)
    groups = {}
    for subset in sorted(set(predictions["subsets"])):
        mask = predictions["subsets"] == subset
        groups[str(subset)] = metrics(scores[mask], labels[mask], threshold)
    if bootstrap:
        from tools.eval_communityforensics import compute_bootstrap_ci
        pooled["ci95"] = compute_bootstrap_ci(scores.astype(np.float64), labels, n_boot=bootstrap, seed=42)
    return dict(pooled=pooled, per_subset=groups)


def evaluate_archive(spec, connection, destination, output, model, device, batch_size=16, bootstrap=1000):
    archive_sha = sha256_file(spec["path"])
    if archive_sha != spec["audited_sha256"]:
        raise ValueError("Archive bytes changed after audit")
    records = extract_audited(spec, connection, destination)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for split, condition, selected in jobs_for(spec, records):
        tag = f'{spec["name"]}__{split}__{condition}'
        result = predict_records(model, selected, condition, device, batch_size=batch_size,
                                 workers=0, precision="fp16" if device.type == "cuda" else "fp32")
        result["paths"] = np.array([spec["name"]+":"+r["member"] for r in selected])
        path = output/(tag+".npz")
        np.savez_compressed(path, **result, package_sha256=PACKAGE_SHA, evidence_level="measured",
                            archive_sha256=archive_sha)
        record = dict(archive=spec["name"], split=split, condition=condition, prediction_file=path.name,
                      prediction_sha256=sha256_file(path), evidence_level="measured",
                      archive_sha256=archive_sha)
        record["protocol"] = ("stored degradation -> canonical resize/crop -> normalize; no second degradation"
                              if spec["kind"] == "stored_degraded" else "canonical resize/crop -> condition -> normalize")
        record["role"] = "evaluation" if spec["kind"] == "test" or (spec["kind"] == "cfe" and split == "evaluation") or spec["kind"] == "stored_degraded" else "diagnostic_only_no_refit"
        record.update(summarize_predictions(result, model.fixed_threshold,
                      bootstrap if spec["kind"] == "cfe" and split == "evaluation" else 0))
        write_json(output/(tag+".json"), record)
        reports.append(record)
    return reports


def final_metrics(reports, output, threshold, bootstrap=1000):
    output = Path(output)
    cfe = {r["condition"]: r for r in reports if r["archive"] == "cfe" and r["split"] == "evaluation"}
    indist = {}
    for condition in CONDITIONS:
        parts = []
        for row in reports:
            if row["archive"] in {"test_gans", "test_diffusion"} and row["condition"] == condition:
                with np.load(output/row["prediction_file"], allow_pickle=False) as data:
                    keep = np.isin(data["subsets"], list(INDIST_COUNTS))
                    parts.append({key: data[key][keep] for key in ("prob_fake", "targets", "subsets")})
        if len(parts) != 2:
            raise ValueError("Both clean test archives and every condition are required")
        merged = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        if Counter(merged["subsets"]) != INDIST_COUNTS:
            raise ValueError("Wrong pooled 18-subset counts")
        summary = summarize_predictions(merged, threshold, bootstrap if condition in ("clean", "q30") else 0)
        summary["macro_balanced_acc"] = float(np.mean([v["balanced_acc"] for v in summary["per_subset"].values()]))
        indist[condition] = summary
    if set(cfe) != set(CONDITIONS):
        raise ValueError("All seven CFE evaluation conditions are required")
    from tools.run_locked_benchmark_phase7 import evaluate_master_gates
    g72 = evaluate_master_gates(dict(macro_acc=indist["clean"]["macro_balanced_acc"],
        real_acc=indist["clean"]["pooled"]["real_acc"]), cfe["clean"]["pooled"], cfe["q30"]["pooled"],
        cfe["clean"]["per_subset"], 0, {})["G7.2"]
    return dict(cfe=cfe, indist_core18=indist, G72=g72)


def export_summary_csv(summary, path):
    fields = ["scope", "condition", "subset", "total_samples", "accuracy", "balanced_acc",
              "real_acc", "fake_acc", "auc", "fake_at_real90", "ece", "nll", "macro_balanced_acc"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scope, conditions in (("cfe_evaluation", summary["cfe"]), ("indist_core18", summary["indist_core18"])):
            for condition, report in conditions.items():
                for subset, values in {"__pooled__": report["pooled"], **report["per_subset"]}.items():
                    row = {key: values[key] for key in fields if key in values}
                    row.update(scope=scope, condition=condition, subset=subset)
                    if scope == "indist_core18" and subset == "__pooled__":
                        row["macro_balanced_acc"] = report["macro_balanced_acc"]
                    writer.writerow(row)
