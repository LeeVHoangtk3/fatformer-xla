"""Run four fresh, comparable ablations with checked epoch resume and held-out CFE."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.training.runtime_state import sha256_file, training_context
from tools.ablation_v2_common import NAMES, load_partition, read_json, validate_history
from tools.audit_ablation_configs import audit_configs, canonical_hash


def run(command, log_file=None):
    command = [str(x) for x in command]
    if log_file is None:
        subprocess.run(command, cwd=ROOT, check=True)
        return
    with Path(log_file).open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + json.dumps(command) + "\n")
        log.flush()
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace",
                              env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line); log.flush()
            status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)


def run_logged(command, folder, backup):
    log = folder/"commands.log"
    try:
        run(command, log)
    finally:
        if log.exists():
            shutil.copy2(log, backup/log.name)


def context_for(config_path):
    from src.training.train_crossdomain import parse_arguments
    original = sys.argv
    try:
        sys.argv = ["train", "--config", str(config_path)]
        args = parse_arguments()
    finally:
        sys.argv = original
    return training_context(args, ROOT)


def checked_snapshot(path, context):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state.get("schema") != "phase7-epoch-resume/v1"
            or state.get("resume_state", {}).get("context") != context
            or state["resume_state"].get("epoch_complete") is not True):
        raise ValueError("Latest is partial/legacy or has a different immutable context; use a new experiment")
    return state


def restore_latest(folder, backup, context):
    """Choose latest validated completed epoch, never infer completion from best existence."""
    candidates = []
    for directory in (folder, backup):
        path = directory/"checkpoint_latest.pth"
        if path.exists():
            if directory == backup:
                sidecar = Path(str(path)+".sha256")
                if not sidecar.exists() or sha256_file(path) != sidecar.read_text().strip():
                    raise ValueError("Remote latest checksum missing/mismatched")
            state = checked_snapshot(path, context)
            candidates.append((int(state["metadata"]["epoch"]), path))
            del state
    if not candidates:
        return None
    _, source = max(candidates, key=lambda x: x[0])
    if source.parent != folder:
        shutil.copy2(source, folder/source.name)
        for filename in ("best.pth", "best_effort.pth", "val_groups.csv", "train_history.csv"):
            if (source.parent/filename).exists():
                shutil.copy2(source.parent/filename, folder/filename)
    return folder/"checkpoint_latest.pth"


def verify_run(folder, context, source_commit):
    """Bind actual selected weights, logs, epoch horizon and finite gradients."""
    folder = Path(folder)
    state = checked_snapshot(folder/"checkpoint_latest.pth", context)
    epochs = int(context["epochs"])
    if state["metadata"]["epoch"] != epochs:
        raise ValueError("Run has not reached the configured epoch horizon")
    history = state["resume_state"]["training_history"]
    steps = validate_history(history, epochs)
    with (folder/"train_history.csv").open(encoding="utf-8", newline="") as handle:
        logged = list(csv.DictReader(handle))
    if len(logged) != len(history):
        raise ValueError("CSV history differs from latest")
    for actual, expected in zip(logged, history):
        if any(not np.isclose(float(actual[k]), float(expected[k]), equal_nan=False)
               for k in ("epoch", "train_steps", "loss_total", "grad_norm")):
            raise ValueError("CSV history differs from latest")
    with (folder/"val_groups.csv").open(encoding="utf-8", newline="") as handle:
        groups = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in groups] != list(range(epochs+1)):
        raise ValueError("Incomplete or duplicate validation epochs")
    validation = state["resume_state"]["validation_history"]
    if [row["epoch"] for row in validation] != list(range(epochs+1)):
        raise ValueError("Snapshot validation history is incomplete")
    for actual, saved in zip(groups, validation):
        for key in ("clean_r", "clean_f", "q30_r", "q30_f", "worst"):
            expected = saved["metrics"]["worst_group" if key == "worst" else key]
            if not np.isclose(float(actual[key]), float(expected)):
                raise ValueError("Validation CSV differs from latest snapshot")
    del state
    best_path = folder/"best.pth"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    metadata = best["metadata"]
    from tools.evaluate_ablation_v2 import parse_val_groups_csv
    selected = parse_val_groups_csv(folder/"val_groups.csv", int(metadata["epoch"]))
    if not metadata.get("is_best"):
        raise ValueError("Selected checkpoint is not an official best")
    for key in ("clean_r", "clean_f", "q30_r", "q30_f", "worst_group"):
        if not np.isclose(float(metadata[key])*100, selected[key]):
            raise ValueError("Best checkpoint metadata differs from validation log")
    if metadata["calibration_sha256"] != context["calibration_file_sha256"]:
        raise ValueError("Best checkpoint calibration mismatch")
    del best
    proof = {"schema": "ablation-run/v1", "evidence_level": "measured", "epochs": epochs,
             "selected_epoch": selected["selected_epoch"], "steps_per_epoch": steps,
             "source_commit": source_commit, "context": context, "best_name": "best.pth",
             "files": {name: sha256_file(folder/name) for name in
                       ("best.pth", "checkpoint_latest.pth", "train_history.csv", "val_groups.csv")}}
    (folder/"run_verification.json").write_text(json.dumps(proof, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data_root", "assets", "cfe_dir", "calibration_split", "evaluation_split", "run_root", "backup_root"):
        parser.add_argument("--"+key, required=True, type=Path)
    parser.add_argument("--calibration", type=Path, default=ROOT/"DATASET/logs/phase7/calibration/baseline.json")
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.run_root == args.backup_root:
        raise ValueError("Runtime and backup directories must differ")
    if not torch.cuda.is_available():
        raise RuntimeError("Real ablation requires CUDA")
    # Earlier full-v2 measurement exceeded 18 GiB; 14 GiB alone is not readiness.
    if torch.cuda.mem_get_info()[0] < 20*1024**3:
        raise RuntimeError("Require >=20 GiB free for this unchanged config; no automatic batch/AMP changes")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "configs", "tools", "tests", "notebooks"], cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError("Publish/checkout the fixed source first; source tree must be clean")
    templates = audit_configs()
    expected_ids, _ = load_partition(args.evaluation_split, args.calibration_split, args.calibration)
    from tools.eval_communityforensics import scan_cfe_dataset
    from tools.ablation_v2_common import filter_records
    filter_records(scan_cfe_dataset(str(args.cfe_dir)), expected_ids)
    for name in NAMES:
        folder, backup = args.run_root/name, args.backup_root/name
        folder.mkdir(parents=True, exist_ok=True); backup.mkdir(parents=True, exist_ok=True)
        config = templates[name]
        config["model"].update(clip_path=str(args.assets/"ViT-L-14.pt"),
                               checkpoint_init=str(args.assets/"fatformer_4class_ckpt.pth"),
                               teacher_checkpoint=str(args.assets/"fatformer_4class_ckpt.pth"))
        config["dataset"].update(data_root=str(args.data_root),
            manifest=str(args.data_root/"DATASET/manifests/data_manifest_v2.json"),
            isolation_report=str(args.data_root/"DATASET/logs/phase7/data_isolation_report.json"),
            audit_config=str(args.data_root/"configs/phase7_data_v2.yaml"),
            degradation_config=str(ROOT/"configs/degradation_v2.yaml"))
        config["training"].update(save_dir=str(folder), backup_dir=str(backup), save_name="best.pth",
            history_csv=str(folder/"train_history.csv"), calibration_file=str(args.calibration))
        path = folder/"run_config.yaml"
        candidate = folder/"candidate_config.yaml"
        candidate.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        context = context_for(candidate)
        identity = {"source_commit": commit, "context": context, "config_hash": canonical_hash(config)}
        for directory in (folder, backup):
            marker = directory/"run_identity.json"
            if marker.exists() and read_json(marker) != identity:
                raise ValueError("Saved experiment identity changed; select a new run_root/backup_root")
            if not marker.exists() and (list(directory.glob("*.pth")) or list(directory.glob("*.csv"))):
                raise ValueError("Legacy run without identity; choose a fresh experiment")
            marker.write_text(json.dumps(identity, indent=2)+"\n", encoding="utf-8")
        os.replace(candidate, path)
        shutil.copy2(path, backup/path.name)
        (backup/"source_commit.txt").write_text(commit+"\n", encoding="ascii")
        preflight = folder/"preflight_train.json"
        run_logged([sys.executable, "tools/preflight_phase7_train.py", "--config", path, "--output", preflight], folder, backup)
        if not read_json(preflight).get("ready_to_train"):
            raise RuntimeError("Preflight is not ready")
        shutil.copy2(preflight, backup/preflight.name)
        latest = restore_latest(folder, backup, context)
        benchmark = folder/"train_benchmark_100steps.json"
        # Measure on the current GPU, including a resumed session; never reuse another GPU's timing.
        run_logged([sys.executable, "-m", "src.training.train_crossdomain", "--config", path,
             "--benchmark_steps", "100", "--benchmark_warmup", "5", "--output", benchmark,
             "--save_dir", folder/"benchmark_scratch"], folder, backup)
        measurement = read_json(benchmark)
        if measurement["evidence_level"] != "measured" or measurement["gpu_name"] != torch.cuda.get_device_name():
            raise ValueError("Benchmark evidence/GPU mismatch")
        shutil.copy2(benchmark, backup/benchmark.name)
        command = [sys.executable, "-m", "src.training.train_crossdomain", "--config", path]
        if latest:
            command += ["--resume_from", latest]
        # Full-state resume also repairs CSV; when already complete it does not retrain epochs.
        run_logged(command, folder, backup)
        verify_run(folder, context, commit)
        run_logged([sys.executable, "tools/eval_communityforensics.py", "--checkpoint", folder/"best.pth",
             "--clip_path", args.assets/"ViT-L-14.pt", "--model_type", "anchored", "--model_name", name,
             "--condition", "all", "--data_dir", args.cfe_dir, "--output_dir", folder,
             "--calibration", args.calibration, "--calibration_split", args.calibration_split,
             "--evaluation_split", args.evaluation_split, "--degrade_order", "post_resize", "--seed", "42"], folder, backup)
        for generated in folder.iterdir():
            if generated.is_file() and generated.suffix in (".json", ".csv", ".yaml", ".npz", ".md"):
                shutil.copy2(generated, backup/generated.name)
    run([sys.executable, "tools/evaluate_ablation_v2.py", "--experiment_dir", args.run_root,
         "--output_dir", args.run_root/"summary"])
    for generated in (args.run_root/"summary").iterdir():
        shutil.copy2(generated, args.backup_root/generated.name)
    print("Ablation artifacts written. Review the limitations; this does not automatically pass G7.2.")


if __name__ == "__main__":
    main()
