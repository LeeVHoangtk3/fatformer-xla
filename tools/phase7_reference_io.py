"""Logged, repeatable reference indexing for Task 7.7 notebooks."""
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from hash_utils import load_hash_index, save_hash_index, sha256_file


def run_logged(command, root):
    root = Path(root)
    log_dir = root / "DATASET/logs/phase7/commands"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".log")
    command = [str(x) for x in command]
    tail = deque(maxlen=40)
    print("Running:", " ".join(command), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + repr(command) + "\n")
        with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1, env=env) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                tail.append(line)
            code = process.wait()
    if code:
        raise RuntimeError(f"Command failed (exit {code}). Full log: {log_path}\n" + "".join(tail))
    return log_path


def index_reference(root, role, tars, out, manifest=None, workers=4):
    """Reuse only a complete index with unchanged inputs, tool code and index bytes.

    Archive identity uses resolved path, size and mtime; this cache is an IO
    optimization, not a cryptographic attestation of the original tar archive.
    """
    root, out = Path(root).resolve(), Path(out)
    if not out.is_absolute():
        out = root / out
    inputs = []
    for path in tars:
        path = Path(path).resolve()
        stat = path.stat()
        inputs.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    signature = {"role": role, "archives": inputs, "schema": 1,
                 "manifest_sha256": sha256_file(manifest) if manifest else None,
                 "tools": {name: sha256_file(root / "tools" / name) for name in
                           ("index_test_sets.py", "hash_utils.py", "build_generator_denylist.py")}}
    marker = Path(str(out) + ".inputs.json")
    if out.is_file() and marker.is_file():
        try:
            cached = json.loads(marker.read_text(encoding="utf-8"))
            payload = load_hash_index(str(out))
            if (cached.get("inputs") == signature and cached.get("index_sha256") == sha256_file(out)
                    and payload.get("metadata", {}).get("complete") is True
                    and payload["metadata"].get("role") == role
                    and payload["metadata"].get("n_images") == len(payload.get("images", []))):
                print(f"Reusing complete {role} index: {out}", flush=True)
                return out
        except (OSError, ValueError, KeyError, EOFError):
            pass
    command = [sys.executable, "tools/index_test_sets.py", "--role", role,
               "--tar", *[str(x) for x in tars], "--workers", str(workers), "--out", str(out)]
    if manifest:
        command += ["--manifest", str(manifest)]
    run_logged(command, root)
    payload = load_hash_index(str(out))
    if payload.get("metadata", {}).get("complete") is not True:
        raise RuntimeError(f"Index process did not produce complete evidence: {out}")
    save_hash_index({"inputs": signature, "index_sha256": sha256_file(out)}, str(marker))
    return out
