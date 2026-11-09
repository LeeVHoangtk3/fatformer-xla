#!/usr/bin/env python3
"""Trích xuất Logits Thực Nghiệm trên Phân Vùng val_calib cho Checkpoint v2 (Task 7.15 / Task 7.4).

Chạy forward pass của checkpoint trên 400 ảnh val_calib độc lập và lưu file NPZ:
- Không dùng chung hoặc đổi tên NPZ của baseline.
- Ghi rõ checkpoint_sha256, manifest_sha256 và tiền xử lý chuẩn.
- Đầu ra là tệp NPZ sẵn sàng để nạp vào tools/calibrate_and_select.py.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.canonical_loader import extract_state_dict, infer_adapter_blocks, resolve_layout, load_canonical
from tools.calibration_data import sample_id, read_partition, validate_partitions


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def get_default_transform():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])


def find_calib_samples(data_dir: str, split_file: Optional[str] = None) -> List[Tuple[str, int, str]]:
    """Resolve every frozen identity exactly; missing/ambiguous files are errors."""
    if not split_file:
        raise ValueError("Measured calibration requires an explicit split_file")
    expected, targets = read_partition(split_file, "calibration")
    index = {}
    for path in Path(data_dir).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            continue
        try:
            sid = sample_id(path)
        except ValueError:
            continue
        if sid in index:
            raise ValueError(f"Ambiguous sample identity: {sid}")
        index[sid] = str(path.resolve())
    missing = set(expected) - set(index)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} frozen calibration samples: {sorted(missing)[:3]}")
    return [(index[sid], int(y), sid) for sid, y in zip(expected, targets)]


def extract_logits(
    checkpoint_path: str,
    data_dir: str,
    output_npz: str,
    split_file: Optional[str] = None,
    clip_path: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 32,
    test_dummy: bool = False,
    evaluation_split: Optional[str] = None,
    precision: str = "fp32"
) -> str:
    """Trích xuất logits và lưu file NPZ."""
    ckpt_sha = compute_sha256(checkpoint_path) if os.path.exists(checkpoint_path) else "dummy_sha"

    if test_dummy:
        print("[*] Chạy chế độ DUMMY: Tạo logits NPZ giả lập phục vụ test...")
        n_samples = 400
        logits = np.random.randn(n_samples, 2).astype(np.float32)
        targets = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.int64)
        sample_ids = np.array([f"dummy_sample_{i:04d}" for i in range(n_samples)])
        paths = np.array([f"/dummy/path/{sid}.png" for sid in sample_ids])

        os.makedirs(os.path.dirname(os.path.abspath(output_npz)), exist_ok=True)
        np.savez_compressed(
            output_npz,
            logits=logits,
            targets=targets,
            split="calibration",
            sample_ids=sample_ids,
            paths=paths,
            evidence_level="synthetic",
            checkpoint_sha256=ckpt_sha,
            manifest_sha256="dummy_manifest_sha",
            preprocessing="Resize(224) -> CenterCrop(224) -> ToTensor() -> Normalize",
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        print(f"[✓] Đã tạo dummy NPZ: {output_npz}")
        return output_npz

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if precision not in {"fp32","fp16"} or (precision == 'fp16' and torch.device(device).type != 'cuda'):
        raise ValueError('FP16 extraction requires a CUDA device; use explicit fp32 on CPU')
    if torch.device(device).type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if not evaluation_split:
        raise ValueError("Measured extraction requires evaluation_split to verify independence")
    samples = find_calib_samples(data_dir, split_file)
    partition = validate_partitions([s[2] for s in samples], np.array([s[1] for s in samples]),
                                    split_file, evaluation_split)
    if not samples:
        raise ValueError(f"Không tìm thấy mẫu ảnh hợp lệ nào trong '{data_dir}' (split_file={split_file}).")

    print(f"[*] Tìm thấy {len(samples)} mẫu hiệu chuẩn (Real: {sum(1 for s in samples if s[1] == 0)}, Fake: {sum(1 for s in samples if s[1] == 1)}).")

    # Nạp checkpoint qua load_canonical
    print(f"[*] Đang nạp checkpoint: {checkpoint_path}")
    raw_sd = extract_state_dict(checkpoint_path)
    blocks = infer_adapter_blocks(raw_sd)
    layout = resolve_layout(blocks)

    model_args = SimpleNamespace(
        arch="ViT-L/14",
        backbone="CLIP:ViT-L/14",
        num_classes=2,
        clip_path=clip_path if clip_path and os.path.exists(clip_path) else None,
        use_srm=True,
        use_gating=True,
        num_adapter_blocks=layout["n"],
        adapter_block_indices=layout["indices"]
    )

    del raw_sd
    model, report = load_canonical(checkpoint_path, model_args)
    model = model.to(device).eval()
    print("[✓] Nạp mô hình thành công. Đang tiến hành forward trích xuất logits...")

    transform = get_default_transform()
    all_logits = []
    all_targets = []
    all_sids = []
    all_paths = []

    dev_obj = torch.device(device)

    for idx in range(0, len(samples), batch_size):
        batch = samples[idx : idx + batch_size]
        tensors = []
        cur_targets = []
        cur_sids = []
        cur_paths = []

        for p, tgt, sid in batch:
            try:
                with Image.open(p) as img:
                    img_rgb = img.convert("RGB")
                    tensor = transform(img_rgb)
                    tensors.append(tensor)
                    cur_targets.append(tgt)
                    cur_sids.append(sid)
                    cur_paths.append(p)
            except Exception as e:
                raise ValueError(f"Cannot read frozen calibration image: {p}") from e

        if not tensors:
            continue

        batch_tensors = torch.stack(tensors, dim=0).to(dev_obj)

        with torch.no_grad():
            with torch.amp.autocast(dev_obj.type,dtype=torch.float16,
                                    enabled=dev_obj.type=='cuda' and precision=='fp16'):
                outputs = model(batch_tensors)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            out_np = outputs.cpu().float().numpy()

        all_logits.append(out_np)
        all_targets.extend(cur_targets)
        all_sids.extend(cur_sids)
        all_paths.extend(cur_paths)

    final_logits = np.concatenate(all_logits, axis=0)
    final_targets = np.array(all_targets, dtype=np.int64)
    final_sids = np.array(all_sids)
    final_paths = np.array(all_paths)

    manifest_sha = compute_sha256(split_file) if split_file and os.path.exists(split_file) else "no_split_file"

    os.makedirs(os.path.dirname(os.path.abspath(output_npz)), exist_ok=True)
    np.savez_compressed(
        output_npz,
        logits=final_logits,
        targets=final_targets,
        split="calibration",
        sample_ids=final_sids,
        paths=final_paths,
        evidence_level="measured",
        precision=precision,
        inference_batch_size=batch_size,
        torch_version=str(torch.__version__),
        partition_validation=json.dumps(partition, sort_keys=True),
        checkpoint_sha256=report.get("sha256", ckpt_sha),
        manifest_sha256=manifest_sha,
        preprocessing="Resize(224) -> CenterCrop(224) -> ToTensor() -> Normalize",
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    print(f"[✓] Đã xuất {len(final_targets)} logits đo thật ra tệp NPZ: {output_npz}")
    return output_npz


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Extract Calibration Logits from Checkpoint (Task 7.15)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn file .pth")
    parser.add_argument("--data_dir", type=str, default="DATASET/val_crossdomain", help="Thư mục dữ liệu validation")
    parser.add_argument("--split_file", type=str, default=None, help="File split phân vùng (NPZ/JSON)")
    parser.add_argument("--output_npz", type=str, default="DATASET/logs/phase7/logits_anchored_v2_clean_calib.npz", help="Đường dẫn lưu tệp NPZ")
    parser.add_argument("--clip_path", type=str, default="DATASET/pretrained/ViT-L-14.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_dummy", action="store_true", help="Chạy chế độ dummy test")
    parser.add_argument("--evaluation_split", help="Frozen evaluation NPZ/JSON; required for measured extraction")
    parser.add_argument("--precision", choices=['fp32','fp16'], default='fp32')
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"

    try:
        extract_logits(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_npz=args.output_npz,
            split_file=args.split_file,
            clip_path=args.clip_path,
            device=device,
            batch_size=args.batch_size,
            test_dummy=args.test_dummy,
            evaluation_split=args.evaluation_split,
            precision=args.precision
        )
    except Exception as e:
        print(f"[!] Thất bại trong quá trình trích xuất logits: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
