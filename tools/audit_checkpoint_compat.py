"""
tools/audit_checkpoint_compat.py
Task 7.2 (Phase 7 — D1-D3)
Audit cấu trúc checkpoint & kiểm tra tính tương thích layout adapter (schema: checkpoint_structure_audit/v1).
Uỷ thác extract_state_dict, infer_adapter_blocks, resolve_layout cho canonical_loader (nguồn sự thật duy nhất).
"""

import os
import sys
import json
import hashlib
import argparse
import datetime
import re
from typing import Dict, List, Any

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm root vào sys.path để import src.models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.canonical_loader import (
    extract_state_dict as canonical_extract,
    infer_adapter_blocks,
    resolve_layout,
)


def compute_sha256(filepath: str, chunk_size: int = 1024 * 1024 * 4) -> str:
    """Tính SHA-256 của file checkpoint theo từng chunk để tiết kiệm RAM."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def audit_single_checkpoint(filepath: str, total_layers: int = 24) -> Dict[str, Any]:
    """Audit một checkpoint; uỷ thác layout analysis cho canonical_loader."""
    print(f"\n[*] Đang audit: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tệp không tồn tại: {filepath}")

    size_bytes = os.path.getsize(filepath)
    print(f"    • Dung lượng: {size_bytes / (1024 * 1024):.2f} MB ({size_bytes} bytes)")

    print("    • Đang tính SHA-256...")
    sha256 = compute_sha256(filepath)
    print(f"    • SHA-256: {sha256[:16]}...")

    print("    • Đang tải state_dict lên CPU...")
    state_dict = canonical_extract(filepath)
    n_keys = len(state_dict)
    print(f"    • Số lượng tensor keys: {n_keys}")

    # 1. Adapter blocks và layout — uỷ thác hoàn toàn cho canonical_loader
    adapter_blocks_set = infer_adapter_blocks(state_dict)
    adapter_blocks = sorted(adapter_blocks_set)
    print(f"    • Khối adapter phát hiện ({len(adapter_blocks)} khối): {adapter_blocks}")

    candidate_n = [3, 4, 6, 8, 12, 24]
    blocks_dropped: Dict[str, List[int]] = {}
    for n in candidate_n:
        if total_layers % n != 0:
            blocks_dropped[str(n)] = list(adapter_blocks)
            continue
        s_n = {(i + 1) * (total_layers // n) - 1 for i in range(n)}
        blocks_dropped[str(n)] = sorted(adapter_blocks_set - s_n)

    expressible_n = [
        n for n in candidate_n
        if total_layers % n == 0
        and {(i + 1) * (total_layers // n) - 1 for i in range(n)} == adapter_blocks_set
    ]

    if adapter_blocks:
        info = resolve_layout(adapter_blocks_set, layers=total_layers)
        layout = {k: info[k] for k in ("n", "indices", "dropped", "partial")}
    else:
        layout = {"n": None, "indices": [], "dropped": [], "partial": True}

    # 2. Audit-specific: tensor count per block, SRM presence
    block_re = re.compile(r"(?:resblocks|transformer\.resblocks)\.(\d+)\.", re.IGNORECASE)
    block_keys: Dict[int, List[str]] = {}
    for k in state_dict:
        m = block_re.search(k)
        if m:
            block_keys.setdefault(int(m.group(1)), []).append(k)

    has_srm_per_block: Dict[str, bool] = {}
    tensor_count_per_block: Dict[str, int] = {}
    for b in adapter_blocks:
        keys_b = block_keys.get(b, [])
        adapter_k = [k for k in keys_b if "adapter" in k.lower() or "srm" in k.lower()]
        has_srm_per_block[str(b)] = any("srm" in k.lower() for k in keys_b)
        tensor_count_per_block[str(b)] = len(adapter_k)

    # 3. Alias equality (audit-specific)
    image_enc_keys = [k for k in state_dict if k.startswith("image_encoder.")]
    eq = cmp = 0
    for k in image_enc_keys:
        clip_k = "clip_model.visual." + k[len("image_encoder."):]
        if clip_k in state_dict:
            cmp += 1
            t1, t2 = state_dict[k], state_dict[clip_k]
            if torch.is_tensor(t1) and torch.is_tensor(t2) and t1.shape == t2.shape and torch.equal(t1, t2):
                eq += 1
    alias_equality = {"image_encoder_vs_clip": round(eq / cmp if cmp else 1.0, 4)}

    # 4. Has gating & top-level SRM (audit-specific)
    has_gating = any(
        re.search(r"(?:gating|gate|alpha)", k, re.IGNORECASE) and "positional" not in k
        for k in state_dict
    )
    has_top_level_srm = any("srm" in k.lower() and not block_re.search(k) for k in state_dict)

    # 5. Dead keys (audit-specific)
    valid_pat = re.compile(
        r"^(?:clip_model|image_encoder|text_encoder|visual|transformer|srm|head|fc|"
        r"classifier|logit_scale|text_projection|language_guided_alignment|"
        r"prompt_learner|text_guided_interactor|gating|linear|norm)",
        re.IGNORECASE,
    )
    dead_keys = [k for k in state_dict if not valid_pat.match(k)]

    return {
        "path": filepath,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "n_keys": n_keys,
        "adapter_blocks": adapter_blocks,
        "expressible_n": expressible_n,
        "layout": layout,
        "blocks_dropped": blocks_dropped,
        "has_srm_per_block": has_srm_per_block,
        "tensor_count_per_block": tensor_count_per_block,
        "alias_equality": alias_equality,
        "dead_keys": dead_keys,
        "has_gating": has_gating,
        "has_top_level_srm": has_top_level_srm,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit Checkpoint Compatibility (Task 7.2)")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Danh sách các tệp checkpoint .pth cần audit")
    parser.add_argument("--layers", type=int, default=24, help="Số layer transformer (mặc định 24 cho ViT-L/14)")
    parser.add_argument("--output", type=str, default="logs/phase7/checkpoint_structure_audit.json", help="Đường dẫn file JSON đầu ra")
    args = parser.parse_args()

    print("=" * 80)
    print("FATFORMER-XLA: AUDIT CẤU TRÚC CHECKPOINT & ĐO LƯỜNG TƯƠNG THÍCH (TASK 7.2)")
    print(f"Thời gian: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"Số layer: {args.layers}")
    print("=" * 80)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    results = []
    for ckpt_path in args.checkpoints:
        try:
            r = audit_single_checkpoint(ckpt_path, total_layers=args.layers)
            results.append(r)
        except Exception as e:
            print(f"[!] Lỗi khi audit {ckpt_path}: {e}")
            import traceback
            traceback.print_exc()

    audit_doc = {
        "schema": "checkpoint_structure_audit/v1",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "checkpoints": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(audit_doc, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"[✓] HOÀN TẤT KIỂM TOÁN! Đã lưu kết quả tại: {args.output}")
    print("=" * 80)

    print("\n| Checkpoint | Số tensor | Khối adapter | n khả biểu | Layout n | Partial? | SRM? | Gating? | Dead keys |")
    print("| :--- | ---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        fname = os.path.basename(r["path"])
        n_val = str(r["layout"]["n"]) if r["layout"]["n"] is not None else "None"
        partial_str = "YES" if r["layout"]["partial"] else "NO"
        srm_str = "YES" if any(r["has_srm_per_block"].values()) or r["has_top_level_srm"] else "NO"
        gating_str = "YES" if r["has_gating"] else "NO"
        print(f"| {fname} | {r['n_keys']:,} | {r['adapter_blocks']} | {r['expressible_n']} | {n_val} | {partial_str} | {srm_str} | {gating_str} | {len(r['dead_keys'])} |")


if __name__ == "__main__":
    main()
