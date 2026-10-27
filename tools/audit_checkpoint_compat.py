"""
tools/audit_checkpoint_compat.py
Task 7.1 (Phase 7 — D1-D2)
Audit cấu trúc checkpoint & kiểm tra tính tương thích layout adapter (schema: checkpoint_structure_audit/v1).
"""

import os
import sys
import json
import hashlib
import argparse
import datetime
import re
from typing import Dict, List, Any, Optional

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


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


def get_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Trích xuất state_dict phẳng từ các dạng bọc khác nhau của .pth."""
    if isinstance(ckpt, dict):
        for candidate_key in ["model", "state_dict", "model_state_dict", "model_state", "net"]:
            if candidate_key in ckpt and isinstance(ckpt[candidate_key], dict):
                return ckpt[candidate_key]
        return ckpt
    raise ValueError(f"Không nhận diện được định dạng checkpoint: {type(ckpt)}")


def get_layout_indices(n: int, total_layers: int = 24) -> List[int]:
    """Tính danh sách index khối theo công thức (i + 1) * (total_layers // n) - 1."""
    if total_layers % n != 0:
        return []
    split = total_layers // n
    return [(i + 1) * split - 1 for i in range(n)]


def audit_single_checkpoint(filepath: str, total_layers: int = 24) -> Dict[str, Any]:
    print(f"\n[*] Đang audit: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tệp không tồn tại: {filepath}")

    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    print(f"    • Dung lượng: {size_mb:.2f} MB ({size_bytes} bytes)")

    print("    • Đang tính SHA-256...")
    sha256 = compute_sha256(filepath)
    print(f"    • SHA-256: {sha256[:16]}...")

    print("    • Đang tải state_dict lên CPU...")
    raw_ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
    state_dict = get_state_dict(raw_ckpt)
    n_keys = len(state_dict)
    print(f"    • Số lượng tensor keys: {n_keys}")

    # 1. Phát hiện các adapter blocks
    adapter_regex = re.compile(r"(?:resblocks|transformer\.resblocks)\.(\d+)\..*(?:adapter|adaptor)", re.IGNORECASE)
    block_regex = re.compile(r"(?:resblocks|transformer\.resblocks)\.(\d+)\.", re.IGNORECASE)

    adapter_blocks_set = set()
    block_tensor_keys: Dict[int, List[str]] = {}

    for k in state_dict.keys():
        m_block = block_regex.search(k)
        if m_block:
            b_idx = int(m_block.group(1))
            if b_idx not in block_tensor_keys:
                block_tensor_keys[b_idx] = []
            block_tensor_keys[b_idx].append(k)

        m_adapter = adapter_regex.search(k)
        if m_adapter:
            b_idx = int(m_adapter.group(1))
            adapter_blocks_set.add(b_idx)

    adapter_blocks = sorted(list(adapter_blocks_set))
    print(f"    • Khối adapter phát hiện ({len(adapter_blocks)} khối): {adapter_blocks}")

    # 2. Candidate layouts n in {3, 4, 6, 8, 12, 24}
    candidate_n = [3, 4, 6, 8, 12, 24]
    blocks_dropped: Dict[str, List[int]] = {}

    for n in candidate_n:
        s_n = set(get_layout_indices(n, total_layers))
        dropped = sorted(list(adapter_blocks_set - s_n))
        blocks_dropped[str(n)] = dropped

    # expressible_n: chỉ chấp nhận layout n biểu diễn trọn vẹn tập adapter_blocks
    exact_n = [n for n in candidate_n if set(get_layout_indices(n, total_layers)) == adapter_blocks_set]
    expressible_n = exact_n

    if exact_n:
        best_n = exact_n[0]
        layout = {
            "n": best_n,
            "indices": get_layout_indices(best_n, total_layers),
            "dropped": [],
            "partial": False
        }
    else:
        # Không có n nào biểu diễn được hoàn toàn (như Gen 1)
        layout = {
            "n": None,
            "indices": adapter_blocks,
            "dropped": blocks_dropped.get("8", []),
            "partial": True
        }

    # 3. has_srm_per_block & tensor_count_per_block
    has_srm_per_block: Dict[str, bool] = {}
    tensor_count_per_block: Dict[str, int] = {}

    for b in adapter_blocks:
        keys_in_b = block_tensor_keys.get(b, [])
        adapter_keys_in_b = [k for k in keys_in_b if "adapter" in k.lower() or "srm" in k.lower()]
        has_srm = any("srm" in k.lower() for k in keys_in_b)
        has_srm_per_block[str(b)] = has_srm
        tensor_count_per_block[str(b)] = len(adapter_keys_in_b)

    # 4. alias_equality: so sánh image_encoder.X vs clip_model.visual.X
    image_enc_keys = [k for k in state_dict.keys() if k.startswith("image_encoder.")]
    equal_count = 0
    compared_count = 0

    for k in image_enc_keys:
        suffix = k[len("image_encoder."):]
        clip_k = f"clip_model.visual.{suffix}"
        if clip_k in state_dict:
            compared_count += 1
            t1 = state_dict[k]
            t2 = state_dict[clip_k]
            if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                if t1.shape == t2.shape and torch.equal(t1, t2):
                    equal_count += 1

    alias_ratio = (equal_count / compared_count) if compared_count > 0 else 1.0
    alias_equality = {"image_encoder_vs_clip": round(alias_ratio, 4)}

    # 5. has_gating & has_top_level_srm
    has_gating = any(
        re.search(r"(?:gating|gate|alpha)", k, re.IGNORECASE) and not re.search(r"positional", k, re.IGNORECASE)
        for k in state_dict.keys()
    )
    has_top_level_srm = any(
        "srm" in k.lower() and not block_regex.search(k)
        for k in state_dict.keys()
    )

    # 6. dead_keys: kiểm tra các key không thuộc kiến trúc hợp lệ của FatFormer
    valid_key_patterns = [
        r"^(?:clip_model|image_encoder|text_encoder|visual|transformer|srm|head|fc|classifier|logit_scale|text_projection|language_guided_alignment|prompt_learner|text_guided_interactor|gating|linear|norm)",
    ]
    dead_keys = []
    for k in state_dict.keys():
        if not any(re.search(p, k, re.IGNORECASE) for p in valid_key_patterns):
            dead_keys.append(k)

    # Giải phóng RAM
    del raw_ckpt
    del state_dict

    res = {
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
        "has_top_level_srm": has_top_level_srm
    }
    return res


def main():
    parser = argparse.ArgumentParser(description="Audit Checkpoint Compatibility (Task 7.1)")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Danh sách các tệp checkpoint .pth cần audit")
    parser.add_argument("--layers", type=int, default=24, help="Số layer transformer (mặc định 24 cho ViT-L/14)")
    parser.add_argument("--output", type=str, default="logs/phase7/checkpoint_structure_audit.json", help="Đường dẫn file JSON đầu ra")
    args = parser.parse_args()

    print("=" * 80)
    print("FATFORMER-XLA: AUDIT CẤU TRÚC CHECKPOINT & ĐO LƯỜNG TƯƠNG THÍCH (TASK 7.1)")
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
        "checkpoints": results
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(audit_doc, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"[✓] HOÀN TẤT KIỂM TOÁN! Đã lưu kết quả tại: {args.output}")
    print("=" * 80)

    # In bảng tóm tắt
    print("\n| Checkpoint | Số tensor | Khối adapter | n khả biểu | Layout n | Partial? | SRM? | Gating? | Dead keys |")
    print("| :--- | ---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        fname = os.path.basename(r["path"])
        n_val = str(r["layout"]["n"]) if r["layout"]["n"] is not None else "None"
        partial_str = "YES" if r["layout"]["partial"] else "NO"
        srm_str = "YES" if any(r["has_srm_per_block"].values()) or r["has_top_level_srm"] else "NO"
        gating_str = "YES" if r["has_gating"] else "NO"
        dead_str = str(len(r["dead_keys"]))
        exp_n_str = str(r["expressible_n"])
        print(f"| {fname} | {r['n_keys']:,} | {r['adapter_blocks']} | {exp_n_str} | {n_val} | {partial_str} | {srm_str} | {gating_str} | {dead_str} |")


if __name__ == "__main__":
    main()

