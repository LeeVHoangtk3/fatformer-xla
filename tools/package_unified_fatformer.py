#!/usr/bin/env python3
"""Công cụ đóng gói UnifiedFatFormerXLA (Task 7.15).

Tích hợp trọng số backbone và siêu dữ liệu (metadata) thành một tệp checkpoint duy nhất.
Metadata bao gồm:
- layout adapter (n=3, indices=[7, 15, 23])
- thông số hiệu chuẩn xác suất (T, T*) từ calibration file chính chủ
- thông số router (nếu có win proof hợp lệ)
- git commit và SHA-256 đối chiếu
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional

import torch

# Hỗ trợ nạp module từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.canonical_loader import extract_state_dict, infer_adapter_blocks, resolve_layout


def compute_file_sha256(filepath: str) -> str:
    """Tính toán mã băm SHA-256 của tệp."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def get_git_commit() -> str:
    """Lấy commit hash hiện tại của git."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown_commit"


def validate_calibration_against_checkpoint(calib: Dict[str, Any], ckpt_sha: str) -> None:
    """Kiểm tra tính nhất quán giữa calibration và checkpoint.
    
    Ngăn chặn việc gán ghép calibration sai nguồn hoặc sửa hash metadata.
    """
    calib_ckpt_sha = calib.get("checkpoint", {}).get("sha256")
    if not calib_ckpt_sha:
        raise ValueError("[!] Lỗi: Tệp calibration thiếu trường 'checkpoint.sha256'.")
    
    if calib_ckpt_sha != ckpt_sha:
        raise ValueError(
            f"[!] Lỗi nghiêm trọng: Calibration thuộc checkpoint khác!\n"
            f"  - Checkpoint SHA-256   : {ckpt_sha}\n"
            f"  - Calibration CKPT SHA : {calib_ckpt_sha}\n"
            f"Không được giả mạo hash hoặc dùng calibration của baseline cho v2!"
        )


def package_checkpoint(
    checkpoint_path: str,
    output_path: str,
    calibration_path: Optional[str] = None,
    router_path: Optional[str] = None,
    router_ablation_path: Optional[str] = None,
    metadata_json_path: Optional[str] = None,
    allow_partial: bool = False,
    allow_uncalibrated_debug: bool = False
) -> Dict[str, Any]:
    """Đóng gói checkpoint và metadata thành file .pth thống nhất."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint nguồn: {checkpoint_path}")

    print(f"[*] Đang đọc checkpoint nguồn: {checkpoint_path}")
    raw_payload = torch.load(checkpoint_path, map_location="cpu")
    src_sd = extract_state_dict(raw_payload)
    ckpt_sha = compute_file_sha256(checkpoint_path)
    print(f"[✓] Checkpoint SHA-256: {ckpt_sha}")

    # Nhận diện layout adapter và kiểm tra tính hoàn chỉnh
    blocks = infer_adapter_blocks(src_sd)
    layout_info = resolve_layout(blocks)
    print(f"[✓] Layout adapter nhận diện: n={layout_info.get('n')}, indices={layout_info.get('indices')}, partial={layout_info.get('partial')}")
    if layout_info.get("partial", False) and not allow_partial:
        raise ValueError(
            f"[!] Lỗi nghiêm trọng: Checkpoint adapter layout không đầy đủ (partial=True): "
            f"blocks={sorted(blocks)}, nearest_n={layout_info.get('n')}, dropped={layout_info.get('dropped')}. "
            f"Bị từ chối đóng gói artifact chính thức (chỉ cho phép nếu bật cờ --allow_partial trong forensic analysis)!"
        )

    # Đọc và xác thực calibration: BẮT BUỘC có calibration hợp lệ khớp SHA-256
    calib_dict = None
    if not calibration_path or not os.path.exists(calibration_path):
        if not allow_uncalibrated_debug:
            raise ValueError(
                "[!] Lỗi nghiêm trọng: Đóng gói artifact chính thức bắt buộc phải truyền --calibration <file.json> "
                "khớp SHA-256 với checkpoint nguồn. (Chỉ bỏ qua nếu bật --allow_uncalibrated_debug)."
            )
        else:
            print("[!] CẢNH BÁO: Đang đóng gói không kèm calibration (--allow_uncalibrated_debug)!")
    else:
        print(f"[*] Đang nạp calibration: {calibration_path}")
        with open(calibration_path, "r", encoding="utf-8") as f:
            calib_dict = json.load(f)
        
        validate_calibration_against_checkpoint(calib_dict, ckpt_sha)
        print("[✓] Xác thực calibration chính chủ: Khớp SHA-256 100%.")

    # Đọc và xác thực router (nếu có)
    router_dict = None
    router_proof = None
    router_enabled = False
    if router_path and os.path.exists(router_path):
        print(f"[*] Đang nạp router params: {router_path}")
        with open(router_path, "r", encoding="utf-8") as f:
            router_dict = json.load(f)
        if router_ablation_path and os.path.exists(router_ablation_path):
            from src.models.unified_fatformer import router_is_approved
            with open(router_ablation_path, "r", encoding="utf-8") as f:
                router_proof = json.load(f)
            router_enabled = router_is_approved(router_dict, router_proof)
            print(f"[✓] Trạng thái router: router_enabled={router_enabled}")

    git_commit = get_git_commit()
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    metadata = {
        "schema": "unified_fatformer/v1",
        "layout": layout_info,
        "calibration": calib_dict,
        "router": router_dict,
        "router_ablation": router_proof,
        "router_enabled": router_enabled,
        "git_commit": git_commit,
        "source_checkpoint_sha256": ckpt_sha,
        "created_at": created_at
    }

    packaged_payload = {
        "model": src_sd,
        "metadata": metadata
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"[*] Đang lưu checkpoint đóng gói ra: {output_path}")
    torch.save(packaged_payload, output_path)

    if metadata_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(metadata_json_path)), exist_ok=True)
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"[✓] Đã xuất metadata JSON ra: {metadata_json_path}")

    # Tự kiểm tra tính toàn vẹn (Self-Verification)
    print("[*] Đang thực hiện tự kiểm chứng trọng số...")
    reloaded = torch.load(output_path, map_location="cpu")
    assert "model" in reloaded and "metadata" in reloaded, "Lỗi: Payload đóng gói thiếu model hoặc metadata"
    reloaded_sd = reloaded["model"]
    assert len(reloaded_sd) == len(src_sd), "Lỗi: Số lượng tensor không khớp"
    for k in src_sd:
        assert k in reloaded_sd, f"Lỗi: Thiếu tensor {k}"
        assert torch.equal(src_sd[k], reloaded_sd[k]), f"Lỗi: Tensor {k} bị biến đổi giá trị sau đóng gói"
    print("[✓] TỰ KIỂM CHỨNG THÀNH CÔNG: Trọng số khớp 100% không bị thay đổi.")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="FatFormer-XLA Checkpoint Packaging Tool (Task 7.15)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn file .pth nguồn")
    parser.add_argument("--output", type=str, default="DATASET/checkpoints/phase7/unified_fatformer_v2.pth", help="Đường dẫn file .pth đích")
    parser.add_argument("--calibration", type=str, default=None, help="Đường dẫn file calibration JSON chính chủ")
    parser.add_argument("--router_params", type=str, default=None, help="Đường dẫn file router_params.json")
    parser.add_argument("--router_ablation", type=str, default=None, help="Đường dẫn file router_ablation.json")
    parser.add_argument("--metadata_out", type=str, default=None, help="Đường dẫn file metadata JSON độc lập")
    parser.add_argument("--allow_partial", action="store_true", help="Cho phép layout adapter không đầy đủ (chỉ dùng cho forensic analysis)")
    parser.add_argument("--allow_uncalibrated_debug", action="store_true", help="Bỏ qua bắt buộc calibration (chỉ dùng cho debug nội bộ)")
    args = parser.parse_args()

    try:
        package_checkpoint(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            calibration_path=args.calibration,
            router_path=args.router_params,
            router_ablation_path=args.router_ablation,
            metadata_json_path=args.metadata_out,
            allow_partial=args.allow_partial,
            allow_uncalibrated_debug=args.allow_uncalibrated_debug
        )
    except Exception as e:
        print(f"[!] Thất bại trong quá trình đóng gói: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
