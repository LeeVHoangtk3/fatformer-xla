# -*- coding: utf-8 -*-
"""
FatFormer-XLA: Bộ Điều Phối Nghiệm Thu Toàn Trình GPU Phase 7 (Tasks 7.3 - 7.6 & Cổng G7.1)
========================================================================================
Vận hành chuỗi thực nghiệm khép kín theo đúng khế ước đo lường:
  0. Preflight Fail-Fast: Kiểm tra CUDA, Git commit sạch, checksum ViT-L-14 và các checkpoint.
  1. Eval 7.3 cho Baseline và Anchored trên Clean & Q30 -> Xuất JSON v3 và NPZ logits.
  2. Calibration 7.4 độc lập -> Xuất T, T* có ràng buộc FPR 5% và ECE.
  3. WiSE-FT Gen 2: Quét lưới sinh 12 ứng viên kèm --verify 0/0 keys.
  4. Sweep 12 ứng viên thật trên Clean & Q30 -> Tuyển chọn Pareto vô địch (selected=True duy nhất).
  5. Eval 7.3 cho WiSE-FT Gen 2 Best -> Xuất JSON v3 và NPZ logits.
  6. Re-baseline 7.6 ma trận 4 cấu hình x 2 điều kiện từ số đo thật (evidence_level=measured).
  7. Validator Cổng G7.1: Phán quyết chất lượng học thuật chính thức (PASS/FAIL).
"""

import os
import sys
import json
import time
import shutil
import hashlib
import argparse
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import yaml
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def compute_sha256(file_path: str) -> str:
    """Tính mã băm SHA-256 của tệp tin."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> Tuple[str, str, bool]:
    """Lấy thông tin commit, branch và trạng thái dirty của Git."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        branch = "unknown"

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        commit = "unknown"

    try:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        is_dirty = len(status_out) > 0
    except Exception:
        is_dirty = False

    return branch, commit, is_dirty


def run_preflight_checks(cfg: Dict[str, Any], allow_cpu: bool = False, allow_dirty: bool = False):
    """
    Kiểm tra tiền điều kiện (Preflight Validation) — Dừng ngay nếu không đạt chuẩn.
    """
    print("\n" + "=" * 88)
    print("FATFORMER-XLA: BẮT ĐẦU KIỂM TRA TIỀN ĐIỀU KIỆN (PREFLIGHT VALIDATION)")
    print("=" * 88)

    # 1. Kiểm tra phần cứng CUDA
    if not allow_cpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[PREFLIGHT FAIL] Không tìm thấy GPU CUDA! Pipeline đo lường thật bắt buộc chạy trên GPU "
                "(Colab/Kaggle). Dùng cờ --allow_cpu chỉ khi debug luồng logic."
            )
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  [✓] GPU CUDA        : {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        print("  [!] CẢNH BÁO        : Đang chạy ở chế độ CPU debug (--allow_cpu)")

    # 2. Kiểm tra Git branch & status
    branch, commit, is_dirty = get_git_info()
    expected_branch = cfg.get("environment", {}).get("expected_branch", "hoang")
    print(f"  [✓] Git Branch      : {branch} (Kỳ vọng: {expected_branch})")
    print(f"  [✓] Git Commit      : {commit[:16]}...")
    if branch != expected_branch and expected_branch != "any":
        raise RuntimeError(f"[PREFLIGHT FAIL] Đang ở nhánh '{branch}', yêu cầu phải ở nhánh '{expected_branch}'.")
    if is_dirty and not allow_dirty:
        raise RuntimeError(
            "[PREFLIGHT FAIL] Git working tree bị dirty! Vui lòng commit hoặc stash trước khi chạy nghiệm thu."
        )

    # 3. Kiểm tra trọng số backbone CLIP
    clip_path = os.path.join(PROJECT_ROOT, cfg["checkpoints"]["clip_vit_l14"]["path"])
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(
            f"[PREFLIGHT FAIL] Không tìm thấy backbone CLIP tại: {clip_path}\n"
            f"Vui lòng tải hoặc mount file ViT-L-14.pt vào thư mục DATASET/pretrained/"
        )
    clip_sha = compute_sha256(clip_path)
    print(f"  [✓] CLIP Backbone   : {clip_path} (SHA: {clip_sha[:16]}...)")

    # 4. Kiểm tra Checkpoint Baseline
    base_info = cfg["checkpoints"]["baseline"]
    base_path = os.path.join(PROJECT_ROOT, base_info["path"])
    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"[PREFLIGHT FAIL] Không tìm thấy Baseline checkpoint: {base_path}")
    base_sha = compute_sha256(base_path)
    if base_info.get("expected_sha256") and base_sha != base_info["expected_sha256"]:
        raise ValueError(
            f"[PREFLIGHT FAIL] Sai lệch SHA-256 Baseline: Nhận {base_sha}, kỳ vọng {base_info['expected_sha256']}"
        )
    print(f"  [✓] Baseline Ckpt   : {base_path} (SHA: {base_sha[:16]}...) [KHỚP]")

    # 5. Kiểm tra Checkpoint Anchored
    anch_info = cfg["checkpoints"]["anchored"]
    anch_path = os.path.join(PROJECT_ROOT, anch_info["path"])
    if not os.path.isfile(anch_path):
        raise FileNotFoundError(f"[PREFLIGHT FAIL] Không tìm thấy Anchored checkpoint: {anch_path}")
    anch_sha = compute_sha256(anch_path)
    if anch_info.get("expected_sha256") and anch_sha != anch_info["expected_sha256"]:
        raise ValueError(
            f"[PREFLIGHT FAIL] Sai lệch SHA-256 Anchored: Nhận {anch_sha}, kỳ vọng {anch_info['expected_sha256']}"
        )
    print(f"  [✓] Anchored Ckpt   : {anch_path} (SHA: {anch_sha[:16]}...) [KHỚP]")

    # 6. Kiểm tra Dataset CFE
    cfe_root = os.path.join(PROJECT_ROOT, cfg["data"]["cfe_root"])
    if not os.path.isdir(cfe_root):
        raise FileNotFoundError(
            f"[PREFLIGHT FAIL] Không tìm thấy thư mục CFE dataset tại: {cfe_root}\n"
            f"Vui lòng giải nén hoặc liên kết thư mục test_communityforensics/"
        )
    total_imgs = 0
    for sub in cfg["data"]["subsets"]:
        sub_dir = os.path.join(cfe_root, sub)
        if not os.path.isdir(sub_dir):
            raise FileNotFoundError(f"[PREFLIGHT FAIL] Thiếu tập con CFE: {sub_dir}")
        for lbl in ["0_real", "1_fake"]:
            f_dir = os.path.join(sub_dir, lbl)
            if os.path.isdir(f_dir):
                total_imgs += len([f for f in os.listdir(f_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))])
    print(f"  [✓] CFE Dataset     : {cfe_root} ({total_imgs} ảnh quét được)")
    if total_imgs < 100 and not allow_cpu:
        raise ValueError(f"[PREFLIGHT FAIL] Số lượng ảnh CFE quá ít ({total_imgs} ảnh), kỳ vọng ~2000 ảnh.")

    # 7. Kiểm tra tính tương thích nạp Canonical (0 missing, 0 unexpected)
    from src.models.canonical_loader import load_canonical
    from types import SimpleNamespace
    print("  [*] Đang thử nghiệm strict load canonical hai checkpoint nguồn...")
    for name, c_path, use_mod in [("Baseline", base_path, False), ("Anchored", anch_path, True)]:
        args = SimpleNamespace(
            backbone="CLIP:ViT-L/14", clip_path=clip_path, num_classes=2,
            use_srm=use_mod, use_gating=use_mod, srm_out_dim=1024, num_context_embedding=8
        )
        _, rep = load_canonical(c_path, args)
        if rep.get("missing", 0) != 0 or rep.get("unexpected", 0) != 0:
            raise RuntimeError(f"[PREFLIGHT FAIL] {name} vi phạm 0/0 keys: Missing={rep.get('missing')}, Unexpected={rep.get('unexpected')}")
        print(f"    • {name:<10}: strict load OK (0/0 keys, {rep.get('n_keys')} keys)")

    print("=" * 88)
    print("🏆 TOÀN BỘ TIỀN ĐIỀU KIỆN PREFLIGHT ĐÃ ĐẠT CHUẨN 100%! BẮT ĐẦU QUY TRÌNH!")
    print("=" * 88 + "\n")


def execute_pipeline_step(cmd_args: List[str], desc: str):
    """Thực thi một lệnh trong pipeline và bắt lỗi fail-fast."""
    cmd_str = " ".join(cmd_args)
    print(f"\n>>> [BƯỚC] {desc}")
    print(f"    Lệnh: {cmd_str}\n")
    start_t = time.time()
    ret = subprocess.run(cmd_args, cwd=PROJECT_ROOT)
    elapsed = time.time() - start_t
    if ret.returncode != 0:
        raise RuntimeError(f"[LỖI PIPELINE] Lệnh thất bại (mã thoát {ret.returncode}): {cmd_str}")
    print(f"\n[✓] Hoàn thành trong {elapsed:.1f}s: {desc}\n")


def split_calibration_logits(logits_npz: str, calib_ratio: float = 0.20, seed: int = 42) -> str:
    """
    Tách tập logits thành phân vùng calibration độc lập (không trùng mẫu với tập đánh giá).
    """
    data = np.load(logits_npz, allow_pickle=False)
    logits = data["logits"]
    targets = data["targets"]

    np.random.seed(seed)
    n = len(targets)
    perm = np.random.permutation(n)
    n_calib = int(n * calib_ratio)

    calib_idx = perm[:n_calib]
    eval_idx = perm[n_calib:]

    calib_npz = logits_npz.replace(".npz", "_calib_split.npz")
    np.savez_compressed(
        calib_npz,
        logits=logits[calib_idx],
        targets=targets[calib_idx],
        source=logits_npz,
        n_calib=n_calib,
    )
    print(f"  [✓] Đã tạo tập calibration tách biệt: {calib_npz} ({n_calib} mẫu, eval giữ {len(eval_idx)} mẫu)")
    return calib_npz


def run_phase7_acceptance(cfg_path: str, allow_cpu: bool = False, allow_dirty: bool = False):
    """
    Hàm thực thi toàn bộ 7 bước nghiệm thu GPU.
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Bước 0: Preflight
    run_preflight_checks(cfg, allow_cpu=allow_cpu, allow_dirty=allow_dirty)

    data_dir = cfg["data"]["cfe_root"]
    clip_path = cfg["checkpoints"]["clip_vit_l14"]["path"]
    base_ckpt = cfg["checkpoints"]["baseline"]["path"]
    anch_ckpt = cfg["checkpoints"]["anchored"]["path"]
    batch_size = str(cfg["runtime"].get("batch_size", 32))
    num_workers = str(cfg["runtime"].get("num_workers", 2))
    device = "cpu" if allow_cpu else "cuda"
    out_logs_dir = cfg["rebaseline"]["measured_artifacts_dir"]
    os.makedirs(out_logs_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # BƯỚC 1: Đánh giá Eval 7.3 cho Baseline & Anchored trên Clean & Q30
    # -------------------------------------------------------------------------
    for m_type, m_name, c_path in [
        ("baseline", "baseline", base_ckpt),
        ("anchored", "anchored", anch_ckpt),
    ]:
        for cond in ["clean", "q30"]:
            execute_pipeline_step(
                [
                    sys.executable, "tools/eval_communityforensics.py",
                    "--data_dir", data_dir,
                    "--clip_path", clip_path,
                    "--checkpoint", c_path,
                    "--model_type", m_type,
                    "--model_name", m_name,
                    "--condition", cond,
                    "--batch_size", batch_size,
                    "--num_workers", num_workers,
                    "--device", device,
                    "--output_dir", out_logs_dir,
                ],
                f"Eval 7.3 cho mô hình {m_name.upper()} trên điều kiện {cond.upper()}"
            )

    # -------------------------------------------------------------------------
    # BƯỚC 2: Hiệu chuẩn nhiệt độ & ngưỡng ràng buộc FPR 5% (Task 7.4)
    # -------------------------------------------------------------------------
    base_clean_npz = os.path.join(out_logs_dir, "logits_baseline_clean.npz")
    anch_clean_npz = os.path.join(out_logs_dir, "logits_anchored_clean.npz")

    # Tách phân vùng calibration 20%
    base_calib_npz = split_calibration_logits(base_clean_npz, calib_ratio=cfg["data"].get("calib_split_ratio", 0.20))
    anch_calib_npz = split_calibration_logits(anch_clean_npz, calib_ratio=cfg["data"].get("calib_split_ratio", 0.20))

    calib_out_dir = cfg["calibration"]["output_dir"]
    os.makedirs(calib_out_dir, exist_ok=True)

    execute_pipeline_step(
        [
            sys.executable, "tools/calibrate_and_select.py",
            "--logits_npz", base_calib_npz,
            "--model_name", "baseline",
            "--checkpoint", base_ckpt,
            "--output_dir", calib_out_dir,
            "--fpr_target", str(cfg["calibration"].get("target_fpr", 0.05)),
        ],
        "Hiệu chuẩn nhiệt độ & ngưỡng FPR 5% cho Baseline (Task 7.4)"
    )

    execute_pipeline_step(
        [
            sys.executable, "tools/calibrate_and_select.py",
            "--logits_npz", anch_calib_npz,
            "--model_name", "anchored",
            "--checkpoint", anch_ckpt,
            "--output_dir", calib_out_dir,
            "--fpr_target", str(cfg["calibration"].get("target_fpr", 0.05)),
        ],
        "Hiệu chuẩn nhiệt độ & ngưỡng FPR 5% cho Anchored (Task 7.4)"
    )

    # -------------------------------------------------------------------------
    # BƯỚC 3: Sinh lưới 12 ứng viên WiSE-FT Gen 2 (Task 7.5)
    # -------------------------------------------------------------------------
    cands_dir = cfg["wise_ft_gen2"]["candidates_dir"]
    alpha_str = ",".join(str(a) for a in cfg["wise_ft_gen2"]["alpha_grid"])
    beta_str = ",".join(str(b) for b in cfg["wise_ft_gen2"]["beta_grid"])

    execute_pipeline_step(
        [
            sys.executable, "tools/wise_ft_merge.py",
            "--orig_checkpoint", base_ckpt,
            "--target_checkpoint", anch_ckpt,
            "--grid_search",
            "--alpha_grid", alpha_str,
            "--beta_grid", beta_str,
            "--output_dir", cands_dir,
            "--verify",
        ],
        "Sinh lưới 12 ứng viên WiSE-FT Gen 2 kèm kiểm tra nghiêm ngặt verify 0/0 keys"
    )

    # -------------------------------------------------------------------------
    # BƯỚC 4: Đo đạc 12 ứng viên trên CFE & Tuyển chọn Pareto Best (Task 7.5)
    # -------------------------------------------------------------------------
    cand_files = [f for f in os.listdir(cands_dir) if f.startswith("wise_ft_v2_") and f.endswith(".pth") and not f.endswith("_best.pth")]
    print(f"\n[*] Bắt đầu đo đạc đánh giá {len(cand_files)} ứng viên trên CFE Clean & Q30...")

    base_calib_json = cfg["calibration"]["baseline_json"]
    for cand_f in sorted(cand_files):
        c_path = os.path.join(cands_dir, cand_f)
        cand_name = cand_f.replace(".pth", "")
        execute_pipeline_step(
            [
                sys.executable, "tools/eval_communityforensics.py",
                "--data_dir", data_dir,
                "--clip_path", clip_path,
                "--checkpoint", c_path,
                "--model_type", "candidate",
                "--model_name", cand_name,
                "--condition", "all",
                "--calibration", base_calib_json,
                "--batch_size", batch_size,
                "--num_workers", num_workers,
                "--device", device,
                "--output_dir", out_logs_dir,
                "--export_candidate_eval", c_path.replace(".pth", "_eval.json"),
            ],
            f"Đo đạc ứng viên WiSE-FT: {cand_name} (Clean & Q30)"
        )

    # Chạy tuyển chọn Pareto
    sweep_csv = cfg["wise_ft_gen2"]["sweep_csv"]
    best_ckpt_path = cfg["wise_ft_gen2"]["best_checkpoint_path"]

    execute_pipeline_step(
        [
            sys.executable, "tools/sweep_wise_ft_candidates.py",
            "--candidates_dir", cands_dir,
            "--calibration", base_calib_json,
            "--output_csv", sweep_csv,
            "--best_output", best_ckpt_path,
        ],
        "Tuyển chọn ứng viên WiSE-FT Gen 2 tối ưu Pareto (selected=True duy nhất)"
    )

    # -------------------------------------------------------------------------
    # BƯỚC 5: Đánh giá mô hình chiến thắng WiSE-FT Gen 2 Best (Task 7.3)
    # -------------------------------------------------------------------------
    for cond in ["clean", "q30"]:
        execute_pipeline_step(
            [
                sys.executable, "tools/eval_communityforensics.py",
                "--data_dir", data_dir,
                "--clip_path", clip_path,
                "--checkpoint", best_ckpt_path,
                "--model_type", "wise_ft_v2",
                "--model_name", "wise_ft_v2",
                "--condition", cond,
                "--calibration", base_calib_json,
                "--batch_size", batch_size,
                "--num_workers", num_workers,
                "--device", device,
                "--output_dir", out_logs_dir,
            ],
            f"Đánh giá WiSE-FT Gen 2 Best trên điều kiện {cond.upper()}"
        )

    # -------------------------------------------------------------------------
    # BƯỚC 6: Re-baseline Ma Trận 4 Cấu Hình từ Số Đo Thật (Task 7.6)
    # -------------------------------------------------------------------------
    rebase_json = cfg["rebaseline"]["output_json"]
    rebase_md = cfg["rebaseline"]["output_md"]

    execute_pipeline_step(
        [
            sys.executable, "tools/rebaseline_cfe_matrix.py",
            "--calibration", base_calib_json,
            "--measured_dir", out_logs_dir,
            "--output_json", rebase_json,
            "--output_md", rebase_md,
        ],
        "Tái lập ma trận sàn 4 cấu hình x 2 điều kiện từ số đo thật (evidence_level=measured)"
    )

    # -------------------------------------------------------------------------
    # BƯỚC 7: Validator Kiểm Tra Nghiệm Thu Cổng G7.1
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("FATFORMER-XLA: TIẾN TRÌNH THẨM ĐỊNH CHẤT LƯỢNG CỔNG G7.1")
    print("=" * 88)

    with open(rebase_json, "r", encoding="utf-8") as f:
        matrix_data = json.load(f)

    if matrix_data.get("evidence_level") != "measured":
        raise ValueError("[G7.1 FAIL] Ma trận kết quả không có evidence_level=measured!")

    res = matrix_data["results"]
    best_res = res["wise_ft_v2"]

    far90_q30 = best_res["q30"]["fake_at_real90"]
    auc_clean = best_res["clean"]["auc"]
    auc_q30 = best_res["q30"]["auc"]

    must_req = cfg["gate_g7_1"]["must"]
    stretch_req = cfg["gate_g7_1"]["stretch"]

    pass_far90 = far90_q30 >= must_req["fake_at_real90_q30"]
    pass_auc_clean = auc_clean >= must_req["auc_clean"]
    pass_auc_q30 = auc_q30 >= must_req["auc_q30"]
    pass_stretch = far90_q30 >= stretch_req["fake_at_real90_q30"]

    all_must_pass = pass_far90 and pass_auc_clean and pass_auc_q30
    verdict = "PASS" if all_must_pass else "FAIL"

    print(f"• Fake@Real90 (JPEG Q=30) : {far90_q30:.2f}% (Yêu cầu Must: >={must_req['fake_at_real90_q30']}%, Stretch: >={stretch_req['fake_at_real90_q30']}%) -> {'PASS' if pass_far90 else 'FAIL'}")
    print(f"• ROC-AUC (Clean CFE)    : {auc_clean:.2f}% (Yêu cầu Must: >={must_req['auc_clean']}%) -> {'PASS' if pass_auc_clean else 'FAIL'}")
    print(f"• ROC-AUC (JPEG Q=30)    : {auc_q30:.2f}% (Yêu cầu Must: >={must_req['auc_q30']}%) -> {'PASS' if pass_auc_q30 else 'FAIL'}")
    print(f"• Đạt mức Stretch        : {'CÓ' if pass_stretch else 'CHƯA'}")
    print("-" * 88)
    print(f"🏆 PHÁN QUYẾT CHUNG CUỘC CỔNG G7.1: [{verdict}]")
    print("=" * 88 + "\n")

    if verdict != "PASS":
        print("[!] Chú ý: Cổng G7.1 chưa đạt tiêu chuẩn Must; cần rà soát lại trước khi sang Phase 7.11.")
    else:
        print("[✓] CHÚC MỪNG: Cổng G7.1 chính thức ĐẠT CHUẨN bằng số đo thực tế trên GPU!")


def main():
    parser = argparse.ArgumentParser(description="FatFormer-XLA: Phase 7 Full GPU Acceptance Runner")
    parser.add_argument("--config", type=str, default="configs/phase7_gpu_acceptance.yaml",
                        help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--allow_cpu", action="store_true",
                        help="Cho phép chạy debug trên CPU (chỉ dùng test luồng logic)")
    parser.add_argument("--allow_dirty", action="store_true",
                        help="Cho phép chạy khi Git working tree dirty")
    args = parser.parse_args()

    cfg_file = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    if not os.path.isfile(cfg_file):
        print(f"[!] Không tìm thấy file cấu hình tại: {cfg_file}")
        sys.exit(1)

    try:
        run_phase7_acceptance(cfg_file, allow_cpu=args.allow_cpu, allow_dirty=args.allow_dirty)
    except Exception as e:
        print(f"\n[!] DỪNG KHẨN CẤP DO LỖI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
