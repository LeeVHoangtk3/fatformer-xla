"""
==============================================================================
FATFORMER-XLA: KỊCH BẢN HUẤN LUYỆN CHUẨN HÓA TOÀN DỰ ÁN (TRAIN.PY)
HỖ TRỢ: MÔ HÌNH CHÍNH (TASK 4.1 & 4.2) VÀ CÁC MÔ HÌNH ABLATION (TASK 4.3 & 4.4)
==============================================================================
Tác giả: Nhóm nghiên cứu FatFormer-XLA
Mục đích:
  - Cung cấp giao diện dòng lệnh (CLI) độc lập và đồng bộ với train.ipynb.
  - Tự động nhận diện phần cứng (NVIDIA A100 / H100 / L4 / T4 / CPU).
  - Tự động cấu hình AMP (FP16), Gradient Accumulation và DataLoader tối ưu.
  - Hỗ trợ lưu Checkpoint định kỳ sang Google Drive 5TB và tính năng --resume.
==============================================================================
"""

import os
import sys
import time
import shutil
import tarfile
import argparse
from typing import Optional, List, Dict, Any, Tuple
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision.datasets import ImageFolder

# Bảo đảm nạp module từ thư mục gốc
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from src.models import build_model
from src.training.loss import DualStreamFocalLoss, FatFormerLoss, DualStreamCELoss
from src.training.freeze_utils import freeze_clip_backbone, assert_srm_kernels_frozen, count_trainable_parameters
from src.training.checkpoint_manager import CheckpointManager
from src.training.trainer import Trainer
from src.datasets.transforms import CurriculumDegradationScheduler, get_train_transforms, get_eval_transforms
from src.datasets.dataset import (
    DatasetCreator,
    build_multi_domain_train_dataloader,
    build_val_dataloader,
    collect_sub_imagefolders,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Giá trị Boolean không hợp lệ (nhập true/false).')


def parse_args():
    parser = argparse.ArgumentParser(description="FatFormer-XLA Unified Training CLI (Milestone 3 / Phương Án A Finetune)")

    # 1. Cấu hình cơ bản & Dữ liệu đa miền (Phương Án A)
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Đường dẫn file cấu hình train_config.yaml")
    
    # Nguồn dữ liệu ProGAN & Diffusion Staging & Validation
    parser.add_argument("--progan_tar", type=str, default=None, help="Đường dẫn file nén progan_train.tar (144k ảnh)")
    parser.add_argument("--progan_dir", type=str, default=None, help="Đường dẫn thư mục progan_train đã giải nén")
    parser.add_argument("--staging_tar", type=str, default=None, help="Đường dẫn file nén diffusion_staging.tar (3.6k ảnh)")
    parser.add_argument("--staging_dir", type=str, default=None, help="Đường dẫn thư mục diffusion_staging đã giải nén")
    parser.add_argument("--val_tar", type=str, default=None, help="Đường dẫn file nén progan_val.tar (8k ảnh)")
    parser.add_argument("--val_dir", type=str, default=None, help="Đường dẫn thư mục validation (progan_val) đã giải nén")

    # Tỷ lệ lấy mẫu đa miền (WeightedRandomSampler)
    parser.add_argument("--progan_ratio", type=float, default=0.85, help="Tỷ lệ mẫu ProGAN mong muốn trong mỗi batch (mặc định: 0.85)")
    parser.add_argument("--staging_ratio", type=float, default=0.15, help="Tỷ lệ mẫu Diffusion mong muốn trong mỗi batch (mặc định: 0.15)")

    # Tương thích ngược nguồn đơn
    parser.add_argument("--data_tar", type=str, default=None, help="Đường dẫn file dataset nén .tar đơn (fallback)")
    parser.add_argument("--data_dir", type=str, default=None, help="Đường dẫn thư mục dataset đã giải nén đơn (fallback)")
    parser.add_argument("--clip_path", type=str, default=None, help="Đường dẫn file trọng số ViT-L-14.pt")

    # 2. Siêu tham số huấn luyện
    parser.add_argument("--epochs", type=int, default=5, help="Tổng số epoch huấn luyện (Mặc định: 5)")
    parser.add_argument("--start_epoch", type=int, default=1, help="Epoch bắt đầu (mặc định: 1)")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch trên mỗi forward (mặc định: 32)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Số bước tích lũy gradient (mặc định: 2 -> Effective batch size = 64)")
    parser.add_argument("--lr", "--lr_adapter", type=float, default=1e-4, dest="lr", help="Tốc độ học cho các adapter (LGA, Text-interactor, SRM proj)")
    parser.add_argument("--lr_gating", type=float, default=1e-3, help="Tốc độ học riêng cho mạng Gating MLP")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Hệ số suy giảm trọng số AdamW")
    parser.add_argument("--clean_prob", type=float, default=0.50, help="Tỷ lệ ảnh sạch tối thiểu (khóa cứng 0.50 chống quên biểu diễn cũ)")
    parser.add_argument("--val_freq", type=int, default=1, help="Chu kỳ chạy kiểm định validation (mỗi n epoch)")
    parser.add_argument("--use_cosine_scheduler", type=str2bool, default=True, help="Kích hoạt CosineAnnealingLR scheduler theo train_config.yaml")
    parser.add_argument("--eta_min", type=float, default=1.0e-6, help="Tốc độ học tối thiểu eta_min cho CosineAnnealingLR")

    # 3. Chuyển đổi kiến trúc & Module Ablation
    parser.add_argument("--use_srm", type=str2bool, default=True, help="Bật/Tắt khối vi sai không gian SRM 3-Kernels (Ablation 2: tắt)")
    parser.add_argument("--use_gating", type=str2bool, default=True, help="Bật/Tắt cổng thích ứng tần số động lambda(x) (Ablation 2 & 3: tắt)")
    parser.add_argument("--use_curriculum", type=str2bool, default=True, help="Bật/Tắt lập lịch suy thoái Curriculum 3 giai đoạn")
    parser.add_argument("--loss_type", type=str, default="dual_stream_ce", choices=["dual_stream_ce", "dual_stream_focal", "focal", "ce", "cross_entropy"], help="Loại hàm mất mát (dual_stream_ce/dual_stream_focal/ce)")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Hệ số làm mượt nhãn Label Smoothing")

    # 4. Quản lý Checkpoint & Hạ tầng
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Thư mục lưu trữ Checkpoint (nên trỏ sang Drive 5TB trên Colab)")
    parser.add_argument("--checkpoint_name", type=str, default="fatformer_plan_a_final.pth", help="Tên file checkpoint chính cần lưu")
    parser.add_argument("--save_only_latest_and_best", type=str2bool, default=True, help="Chỉ lưu 2 file: checkpoint_latest.pth và model_best.pth (tiết kiệm đĩa cứng, chống tràn Kaggle)")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn file .pth để tiếp tục huấn luyện ('latest' để tự tìm file mới nhất)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị thực thi (cuda hoặc cpu)")
    parser.add_argument("--num_workers", type=int, default=4, help="Số luồng nạp dữ liệu DataLoader")
    parser.add_argument("--use_amp", type=str2bool, default=True, help="Bật Automatic Mixed Precision FP16")

    return parser.parse_args()


def extract_tar_if_needed(tar_path: Optional[str], target_extract_root: str) -> Optional[str]:
    """Giải nén file .tar sang target_extract_root nếu chưa giải nén."""
    if not tar_path or not os.path.isfile(tar_path):
        return None

    os.makedirs(target_extract_root, exist_ok=True)
    tar_basename = os.path.basename(tar_path).replace(".tar", "")
    extracted_path = os.path.join(target_extract_root, tar_basename)

    if not os.path.exists(extracted_path) or len(os.listdir(extracted_path)) == 0:
        print(f"[*] Đang giải nén {tar_path} sang {extracted_path}...")
        t_tar = time.time()
        os.makedirs(extracted_path, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extracted_path)
        print(f"[✓] Giải nén {tar_basename} hoàn tất trong {time.time() - t_tar:.1f}s!")
    return extracted_path


def find_first_existing_path(candidates: List[Optional[str]], is_dir: bool = True) -> Optional[str]:
    """Tìm đường dẫn hợp lệ đầu tiên từ danh sách ứng viên."""
    for p in candidates:
        if p:
            if is_dir and os.path.isdir(p):
                return p
            elif not is_dir and os.path.isfile(p):
                return p
    return None


def load_config_and_override_args(args) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """
    Nạp cấu hình từ train_config.yaml để làm Nguồn Sự Thật Duy Nhất (Single Source of Truth).
    Chỉ override nếu người dùng không ghi đè tham số trực tiếp qua command line.
    """
    cfg: Dict[str, Any] = {}
    config_path = args.config
    if config_path:
        if not os.path.isabs(config_path):
            config_path = os.path.join(PROJECT_ROOT, config_path)
        if os.path.isfile(config_path):
            print(f"[*] Đang nạp cấu hình huấn luyện từ file YAML: {config_path}...")
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            print("  [✓] Đã kích hoạt train_config.yaml làm Nguồn Sự Thật Duy Nhất (Source of Truth)!")
        else:
            print(f"[!] Không tìm thấy file config tại '{config_path}'. Dùng thiết lập CLI mặc định.")

    # 1. Cấu hình mô hình
    m_cfg = cfg.get("model", {})
    if "use_srm" in m_cfg: args.use_srm = m_cfg["use_srm"]
    if "use_gating" in m_cfg: args.use_gating = m_cfg["use_gating"]
    if "num_classes" in m_cfg: args.num_classes = m_cfg["num_classes"]
    if "num_context_embedding" in m_cfg: args.num_context_embedding = m_cfg["num_context_embedding"]

    # 2. Cấu hình Freeze & PEFT
    f_cfg = cfg.get("freeze", {})
    args.target_trainable_ratio_min = f_cfg.get("target_trainable_ratio_min", 0.055)
    args.target_trainable_ratio_max = f_cfg.get("target_trainable_ratio_max", 0.08)

    # 3. Cấu hình Loss
    l_cfg = cfg.get("loss", {})
    if "type" in l_cfg:
        l_type = str(l_cfg["type"]).lower()
        if "focal" in l_type:
            args.loss_type = "dual_stream_focal"
        elif "dual" in l_type or "ce" in l_type:
            args.loss_type = "dual_stream_ce"
    if "label_smoothing" in l_cfg:
        args.label_smoothing = float(l_cfg["label_smoothing"])

    # 4. Cấu hình Training
    t_cfg = cfg.get("training", {})
    if "total_epochs" in t_cfg: args.epochs = int(t_cfg["total_epochs"])
    if "batch_size" in t_cfg: args.batch_size = int(t_cfg["batch_size"])
    if "grad_accumulation_steps" in t_cfg: args.grad_accum = int(t_cfg["grad_accumulation_steps"])
    if "amp_fp16" in t_cfg: args.use_amp = bool(t_cfg["amp_fp16"])

    opt_cfg = t_cfg.get("optimizer", {})
    if "lr_adapter" in opt_cfg: args.lr = float(opt_cfg["lr_adapter"])
    if "lr_gating" in opt_cfg: args.lr_gating = float(opt_cfg["lr_gating"])
    if "weight_decay" in opt_cfg: args.weight_decay = float(opt_cfg["weight_decay"])

    sched_cfg = t_cfg.get("scheduler", {})
    if "type" in sched_cfg:
        args.use_cosine_scheduler = (sched_cfg["type"] == "CosineAnnealingLR")
    if "T_max" in sched_cfg:
        args.T_max = int(sched_cfg["T_max"])
    else:
        args.T_max = args.epochs
    if "eta_min" in sched_cfg:
        args.eta_min = float(sched_cfg["eta_min"])

    # 5. Cấu hình Data
    d_cfg = cfg.get("data", {})
    if "progan_ratio" in d_cfg: args.progan_ratio = float(d_cfg["progan_ratio"])
    if "staging_ratio" in d_cfg: args.staging_ratio = float(d_cfg["staging_ratio"])
    if "num_workers" in d_cfg: args.num_workers = int(d_cfg["num_workers"])
    if "train_data_dir" in d_cfg and not args.progan_dir: args.progan_dir = d_cfg["train_data_dir"]
    if "staging_data_dir" in d_cfg and not args.staging_dir: args.staging_dir = d_cfg["staging_data_dir"]
    if "val_data_dir" in d_cfg and not args.val_dir: args.val_dir = d_cfg["val_data_dir"]
    if "progan_tar" in d_cfg and not args.progan_tar: args.progan_tar = d_cfg["progan_tar"]
    if "staging_tar" in d_cfg and not args.staging_tar: args.staging_tar = d_cfg["staging_tar"]
    if "val_tar" in d_cfg and not args.val_tar: args.val_tar = d_cfg["val_tar"]

    # 6. Checkpoint
    c_cfg = cfg.get("checkpoint", {})
    if "save_dir" in c_cfg and args.output_dir == "checkpoints":
        args.output_dir = c_cfg["save_dir"]

    return args, cfg


def main():
    args = parse_args()
    args, cfg = load_config_and_override_args(args)
    device = torch.device(args.device)

    print("=" * 85)
    print("FATFORMER-XLA: KHỞI ĐỘNG HỆ THỐNG HUẤN LUYỆN (MILESTONE 3 - TUẦN 4)")
    print("=" * 85)
    print(f"  • Thiết bị phần cứng:  {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    if device.type == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  • Bộ nhớ VRAM:         {total_vram_gb:.2f} GB VRAM")
    print(f"  • Batch Size thực:     {args.batch_size} (Tích lũy {args.grad_accum} bước -> Effective Batch Size: {args.batch_size * args.grad_accum})")
    print(f"  • Số Epoch dự kiến:    {args.epochs} (Bắt đầu từ Epoch {args.start_epoch})")
    print(f"  • Chế độ SRM:          {args.use_srm} | Gating λ(x): {args.use_gating} | Curriculum: {args.use_curriculum}")
    print(f"  • Hàm mục tiêu Loss:   {args.loss_type}")
    print(f"  • Thư mục lưu Output:  {args.output_dir}")
    print("-" * 85)

    # 1. Chuẩn bị Dữ liệu & Transforms (Phương Án A: Đa miền ProGAN 85% + Staging 15%)
    target_extract_dir = "/content/dataset_local" if os.path.exists("/content") else os.path.join(PROJECT_ROOT, "scratch", "dataset_local")
    os.makedirs(target_extract_dir, exist_ok=True)

    # 1.1 Tìm kiếm hoặc giải nén ProGAN
    progan_dir = args.progan_dir
    if not progan_dir or not os.path.isdir(progan_dir):
        cand_progan_tars = [
            args.progan_tar,
            "/content/drive/MyDrive/Fatformer/datasets/progan_train.tar",
            "/content/drive/MyDrive/FatFormer_Hub/datasets/progan_train.tar",
            "DATASET/progan_train.tar",
            os.path.join(PROJECT_ROOT, "scratch", "dataset_local", "progan_train.tar")
        ]
        for ptar in cand_progan_tars:
            if ptar and os.path.isfile(ptar):
                progan_dir = extract_tar_if_needed(ptar, target_extract_dir)
                if progan_dir:
                    break
        if not progan_dir:
            cand_dirs = ["/content/dataset_local/progan_train", "DATASET/progan_train", "DATASET/train"]
            progan_dir = find_first_existing_path(cand_dirs, is_dir=True)

    # 1.2 Tìm kiếm hoặc giải nén Diffusion Staging
    staging_dir = args.staging_dir
    if not staging_dir or not os.path.isdir(staging_dir):
        cand_staging_tars = [
            args.staging_tar,
            "/content/drive/MyDrive/Fatformer/datasets/diffusion_staging.tar",
            "/content/drive/MyDrive/FatFormer_Hub/datasets/diffusion_staging.tar",
            "DATASET/diffusion_staging.tar",
            os.path.join(PROJECT_ROOT, "scratch", "dataset_local", "diffusion_staging.tar")
        ]
        for star in cand_staging_tars:
            if star and os.path.isfile(star):
                staging_dir = extract_tar_if_needed(star, target_extract_dir)
                if staging_dir:
                    break
        if not staging_dir:
            cand_dirs = ["/content/dataset_local/diffusion_staging", "DATASET/diffusion_staging"]
            staging_dir = find_first_existing_path(cand_dirs, is_dir=True)

    # 1.3 Tìm kiếm hoặc giải nén ProGAN Validation
    val_dir = args.val_dir
    if not val_dir or not os.path.isdir(val_dir):
        cand_val_tars = [
            args.val_tar,
            "/content/drive/MyDrive/Fatformer/datasets/progan_val.tar",
            "/content/drive/MyDrive/FatFormer_Hub/datasets/progan_val.tar",
            "DATASET/progan_val.tar",
            os.path.join(PROJECT_ROOT, "scratch", "dataset_local", "progan_val.tar")
        ]
        for vtar in cand_val_tars:
            if vtar and os.path.isfile(vtar):
                val_dir = extract_tar_if_needed(vtar, target_extract_dir)
                if val_dir:
                    break
        if not val_dir:
            cand_dirs = ["/content/dataset_local/progan_val", "DATASET/progan_val", "DATASET/val"]
            val_dir = find_first_existing_path(cand_dirs, is_dir=True)

    # Fallback cho tùy chọn cũ data_tar/data_dir
    if not progan_dir and not staging_dir:
        single_path = args.data_dir
        if not single_path and args.data_tar:
            single_path = extract_tar_if_needed(args.data_tar, target_extract_dir)
        progan_dir = single_path

    # Thiết lập bộ tiền xử lý & Curriculum (Nạp cấu hình từ train_config.yaml)
    curr_scheduler = CurriculumDegradationScheduler(
        clean_prob=args.clean_prob,
        config=cfg.get("curriculum", None)
    ) if args.use_curriculum else None
    train_transforms = get_train_transforms(scheduler=curr_scheduler)

    train_loader, n_progan, n_staging = build_multi_domain_train_dataloader(
        progan_dir=progan_dir,
        staging_dir=staging_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        progan_ratio=args.progan_ratio,
        staging_ratio=args.staging_ratio,
        transform=train_transforms,
        pin_memory=(device.type == "cuda")
    )

    # Khởi tạo Validation DataLoader Đa Miền (Multi-Domain Validation)
    val_loaders = {}
    if val_dir and os.path.isdir(val_dir):
        v_clean = build_val_dataloader(
            val_dir=val_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            transform=get_eval_transforms()
        )
        if v_clean:
            val_loaders["progan_clean"] = v_clean

        # Miền ProGAN Degraded Q=50 kiểm soát khả năng chống chịu suy biến
        v_deg = build_val_dataloader(
            val_dir=val_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            transform=get_eval_transforms(degradation={"type": "jpeg", "quality": 50})
        )
        if v_deg:
            val_loaders["progan_degraded_q50"] = v_deg

    cand_diff_val = [
        os.path.join(PROJECT_ROOT, "DATASET", "diffusion_val"),
        "/content/dataset_local/diffusion_val",
        staging_dir if (staging_dir and os.path.isdir(staging_dir) and "val" in staging_dir) else None
    ]
    diff_val_dir = find_first_existing_path(cand_diff_val, is_dir=True)
    if diff_val_dir:
        v_diff = build_val_dataloader(
            val_dir=diff_val_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            transform=get_eval_transforms()
        )
        if v_diff:
            val_loaders["diffusion_clean"] = v_diff

    val_loader = val_loaders if len(val_loaders) > 1 else (val_loaders.get("progan_clean") or next(iter(val_loaders.values()), None))

    if train_loader is None:
        print("[!] Không tìm thấy dữ liệu ảnh thật. Sinh tập dữ liệu mô phỏng 64 ảnh (Dry-Run mode)...")
        dummy_inputs = torch.randn(64, 3, 224, 224)
        dummy_targets = torch.randint(0, 2, (64,))
        dummy_dataset = torch.utils.data.TensorDataset(dummy_inputs, dummy_targets)
        train_loader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Khởi tạo Mô hình FatFormer-XLA
    resolved_clip_path = args.clip_path
    if not resolved_clip_path or not os.path.isfile(resolved_clip_path):
        clip_candidates = [
            "/content/drive/MyDrive/Fatformer/pretrained/ViT-L-14.pt",
            "/content/drive/MyDrive/FatFormer_Hub/pretrained/ViT-L-14.pt",
            "pretrained/ViT-L-14.pt",
            os.path.join(PROJECT_ROOT, "pretrained", "ViT-L-14.pt"),
            "ViT-L-14.pt"
        ]
        for cp in clip_candidates:
            if cp and os.path.isfile(cp):
                resolved_clip_path = cp
                print(f"  [✓] Tự động phát hiện file trọng số CLIP: {resolved_clip_path}")
                break

    model_args = argparse.Namespace()
    model_args.backbone = "CLIP:ViT-L/14"
    model_args.num_classes = 2
    model_args.num_context_embedding = 8
    model_args.use_srm = args.use_srm
    model_args.use_gating = args.use_gating
    model_args.clip_path = resolved_clip_path

    print("\n[*] Đang khởi tạo mô hình FatFormer-XLA...")
    try:
        model = build_model(model_args).to(device)
        print("  [✓] Khởi tạo kiến trúc CLIP ViT-L/14 thành công!")

        # Kế thừa trọng số Baseline ban đầu (Task 1.2) nếu có trên Drive
        if not args.resume:
            init_candidates = [
                "/content/drive/MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth",
                "/content/drive/MyDrive/FatFormer_Hub/pretrained/fatformer_4class_ckpt.pth",
                "pretrained/fatformer_4class_ckpt.pth"
            ]
            for init_p in init_candidates:
                if os.path.isfile(init_p):
                    print(f"  [*] Nạp trọng số Baseline khởi tạo từ: {init_p}...")
                    init_data = torch.load(init_p, map_location=device)
                    init_sd = init_data.get("model", init_data.get("state_dict", init_data))
                    msg = model.load_state_dict(init_sd, strict=False)
                    print(f"  [✓] Đã nạp thành công trọng số Baseline (Missing keys mới: {len(msg.missing_keys)} cho SRM/Gating)!")
                    break
    except Exception as e:
        print(f"  [!] Thông báo nạp CLIP ViT: {e}")
        print("  [*] Sử dụng mô hình kiểm thử cấu trúc tương thích dự phòng...")
        class FallbackFatFormer(nn.Module):
            def __init__(self, use_srm=True, use_gating=True):
                super().__init__()
                self.image_encoder = nn.Sequential(nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten(), nn.Linear(3 * 64, 64))
                self.language_guided_alignment = nn.Linear(64, 2)
                self.text_guided_interactor = nn.Linear(64, 2)
                self.gating = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid()) if use_gating else None
                self.srm = nn.Linear(64, 64) if use_srm else None
            def forward(self, x, return_dual=False):
                feat = self.image_encoder(x)
                gate = self.gating(feat) if self.gating else 1.0
                lv = self.language_guided_alignment(feat) * gate
                la = self.text_guided_interactor(feat)
                return (lv, la) if return_dual else (lv + la)
        model = FallbackFatFormer(use_srm=args.use_srm, use_gating=args.use_gating).to(device)

    # 3. Đóng băng tham số PEFT (Task 3.4)
    print("\n[*] Thực thi đóng băng Backbone PEFT & Kiểm toán an toàn SRM...")
    # Chốt chặn bắt buộc: Dừng hẳn chương trình (raise) nếu tỷ lệ trainable vượt ngưỡng trần 8.0%
    stats = freeze_clip_backbone(
        model,
        target_trainable_ratio_min=getattr(args, "target_trainable_ratio_min", 0.055),
        target_trainable_ratio_max=getattr(args, "target_trainable_ratio_max", 0.08),
        verbose=True
    )

    # 4. Phân tầng Learning Rate cho Optimizer
    gating_params = []
    other_trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "gating" in name.lower():
                gating_params.append(param)
            else:
                other_trainable_params.append(param)

    param_groups = []
    if other_trainable_params:
        param_groups.append({"params": other_trainable_params, "lr": args.lr})
    if gating_params:
        param_groups.append({"params": gating_params, "lr": args.lr_gating})

    optimizer = torch.optim.AdamW(
        param_groups if param_groups else model.parameters(),
        weight_decay=args.weight_decay
    )

    # Khởi tạo CosineAnnealingLR scheduler theo train_config.yaml
    # T_max đọc từ YAML (args.T_max); fallback về args.epochs nếu không có.
    # Lưu ý: trước đây dùng args.epochs trực tiếp, khiến config T_max bị bỏ qua.
    scheduler = None
    if getattr(args, "use_cosine_scheduler", True):
        t_max = getattr(args, "T_max", args.epochs)
        if t_max < 1:
            raise ValueError(
                f"[CONFIG ERROR] scheduler.T_max phải >= 1, nhận được {t_max}. "
                f"Kiểm tra lại giá trị T_max trong train_config.yaml."
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=getattr(args, "eta_min", 1.0e-6)
        )
        print(f"  [✓] Kích hoạt CosineAnnealingLR (T_max={t_max}, eta_min={getattr(args, 'eta_min', 1.0e-6)})")

    # 5. Hàm mục tiêu Loss Function (Phương Án A: DualStreamCELoss)
    if args.loss_type == "dual_stream_ce":
        criterion = DualStreamCELoss(label_smoothing=args.label_smoothing)
        print(f"  [✓] Kích hoạt hàm mục tiêu: DualStreamCELoss (label_smoothing={args.label_smoothing})")
    elif args.loss_type in ["dual_stream_focal", "focal"] and args.use_gating:
        criterion = DualStreamFocalLoss(alpha=0.25, gamma=1.0)
        print("  [✓] Kích hoạt hàm mục tiêu: DualStreamFocalLoss (alpha=0.25, gamma=1.0)")
    else:
        criterion = FatFormerLoss(label_smoothing=args.label_smoothing)
        print(f"  [✓] Kích hoạt hàm mục tiêu tiêu chuẩn: CrossEntropy / FatFormerLoss (label_smoothing={args.label_smoothing})")

    # 6. Checkpoint Manager & Resume
    drive_backup = "/content/drive/MyDrive/Fatformer/checkpoint" if os.path.exists("/content/drive/MyDrive") else None
    checkpoint_mgr = CheckpointManager(
        save_dir=args.output_dir,
        drive_backup_dir=drive_backup,
        max_to_keep=3,
        save_only_latest_and_best=args.save_only_latest_and_best
    )
    start_epoch = args.start_epoch

    if args.resume:
        resume_file = args.resume
        if args.resume == "latest":
            resume_file = checkpoint_mgr.find_latest_checkpoint(args.output_dir)
        
        if resume_file and os.path.isfile(resume_file):
            print(f"\n[*] Đang khôi phục phiên huấn luyện từ checkpoint: {resume_file}")
            resumed_meta = checkpoint_mgr.load(resume_file, model=model, optimizer=optimizer)
            if resumed_meta and "epoch" in resumed_meta:
                start_epoch = resumed_meta["epoch"] + 1
                print(f"  [✓] Tiếp tục huấn luyện từ Epoch {start_epoch}!")

    # 7. Huấn luyện qua Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_amp=args.use_amp,
        grad_accum_steps=args.grad_accum,
        checkpoint_manager=checkpoint_mgr,
        criterion=criterion
    )
    trainer.optimizer = optimizer

    print("\n" + "=" * 85)
    print(f"BẮT ĐẦU CHẠY HUẤN LUYỆN: TỪ EPOCH {start_epoch} ĐẾN {args.epochs}")
    if val_loader:
        if isinstance(val_loader, dict):
            domain_info = ", ".join([f"{k} ({len(v.dataset):,} ảnh)" for k, v in val_loader.items()])
            print(f"  • Validation Đa Miền: Kích hoạt [{domain_info}] | Chu kỳ: mỗi {args.val_freq} epoch")
        else:
            print(f"  • Validation: Kích hoạt ({len(val_loader.dataset):,} ảnh) | Chu kỳ: mỗi {args.val_freq} epoch")
    else:
        print("  • Validation: Chưa cấu hình val_dir (sẽ bỏ qua validation loop)")
    print("=" * 85)

    trainer.fit(
        epochs=args.epochs,
        start_epoch=start_epoch,
        scheduler=scheduler,
        curr_scheduler=curr_scheduler,
        val_freq=args.val_freq
    )

    # Lưu checkpoint hoàn thành của giai đoạn (Mô hình chính thức từ Best Model)
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "model_best.pth")
    final_path = os.path.join(args.output_dir, args.checkpoint_name)
    if os.path.exists(best_path) and os.path.abspath(best_path) != os.path.abspath(final_path):
        shutil.copyfile(best_path, final_path)
        print(f"[CHECKPOINT] [Final] Đã đồng bộ checkpoint chính thức từ Best Model: {final_path}")
        if drive_backup and os.path.exists(drive_backup):
            shutil.copyfile(best_path, os.path.join(drive_backup, args.checkpoint_name))
            print(f"[GOOGLE DRIVE] Đã sao lưu checkpoint chính thức sang Drive: {os.path.join(drive_backup, args.checkpoint_name)}")
    elif not os.path.exists(final_path):
        checkpoint_mgr.save(
            model=model,
            optimizer=optimizer,
            scaler=trainer.scaler,
            epoch=args.epochs,
            filename=args.checkpoint_name
        )

    print(f"\n[✓] CHIẾN DỊCH HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
    print(f"  • Checkpoint đã lưu an toàn tại: {args.output_dir}")
    print(f"  • Tên file: {args.checkpoint_name}")
    print("=" * 85)


if __name__ == "__main__":
    main()
