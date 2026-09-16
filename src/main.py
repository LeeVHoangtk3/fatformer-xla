import os
import sys
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

# Thêm đường dẫn gốc vào sys.path để import chuẩn module src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import build_model
from src.datasets.dataset import DatasetCreator, ALL_TEST_SUBSETS
from src.evaluation.fast_eval import run_benchmark
from src.training.trainer import Trainer
from src.training.checkpoint_manager import CheckpointManager


def get_args_parser():
    parser = argparse.ArgumentParser("FatFormer-XLA Unified Framework", add_help=True)

    # General
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--print_freq", default=50, type=int)

    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, default="./dataset")
    parser.add_argument("--img_resolution", type=int, default=256)
    parser.add_argument("--crop_resolution", type=int, default=224)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--test_selected_subsets", nargs="+", default=None,
                        help="Danh sách các subset cần test, hoặc 'all', 'gans', 'diffusion'")

    # Model architecture
    parser.add_argument("--backbone", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--clip_path", type=str, default=None, help="Đường dẫn đến file ViT-L-14.pt")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_vit_adapter", type=int, default=3)
    parser.add_argument("--num_context_embedding", type=int, default=8)
    parser.add_argument("--init_context_embedding", type=str, default="")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--clip_vision_width", type=int, default=1024)
    parser.add_argument("--frequency_encoder_layer", type=int, default=2)
    parser.add_argument("--decoder_layer", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=12)

    # Robustness & Novel Modules (Phương Án 2)
    parser.add_argument("--use_srm", action="store_true", help="Kích hoạt khối SRM 3-Kernels")
    parser.add_argument("--use_gating", action="store_true", help="Kích hoạt Dynamic Frequency Gating")

    # Evaluation mode
    parser.add_argument("--eval", action="store_true", help="Chế độ đánh giá kiểm thử")
    parser.add_argument("--fast_eval", action="store_true", help="Đánh giá nhanh (mặc định lấy 500 ảnh/subset)")
    parser.add_argument("--max_samples", type=int, default=500, help="Số ảnh tối đa mỗi subset khi fast_eval")
    parser.add_argument("--degradation_type", type=str, default="none", choices=["none", "jpeg", "blur", "down_up"])
    parser.add_argument("--jpeg_quality", type=int, default=50)
    parser.add_argument("--blur_radius", type=float, default=1.5)
    parser.add_argument("--down_up_ratio", type=float, default=0.5)

    # Training mode
    parser.add_argument("--train", action="store_true", help="Chế độ huấn luyện / fine-tuning")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_amp", action="store_true", default=True, help="Sử dụng AMP FP16")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--pretrained_model", type=str, default="", help="Đường dẫn checkpoint nạp vào")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--drive_backup_dir", type=str, default=None, help="Đường dẫn thư mục Google Drive để tự động backup")

    return parser


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[INFO] Thiết bị thực thi: {device}")

    # 1. Khởi tạo mô hình
    print(f"[INFO] Đang khởi tạo mô hình với backbone: {args.backbone} | SRM: {args.use_srm} | Gating: {args.use_gating}")
    model = build_model(args)
    model = model.to(device)

    # 2. Nạp checkpoint nếu được chỉ định
    if args.pretrained_model:
        print(f"[INFO] Đang nạp checkpoint từ: {args.pretrained_model}")
        # Nếu bật SRM/Gating mà nạp checkpoint gốc tác giả thì dùng strict=False để giữ trọng số mới khởi tạo
        strict = not (args.use_srm or args.use_gating)
        CheckpointManager.load(args.pretrained_model, model, device=device, strict=strict)

    # 3. Chế độ đánh giá (Evaluation)
    if args.eval:
        degradation = None
        if args.degradation_type == "jpeg":
            degradation = {"type": "jpeg", "quality": args.jpeg_quality}
        elif args.degradation_type == "blur":
            degradation = {"type": "blur", "radius": args.blur_radius}
        elif args.degradation_type == "down_up":
            degradation = {"type": "down_up", "ratio": args.down_up_ratio}

        max_samples = args.max_samples if args.fast_eval else None
        run_benchmark(
            model=model,
            dataset_path=args.dataset_path,
            device=device,
            selected_subsets=args.test_selected_subsets,
            degradation=degradation,
            max_samples_per_subset=max_samples,
            batch_size=args.batchsize,
            num_workers=args.num_workers
        )
        return

    # 4. Chế độ huấn luyện (Training / Fine-Tuning)
    if args.train:
        creator = DatasetCreator(
            dataset_path=args.dataset_path,
            img_resolution=args.img_resolution,
            crop_resolution=args.crop_resolution,
            batch_size=args.batchsize,
            num_workers=args.num_workers
        )

        train_dataset = creator.build_train_dataset(use_dual_stream=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True
        )

        ckpt_mgr = CheckpointManager(
            save_dir=args.checkpoint_dir,
            drive_backup_dir=args.drive_backup_dir
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            use_amp=args.use_amp,
            grad_accum_steps=args.grad_accum_steps,
            checkpoint_manager=ckpt_mgr,
            label_smoothing=args.label_smoothing
        )

        trainer.fit(epochs=args.epochs)
        return

    print("[CẢNH BÁO] Vui lòng chỉ định cờ --eval hoặc --train để thực thi. Sử dụng --help để xem chi tiết.")


if __name__ == "__main__":
    main()
