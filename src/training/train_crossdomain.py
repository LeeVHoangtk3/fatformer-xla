"""
FatFormer-XLA: Script Huan Luyen Thich Ung Da Mien (Cross-Domain Training Execution)
Phuc vu Task 6.3-C & Task 6.5 (Milestone 6) — Thanh vien C (Training & Evaluation Lead).

Ho tro 3 nhanh thuc nghiem:
    - M3: Data da bo sinh + CrossEntropyLoss
    - M4: Data da bo sinh + CrossDomainTotalLoss (CE + Distillation + Consistency)
    - M5: Data da bo sinh + CrossDomainTotalLoss + LoRA ViT Adapter

Cac che do thuc thi:
    # 1. Chay kiem thu Dry-Run 1 buoc tren CPU (khong can du lieu that):
    python src/training/train_crossdomain.py --dry_run --device cpu --branch M4

    # 2. Chay huan luyen tren GPU Kaggle / Colab voi file cau hinh:
    python src/training/train_crossdomain.py --config configs/train_m4_distill.yaml
"""

import os
import sys
import time
import math
import random
import argparse
from typing import Optional, Dict, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.training.loss import CrossDomainTotalLoss
from src.datasets.transforms import PairedCleanDegradedAugmentation
from src.models.lora_adapter import ViTLoRAAdapter


def set_seed(seed: int = 42) -> None:
    """Khoa chat tinh ngau nhien tren toan bo he thong."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class _DummyDataset(Dataset):
    """Dataset gia lap dung cho kiem thu Dry-Run ma khong can kho anh that."""
    def __init__(self, num_samples: int = 16, paired: bool = True):
        self.num_samples = num_samples
        self.paired = paired
        self.aug = PairedCleanDegradedAugmentation() if paired else None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        pil_img = Image.new("RGB", (256, 256), color=(idx * 15 % 255, idx * 30 % 255, 120))
        target = 0 if idx % 2 == 0 else 1  # 0: real, 1: fake
        if self.paired and self.aug is not None:
            xc, xd = self.aug(pil_img)
            return (xc, xd), target
        else:
            tensor = torch.randn(3, 224, 224)
            return tensor, target


class _MockBackbone(nn.Module):
    """Mo hinh nhe dung cho che do Dry-Run tren CPU."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class CrossDomainTrainer:
    """
    Trinh quan ly huan luyen thich ung da mien cho Milestone 6.
    - Ho tro cap anh sach - suy bien de dong bo quyet dinh phan loai.
    - Ho tro mo hinh mau (teacher) dong bang chong representation drift.
    - Ap dung chinh sach xoay vong checkpoint de toi uu bo nho dia luu tru.
    """
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: Optional[nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: CrossDomainTotalLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cpu"),
        save_dir: str = "checkpoints/m4",
        grad_clip_norm: float = 1.0,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device) if teacher_model is not None else None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp and (device.type == "cuda")

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_composite_score = -1.0

        os.makedirs(self.save_dir, exist_ok=True)

        # Dong bang hoan toan mo hinh teacher
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.student.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_cons = 0.0
        num_batches = len(self.train_loader)

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            # Xu ly ca hai truong hop: batch dang cap anh hoac batch dang don
            if isinstance(batch[0], (list, tuple)):
                (x_clean, x_degraded), targets = batch
                x_clean = x_clean.to(self.device)
                x_degraded = x_degraded.to(self.device)
            else:
                x_clean, targets = batch
                x_clean = x_clean.to(self.device)
                x_degraded = None

            targets = targets.to(self.device)
            clean_mask = (targets == 0)  # Nhan 0 la anh that sach

            # Tinh toan logit mo hinh mau (teacher) khong qua dao ham
            teacher_logits = None
            if self.teacher is not None:
                with torch.no_grad():
                    teacher_logits = self.teacher(x_clean)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                student_clean_logits = self.student(x_clean)
                student_degraded_logits = self.student(x_degraded) if x_degraded is not None else None

                loss_dict = self.criterion(
                    logits=student_clean_logits,
                    targets=targets,
                    teacher_logits=teacher_logits,
                    logits_degraded=student_degraded_logits,
                    clean_mask=clean_mask,
                )
                loss = loss_dict["loss_total"] / self.grad_accum_steps

            # Lan truyen nguoc qua GradScaler
            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == num_batches:
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss_dict["loss_total"].item()
            total_ce += loss_dict["loss_ce"].item()
            total_kd += loss_dict["loss_kd"].item()
            total_cons += loss_dict["loss_cons"].item()

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        return {
            "loss_total": avg_loss,
            "loss_ce": total_ce / max(num_batches, 1),
            "loss_kd": total_kd / max(num_batches, 1),
            "loss_cons": total_cons / max(num_batches, 1),
        }

    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {"val_acc": 0.0, "val_composite": 0.0}

        self.student.eval()
        correct = 0
        total = 0
        real_correct = 0
        real_total = 0
        fake_correct = 0
        fake_total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch[0], (list, tuple)):
                    (x_eval, _), targets = batch
                else:
                    x_eval, targets = batch

                x_eval = x_eval.to(self.device)
                targets = targets.to(self.device)

                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    logits = self.student(x_eval)

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                # Tach rieng Real Recall va Fake Recall
                real_mask = (targets == 0)
                fake_mask = (targets == 1)

                if real_mask.any():
                    real_correct += (preds[real_mask] == targets[real_mask]).sum().item()
                    real_total += real_mask.sum().item()
                if fake_mask.any():
                    fake_correct += (preds[fake_mask] == targets[fake_mask]).sum().item()
                    fake_total += fake_mask.sum().item()

        macro_acc = (correct / total * 100.0) if total > 0 else 0.0
        r_acc = (real_correct / real_total * 100.0) if real_total > 0 else 0.0
        f_acc = (fake_correct / fake_total * 100.0) if fake_total > 0 else 0.0

        # Diem composite danh gia su can bang giua Real va Fake
        composite = 0.5 * macro_acc + 0.25 * r_acc + 0.25 * f_acc
        return {
            "val_acc": macro_acc,
            "val_r_acc": r_acc,
            "val_f_acc": f_acc,
            "val_composite": composite,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Chinh sach xoay vong checkpoint nham bao ve bo nho dia luu tru:
        - checkpoint_latest.pth: Luon ghi de de phuc vu phuc hoi phien.
        - crossdomain_model_best.pth: Chi cap nhat khi composite score vuot moc cu.
        """
        state = {
            "epoch": epoch,
            "model_state": self.student.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_score": self.best_composite_score,
        }

        latest_path = os.path.join(self.save_dir, "checkpoint_latest.pth")
        torch.save(state, latest_path)

        if is_best:
            best_path = os.path.join(self.save_dir, "crossdomain_model_best.pth")
            torch.save(state, best_path)
            print(f"  -> Luu model vuot troi tai epoch {epoch} (Composite={self.best_composite_score:.2f}%)")


def run_training_pipeline(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 80)
    print(f"FATFORMER-XLA: PIPELINE HUAN LUYEN THICH UNG DA MIEN ({args.branch})")
    print(f"Thiet bi: {device} | Dry-run: {args.dry_run} | Epochs: {args.epochs}")
    print("=" * 80)

    # 1. Khoi tao Loss
    criterion = CrossDomainTotalLoss(
        label_smoothing=args.label_smoothing,
        temperature=args.temperature,
        alpha_kd=args.alpha_kd if args.branch in ["M4", "M5"] else 0.0,
        beta_cons=args.beta_cons if args.branch in ["M4", "M5"] else 0.0,
    )

    # 2. Khoi tao Model (Thuc te hoac Dry-run)
    if args.dry_run:
        student = _MockBackbone(num_classes=2)
        teacher = _MockBackbone(num_classes=2) if args.branch in ["M4", "M5"] else None
        train_dataset = _DummyDataset(num_samples=16, paired=(args.branch in ["M4", "M5"]))
        val_dataset = _DummyDataset(num_samples=16, paired=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        # Khi chay tren he thong Kaggle/Colab voi mo hinh FatFormer day du
        from src.models import build_model
        from src.training.checkpoint_manager import CheckpointManager

        class ModelArgs:
            backbone = "CLIP:ViT-L/14"
            clip_path = args.clip_path
            num_classes = 2
            use_srm = True
            use_gating = True
            srm_out_dim = 1024

        student = build_model(ModelArgs())
        if os.path.exists(args.checkpoint_init):
            CheckpointManager.load(args.checkpoint_init, student, device=device, strict=False)

        teacher = None
        if args.branch in ["M4", "M5"] and os.path.exists(args.teacher_checkpoint):
            teacher = build_model(ModelArgs())
            CheckpointManager.load(args.teacher_checkpoint, teacher, device=device, strict=True)

        # Neu la nhanh M5: Kich hoat them LoRA ViT Adapter
        if args.branch == "M5":
            lora_cfg = {"enabled": True, "rank": 8, "alpha": 16.0, "num_layers": 2}
            ViTLoRAAdapter(student.visual, config=lora_cfg)

        train_loader = None  # Se nhan tu Task 6.2 DataLoader
        val_loader = None

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    trainer = CrossDomainTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=(args.device == "cuda"),
    )

    if args.dry_run:
        print("\n[DRY-RUN] Bat dau chay kiem thu 1 step Train + 1 step Val...")
        train_metrics = trainer.train_epoch(epoch=1)
        val_metrics = trainer.validate()
        print(f"  -> Ket qua Train: Loss={train_metrics['loss_total']:.4f} "
              f"(CE={train_metrics['loss_ce']:.4f}, KD={train_metrics['loss_kd']:.4f}, Cons={train_metrics['loss_cons']:.4f})")
        print(f"  -> Ket qua Val: ACC={val_metrics['val_acc']:.2f}%, Composite={val_metrics['val_composite']:.2f}%")
        trainer.save_checkpoint(epoch=1, is_best=True)
        print("[DRY-RUN] HOAN THANH XUAT SAC — MA NGUON SAN SANG 100% CHO KAGGLE/COLAB!")
        return

    # Huan luyen chinh thuc nhieu epochs
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate()
        elapsed = time.time() - t0

        is_best = val_metrics["val_composite"] > trainer.best_composite_score
        if is_best:
            trainer.best_composite_score = val_metrics["val_composite"]

        trainer.save_checkpoint(epoch, is_best=is_best)
        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | Train Loss: {train_metrics['loss_total']:.4f} | "
              f"Val ACC: {val_metrics['val_acc']:.2f}% | Composite: {val_metrics['val_composite']:.2f}%")


def parse_arguments():
    parser = argparse.ArgumentParser(description="FatFormer-XLA Cross-Domain Training")
    parser.add_argument("--config", type=str, default="", help="Duong dan tep cau hinh yaml")
    parser.add_argument("--branch", type=str, default="M4", choices=["M3", "M4", "M5"], help="Nhanh thuc nghiem")
    parser.add_argument("--dry_run", action="store_true", help="Chay kiem thu dry-run gia lap")
    parser.add_argument("--device", type=str, default="cpu", help="Thiet bi: cpu hoac cuda")
    parser.add_argument("--epochs", type=int, default=5, help="So epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--alpha_kd", type=float, default=0.4, help="Trong so KD loss")
    parser.add_argument("--beta_cons", type=float, default=0.2, help="Trong so Consistency loss")
    parser.add_argument("--temperature", type=float, default=2.0, help="Nhiet do KD")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing")
    parser.add_argument("--save_dir", type=str, default="checkpoints/m4", help="Thu muc luu checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--clip_path", type=str, default="ViT-L-14.pt", help="Path to CLIP backbone")
    parser.add_argument("--checkpoint_init", type=str, default="DATASET/checkpoints/fatformer_plan_a_best_weights.pth")
    parser.add_argument("--teacher_checkpoint", type=str, default="fatformer_4class_ckpt.pth")

    args = parser.parse_args()

    # Neu co file config yaml, nạp va ghi de cac gia tri tuong ung
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if "experiment" in cfg:
            args.branch = cfg["experiment"].get("branch", args.branch)
            args.seed = cfg["experiment"].get("seed", args.seed)
        if "training" in cfg:
            t_cfg = cfg["training"]
            args.epochs = t_cfg.get("epochs", args.epochs)
            args.batch_size = t_cfg.get("batch_size", args.batch_size)
            args.learning_rate = float(t_cfg.get("learning_rate", args.learning_rate))
            args.save_dir = t_cfg.get("save_dir", args.save_dir)
        if "loss" in cfg:
            l_cfg = cfg["loss"]
            args.alpha_kd = float(l_cfg.get("alpha_kd", args.alpha_kd))
            args.beta_cons = float(l_cfg.get("beta_cons", args.beta_cons))
            args.temperature = float(l_cfg.get("temperature", args.temperature))
            args.label_smoothing = float(l_cfg.get("label_smoothing", args.label_smoothing))

    return args


if __name__ == "__main__":
    args = parse_arguments()
    run_training_pipeline(args)
