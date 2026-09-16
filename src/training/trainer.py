import time
from typing import Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .loss import FatFormerLoss
from .checkpoint_manager import CheckpointManager
from ..evaluation.fast_eval import evaluate_dataloader


class Trainer:
    """
    Vòng lặp huấn luyện chuẩn hóa cho FatFormer:
    - Hỗ trợ Automatic Mixed Precision (AMP FP16) giúp tối ưu VRAM và tốc độ trên GPU A100.
    - Hỗ trợ Gradient Accumulation cho batch size lớn.
    - Tự động đánh giá trên tập validation và lưu checkpoint tốt nhất.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 4e-4,
        weight_decay: float = 1e-4,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        checkpoint_manager: Optional[CheckpointManager] = None,
        label_smoothing: float = 0.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp and (device.type == "cuda")
        self.grad_accum_steps = grad_accum_steps
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()

        # Cấu hình Loss
        self.criterion = FatFormerLoss(label_smoothing=label_smoothing)

        # Lọc các tham số cần huấn luyện (requires_grad=True)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"[TRAINER] Tổng số tensors cần tối ưu: {len(trainable_params)}")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.best_val_ap = 0.0

    def train_epoch(self, epoch: int, print_freq: int = 50) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        
        self.optimizer.zero_grad()
        num_batches = len(self.train_loader)

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_to_backward = loss / self.grad_accum_steps

            self.scaler.scale(loss_to_backward).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if (batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == num_batches:
                avg_loss = total_loss / total_samples
                elapsed = time.time() - start_time
                print(
                    f"Epoch [{epoch}] [{batch_idx + 1}/{num_batches}] "
                    f"Loss: {avg_loss:.4f} | Tốc độ: {total_samples / elapsed:.1f} ảnh/giây"
                )

        return total_loss / max(1, total_samples)

    def fit(
        self,
        epochs: int = 20,
        val_freq: int = 1,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Thực hiện toàn bộ quá trình huấn luyện qua các epoch.
        """
        print("\n" + "=" * 80)
        print(f"BẮT ĐẦU HUẤN LUYỆN: {epochs} EPOCHS | Thiết bị: {self.device} | AMP: {self.use_amp}")
        print("=" * 80)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            epoch_time = time.time() - t0

            val_metrics = {}
            is_best = False

            if self.val_loader and (epoch % val_freq == 0 or epoch == epochs):
                print(f"[EVALUATION] Đang kiểm tra validation cho Epoch {epoch}...")
                val_metrics = evaluate_dataloader(self.model, self.val_loader, self.device)
                print(f"  -> Validation ACC: {val_metrics['acc']:.2f}% | AP: {val_metrics['ap']:.2f}%")
                
                if val_metrics.get("ap", 0.0) > self.best_val_ap:
                    self.best_val_ap = val_metrics["ap"]
                    is_best = True

            if scheduler:
                scheduler.step()

            # Lưu checkpoint
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_metrics=val_metrics,
                is_best=is_best
            )

            print(f"Epoch {epoch} hoàn thành trong {epoch_time:.1f}s | Train Loss: {train_loss:.4f}\n")
