import time
import warnings
from typing import Optional, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .loss import FatFormerLoss
from .metrics import group_recalls, worst_group_score
from .selection import OperatingPoint, checkpoint_decision, write_group_log, selection_loader_names
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
        label_smoothing: float = 0.0,
        criterion: Optional[nn.Module] = None,
        grad_clip_norm: Optional[float] = 1.0,
        best_val_score: float = 0.0,
        best_val_ap: float = 0.0,
        calibration_file: Optional[str] = None,
        selection_threshold: Optional[float] = None,
        real_guard_threshold: float = .90,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp and (device.type == "cuda")
        self.grad_accum_steps = grad_accum_steps
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.grad_clip_norm = grad_clip_norm

        # Cấu hình Loss: Hỗ trợ DualStreamFocalLoss hoặc FatFormerLoss mặc định
        if criterion is not None:
            self.criterion = criterion
        else:
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

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_val_ap = float(best_val_ap)
        self.best_val_score = float(best_val_score)
        self.best_worst_group = -1.0
        self.best_effort_worst_group = -1.0
        self.operating_point = OperatingPoint(calibration_file, selection_threshold)
        self.real_guard_threshold = real_guard_threshold
        if isinstance(self.val_loader, dict):
            clean_name, q30_name = selection_loader_names(self.val_loader)
            q30 = self.val_loader[q30_name]
            if len(q30.dataset) > 300:
                raise ValueError("Q30 loader must contain at most 300 stratified samples")
            # Caller supplies JPEG30 tensors; materialize once, never recompress them.
            tensors, labels = [], []
            for images, targets in q30:
                tensors.append(images.cpu())
                labels.append(targets.cpu())
            if not labels or set(torch.cat(labels).tolist()) != {0, 1}:
                raise ValueError("Q30 loader requires both Real and Fake")
            self.val_loader = dict(self.val_loader)
            cached = torch.utils.data.TensorDataset(torch.cat(tensors), torch.cat(labels))
            self.val_loader[q30_name] = DataLoader(cached, batch_size=q30.batch_size or 16)

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
                try:
                    outputs = self.model(images, return_dual=True)
                except TypeError:
                    outputs = self.model(images)

                if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                    try:
                        loss = self.criterion(outputs[0], outputs[1], labels)
                    except TypeError:
                        loss = self.criterion(outputs[0] + outputs[1], labels)
                else:
                    loss = self.criterion(outputs, labels)
                loss_to_backward = loss / self.grad_accum_steps

            self.scaler.scale(loss_to_backward).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

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
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        curr_scheduler: Optional[object] = None,
        start_epoch: int = 1
    ):
        """
        Thực hiện toàn bộ quá trình huấn luyện qua các epoch.
        """
        print("\n" + "=" * 80)
        print(f"BẮT ĐẦU HUẤN LUYỆN: EPOCHS {start_epoch} -> {epochs} | Thiết bị: {self.device} | AMP: {self.use_amp}")
        print("=" * 80)

        history_csv_path = None
        if hasattr(self.checkpoint_manager, "save_dir"):
            import os
            history_csv_path = os.path.join(self.checkpoint_manager.save_dir, "train_val_history.csv")
            if not os.path.exists(history_csv_path):
                try:
                    with open(history_csv_path, "w", encoding="utf-8") as f:
                        f.write("epoch,train_loss,val_acc,val_ap,val_auc,val_r_acc,val_f_acc,epoch_time_s\n")
                except Exception:
                    pass

        for epoch in range(start_epoch, epochs + 1):
            if curr_scheduler and hasattr(curr_scheduler, "set_epoch"):
                curr_scheduler.set_epoch(epoch)
                stage_idx = getattr(curr_scheduler, "stage", 1)
                print(f"[CURRICULUM] Epoch {epoch}: Thiết lập Giáo trình Giai đoạn {stage_idx}")
                
                # Áp dụng lr_multiplier (0.5) cho Phase 3 theo đúng configs/train_config.yaml
                if stage_idx == 3 and not getattr(self, "_phase3_lr_scaled", False):
                    lr_mult = getattr(curr_scheduler, "lr_multiplier", 0.5)
                    print(f"  [CURRICULUM] Giai đoạn 3 (Suy biến sâu): Áp dụng lr_multiplier={lr_mult} từ cấu hình cho Optimizer!")
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = param_group["lr"] * lr_mult
                    self._phase3_lr_scaled = True

            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            epoch_time = time.time() - t0

            val_metrics = {}
            is_best = False

            if self.val_loader and (epoch % val_freq == 0 or epoch == epochs):
                print(f"[EVALUATION] Đang kiểm tra validation cho Epoch {epoch}...")
                if isinstance(self.val_loader, dict):
                    # Đánh giá đa miền đại diện cho cả Clean và Robustness
                    domain_scores = []
                    clean_scores = []
                    robust_scores = []
                    diff_scores = []
                    val_metrics = {}

                    loader_outputs = {}
                    for vname, vloader in self.val_loader.items():
                        m = evaluate_dataloader(self.model, vloader, self.device, operating_point=self.operating_point)
                        loader_outputs[vname] = m
                        acc = m.get("acc", 0.0)
                        ap = m.get("ap", 0.0)
                        r_acc = m.get("r_acc", 0.0)
                        f_acc = m.get("f_acc", 0.0)
                        auc = m.get("auc", 0.0)
                        print(f"  -> [{vname}] ACC: {acc:.2f}% | AP: {ap:.2f}% | AUC: {auc:.2f}% | Real: {r_acc:.2f}% | Fake: {f_acc:.2f}%")

                        val_metrics[f"{vname}_acc"] = acc
                        val_metrics[f"{vname}_ap"] = ap
                        val_metrics[f"{vname}_r_acc"] = r_acc
                        val_metrics[f"{vname}_f_acc"] = f_acc

                        # Điểm cân bằng của từng loader
                        loader_score = 0.5 * acc + 0.5 * ap
                        domain_scores.append(loader_score)

                        # Phân loại miền để tính trọng số composite
                        v_lower = vname.lower()
                        if "deg" in v_lower or "q30" in v_lower or "q50" in v_lower:
                            robust_scores.append(loader_score)
                        elif "diff" in v_lower:
                            diff_scores.append(loader_score)
                        else:
                            clean_scores.append(loader_score)

                    # Điểm composite tham khảo (legacy)
                    if clean_scores and robust_scores and diff_scores:
                        val_score = 0.40 * float(np.mean(clean_scores)) + 0.40 * float(np.mean(robust_scores)) + 0.20 * float(np.mean(diff_scores))
                    elif clean_scores and robust_scores:
                        val_score = 0.50 * float(np.mean(clean_scores)) + 0.50 * float(np.mean(robust_scores))
                    else:
                        val_score = float(np.mean(domain_scores)) if domain_scores else 0.0

                    val_acc = float(np.mean([val_metrics[k] for k in val_metrics if k.endswith("_acc") and not (k.endswith("_r_acc") or k.endswith("_f_acc"))]))
                    val_ap = float(np.mean([val_metrics[k] for k in val_metrics if k.endswith("_ap")]))
                    val_metrics["acc"] = val_acc
                    val_metrics["ap"] = val_ap
                    val_metrics["val_score"] = val_score

                    clean_name, q30_name = selection_loader_names(self.val_loader)
                    true, predicted, conditions = [], [], []
                    for condition, name in enumerate([clean_name, q30_name]):
                        output = loader_outputs[name]
                        true.extend(output["y_true"])
                        predicted.extend(output["y_pred_selection"])
                        conditions.extend([condition] * len(output["y_true"]))
                    recalls = group_recalls(true, predicted, conditions)
                    val_metrics.update(recalls)
                    val_metrics["worst_group"] = worst_group_score(recalls)
                    val_metrics["n_eval_clean"] = len(loader_outputs[clean_name]["y_true"])
                    val_metrics["n_eval_q30"] = len(loader_outputs[q30_name]["y_true"])
                    val_metrics["threshold"] = self.operating_point.threshold
                    val_metrics["temperature"] = self.operating_point.temperature
                    val_metrics["calibration_source"] = self.operating_point.source
                    val_metrics["calibration_sha256"] = self.operating_point.sha256
                    val_metrics["real_guard_threshold"] = self.real_guard_threshold
                    is_best, effort = checkpoint_decision(val_metrics, self.best_worst_group,
                                                          self.best_effort_worst_group, self.real_guard_threshold)
                    val_metrics["is_best_effort"] = effort
                    if is_best:
                        self.best_worst_group = val_metrics["worst_group"]
                    if effort:
                        self.best_effort_worst_group = val_metrics["worst_group"]
                    write_group_log(os.path.join(self.checkpoint_manager.save_dir, "val_groups.csv"),
                                    epoch, val_metrics, effort, self.operating_point)
                else:
                    val_metrics = evaluate_dataloader(self.model, self.val_loader, self.device)
                    val_acc = val_metrics.get("acc", 0.0)
                    val_ap = val_metrics.get("ap", 0.0)
                    val_auc = val_metrics.get("auc", 0.0)
                    warnings.warn("Single validation loader: keeping legacy ACC/AP selection; no Q30 score is reported", RuntimeWarning)
                    val_score = 0.5 * val_acc + 0.5 * val_ap
                    val_metrics["val_score"] = val_score
                    is_best = val_score > self.best_val_score
                    if is_best:
                        self.best_val_score = val_score
                        self.best_val_ap = val_ap

            if scheduler:
                scheduler.step()

            # Ghi nhật ký vào file CSV
            if history_csv_path:
                try:
                    with open(history_csv_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"{epoch},{train_loss:.6f},"
                            f"{val_metrics.get('acc', 0.0):.2f},{val_metrics.get('ap', 0.0):.2f},"
                            f"{val_metrics.get('auc', 0.0):.2f},{val_metrics.get('r_acc', 0.0):.2f},"
                            f"{val_metrics.get('f_acc', 0.0):.2f},{epoch_time:.2f}\n"
                        )
                    # Đồng bộ sang Google Drive nếu có
                    if hasattr(self.checkpoint_manager, "drive_backup_dir") and self.checkpoint_manager.drive_backup_dir:
                        import shutil
                        drive_csv = os.path.join(self.checkpoint_manager.drive_backup_dir, "train_val_history.csv")
                        shutil.copyfile(history_csv_path, drive_csv)
                except Exception as e:
                    print(f"  [!] Lưu lịch sử CSV: {e}")

            # Lưu checkpoint đồng bộ kèm GradScaler (Task 3.5)
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=scheduler,
                scaler=self.scaler,
                epoch=epoch,
                val_metrics=val_metrics,
                is_best=is_best,
                best_val_score=self.best_val_score,
                best_val_ap=self.best_val_ap
            )

            print(f"Epoch {epoch} hoàn thành trong {epoch_time:.1f}s | Train Loss: {train_loss:.4f}\n")
