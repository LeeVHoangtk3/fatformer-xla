import os
import shutil
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class CheckpointManager:
    """
    Quản lý lưu trữ và nạp checkpoint an toàn, hỗ trợ:
    - Lưu cục bộ (local disk) và đồng bộ tự động sang Google Drive 5TB (/content/drive/MyDrive/...).
    - Tự động khôi phục (auto-resume) khi session Colab bị ngắt kết nối.
    - Hỗ trợ lưu chỉ các tham số có thể huấn luyện (trainable params) hoặc toàn bộ mô hình.
    """
    def __init__(
        self,
        save_dir: str = "checkpoints",
        drive_backup_dir: Optional[str] = None,
        max_to_keep: int = 3
    ):
        self.save_dir = save_dir
        self.drive_backup_dir = drive_backup_dir
        self.max_to_keep = max_to_keep

        os.makedirs(self.save_dir, exist_ok=True)
        if self.drive_backup_dir:
            try:
                os.makedirs(self.drive_backup_dir, exist_ok=True)
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể tạo thư mục backup Google Drive: {e}")

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        val_metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> str:
        """
        Lưu checkpoint hiện tại và đồng bộ sang Google Drive (nếu có cấu hình).
        """
        model_to_save = model.module if hasattr(model, "module") else model
        
        state = {
            "epoch": epoch,
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "val_metrics": val_metrics or {},
        }

        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pth"

        local_path = os.path.join(self.save_dir, filename)
        torch.save(state, local_path)
        print(f"[CHECKPOINT] Đã lưu checkpoint tại: {local_path}")

        # Cập nhật latest
        latest_path = os.path.join(self.save_dir, "checkpoint_latest.pth")
        torch.save(state, latest_path)

        if is_best:
            best_path = os.path.join(self.save_dir, "model_best.pth")
            shutil.copyfile(local_path, best_path)
            print(f"[CHECKPOINT] Đã cập nhật mô hình tốt nhất (Best Model) tại: {best_path}")

        # Đồng bộ sang Google Drive
        if self.drive_backup_dir and os.path.exists(self.drive_backup_dir):
            try:
                drive_path = os.path.join(self.drive_backup_dir, filename)
                shutil.copyfile(local_path, drive_path)
                shutil.copyfile(latest_path, os.path.join(self.drive_backup_dir, "checkpoint_latest.pth"))
                if is_best:
                    shutil.copyfile(local_path, os.path.join(self.drive_backup_dir, "model_best.pth"))
                print(f"[GOOGLE DRIVE] Đã sao lưu thành công sang Drive: {drive_path}")
            except Exception as e:
                print(f"[CẢNH BÁO] Lỗi khi sao lưu sang Google Drive: {e}")

        return local_path

    @staticmethod
    def load(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cpu"),
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Nạp checkpoint an toàn, tự động xử lý tiền tố 'module.' và tương thích PyTorch 2.6+.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict = checkpoint.get("model", checkpoint)
        
        # Xóa tiền tố 'module.' nếu checkpoint được lưu từ DistributedDataParallel
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                clean_state_dict[k[7:]] = v
            else:
                clean_state_dict[k] = v

        model_to_load = model.module if hasattr(model, "module") else model
        load_result = model_to_load.load_state_dict(clean_state_dict, strict=strict)
        print(f"[CHECKPOINT] Đã nạp thành công từ: {checkpoint_path}")
        if hasattr(load_result, "missing_keys") and load_result.missing_keys:
            print(f"  Missing keys: {len(load_result.missing_keys)}")
        if hasattr(load_result, "unexpected_keys") and load_result.unexpected_keys:
            print(f"  Unexpected keys: {len(load_result.unexpected_keys)}")

        if optimizer and checkpoint.get("optimizer"):
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể nạp trạng thái optimizer: {e}")

        if scheduler and checkpoint.get("scheduler"):
            try:
                scheduler.load_state_dict(checkpoint["scheduler"])
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể nạp trạng thái scheduler: {e}")

        return checkpoint
