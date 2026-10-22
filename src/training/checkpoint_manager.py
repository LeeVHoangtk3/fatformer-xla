import os
import sys
import shutil
import json
import time
from typing import Optional, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
        max_to_keep: int = 3,
        save_only_latest_and_best: bool = True
    ):
        self.save_dir = save_dir
        self.drive_backup_dir = drive_backup_dir
        self.max_to_keep = max_to_keep
        self.save_only_latest_and_best = save_only_latest_and_best

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
        scaler: Optional[Any] = None,
        epoch: int = 0,
        val_metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        filename: Optional[str] = None,
        best_val_score: Optional[float] = None,
        best_val_ap: Optional[float] = None
    ) -> str:
        """
        Lưu checkpoint hiện tại và đồng bộ sang Google Drive (nếu có cấu hình).
        Bao gồm: model weights, optimizer, scheduler, scaler (AMP), epoch, metrics, best_val_score.
        Mặc định: Chỉ lưu checkpoint_latest.pth và model_best.pth (tiết kiệm đĩa cứng tối đa).
        """
        model_to_save = model.module if hasattr(model, "module") else model
        
        state = {
            "epoch": epoch,
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "val_metrics": val_metrics or {},
            "best_val_score": best_val_score,
            "best_val_ap": best_val_ap,
        }

        # 1. Chế độ tối ưu dung lượng: Chỉ lưu latest và model best
        if self.save_only_latest_and_best and filename is None:
            filename = "checkpoint_latest.pth"
            local_path = os.path.join(self.save_dir, filename)

            # Dọn dẹp bất kỳ file checkpoint_epoch_*.pth cũ nào còn sót
            self._purge_epoch_checkpoints(self.save_dir)
            if self.drive_backup_dir and os.path.exists(self.drive_backup_dir):
                self._purge_epoch_checkpoints(self.drive_backup_dir)

            # Ghi đè an toàn checkpoint_latest.pth
            temp_path = os.path.join(self.save_dir, "checkpoint_latest.pth.tmp")
            torch.save(state, temp_path)
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass
            os.replace(temp_path, local_path)
            print(f"[CHECKPOINT] [Latest] Đã lưu/ghi đè epoch {epoch} tại: {local_path}")

            # Cập nhật model_best nếu đạt kỷ lục
            if is_best:
                best_path = os.path.join(self.save_dir, "model_best.pth")
                shutil.copyfile(local_path, best_path)
                print(f"[CHECKPOINT] [Best] ★ Đã cập nhật mô hình tốt nhất (Kỷ lục mới) tại: {best_path}")

            # Sao lưu sang Google Drive (nếu có)
            if self.drive_backup_dir and os.path.exists(self.drive_backup_dir):
                try:
                    drive_latest = os.path.join(self.drive_backup_dir, "checkpoint_latest.pth")
                    shutil.copyfile(local_path, drive_latest)
                    if is_best:
                        drive_best = os.path.join(self.drive_backup_dir, "model_best.pth")
                        shutil.copyfile(local_path, drive_best)
                    print(f"[GOOGLE DRIVE] Đã sao lưu thành công sang Drive: {drive_latest}")
                except Exception as e:
                    print(f"[CẢNH BÁO] Lỗi khi sao lưu sang Google Drive: {e}")

            return local_path

        # 2. Chế độ truyền thống (Lưu riêng từng epoch nếu được yêu cầu rõ ràng)
        # Dọn dẹp trước để giải phóng đĩa cứng trước khi ghi file mới
        self._prune_old_checkpoints(self.save_dir)
        if self.drive_backup_dir and os.path.exists(self.drive_backup_dir):
            self._prune_old_checkpoints(self.drive_backup_dir)

        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pth"

        local_path = os.path.join(self.save_dir, filename)
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception:
                pass
        torch.save(state, local_path)
        print(f"[CHECKPOINT] Đã lưu checkpoint tại: {local_path}")

        # Cập nhật latest
        latest_path = os.path.join(self.save_dir, "checkpoint_latest.pth")
        if os.path.exists(latest_path):
            try:
                os.remove(latest_path)
            except Exception:
                pass
        shutil.copyfile(local_path, latest_path)

        # Cập nhật model_best nếu đạt kỷ lục
        if is_best:
            best_path = os.path.join(self.save_dir, "model_best.pth")
            if os.path.exists(best_path):
                try:
                    os.remove(best_path)
                except Exception:
                    pass
            shutil.copyfile(local_path, best_path)
            print(f"[CHECKPOINT] Đã cập nhật mô hình tốt nhất (Best Model) tại: {best_path}")

        # Đồng bộ sang Google Drive
        if self.drive_backup_dir and os.path.exists(self.drive_backup_dir):
            try:
                drive_path = os.path.join(self.drive_backup_dir, filename)
                if os.path.abspath(local_path) != os.path.abspath(drive_path):
                    shutil.copyfile(local_path, drive_path)
                    shutil.copyfile(latest_path, os.path.join(self.drive_backup_dir, "checkpoint_latest.pth"))
                    if is_best:
                        shutil.copyfile(local_path, os.path.join(self.drive_backup_dir, "model_best.pth"))
                    print(f"[GOOGLE DRIVE] Đã sao lưu thành công sang Drive: {drive_path}")
            except Exception as e:
                print(f"[CẢNH BÁO] Lỗi khi sao lưu sang Google Drive: {e}")

        # Dọn dẹp giữ lại max_to_keep checkpoint gần nhất
        self._prune_old_checkpoints(self.save_dir)
        if self.drive_backup_dir and os.path.exists(self.drive_backup_dir):
            self._prune_old_checkpoints(self.drive_backup_dir)

        return local_path

    def _purge_epoch_checkpoints(self, directory: str):
        """Xóa sạch các file checkpoint_epoch_*.pth để giải phóng triệt để dung lượng."""
        try:
            for f in os.listdir(directory):
                if f.startswith("checkpoint_epoch_") and f.endswith(".pth"):
                    try:
                        os.remove(os.path.join(directory, f))
                    except Exception:
                        pass
        except Exception:
            pass

    def _prune_old_checkpoints(self, directory: str):
        """Giữ lại tối đa self.max_to_keep file checkpoint theo thứ tự epoch."""
        if not self.max_to_keep or self.max_to_keep <= 0:
            return
        try:
            files = [
                f for f in os.listdir(directory)
                if f.startswith("checkpoint_epoch_") and f.endswith(".pth")
            ]
            files.sort()
            if len(files) > self.max_to_keep:
                to_remove = files[:-self.max_to_keep]
                for f in to_remove:
                    os.remove(os.path.join(directory, f))
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể dọn dẹp checkpoint cũ tại {directory}: {e}")

    @staticmethod
    def load(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
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

        state_dict = checkpoint
        if isinstance(state_dict, dict):
            for key in ["model", "model_state", "state_dict", "model_state_dict"]:
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
        
        model_to_load = model.module if hasattr(model, "module") else model
        model_keys = set(model_to_load.state_dict().keys())

        # Xóa tiền tố 'module.' hoặc 'model.' nếu checkpoint được lưu từ Wrapper/DDP
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k
            if clean_k.startswith("module."):
                clean_k = clean_k[7:]
            if clean_k.startswith("model.") and clean_k not in model_keys and clean_k[6:] in model_keys:
                clean_k = clean_k[6:]
            clean_state_dict[clean_k] = v

        load_result = model_to_load.load_state_dict(clean_state_dict, strict=strict)
        print(f"[CHECKPOINT] Đã nạp thành công từ: {checkpoint_path} (strict={strict})")
        missing = list(getattr(load_result, "missing_keys", []))
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        # Ghi log kiểm toán missing/unexpected keys ra file JSON phục vụ nghiệm thu
        try:
            audit_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint_path": str(checkpoint_path),
                "strict_mode": bool(strict),
                "missing_keys_count": len(missing),
                "missing_keys": missing,
                "unexpected_keys_count": len(unexpected),
                "unexpected_keys": unexpected,
            }
            os.makedirs("logs", exist_ok=True)
            log_path = os.path.join("logs", "checkpoint_keys_audit.json")
            history = []
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as af:
                        history = json.load(af)
                    if not isinstance(history, list):
                        history = [history]
                except Exception:
                    history = []
            history.append(audit_entry)
            with open(log_path, "w", encoding="utf-8") as af:
                json.dump(history, af, indent=2, ensure_ascii=False)
            print(f"  [✓] Đã ghi log kiểm toán keys vào: {log_path}")
        except Exception as e_log:
            print(f"  [!] Ghi log kiểm toán keys thất bại: {e_log}")

        if optimizer and checkpoint.get("optimizer"):
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("  [✓] Đã khôi phục trạng thái optimizer.")
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể nạp trạng thái optimizer: {e}")

        if scheduler and checkpoint.get("scheduler"):
            try:
                scheduler.load_state_dict(checkpoint["scheduler"])
                print("  [✓] Đã khôi phục trạng thái scheduler.")
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể nạp trạng thái scheduler: {e}")

        if scaler and checkpoint.get("scaler"):
            try:
                scaler.load_state_dict(checkpoint["scaler"])
                print("  [✓] Đã khôi phục trạng thái GradScaler (AMP FP16).")
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể nạp trạng thái GradScaler: {e}")

        if not isinstance(checkpoint, dict):
            checkpoint = {"model": checkpoint}

        class CheckpointLoadResult(dict):
            @property
            def missing_keys(self):
                return self.get("missing_keys", [])
            @property
            def unexpected_keys(self):
                return self.get("unexpected_keys", [])

        result = CheckpointLoadResult(checkpoint)
        result["missing_keys"] = missing
        result["unexpected_keys"] = unexpected
        result["load_result"] = load_result
        return result

    @staticmethod
    def find_latest_checkpoint(dir_path: str) -> Optional[str]:
        """Tự động tìm kiếm checkpoint mới nhất trong thư mục."""
        if not os.path.exists(dir_path):
            return None
        # Ưu tiên checkpoint_latest.pth
        latest_file = os.path.join(dir_path, "checkpoint_latest.pth")
        if os.path.isfile(latest_file):
            return latest_file
        # Hoặc tìm checkpoint_epoch_*.pth có số epoch cao nhất
        epoch_files = [
            f for f in os.listdir(dir_path)
            if f.startswith("checkpoint_epoch_") and f.endswith(".pth")
        ]
        if epoch_files:
            epoch_files.sort()
            return os.path.join(dir_path, epoch_files[-1])
        return None
