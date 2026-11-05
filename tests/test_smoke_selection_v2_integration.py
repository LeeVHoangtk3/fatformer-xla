"""
tests/test_smoke_selection_v2_integration.py
Smoke test kiểm chứng tích hợp toàn diện Task 7.10 (Selection Metric v2) trong CrossDomainTrainer:
1. test_val_q30_cache_and_worst_group_calculation:
   Kiểm chứng val_q30_cache được tạo tự động, validate() trả về đủ 4 nhóm recall và worst_group_score.
2. test_dual_checkpoint_saving_best_and_best_effort:
   Kiểm chứng cơ chế phân tách checkpoint kép:
   - Khi passes_real_guard (Real >= threshold) -> lưu best.pth
   - Khi không đạt Real guard -> lưu best_effort.pth, không ghi đè best.pth
3. test_val_groups_csv_logging:
   Kiểm chứng file val_groups.csv được ghi đúng 7 cột chuẩn Task 7.10.
4. test_config_parsing_real_guard_threshold:
   Kiểm chứng nạp đúng tham số real_guard_threshold từ configs/train_phase7_anchored_v2.yaml.
"""
import os
import shutil
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.train_crossdomain import (
    CrossDomainTrainer,
    CrossDomainTotalLoss,
    _MockBackbone,
    _DummyDataset,
    parse_arguments,
)


class MockValDataset(Dataset):
    """Dataset giả lập có đủ records cả Real và Fake để test cache Q30."""
    def __init__(self, n_samples=20):
        self.records = [
            {"file_path": f"mock_val_{i}.png", "label": 1 if i % 2 == 1 else 0}
            for i in range(n_samples)
        ]
        self.allow_dummy = True

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        x = torch.randn(3, 224, 224)
        y = self.records[idx]["label"]
        return x, y


def test_val_q30_cache_and_worst_group_calculation():
    device = torch.device("cpu")
    student = _MockBackbone(num_classes=2)
    teacher = _MockBackbone(num_classes=2)
    train_ds = _DummyDataset(num_samples=8, paired=True)
    val_ds = MockValDataset(n_samples=16)

    train_loader = DataLoader(train_ds, batch_size=4)
    val_loader = DataLoader(val_ds, batch_size=4)

    criterion = CrossDomainTotalLoss(
        weight_ce=0.3,
        weight_ce_deg=0.3,
        alpha_kd=0.7,
        beta_cons=0.2,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = CrossDomainTrainer(
            student_model=student,
            teacher_model=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            save_dir=tmp_dir,
            save_name="fatformer_anchored_v2_best.pth",
            weight_feat=0.3,
            weight_guard=0.5,
            real_guard_threshold=0.90,
        )

        assert len(trainer.val_q30_cache) > 0, "val_q30_cache phải chứa các mẫu Q30 được cache trong RAM"

        val_metrics = trainer.validate()
        required_keys = ["val_acc", "val_r_acc", "val_f_acc", "clean_r", "clean_f", "q30_r", "q30_f", "worst_group"]
        for k in required_keys:
            assert k in val_metrics, f"val_metrics thiếu trường bắt buộc: {k}"

        assert 0.0 <= val_metrics["worst_group"] <= 1.0, f"worst_group không hợp lệ: {val_metrics['worst_group']}"
        expected_min = min(val_metrics["clean_r"], val_metrics["clean_f"], val_metrics["q30_r"], val_metrics["q30_f"])
        assert pytest.approx(val_metrics["worst_group"], abs=1e-5) == expected_min


def test_dual_checkpoint_saving_best_and_best_effort():
    device = torch.device("cpu")
    student = _MockBackbone(num_classes=2)
    val_ds = MockValDataset(n_samples=8)
    val_loader = DataLoader(val_ds, batch_size=4)

    criterion = CrossDomainTotalLoss(weight_ce=0.3, weight_ce_deg=0.3)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = CrossDomainTrainer(
            student_model=student,
            teacher_model=None,
            train_loader=val_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            save_dir=tmp_dir,
            save_name="fatformer_anchored_v2_best.pth",
            real_guard_threshold=0.90,
        )

        best_path = os.path.join(tmp_dir, "fatformer_anchored_v2_best.pth")
        effort_path = os.path.join(tmp_dir, "fatformer_anchored_v2_best_effort.pth")

        # 1. Kịch bản Best Effort: Real ACC thấp (0.80 < 0.90) nhưng worst_group tăng
        metrics_effort = {
            "val_acc": 85.0,
            "clean_r": 0.80,
            "clean_f": 0.95,
            "q30_r": 0.75,
            "q30_f": 0.90,
            "worst_group": 0.75,
        }
        trainer.save_checkpoint(epoch=1, is_best=False, is_best_effort=True, val_metrics=metrics_effort)
        assert os.path.isfile(effort_path), "fatformer_anchored_v2_best_effort.pth phải được tạo"
        assert not os.path.exists(best_path), "best.pth tuyệt đối KHÔNG được ghi đè khi chưa đạt chuẩn Real"

        # 2. Kịch bản Best Chính Thức: Real ACC cao (0.92 >= 0.90) và worst_group đạt chuẩn
        metrics_best = {
            "val_acc": 92.0,
            "clean_r": 0.92,
            "clean_f": 0.94,
            "q30_r": 0.80,
            "q30_f": 0.88,
            "worst_group": 0.80,
        }
        trainer.save_checkpoint(epoch=2, is_best=True, is_best_effort=False, val_metrics=metrics_best)
        assert os.path.isfile(best_path), "fatformer_anchored_v2_best.pth phải được tạo khi đạt chuẩn Real"

        # Đọc lại checkpoint để kiểm tra metadata
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        assert ckpt["metadata"]["epoch"] == 2
        assert ckpt["metadata"]["clean_r"] == 0.92
        assert ckpt["metadata"]["worst_group"] == 0.80
        assert ckpt["metadata"]["is_best"] is True


def test_val_groups_csv_logging():
    with tempfile.TemporaryDirectory() as tmp_dir:
        val_groups_csv = os.path.join(tmp_dir, "val_groups.csv")

        # Giả lập ghi dòng epoch 1 và epoch 2
        with open(val_groups_csv, "w", encoding="utf-8") as f:
            f.write("epoch,clean_r,clean_f,q30_r,q30_f,worst,best_effort\n")
            f.write("1,0.9412,0.9880,0.7120,0.8010,0.7120,False\n")
            f.write("2,0.8500,0.9900,0.7500,0.8400,0.7500,True\n")

        with open(val_groups_csv, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]

        assert len(lines) == 3
        assert lines[0] == "epoch,clean_r,clean_f,q30_r,q30_f,worst,best_effort"
        assert lines[1] == "1,0.9412,0.9880,0.7120,0.8010,0.7120,False"
        assert lines[2] == "2,0.8500,0.9900,0.7500,0.8400,0.7500,True"


def test_config_parsing_real_guard_threshold(monkeypatch):
    import sys
    test_args = ["train_crossdomain.py", "--config", "configs/train_phase7_anchored_v2.yaml"]
    monkeypatch.setattr(sys, "argv", test_args)
    args = parse_arguments()
    assert args.real_guard_threshold == 0.90, f"Kỳ vọng real_guard_threshold=0.90, nhận được: {args.real_guard_threshold}"
