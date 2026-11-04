"""
tests/test_smoke_loss_v2_integration.py
Smoke test kiểm thử tích hợp toàn diện Task 7.9:
1. Luồng Huấn luyện khép kín: Model(return_features) -> CrossDomainTrainer -> CrossDomainTotalLoss (v2).
2. Kiểm tra nạp cấu hình configs/train_phase7_anchored_v2.yaml qua parser.
3. Kiểm tra tính tương thích ngược khi weight_feat=weight_guard=0 (Legacy mode).
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.loss import CrossDomainTotalLoss, RealGuardLoss, FeatureConsistencyLoss
from src.training.train_crossdomain import CrossDomainTrainer, parse_arguments


class MockStudentModel(nn.Module):
    """Mô phỏng CLIPModel của FatFormer hỗ trợ cờ return_features."""
    def __init__(self, in_features=3 * 224 * 224, hidden_dim=768, num_classes=2):
        super().__init__()
        self.encoder = nn.Linear(in_features, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_features=False):
        flat_x = x.view(x.size(0), -1)
        feat = torch.relu(self.encoder(flat_x))
        feat_norm = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        logits = self.head(feat_norm)
        if return_features:
            return logits, feat_norm
        return logits


class MockTeacherModel(nn.Module):
    """Mô phỏng Teacher Model đóng băng."""
    def __init__(self, in_features=3 * 224 * 224, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        flat_x = x.view(x.size(0), -1)
        return self.linear(flat_x)


class MockPairedDataset(Dataset):
    """Dataset giả lập 8 ảnh ghép cặp (4 Real, 4 Fake)."""
    def __init__(self):
        self.samples = []
        for i in range(8):
            label = 0 if i < 4 else 1  # 4 real, 4 fake
            x_clean = torch.randn(3, 224, 224)
            # degraded là ảnh sạch thêm chút nhiễu
            x_deg = x_clean + 0.1 * torch.randn(3, 224, 224)
            self.samples.append(((x_clean, x_deg), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Test 1: Luồng Huấn luyện khép kín với Loss v2
# ---------------------------------------------------------------------------

def test_smoke_loss_v2_trainer_integration():
    """Kiểm tra CrossDomainTrainer tích hợp hoàn chỉnh với Loss v2."""
    device = torch.device("cpu")
    student = MockStudentModel().to(device)
    teacher = MockTeacherModel().to(device)

    dataset = MockPairedDataset()
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    criterion = CrossDomainTotalLoss(
        weight_ce=0.3,
        alpha_kd=0.7,
        beta_cons=0.2,
        temperature=2.0,
        label_smoothing=0.05,
        fake_index=1,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    # Khởi tạo Trainer với trọng số v2
    trainer = CrossDomainTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=None,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=False,
        weight_feat=0.3,
        weight_guard=0.5,
    )

    assert trainer.use_features is True, "use_features phải True khi có weight_feat hoặc weight_guard > 0"

    # Lưu trọng số trước khi huấn luyện để kiểm tra optimizer.step()
    weight_before = student.head.weight.clone()

    # Chạy 1 epoch huấn luyện (2 batches)
    metrics = trainer.train_epoch(epoch=1)

    assert "loss_total" in metrics and metrics["loss_total"] > 0
    assert "loss_ce" in metrics and metrics["loss_ce"] > 0
    assert "loss_kd" in metrics
    assert "loss_cons" in metrics
    assert "loss_feat" in metrics and metrics["loss_feat"] > 0, "loss_feat phải được tính và > 0"
    assert "loss_guard" in metrics and metrics["loss_guard"] >= 0, "loss_guard phải được tính và >= 0"
    assert "guard_active_ratio" in metrics

    # Xác nhận trọng số mô hình đã được tối ưu hóa qua optimizer.step()
    assert not torch.equal(student.head.weight, weight_before), "Trọng số student phải thay đổi sau bước tối ưu"


# ---------------------------------------------------------------------------
# Test 2: Nạp cấu hình configs/train_phase7_anchored_v2.yaml qua parser
# ---------------------------------------------------------------------------

def test_smoke_config_yaml_loss_v2_parsing(monkeypatch):
    """Kiểm tra parser đọc chính xác các tham số Loss v2 từ file YAML."""
    config_file = "configs/train_phase7_anchored_v2.yaml"
    assert os.path.exists(config_file), f"Không tìm thấy file cấu hình {config_file}"

    # Giả lập tham số dòng lệnh CLI
    test_args = ["train_crossdomain.py", "--config", config_file]
    monkeypatch.setattr(sys, "argv", test_args)

    args = parse_arguments()

    assert args.weight_feat == 0.3, f"weight_feat kỳ vọng 0.3, nhận {args.weight_feat}"
    assert args.weight_guard == 0.5, f"weight_guard kỳ vọng 0.5, nhận {args.weight_guard}"
    assert args.fake_index == 1, f"fake_index kỳ vọng 1, nhận {args.fake_index}"
    assert args.alpha_kd == 0.7, f"alpha_kd kỳ vọng 0.7, nhận {args.alpha_kd}"
    assert args.beta_cons == 0.2, f"beta_cons kỳ vọng 0.2, nhận {args.beta_cons}"


# ---------------------------------------------------------------------------
# Test 3: Tương thích ngược khi weight_feat=weight_guard=0
# ---------------------------------------------------------------------------

def test_smoke_backward_compat_legacy_mode():
    """Khi weight_feat=weight_guard=0, use_features=False và không trích xuất features."""
    device = torch.device("cpu")
    student = MockStudentModel().to(device)
    dataset = MockPairedDataset()
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    criterion = CrossDomainTotalLoss(weight_ce=0.7, alpha_kd=0.3, beta_cons=0.2)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    trainer = CrossDomainTrainer(
        student_model=student,
        teacher_model=None,
        train_loader=train_loader,
        val_loader=None,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=False,
        weight_feat=0.0,
        weight_guard=0.0,
    )

    assert trainer.use_features is False
    metrics = trainer.train_epoch(epoch=1)
    assert "loss_feat" not in metrics
    assert "loss_guard" not in metrics
    assert metrics["loss_total"] > 0

