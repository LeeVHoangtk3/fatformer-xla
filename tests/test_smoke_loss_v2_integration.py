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
    def __init__(self, in_features=3, hidden_dim=12, num_classes=2):
        super().__init__()
        self.encoder = nn.Linear(in_features, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_features=False):
        flat_x = x.mean(dim=(-2, -1))
        feat = torch.relu(self.encoder(flat_x))
        feat_norm = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        logits = self.head(feat_norm)
        if return_features:
            return logits, feat_norm
        return logits


class MockTeacherModel(nn.Module):
    """Mô phỏng Teacher Model đóng băng."""
    def __init__(self, in_features=3, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        flat_x = x.mean(dim=(-2, -1))
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

def test_smoke_loss_v2_trainer_integration(tmp_path):
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
        save_dir=str(tmp_path),
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
    for key in ("ce_clean", "ce_deg", "kd", "cons", "feat_cos", "feat_kl", "guard_excess_mean", "grad_norm"):
        assert key in metrics and torch.isfinite(torch.tensor(metrics[key]))
    assert not teacher.training
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert metrics["loss_total"] == pytest.approx(
        metrics["loss_ce"] + metrics["loss_kd"] + metrics["loss_cons"]
        + metrics["weighted_feat"] + metrics["weighted_guard"], abs=1e-6)

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
    assert args.weight_ce_deg == .3
    assert args.degradation_config == "configs/degradation_v2.yaml"
    assert args.train_manifest == "DATASET/manifests/data_manifest_v2.json"
    assert args.val_split == "val_select"
    assert args.checkpoint_init == args.teacher_checkpoint == "DATASET/pretrained/fatformer_4class_ckpt.pth"
    assert args.save_name == "fatformer_anchored_v2_best.pth"
    assert args.anchor_learning_rate == 1e-5


# ---------------------------------------------------------------------------
# Test 3: Tương thích ngược khi weight_feat=weight_guard=0
# ---------------------------------------------------------------------------

def test_smoke_backward_compat_legacy_mode(tmp_path):
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
        save_dir=str(tmp_path),
    )

    assert trainer.use_features is False
    metrics = trainer.train_epoch(epoch=1)
    assert "loss_feat" not in metrics
    assert "loss_guard" not in metrics
    assert metrics["loss_total"] > 0


def test_guard_only_uses_logits_without_feature_api(tmp_path):
    student = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))
    teacher = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))
    trainer = CrossDomainTrainer(student, teacher, DataLoader(MockPairedDataset(), batch_size=2),
                                 None, CrossDomainTotalLoss(weight_ce=.3),
                                 torch.optim.SGD(student.parameters(), lr=.01),
                                 weight_guard=.5, save_dir=str(tmp_path), grad_accum_steps=2)
    assert not trainer.use_features and trainer.use_v2
    metrics = trainer.train_epoch(1, max_steps=3)
    assert metrics["train_steps"] == 3
    assert metrics["guard_active_ratio"] > 0


def test_missing_teacher_is_rejected_before_training(tmp_path):
    student = MockStudentModel()
    with pytest.raises(ValueError, match="teacher"):
        CrossDomainTrainer(student, None, DataLoader(MockPairedDataset(), batch_size=2),
                           None, CrossDomainTotalLoss(), torch.optim.SGD(student.parameters(), lr=.01),
                           weight_guard=.5, save_dir=str(tmp_path))


def test_yaml_cli_dataset_trainer_flow_preserves_calibration_split(tmp_path, monkeypatch):
    import json
    import yaml
    from PIL import Image
    from src.datasets.transforms import DegradationPipelineV2
    from src.training.train_crossdomain import build_crossdomain_dataloaders
    rows = []
    for i, split in enumerate(["train", "train", "train", "train", "val_select", "val_select", "val_calib"]):
        path = tmp_path / f"image_{i}.png"
        Image.effect_noise((256, 256), 60).convert("RGB").save(path)
        rows.append({"path": str(path), "split": split, "label": i % 2, "generator": "real" if i % 2 == 0 else "test"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": rows}), encoding="utf-8")
    with open("configs/train_phase7_anchored_v2.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"].update(manifest=str(manifest), num_workers=0)
    cfg["training"].update(device="cpu", batch_size=2)
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config), "--max_steps", "1"])
    args = parse_arguments()
    train, val = build_crossdomain_dataloaders(args)
    assert isinstance(train.dataset.paired_aug, DegradationPipelineV2)
    assert {r["split"] for r in val.dataset.records} == {"val_select"}
    assert len(val.dataset) == 2
    student, teacher = MockStudentModel(), MockTeacherModel()
    loss = CrossDomainTotalLoss(weight_ce=args.weight_ce, weight_ce_deg=args.weight_ce_deg,
                                alpha_kd=args.alpha_kd, beta_cons=args.beta_cons)
    trainer = CrossDomainTrainer(student, teacher, train, val, loss,
                                 torch.optim.SGD(student.parameters(), lr=.01),
                                 weight_feat=args.weight_feat, weight_guard=args.weight_guard,
                                 save_dir=str(tmp_path / "checkpoints"))
    metrics = trainer.train_epoch(1, max_steps=args.max_steps)
    assert metrics["train_steps"] == 1
    assert torch.isfinite(torch.tensor(metrics["loss_total"]))
    assert trainer.validate()["val_acc"] >= 0


def test_missing_image_never_becomes_synthetic_training_data(tmp_path):
    from src.training.train_crossdomain import CrossDomainDataset
    ds = CrossDomainDataset([{"path": str(tmp_path / "missing.png"), "label": 0}])
    with pytest.raises(FileNotFoundError, match="training image"):
        ds[0]


def test_mock_cli_pipeline_supports_loss_v2_and_max_steps(tmp_path, monkeypatch):
    import src.training.train_crossdomain as train
    monkeypatch.setattr(sys, "argv", ["train", "--config", "configs/train_phase7_anchored_v2.yaml",
                                      "--dry_run", "--device", "cpu", "--max_steps", "1",
                                      "--save_dir", str(tmp_path)])
    args = parse_arguments()
    monkeypatch.setattr(train, "find_first_existing", lambda _: None)
    monkeypatch.setattr(train.CrossDomainTrainer, "save_checkpoint", lambda *a, **kw: None)
    train.run_training_pipeline(args)


def test_production_clip_forward_features_preserve_logits():
    """Exercise the actual wrapper forward with tiny encoders, without weights/GPU."""
    from src.models.clip_models import CLIPModel

    class ImageEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 12)

        def forward(self, image, return_full=False):
            features = self.proj(image.mean(dim=(-2, -1)))
            return torch.stack([features, features * .5, features * 1.5], dim=1)

    class Alignment(nn.Module):
        def forward(self, features):
            return features.unsqueeze(1).expand(-1, 2, -1, -1).contiguous()

    class TextEncoder(nn.Module):
        def forward(self, prompts, tokenized):
            return prompts.mean(dim=1)

    # Bypass checkpoint construction only; all logit/feature math is production code.
    model = CLIPModel.__new__(CLIPModel)
    nn.Module.__init__(model)
    model.logit_scale = nn.Parameter(torch.zeros(()))
    model.tokenized_prompts = torch.zeros(2, 3, dtype=torch.long)
    model.image_encoder = ImageEncoder()
    model.language_guided_alignment = Alignment()
    model.text_encoder = TextEncoder()
    model.text_guided_interactor = nn.MultiheadAttention(12, 3)
    model.norm1, model.norm2 = nn.LayerNorm(12), nn.LayerNorm(12)
    model.linear1, model.linear2 = nn.Linear(12, 24), nn.Linear(24, 12)
    model.activation = nn.ReLU()
    model.eval()
    image = torch.randn(2, 3, 16, 16)
    state_keys = set(model.state_dict())
    logits = model(image)
    with_features, features = model(image, return_features=True)
    visual, alignment = model(image, return_dual=True)
    assert torch.equal(logits, with_features)
    assert torch.equal(logits, visual + alignment)
    assert features.shape == (2, 12)
    features.square().sum().backward()
    assert model.image_encoder.proj.weight.grad is not None
    assert set(model.state_dict()) == state_keys

