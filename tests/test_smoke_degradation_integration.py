"""
tests/test_smoke_degradation_integration.py
Smoke test kiểm thử tích hợp toàn diện Task 7.8:
1. Luồng Paired: YAML config -> CrossDomainDataset -> DataLoader -> 1 bước forward/loss.
2. Luồng Single-Tensor: DatasetCreator.build_train_dataset -> DataLoader -> 1 bước Trainer forward (bảo toàn hợp đồng tensor cũ).
3. Thống nhất Single Source of Truth (YAML 224, q_range binding).
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.transforms import DegradationPipelineV2, PairedCleanDegradedAugmentation
from src.datasets.dataset import DatasetCreator
from src.training.train_crossdomain import CrossDomainDataset
from src.training.loss import CrossDomainTotalLoss


# ---------------------------------------------------------------------------
# Fixture: Tạo thư mục dữ liệu giả lập có ảnh thật và ảnh giả
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dataset_dir():
    """Tạo cây thư mục tạm với cấu trúc chuẩn 0_real và 1_fake để test DatasetCreator."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = os.path.join(tmpdir, "train")
        real_dir = os.path.join(train_dir, "0_real")
        fake_dir = os.path.join(train_dir, "1_fake")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

        # Tạo 2 ảnh real và 2 ảnh fake
        for i in range(2):
            img_r = Image.new("RGB", (256, 256), color=(50 * (i + 1), 100, 150))
            img_r.save(os.path.join(real_dir, f"real_{i}.png"))
            img_f = Image.new("RGB", (256, 256), color=(150, 100, 50 * (i + 1)))
            img_f.save(os.path.join(fake_dir, f"fake_{i}.png"))

        yield tmpdir


# ---------------------------------------------------------------------------
# Test 1: Luồng Paired CrossDomain (v2 + Teacher KD + Consistency)
# ---------------------------------------------------------------------------

def test_smoke_paired_crossdomain_flow():
    """
    Kiểm thử luồng Paired:
    YAML config -> CrossDomainDataset(paired=True) -> DataLoader -> Forward + Loss.
    """
    config_path = "configs/degradation_v2.yaml"
    assert os.path.exists(config_path), f"Không tìm thấy file cấu hình {config_path}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Tạo 4 ảnh mẫu
        records = []
        for i in range(4):
            img_path = os.path.join(tmpdir, f"sample_{i}.png")
            img = Image.new("RGB", (256, 256), color=(i * 40, 120, 200))
            img.save(img_path)
            records.append({
                "file_path": img_path,
                "label": i % 2,
                "generator": "sdv15" if i % 2 == 1 else "real",
            })

        # 1. Khởi tạo CrossDomainDataset nhận degradation_config v2
        ds = CrossDomainDataset(
            records=records,
            paired=True,
            img_resolution=224,
            crop_resolution=224,
            degradation_config=config_path,
        )
        assert isinstance(ds.paired_aug, DegradationPipelineV2), \
            "CrossDomainDataset paired_aug phải là DegradationPipelineV2 khi truyền degradation_config"

        # 2. DataLoader nạp batch dạng cặp
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        assert isinstance(batch[0], (list, tuple)), "Hợp đồng paired: batch[0] phải là tuple/list (x_clean, x_degraded)"
        (x_clean, x_degraded), targets = batch

        assert x_clean.shape == (2, 3, 224, 224), f"Shape x_clean sai: {x_clean.shape}"
        assert x_degraded.shape == (2, 3, 224, 224), f"Shape x_degraded sai: {x_degraded.shape}"
        assert targets.shape == (2,), f"Shape targets sai: {targets.shape}"

        # 3. Một bước tính loss (Student + Teacher + Paired Consistency)
        criterion = CrossDomainTotalLoss(
            weight_ce=0.7,
            alpha_kd=0.3,
            beta_cons=0.2,
            temperature=2.0,
            label_smoothing=0.05,
        )

        # Mô phỏng logits đầu ra của student và teacher
        student_clean_logits = torch.randn(2, 2, requires_grad=True)
        student_degraded_logits = torch.randn(2, 2, requires_grad=True)
        teacher_logits = torch.randn(2, 2)

        loss_dict = criterion(
            logits=student_clean_logits,
            targets=targets,
            teacher_logits=teacher_logits,
            logits_degraded=student_degraded_logits,
        )
        loss = loss_dict["loss_total"]

        assert not torch.isnan(loss) and loss.item() > 0, "Loss tính toán không hợp lệ"
        assert "loss_ce" in loss_dict and "loss_kd" in loss_dict and "loss_cons" in loss_dict
        loss.backward()
        assert student_clean_logits.grad is not None, "Backward gradient cho student_clean_logits thất bại"


# ---------------------------------------------------------------------------
# Test 2: Luồng Đơn Tensor của src.main / Trainer (Bảo toàn hợp đồng cũ)
# ---------------------------------------------------------------------------

def test_smoke_single_tensor_main_flow(mock_dataset_dir):
    """
    Kiểm thử luồng đơn tensor của DatasetCreator & Trainer:
    Mặc định phải trả về 1 tensor (3, 224, 224), không được trả về tuple làm vỡ Trainer.
    """
    creator = DatasetCreator(
        dataset_path=mock_dataset_dir,
        img_resolution=256,
        crop_resolution=224,
        batch_size=2,
        num_workers=0,
    )

    # 1. Luồng mặc định (như src/main.py dòng 141)
    train_dataset = creator.build_train_dataset(use_dual_stream=True)
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False)

    images, labels = next(iter(loader))
    assert isinstance(images, torch.Tensor), \
        f"Lỗi hợp đồng tensor: images phải là torch.Tensor, nhận được {type(images)}"
    assert images.shape == (2, 3, 224, 224), f"Shape images sai: {images.shape}"
    assert labels.shape == (2,), f"Shape labels sai: {labels.shape}"

    # Kiểm tra phương thức images.to(device) hoạt động chuẩn (điểm crash cũ của Trainer:75)
    device = torch.device("cpu")
    images_dev = images.to(device)
    assert images_dev.device == device

    # 2. Mô phỏng 1 forward step của Trainer
    linear_head = nn.Linear(3 * 224 * 224, 2)
    flat_img = images_dev.view(2, -1)
    logits = linear_head(flat_img)
    ce_loss = nn.CrossEntropyLoss()(logits, labels)
    ce_loss.backward()
    assert linear_head.weight.grad is not None


# ---------------------------------------------------------------------------
# Test 3: Cấu hình Single Source of Truth & q_range Binding
# ---------------------------------------------------------------------------

def test_smoke_single_source_of_truth():
    """
    Kiểm tra:
    - Default img_resolution == 224 (khớp YAML và ViT-L/14).
    - jpeg.q_range điều khiển được dải phân bổ heavy_range và light_range.
    """
    # 1. Pipeline từ YAML chuẩn
    pipe_yaml = DegradationPipelineV2.from_config("configs/degradation_v2.yaml")
    assert pipe_yaml.img_resolution == 224
    assert pipe_yaml.crop_resolution == 224

    # 2. Pipeline khởi tạo không tham số
    pipe_def = DegradationPipelineV2()
    assert pipe_def.img_resolution == 224
    assert pipe_def.crop_resolution == 224

    # 3. jpeg.q_range tuỳ biến
    custom_cfg = {
        "jpeg": {
            "q_range": [20, 60],
            "heavy_share": 0.40,
        }
    }
    pipe_custom = DegradationPipelineV2(config=custom_cfg)
    assert pipe_custom._jpeg_q_range == (20, 60)
    assert pipe_custom._jpeg_heavy_range[0] == 20
    assert pipe_custom._jpeg_light_range[1] == 60
    assert pipe_custom._jpeg_heavy_range[1] <= pipe_custom._jpeg_light_range[0]

