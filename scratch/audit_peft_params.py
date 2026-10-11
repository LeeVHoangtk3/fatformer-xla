import sys
import os

PROJECT_ROOT = r"d:\GIT REPO\.nam4\fatformer-xla"
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from src.models.clip_models import CLIPModel
from src.training.freeze_utils import freeze_clip_backbone, count_trainable_parameters

class Args:
    use_srm = True
    use_gating = True
    num_classes = 2
    num_vit_adapter = 8
    clip_path = r"d:\GIT REPO\.nam4\fatformer-xla\DATASET\pretrained\ViT-L-14.pt"

args = Args()
print("[*] Đang khởi tạo CLIPModel...")
model = CLIPModel(name=args.clip_path, args=args)

print("\n[*] Chạy freeze_clip_backbone...")
try:
    stats = freeze_clip_backbone(model, verbose=True)
except Exception as e:
    print(f"[!] Bắt ngoại lệ: {e}")
    stats = count_trainable_parameters(model)

print("\n[*] Chi tiết các tham số ĐANG ĐƯỢC MỞ KHÓA (requires_grad = True):")
trainable_by_module = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        prefix = name.split(".")[0]
        if prefix not in trainable_by_module:
            trainable_by_module[prefix] = {"count": 0, "params": 0}
        trainable_by_module[prefix]["count"] += 1
        trainable_by_module[prefix]["params"] += param.numel()

print("\n[*] 20 tham số đầu tiên trong clip_model có requires_grad=True:")
count = 0
for name, param in model.named_parameters():
    if param.requires_grad and name.startswith("clip_model"):
        print(f"  - {name} ({param.shape}, {param.numel():,} params)")
        count += 1
        if count >= 20:
            break

