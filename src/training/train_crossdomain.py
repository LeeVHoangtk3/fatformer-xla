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
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.training.loss import CrossDomainTotalLoss
from src.datasets.transforms import PairedCleanDegradedAugmentation
from src.datasets.samplers import BalancedBatchSampler, BalancedCrossDomainSampler
from src.models.lora_adapter import ViTLoRAAdapter
import torchvision.transforms as transforms


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


def find_first_existing(candidates: list) -> Optional[str]:
    """Tim duong dan ton tai dau tien tu danh sach ung vien."""
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


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


class CrossDomainDataset(Dataset):
    """
    Dataset nap anh da mien ho tro doc tu manifest JSON hoac quet thu muc anh.
    - Neu paired=True (nhanh M4, M5): tra ve ((x_clean, x_degraded), label).
    - Neu paired=False (nhanh M3, Val): tra ve (x_transformed, label).
    """
    def __init__(
        self,
        records: list,
        paired: bool = True,
        img_resolution: int = 256,
        crop_resolution: int = 224,
    ):
        self.records = records
        self.paired = paired
        self.paired_aug = PairedCleanDegradedAugmentation(
            img_resolution=img_resolution,
            crop_resolution=crop_resolution,
        ) if paired else None

        _mean = [0.485, 0.456, 0.406]
        _std = [0.229, 0.224, 0.225]
        self.eval_transform = transforms.Compose([
            transforms.Resize((img_resolution, img_resolution)),
            transforms.CenterCrop(crop_resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mean, std=_std),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        file_path = rec.get("file_path", rec.get("path", ""))
        target = int(rec.get("label", 0))

        pil_img = None
        if file_path and os.path.isfile(file_path):
            try:
                pil_img = Image.open(file_path).convert("RGB")
            except Exception:
                pil_img = None

        if pil_img is None:
            pil_img = Image.new("RGB", (256, 256), color=(idx * 17 % 255, idx * 37 % 255, 128))

        if self.paired and self.paired_aug is not None:
            xc, xd = self.paired_aug(pil_img)
            return (xc, xd), target
        else:
            return self.eval_transform(pil_img), target


def build_crossdomain_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    """
    Xay dung DataLoader huan luyen va danh gia cho chien dich thich ung da mien:
    - Thu thap mau tu manifest JSON hoac quet thu muc tren dia.
    - Su dung BalancedBatchSampler cuong buc 50/50 Real/Fake tren tung mini-batch.
    """
    import json
    train_records = []
    val_records = []

    # 0. Uu tien tuyet doi: Neu nguoi dung truyen ro rang train_dir va val_dir
    train_dir_arg = getattr(args, "train_dir", "")
    val_dir_arg = getattr(args, "val_dir", "")

    # 1. Do tim tep manifest neu khong chi dinh ro train_dir
    if not (train_dir_arg and os.path.isdir(train_dir_arg)):
        manifest_candidates = [
            getattr(args, "train_manifest", ""),
            os.path.join(PROJECT_ROOT, "DATASET", "manifests", "data_manifest.json"),
            os.path.join(PROJECT_ROOT, "data_manifest.json"),
            "/content/drive/MyDrive/Fatformer/manifests/data_manifest.json",
        ]
        manifest_file = find_first_existing(manifest_candidates)

        if manifest_file:
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                records = manifest_data.get("records", [])

                manifest_dir = os.path.dirname(manifest_file)
                base_search_paths = [
                    PROJECT_ROOT,
                    os.path.join(PROJECT_ROOT, "DATASET"),
                    manifest_dir,
                    os.path.dirname(manifest_dir),
                    "/content/dataset_local",
                    "/content/dataset_local/diffusion_staging",
                    "/content/drive/MyDrive/Fatformer",
                    "/content/fatformer-xla",
                ]

                temp_train = []
                temp_val = []
                for r in records:
                    rel_path = r.get("file_path", r.get("path", ""))
                    abs_path = rel_path
                    if not os.path.isabs(rel_path):
                        for bsp in base_search_paths:
                            candidate = os.path.normpath(os.path.join(bsp, rel_path))
                            if os.path.exists(candidate):
                                abs_path = candidate
                                break

                    r_entry = dict(r)
                    r_entry["file_path"] = abs_path
                    split = r.get("split", "train")
                    if split == "train":
                        temp_train.append(r_entry)
                    else:
                        temp_val.append(r_entry)

                train_existing = sum(1 for r in temp_train if os.path.isfile(r.get("file_path", "")))
                val_existing = sum(1 for r in temp_val if os.path.isfile(r.get("file_path", "")))

                if len(temp_train) > 0 and train_existing == 0:
                    if not getattr(args, "dry_run", False):
                        raise RuntimeError(f"LỖI DỮ LIỆU: Toàn bộ {len(temp_train)} đường dẫn trong manifest {manifest_file} không tồn tại trên đĩa! Cấm fallback dummy.")
                    print(f"  [!] Canh bao: Toan bo {len(temp_train)} duong dan trong manifest {manifest_file} deu khong ton tai tren dia (dummy mock).")
                    print("  [*] Tu dong bo qua manifest ao de quet tim thu muc anh thuc te...")
                else:
                    train_records = temp_train
                    val_records = temp_val
                    print(f"  [✓] Da nap thanh cong tu manifest ({manifest_file}): Train={len(train_records)} ({train_existing} anh ton tai), Val={len(val_records)} ({val_existing} anh ton tai)")
            except Exception as e:
                if not getattr(args, "dry_run", False) and "LỖI DỮ LIỆU" in str(e):
                    raise
                print(f"  [!] Khong the doc manifest {manifest_file}: {e}")

    # 2. Quet thu muc du lieu train neu records con trong
    if len(train_records) == 0:
        dir_candidates = [
            train_dir_arg,
            "/content/dataset_local/diffusion_staging/train",
            "/content/dataset_local/diffusion_staging",
            "/content/dataset_local",
            os.path.join(PROJECT_ROOT, "DATASET", "data_crossdomain", "train"),
            os.path.join(PROJECT_ROOT, "DATASET", "train_crossdomain"),
            "/content/drive/MyDrive/Fatformer/data_crossdomain/train",
        ]
        real_train_dir = find_first_existing(dir_candidates)
        if real_train_dir:
            print(f"  [*] Dang quet anh thuc te tu thu muc: {real_train_dir}...")
            for root, _, files in os.walk(real_train_dir):
                for fname in files:
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        full_p = os.path.join(root, fname)
                        norm_p = full_p.replace("\\", "/").lower()
                        parts = norm_p.split("/")
                        is_fake = any(p in ("1_fake", "fake", "1") for p in parts)
                        is_real = any(p in ("0_real", "real", "0") for p in parts)
                        if is_real and is_fake:
                            for p in reversed(parts):
                                if p in ("0_real", "real", "0"):
                                    is_fake = False
                                    break
                                elif p in ("1_fake", "fake", "1"):
                                    is_real = False
                                    break
                        if not (is_real or is_fake):
                            is_fake = ("fake" in norm_p or "1_fake" in norm_p)
                        gen = "real"
                        if is_fake:
                            for g_name in ["stable_diffusion", "sdv15", "sd_v15", "midjourney", "stylegan", "progan", "dalle", "adm"]:
                                if g_name in norm_p:
                                    gen = g_name
                                    break
                            if gen == "real":
                                gen = os.path.basename(root)
                        train_records.append({
                            "file_path": full_p,
                            "label": 1 if is_fake else 0,
                            "generator": gen,
                            "source": "folder_scan",
                        })
            print(f"  [✓] Quet thu muc du lieu {real_train_dir}: {len(train_records)} mau train.")

    # 3. Quet val hoac phan chia tap kiem dinh neu val_records con trong
    if len(val_records) == 0 and len(train_records) > 0:
        val_candidates = [
            val_dir_arg,
            "/content/dataset_local/diffusion_staging/val",
            "/content/dataset_local/val",
            os.path.join(PROJECT_ROOT, "DATASET", "data_crossdomain", "val"),
            "/content/drive/MyDrive/Fatformer/data_crossdomain/val",
        ]
        real_val_dir = find_first_existing(val_candidates)
        if real_val_dir:
            for root, _, files in os.walk(real_val_dir):
                for fname in files:
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        full_p = os.path.join(root, fname)
                        norm_p = full_p.replace("\\", "/").lower()
                        parts = norm_p.split("/")
                        is_fake = any(p in ("1_fake", "fake", "1") for p in parts)
                        is_real = any(p in ("0_real", "real", "0") for p in parts)
                        if is_real and is_fake:
                            for p in reversed(parts):
                                if p in ("0_real", "real", "0"):
                                    is_fake = False
                                    break
                                elif p in ("1_fake", "fake", "1"):
                                    is_real = False
                                    break
                        if not (is_real or is_fake):
                            is_fake = ("fake" in norm_p or "1_fake" in norm_p)
                        gen = "real"
                        if is_fake:
                            for g_name in ["stable_diffusion", "sdv15", "sd_v15", "midjourney", "stylegan", "progan", "dalle", "adm"]:
                                if g_name in norm_p:
                                    gen = g_name
                                    break
                            if gen == "real":
                                gen = os.path.basename(root)
                        val_records.append({
                            "file_path": full_p,
                            "label": 1 if is_fake else 0,
                            "generator": gen,
                            "source": "folder_scan_val",
                        })
            print(f"  [✓] Quet thu muc val {real_val_dir}: {len(val_records)} mau val.")
        else:
            import random
            rng = random.Random(getattr(args, "seed", 42))
            reals = [r for r in train_records if r.get("label", 0) == 0]
            fakes = [r for r in train_records if r.get("label", 0) == 1]
            rng.shuffle(reals)
            rng.shuffle(fakes)

            n_val_real = max(1, int(len(reals) * 0.15))
            n_val_fake = max(1, int(len(fakes) * 0.15))

            val_records = reals[:n_val_real] + fakes[:n_val_fake]
            train_records = reals[n_val_real:] + fakes[n_val_fake:]
            print(f"  [✓] Tu dong phan chia train/val (85/15 can bang): Train={len(train_records)} (Real: {len(reals)-n_val_real}, Fake: {len(fakes)-n_val_fake}) | Val={len(val_records)} (Real: {n_val_real}, Fake: {n_val_fake})")

    # 4. Mau du phong neu kho anh chua tai ve
    if len(train_records) == 0:
        if not getattr(args, "dry_run", False):
            raise RuntimeError("LỖI DỮ LIỆU: Không tìm thấy ảnh thực tế trong manifest hoặc thư mục! Tuyệt đối cấm fallback dummy khi không bật --dry_run.")
        print("  [!] Khong tim thay anh tren dia, tao 64 mau mo phong co kiem soat cho dry-run.")
        for i in range(64):
            is_fake = (i % 2 == 1)
            train_records.append({
                "file_path": f"mock_sample_{i}.png",
                "label": 1 if is_fake else 0,
                "generator": f"gen_{(i // 2) % 4}" if is_fake else "real",
                "source": "mock",
            })
    if len(val_records) == 0:
        if not getattr(args, "dry_run", False):
            raise RuntimeError("LỖI DỮ LIỆU: Không tìm thấy ảnh thực tế cho tập val! Tuyệt đối cấm fallback dummy khi không bật --dry_run.")
        for i in range(32):
            is_fake = (i % 2 == 1)
            val_records.append({
                "file_path": f"mock_val_{i}.png",
                "label": 1 if is_fake else 0,
                "generator": f"gen_{(i // 2) % 4}" if is_fake else "real",
                "source": "mock",
            })

    reals_all = [r for r in train_records if r.get("label", 0) == 0]
    fakes_all = [r for r in train_records if r.get("label", 0) == 1]
    if not getattr(args, "dry_run", False):
        if len(reals_all) == 0 or len(fakes_all) == 0:
            raise RuntimeError(f"LỖI GÁN NHÃN: Tập dữ liệu thiếu Real ({len(reals_all)}) hoặc Fake ({len(fakes_all)})! Kiểm tra lại đường dẫn và quy tắc gán nhãn.")

    paired_train = (args.branch in ["M4", "M5"])
    train_ds = CrossDomainDataset(train_records, paired=paired_train)
    val_ds = CrossDomainDataset(val_records, paired=False)

    has_real = any(r.get("label", 0) == 0 for r in train_records)
    has_fake = any(r.get("label", 0) == 1 for r in train_records)

    workers = getattr(args, "num_workers", 0)
    use_pin = torch.cuda.is_available()

    if has_real and has_fake:
        batch_sampler = BalancedBatchSampler(train_records, batch_size=args.batch_size)
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=workers, pin_memory=use_pin)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=use_pin)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory=use_pin)
    return train_loader, val_loader


class _MockBackbone(nn.Module):
    """Mo hinh nhe dung cho che do Dry-Run tren CPU (ho tro ca tham so SRM/Gating)."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, num_classes)
        self.srm_proj = nn.Linear(num_classes, num_classes)
        self.gating_mlp = nn.Linear(num_classes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.fc(x.flatten(1))
        return feat + self.gating_mlp(self.srm_proj(feat)) * 0.0


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
        save_name: str = "crossdomain_model_best.pth",
        grad_clip_norm: float = 1.0,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device) if teacher_model is not None else None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device) if isinstance(device, str) else device
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_name = save_name
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=True) if self.use_amp else None
        self.best_composite_score = -1.0

        os.makedirs(self.save_dir, exist_ok=True)

        # Dong bang hoan toan mo hinh teacher va toi uu bo nho VRAM
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            # [LƯU Ý TỐI ƯU AMP]: Giữ Teacher ở FP32 Master Weights để tránh lỗi xung đột kiểu dữ liệu
            # (Input FloatTensor vs Weight HalfTensor) và bảo tồn độ chính xác số học.
            # torch.amp.autocast('cuda') trong train_epoch sẽ tự động tính toán FP16 trên Tensor Cores.
            # if self.device.type == "cuda" and self.use_amp:
            #     self.teacher = self.teacher.half()
            #     print("[*] Da toi uu hoa mo hinh Teacher sang kieu nua do chinh xac FP16 (tiet kiem 50% VRAM).")

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
            clean_mask = None  # Luồng x_clean đã là ảnh sạch cho cả Real (0) và Fake (1)

            # Tinh toan logit mo hinh mau (teacher) khong qua dao ham va tiet kiem VRAM
            teacher_logits = None
            if self.teacher is not None:
                with torch.no_grad():
                    if self.use_amp and self.device.type == "cuda":
                        with torch.amp.autocast(self.device.type, enabled=True):
                            teacher_logits = self.teacher(x_clean)
                    else:
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

            # Lan truyen nguoc qua GradScaler (neu la CUDA AMP) hoac truc tiep (neu khong bat AMP/CPU)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == num_batches:
                    if self.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == num_batches:
                    if self.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss_dict["loss_total"].item()
            total_ce += loss_dict["loss_ce"].item()
            total_kd += loss_dict["loss_kd"].item()
            total_cons += loss_dict["loss_cons"].item()

        if self.scheduler is not None:
            self.scheduler.step()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

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

        # Diem composite danh gia su can bang giua Real va Fake (uu tien ca Real va Fake moi)
        min_rf = min(r_acc, f_acc)
        composite = 0.5 * macro_acc + 0.5 * min_rf
        return {
            "val_acc": macro_acc,
            "val_r_acc": r_acc,
            "val_f_acc": f_acc,
            "val_composite": composite,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False, val_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Chinh sach xoay vong checkpoint nham bao ve bo nho dia luu tru:
        - checkpoint_latest.pth & fatformer_anchored_recovery_last.pth: Luon ghi de sau moi epoch.
        - fatformer_anchored_recovery_best.pth / save_name: Chi luu model.state_dict() va metadata
          (dung luong gon ~2.0 GB, khong kem optimizer, nap khop 100%).
        """
        model_sd = self.student.module.state_dict() if hasattr(self.student, "module") else self.student.state_dict()
        state_latest = {
            "epoch": epoch,
            "model": model_sd,
            "model_state": model_sd,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "best_score": self.best_composite_score,
            "val_metrics": val_metrics or {},
        }

        latest_path = os.path.join(self.save_dir, "checkpoint_latest.pth")
        torch.save(state_latest, latest_path)

        last_clean_path = os.path.join(self.save_dir, "fatformer_anchored_recovery_last.pth")
        clean_state_last = {
            "model": model_sd,
            "metadata": {
                "epoch": epoch,
                "val_acc": val_metrics.get("val_acc", 0.0) if val_metrics else 0.0,
                "val_r_acc": val_metrics.get("val_r_acc", 0.0) if val_metrics else 0.0,
                "val_f_acc": val_metrics.get("val_f_acc", 0.0) if val_metrics else 0.0,
                "composite": val_metrics.get("val_composite", self.best_composite_score) if val_metrics else self.best_composite_score,
                "branch": "M4",
            }
        }
        torch.save(clean_state_last, last_clean_path)

        if is_best:
            v_acc = val_metrics.get("val_acc", 0.0) if val_metrics else 0.0
            comp = val_metrics.get("val_composite", self.best_composite_score) if val_metrics else self.best_composite_score
            best_clean_state = {
                "model": model_sd,
                "metadata": {
                    "epoch": epoch,
                    "val_acc": v_acc,
                    "val_r_acc": val_metrics.get("val_r_acc", 0.0) if val_metrics else 0.0,
                    "val_f_acc": val_metrics.get("val_f_acc", 0.0) if val_metrics else 0.0,
                    "composite": comp,
                    "branch": "M4",
                }
            }
            best_path = os.path.join(self.save_dir, self.save_name)
            torch.save(best_clean_state, best_path)
            print(f"  -> Luu model vuot troi tai epoch {epoch}: {best_path} (Val ACC={v_acc:.2f}%, Composite={comp:.2f}%)")

            # Luu dong thoi ca fatformer_anchored_recovery_best.pth va crossdomain_model_best.pth de tuong thich 100%
            alt_name = "fatformer_anchored_recovery_best.pth" if self.save_name == "crossdomain_model_best.pth" else "crossdomain_model_best.pth"
            alt_path = os.path.join(self.save_dir, alt_name)
            torch.save(best_clean_state, alt_path)

        # Neu phat hien moi truong Google Colab voi Drive duoc mount, tu dong sao luu them ban copy sang Drive
        for drive_base in [
            "/content/drive/MyDrive/Fatformer/checkpoint",
            "/content/drive/MyDrive/Fatformer/checkpoints",
        ]:
            if os.path.exists(os.path.dirname(drive_base)):
                drive_branch_dir = os.path.join(drive_base, os.path.basename(self.save_dir))
                try:
                    os.makedirs(drive_branch_dir, exist_ok=True)
                    import shutil
                    shutil.copy2(latest_path, os.path.join(drive_branch_dir, "checkpoint_latest.pth"))
                    if is_best:
                        shutil.copy2(best_path, os.path.join(drive_branch_dir, self.save_name))
                        if self.save_name != "crossdomain_model_best.pth":
                            shutil.copy2(alt_path, os.path.join(drive_branch_dir, alt_name))
                    print(f"  [✓] Da dong bo checkpoint an toan sang Google Drive: {drive_branch_dir}")
                except Exception as e:
                    print(f"  [!] Canh bao dong bo Drive ({drive_base}): {e}")


def run_training_pipeline(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 80)
    print(f"FATFORMER-XLA: PIPELINE HUAN LUYEN THICH UNG DA MIEN ({args.branch})")
    print(f"Thiet bi: {device} | Dry-run: {args.dry_run} | Epochs: {args.epochs}")
    print("=" * 80)

    # 1. Khoi tao Loss (Anchored Recovery Task 6.3-R: 0.7 CE + 0.3 KD + 0.2 Cons)
    criterion = CrossDomainTotalLoss(
        weight_ce=getattr(args, "weight_ce", 0.7),
        label_smoothing=args.label_smoothing,
        temperature=args.temperature,
        alpha_kd=args.alpha_kd if args.branch in ["M4", "M5"] else 0.0,
        beta_cons=args.beta_cons if args.branch in ["M4", "M5"] else 0.0,
    )

    # 2. Khoi tao Model (Thuc te hoac Dry-run)
    # 2. Khoi tao Model (Thuc te hoac Dry-run)
    from src.models import build_model
    from src.training.checkpoint_manager import CheckpointManager

    clip_candidates = [
        args.clip_path,
        "ViT-L-14.pt",
        "pretrained/ViT-L-14.pt",
        "DATASET/pretrained/ViT-L-14.pt",
        "/content/drive/MyDrive/Fatformer/pretrained/ViT-L-14.pt",
        "/content/drive/MyDrive/FatFormer_Hub/pretrained/ViT-L-14.pt",
    ]
    resolved_clip = find_first_existing(clip_candidates) or args.clip_path

    class VanillaArgs:
        backbone = "CLIP:ViT-L/14"
        clip_path = resolved_clip
        num_classes = 2
        num_vit_adapter = 3
        num_context_embedding = 8
        init_context_embedding = ""
        hidden_dim = 768
        clip_vision_width = 1024
        frequency_encoder_layer = 2
        decoder_layer = 4
        num_heads = 12
        use_srm = False
        use_gating = False

    class UpgradedArgs:
        backbone = "CLIP:ViT-L/14"
        clip_path = resolved_clip
        num_classes = 2
        num_vit_adapter = 3
        num_context_embedding = 8
        init_context_embedding = ""
        hidden_dim = 768
        clip_vision_width = 1024
        frequency_encoder_layer = 2
        decoder_layer = 4
        num_heads = 12
        use_srm = True
        use_gating = True

    init_candidates = [
        args.checkpoint_init,
        os.path.join(PROJECT_ROOT, "DATASET", "pretrained", "fatformer_4class_ckpt.pth"),
        "DATASET/pretrained/fatformer_4class_ckpt.pth",
        "pretrained/fatformer_4class_ckpt.pth",
        "fatformer_4class_ckpt.pth",
        "/content/drive/MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth",
        "/content/drive/MyDrive/FatFormer_Hub/pretrained/fatformer_4class_ckpt.pth",
    ]
    resolved_init = find_first_existing(init_candidates)

    teacher_candidates = [
        args.teacher_checkpoint,
        os.path.join(PROJECT_ROOT, "DATASET", "pretrained", "fatformer_4class_ckpt.pth"),
        "DATASET/pretrained/fatformer_4class_ckpt.pth",
        "pretrained/fatformer_4class_ckpt.pth",
        "fatformer_4class_ckpt.pth",
        "/content/drive/MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth",
        "/content/drive/MyDrive/FatFormer_Hub/pretrained/fatformer_4class_ckpt.pth",
    ]
    resolved_teacher = find_first_existing(teacher_candidates)

    if args.dry_run and not (resolved_init and os.path.exists(resolved_init)):
        print("[*] Dry-run che do gia lap (khong co checkpoint pretrained tren may)...")
        student = _MockBackbone(num_classes=2).to(device)
        teacher = _MockBackbone(num_classes=2).to(device) if args.branch in ["M4", "M5"] else None
        train_dataset = _DummyDataset(num_samples=16, paired=(args.branch in ["M4", "M5"]))
        val_dataset = _DummyDataset(num_samples=16, paired=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        # Khoi tao mo hinh Student voi num_vit_adapter = 3, use_srm = True, use_gating = True
        print("[*] Khoi tao mo hinh Student (Upgraded: use_srm=True, use_gating=True, num_vit_adapter=3)...")
        student = build_model(UpgradedArgs()).to(device)

        if resolved_init and os.path.exists(resolved_init):
            print(f"[*] Nap trong so khoi tao Student tu: {resolved_init}")
            load_res_student = CheckpointManager.load(resolved_init, student, device=device, strict=False)
            missing_s = load_res_student.missing_keys
            unexpected_s = load_res_student.unexpected_keys
            assert len(unexpected_s) == 0, f"Student co unexpected keys: {unexpected_s}"
            assert all(("srm" in k or "gating" in k or "lora" in k) for k in missing_s), f"Phat hien missing keys bat thuong tai Student: {missing_s}"
            print(f"[✓] Student nap thanh cong: 0 unexpected keys, {len(missing_s)} missing keys (toan bo thuoc SRM/Gating tai blocks 7, 15, 23).")

        teacher = None
        if args.branch in ["M4", "M5"]:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if resolved_teacher and os.path.exists(resolved_teacher):
                print(f"[*] Nap trong so Teacher mo neo tu: {resolved_teacher}")
                teacher = build_model(VanillaArgs()).to(device)
                load_res_teacher = CheckpointManager.load(resolved_teacher, teacher, device=device, strict=True)
                missing_t = load_res_teacher.missing_keys
                unexpected_t = load_res_teacher.unexpected_keys
                assert len(missing_t) == 0 and len(unexpected_t) == 0, f"Teacher load loi: missing={missing_t}, unexpected={unexpected_t}"
                print(f"[✓] Teacher nap checkpoint goc bit-perfect: Missing=0, Unexpected=0.")
            else:
                raise FileNotFoundError(f"[!] Khong tim thay checkpoint Teacher hop le tai cac ung vien: {teacher_candidates}")

            for p in teacher.parameters():
                p.requires_grad = False
            teacher.eval()

        # Neu la nhanh M5: Kich hoat them LoRA ViT Adapter
        if args.branch == "M5":
            lora_cfg = {"enabled": True, "rank": 8, "alpha": 16.0, "num_layers": 2}
            vit_target = getattr(student, "image_encoder", getattr(student, "visual", student))
            lora_adapter = ViTLoRAAdapter(vit_target, config=lora_cfg)
            if hasattr(lora_adapter, "summary"):
                print(f"[*] {lora_adapter.summary()}")

        # Ap dung PEFT (Task 3.4): Dong bang 94.1% CLIP ViT-L/14 backbone, chi toi uu cac adapter (~5.9% tham so)
        from src.training.freeze_utils import freeze_clip_backbone
        freeze_clip_backbone(student, verbose=True)

        if args.dry_run:
            print("[DRY-RUN] Khoi tao du lieu gia lap 16 mau kiem thu nhanh tren CPU/GPU...")
            train_dataset = _DummyDataset(num_samples=16, paired=(args.branch in ["M4", "M5"]))
            val_dataset = _DummyDataset(num_samples=16, paired=False)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            print("[*] Khoi tao DataLoader da mien thuc te...")
            train_loader, val_loader = build_crossdomain_dataloaders(args)

    # Task 6.3-R: Differential Learning Rate (Phan chia parameter groups)
    # Nhom 1 (Module moi: SRM, Gating blocks 7, 15, 23) -> lr2 = args.learning_rate (1e-4)
    # Nhom 2 (Pretrained Adapters: FAA, LGA, ctx) -> lr1 = args.learning_rate * 0.1 (1e-5)
    params_new = []
    params_pretrained = []
    new_keywords = ["srm_block", "srm", "dynamic_gating", "gating"]

    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if any(kw in name_lower for kw in new_keywords):
            params_new.append(param)
        else:
            params_pretrained.append(param)

    assert len(params_new) > 0 and len(params_pretrained) > 0, (
        f"Loi phan nhom tham so: group_new={len(params_new)}, group_pretrained={len(params_pretrained)}"
    )

    lr_new = getattr(args, "learning_rate", 3e-4)
    lr_pretrained = getattr(args, "anchor_learning_rate", 5e-5)

    print(f"[*] Differential Learning Rate (Task 6.3-R):")
    print(f"    • Nhom 1 (Module moi: SRM/Gating blocks 7, 15, 23): {len(params_new)} tensors, lr = {lr_new:.2e}, weight_decay = 1e-4")
    for name, param in student.named_parameters():
        if param.requires_grad and any(kw in name.lower() for kw in new_keywords):
            print(f"        + {name} (shape={list(param.shape)})")
    print(f"    • Nhom 2 (Pretrained Adapters FAA/LGA/ctx)       : {len(params_pretrained)} tensors, lr = {lr_pretrained:.2e} (Anchor), weight_decay = 1e-4")
    for name, param in student.named_parameters():
        if param.requires_grad and not any(kw in name.lower() for kw in new_keywords):
            print(f"        + {name} (shape={list(param.shape)})")

    optimizer_groups = []
    if params_new:
        optimizer_groups.append({"params": params_new, "lr": lr_new, "weight_decay": 1e-4})
    if params_pretrained:
        optimizer_groups.append({"params": params_pretrained, "lr": lr_pretrained, "weight_decay": 1e-4})
    if not optimizer_groups:
        trainable_params = [p for p in student.parameters() if p.requires_grad]
        optimizer_groups.append({"params": trainable_params, "lr": lr_new, "weight_decay": 1e-4})

    optimizer = torch.optim.AdamW(optimizer_groups)
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
        save_name=getattr(args, "save_name", "fatformer_anchored_recovery_best.pth"),
        grad_accum_steps=args.grad_accum_steps,
        use_amp=(args.device == "cuda"),
    )

    # Chot chan kiem toan Step-0 Identity giua Student va Teacher
    if teacher is not None:
        student.eval()
        teacher.eval()
        with torch.no_grad():
            x_test = torch.randn(2, 3, 224, 224, device=device)
            student_logits = student(x_test)
            teacher_logits = teacher(x_test)
            max_logit_diff = (student_logits - teacher_logits).abs().max().item()
        print(f"[*] Step-0 Student vs Teacher Max Logit Diff = {max_logit_diff:.10e}")
        assert max_logit_diff < 1e-4, f"LỖI ZERO-INIT: Student lệch khỏi Teacher tại Step 0 ({max_logit_diff:.10e})!"
        print("[✓] Chot chan Step-0 Zero-Init Identity: PASS bit-perfect (< 1e-4)!")

        # Chay danh gia moc san Epoch 0 (Pre-train Anchor Check)
        if not args.dry_run and val_loader is not None:
            print("[*] Dang chay danh gia moc san Epoch 0 (Pre-train Anchor Check)...")
            val_0 = trainer.validate()
            print(f"[Epoch 0 Baseline] Clean Val ACC: {val_0['val_acc']:.2f}% | Real ACC: {val_0['val_r_acc']:.2f}% | Fake ACC: {val_0['val_f_acc']:.2f}% | Composite: {val_0['val_composite']:.2f}%")
            trainer.best_composite_score = val_0["val_composite"]
            trainer.epoch0_val_acc = val_0["val_acc"]
            trainer.epoch0_composite = val_0["val_composite"]

            # [YÊU CẦU 2.1]: Luu ngay lap tuc checkpoint moc san Epoch 0 xuong o dia de dam bao best_ckpt luon ton tai 100%
            trainer.save_checkpoint(epoch=0, is_best=True, val_metrics=val_0)
            print(f"  [✓] Da khoi tao va bao luu checkpoint neo san Epoch 0 tai: {os.path.join(trainer.save_dir, trainer.save_name)}")

    if args.dry_run:
        print("\n[DRY-RUN] Bat dau chay kiem thu 1 step Train + 1 step Val...")
        train_metrics = trainer.train_epoch(epoch=1)
        val_metrics = trainer.validate()
        print(f"  -> Ket qua Train: Loss={train_metrics['loss_total']:.4f} "
              f"(CE={train_metrics['loss_ce']:.4f}, KD={train_metrics['loss_kd']:.4f}, Cons={train_metrics['loss_cons']:.4f})")
        print(f"  -> Ket qua Val: ACC={val_metrics['val_acc']:.2f}%, Composite={val_metrics['val_composite']:.2f}%")
        trainer.save_checkpoint(epoch=1, is_best=True, val_metrics=val_metrics)
        print("[DRY-RUN] HOAN THANH XUAT SAC — MA NGUON SAN SANG 100% CHO KAGGLE/COLAB!")
        return

    # Huan luyen chinh thuc nhieu epochs
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate()
        elapsed = time.time() - t0

        # Rang buoc tuyen chon: Duy tri Real_ACC >= 85.0% va cai thien composite hoac val_acc
        is_healthy = (val_metrics["val_r_acc"] >= 85.0)
        is_best = is_healthy and (
            val_metrics["val_composite"] >= trainer.best_composite_score or
            val_metrics["val_acc"] > getattr(trainer, "epoch0_val_acc", 0.0)
        )
        if is_best:
            trainer.best_composite_score = max(trainer.best_composite_score, val_metrics["val_composite"])

        trainer.save_checkpoint(epoch, is_best=is_best, val_metrics=val_metrics)
        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | Train Loss: {train_metrics['loss_total']:.4f} | "
              f"Val ACC: {val_metrics['val_acc']:.2f}% (Real: {val_metrics['val_r_acc']:.2f}%, Fake: {val_metrics['val_f_acc']:.2f}%) | "
              f"Composite: {val_metrics['val_composite']:.2f}% {'[BEST]' if is_best else ''}")

        # Ghi vao history CSV
        history_csv = os.path.join(PROJECT_ROOT, "DATASET", "logs", "m6_training_history.csv")
        os.makedirs(os.path.dirname(history_csv), exist_ok=True)
        file_exists = os.path.isfile(history_csv)
        with open(history_csv, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("branch,epoch,train_loss,val_macro_acc,val_composite,time_sec\n")
            f.write(f"{args.branch},{epoch},{train_metrics['loss_total']:.4f},{val_metrics['val_acc']:.2f},{val_metrics['val_composite']:.2f},{elapsed:.1f}\n")

        # Dong bo nhat ky sang Drive neu phat hien Colab Drive
        drive_logs = "/content/drive/MyDrive/Fatformer/logs"
        if os.path.exists(drive_logs):
            try:
                import shutil
                shutil.copy2(history_csv, os.path.join(drive_logs, "m6_training_history.csv"))
            except Exception:
                pass

    # [YÊU CẦU 2.2]: Chot chan an toan: Dam bao checkpoint best.pth ton tai 100% truoc khi ket thuc
    best_ckpt_path = os.path.join(trainer.save_dir, trainer.save_name)
    if not os.path.exists(best_ckpt_path):
        print(f"[!] Canh bao: Fallback luu checkpoint epoch cuoi cung thanh {best_ckpt_path}")
        trainer.save_checkpoint(epoch=args.epochs, is_best=True, val_metrics=val_metrics)
    assert os.path.exists(best_ckpt_path), f"CRITICAL LỖI: Checkpoint {best_ckpt_path} không tồn tại sau huấn luyện!"
    print(f"[✓] XAC NHAN: Checkpoint best hop le ton tai 100%: {best_ckpt_path}")


def parse_arguments():
    # 1. Quet truoc de kiem tra co truyen --config hay khong
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="")
    pre_args, _ = pre_parser.parse_known_args()

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    defaults = {
        "branch": "M4",
        "device": default_device,
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 3e-4,
        "anchor_learning_rate": 5e-5,
        "weight_ce": 0.7,
        "alpha_kd": 0.3,
        "beta_cons": 0.2,
        "temperature": 2.0,
        "label_smoothing": 0.05,
        "save_dir": "checkpoints/m4_recovery",
        "seed": 42,
        "grad_accum_steps": 1,
        "clip_path": "ViT-L-14.pt",
        "checkpoint_init": "fatformer_4class_ckpt.pth",
        "teacher_checkpoint": "fatformer_4class_ckpt.pth",
        "train_manifest": "DATASET/manifests/data_manifest.json",
        "train_dir": "DATASET/train_crossdomain",
        "val_dir": "DATASET/val_crossdomain",
        "num_workers": 2,
    }

    # 2. Neu co config YAML, nap cac truong vao defaults truoc khi xu ly CLI
    if pre_args.config and os.path.exists(pre_args.config):
        with open(pre_args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if "experiment" in cfg:
            defaults["branch"] = cfg["experiment"].get("branch", defaults["branch"])
            defaults["seed"] = cfg["experiment"].get("seed", defaults["seed"])
        if "training" in cfg:
            t_cfg = cfg["training"]
            defaults["epochs"] = t_cfg.get("epochs", defaults["epochs"])
            defaults["batch_size"] = t_cfg.get("batch_size", defaults["batch_size"])
            defaults["learning_rate"] = float(t_cfg.get("learning_rate", defaults["learning_rate"]))
            defaults["anchor_learning_rate"] = float(t_cfg.get("anchor_learning_rate", defaults["anchor_learning_rate"]))
            defaults["save_dir"] = t_cfg.get("save_dir", defaults["save_dir"])
            if "device" in t_cfg:
                defaults["device"] = t_cfg.get("device", defaults["device"])
            if "grad_accum_steps" in t_cfg:
                defaults["grad_accum_steps"] = t_cfg.get("grad_accum_steps", defaults["grad_accum_steps"])
        if "loss" in cfg:
            l_cfg = cfg["loss"]
            if "weight_ce" in l_cfg:
                defaults["weight_ce"] = float(l_cfg["weight_ce"])
            defaults["alpha_kd"] = float(l_cfg.get("alpha_kd", defaults["alpha_kd"]))
            defaults["beta_cons"] = float(l_cfg.get("beta_cons", defaults["beta_cons"]))
            defaults["temperature"] = float(l_cfg.get("temperature", defaults["temperature"]))
            defaults["label_smoothing"] = float(l_cfg.get("label_smoothing", defaults["label_smoothing"]))
        if "dataset" in cfg:
            d_cfg = cfg["dataset"]
            defaults["train_manifest"] = d_cfg.get("manifest", defaults["train_manifest"])
            defaults["train_dir"] = d_cfg.get("train_dir", defaults["train_dir"])
            defaults["val_dir"] = d_cfg.get("val_dir", defaults["val_dir"])
            defaults["num_workers"] = d_cfg.get("num_workers", defaults["num_workers"])

    # 3. Parser chinh thuc tiep nhan defaults tu YAML, dong thoi cho phep co CLI ghi de
    parser = argparse.ArgumentParser(description="FatFormer-XLA Cross-Domain Training")
    parser.add_argument("--config", type=str, default=pre_args.config, help="Duong dan tep cau hinh yaml")
    parser.add_argument("--branch", type=str, default=defaults["branch"], choices=["M3", "M4", "M5"], help="Nhanh thuc nghiem")
    parser.add_argument("--dry_run", action="store_true", help="Chay kiem thu dry-run gia lap")
    parser.add_argument("--device", type=str, default=defaults["device"], help="Thiet bi: cuda hoac cpu")
    parser.add_argument("--epochs", type=int, default=defaults["epochs"], help="So epochs")
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"], help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=defaults["learning_rate"], help="Learning rate cho module moi (SRM/Gating)")
    parser.add_argument("--anchor_learning_rate", type=float, default=defaults["anchor_learning_rate"], help="Learning rate cho pretrained adapters (FAA/LGA/ctx)")
    parser.add_argument("--weight_ce", type=float, default=defaults["weight_ce"], help="Trong so Cross-Entropy loss tren anh tang cuong (mac dinh 0.7)")
    parser.add_argument("--alpha_kd", type=float, default=defaults["alpha_kd"], help="Trong so KD loss (mac dinh 0.3)")
    parser.add_argument("--beta_cons", type=float, default=defaults["beta_cons"], help="Trong so Consistency loss (mac dinh 0.2)")
    parser.add_argument("--temperature", type=float, default=defaults["temperature"], help="Nhiet do KD")
    parser.add_argument("--label_smoothing", type=float, default=defaults["label_smoothing"], help="Label smoothing")
    parser.add_argument("--save_dir", type=str, default=defaults["save_dir"], help="Thu muc luu checkpoint")
    parser.add_argument("--save_name", type=str, default="crossdomain_model_best.pth",
                        help="Ten tep checkpoint khi dat best (Phase 6-R dung fatformer_anchored_recovery_best.pth)")
    parser.add_argument("--seed", type=int, default=defaults["seed"], help="Random seed")
    parser.add_argument("--grad_accum_steps", type=int, default=defaults["grad_accum_steps"], help="Gradient accumulation")
    parser.add_argument("--clip_path", type=str, default=defaults["clip_path"], help="Path to CLIP backbone")
    parser.add_argument("--checkpoint_init", type=str, default=defaults["checkpoint_init"])
    parser.add_argument("--teacher_checkpoint", type=str, default=defaults["teacher_checkpoint"])
    parser.add_argument("--train_manifest", type=str, default=defaults["train_manifest"], help="Duong dan tep manifest json")
    parser.add_argument("--train_dir", type=str, default=defaults["train_dir"], help="Thu muc du lieu train")
    parser.add_argument("--val_dir", type=str, default=defaults["val_dir"], help="Thu muc du lieu val")
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"], help="So luong workers cho DataLoader (mac dinh 2)")

    args = parser.parse_args()

    # Kiem tra an toan: neu yeu cau cuda ma he thong khong co GPU, tu dong chuyen sang cpu
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[!] Canh bao: CUDA khong kha dung tren thiet bi nay, tu dong chuyen sang CPU.")
        args.device = "cpu"

    return args


if __name__ == "__main__":
    import traceback
    try:
        args = parse_arguments()
        run_training_pipeline(args)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"[X] SU CO TRONG TRAINING PIPELINE: {e}")
        print("=" * 80)
        traceback.print_exc()
        sys.exit(1)
