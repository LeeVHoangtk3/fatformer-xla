"""
FatFormer-XLA: Bo Lay Mau Can Bang Da Mien (Balanced Cross-Domain Sampler)
Phuc vu Task 6.2 (Milestone 6) — Thanh vien A (Data & Infrastructure Lead).

Muc dich:
    1. Can bang ty le 1:1 giua anh That (Real) va anh Gia (Fake) trong moi batch.
    2. Can bang xac suat giua cac bo sinh (Generators) trong nhom Fake (w_g = 1 / N_g)
       nham triet tieu thien lech bo sinh ap dao (nhu ProGAN chiem so luong lon).
    3. Can bang phan bo giua cac nguon anh That (ImageNet, COCO, FFHQ) de chong
       hoc thuoc dac trung nguon camera (domain shortcut learning).
"""

import math
import sys
from typing import List, Dict, Optional, Iterator
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Sampler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class BalancedCrossDomainSampler(Sampler[int]):
    """
    Bo lay mau can bang da mien cho DataLoader PyTorch.

    Args:
        dataset_records: Danh sach cac dict chua thong tin mau:
                         [{'label': 0/1, 'generator': str, 'source': str}, ...]
        batch_size:      Kich thuoc lo (batch size).
        num_samples:     Tong so mau can lay trong 1 epoch (mac dinh = len(dataset)).
        replacement:     Co cho phep lay mau lap lai khong (mac dinh True de can bang).
    """

    def __init__(
        self,
        dataset_records: List[Dict],
        batch_size: int = 64,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        self.records = dataset_records
        self.batch_size = batch_size
        self.num_samples = num_samples or len(dataset_records)
        self.replacement = replacement

        self.weights = self._calculate_sample_weights()

    def _calculate_sample_weights(self) -> torch.DoubleTensor:
        """
        Tinh toan trong so lay mau nghich dao tan suat da tang:
            Weight(i) = W_class · W_group
        """
        n_total = len(self.records)
        if n_total == 0:
            return torch.DoubleTensor([])

        # 1. Dem so luong Real va Fake
        real_indices = [i for i, r in enumerate(self.records) if r.get("label", 0) == 0]
        fake_indices = [i for i, r in enumerate(self.records) if r.get("label", 0) == 1]

        n_real = len(real_indices)
        n_fake = len(fake_indices)

        # Trọng số nhóm lớn (50% Real, 50% Fake)
        w_class_real = 0.5 / max(n_real, 1)
        w_class_fake = 0.5 / max(n_fake, 1)

        # 2. Dem tan suat theo bo sinh trong nhom Fake
        fake_generators = [self.records[i].get("generator", "unknown") for i in fake_indices]
        gen_counts = Counter(fake_generators)
        num_generators = len(gen_counts)

        # 3. Dem tan suat theo nguon anh trong nhom Real
        real_sources = [self.records[i].get("source", "unknown") for i in real_indices]
        source_counts = Counter(real_sources)
        num_sources = len(source_counts)

        weights = np.zeros(n_total, dtype=np.float64)

        # Tinh trong so cho tung mau Real
        for i in real_indices:
            src = self.records[i].get("source", "unknown")
            count = source_counts.get(src, 1)
            # Phan bo deu giua cac nguon Real
            w_src = (1.0 / max(num_sources, 1)) / max(count, 1)
            weights[i] = 0.5 * w_src

        # Tinh trong so cho tung mau Fake
        for i in fake_indices:
            gen = self.records[i].get("generator", "unknown")
            count = gen_counts.get(gen, 1)
            # Phan bo deu giua cac bo sinh Fake
            w_gen = (1.0 / max(num_generators, 1)) / max(count, 1)
            weights[i] = 0.5 * w_gen

        # Chuan hoa tong trong so ve 1.0
        total_w = np.sum(weights)
        if total_w > 0:
            weights = weights / total_w

        return torch.DoubleTensor(weights)

    def __iter__(self) -> Iterator[int]:
        if len(self.weights) == 0:
            return iter([])

        indices = torch.multinomial(
            self.weights,
            num_samples=self.num_samples,
            replacement=self.replacement,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class BalancedBatchSampler(Sampler[List[int]]):
    """
    Bộ lấy mẫu phân lô cân bằng đa miền (Strict Balanced Batch Sampler).
    Cưỡng bức mỗi mini-batch luôn có chính xác tỷ lệ 50/50 giữa Real và Fake:
      - batch_size // 2 mẫu Real (phân bổ đều theo nguồn camera/kho ảnh).
      - batch_size // 2 mẫu Fake (phân bổ đều theo các bộ sinh).
    Giải quyết triệt để hiện tượng lệch lớp trong các mini-batch kích thước nhỏ.
    """

    def __init__(
        self,
        dataset_records: List[Dict],
        batch_size: int = 64,
        num_batches: Optional[int] = None,
        drop_last: bool = False,
    ):
        self.records = dataset_records
        self.batch_size = max(2, batch_size)
        self.drop_last = drop_last

        # Tách chỉ số Real và Fake
        self.real_indices = [i for i, r in enumerate(self.records) if r.get("label", 0) == 0]
        self.fake_indices = [i for i, r in enumerate(self.records) if r.get("label", 0) == 1]

        if len(self.real_indices) == 0 or len(self.fake_indices) == 0:
            raise ValueError("Dataset phải chứa đồng thời cả mẫu Real và mẫu Fake.")

        # Trọng số nội bộ Real
        real_sources = [self.records[i].get("source", "unknown") for i in self.real_indices]
        src_counts = Counter(real_sources)
        num_src = len(src_counts)
        real_w = np.array([(1.0 / max(num_src, 1)) / max(src_counts[s], 1) for s in real_sources], dtype=np.float64)
        real_w = real_w / real_w.sum()
        self.real_weights = torch.DoubleTensor(real_w)

        # Trọng số nội bộ Fake
        fake_gens = [self.records[i].get("generator", "unknown") for i in self.fake_indices]
        gen_counts = Counter(fake_gens)
        num_gen = len(gen_counts)
        fake_w = np.array([(1.0 / max(num_gen, 1)) / max(gen_counts[g], 1) for g in fake_gens], dtype=np.float64)
        fake_w = fake_w / fake_w.sum()
        self.fake_weights = torch.DoubleTensor(fake_w)

        # Xác định số lượng batch mỗi epoch
        if num_batches is not None:
            self.total_batches = num_batches
        else:
            self.total_batches = max(1, len(self.records) // self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        n_real = self.batch_size // 2
        n_fake = self.batch_size - n_real

        for _ in range(self.total_batches):
            # Lấy mẫu Real có trọng số
            sub_real_idx = torch.multinomial(self.real_weights, num_samples=n_real, replacement=True)
            batch_real = [self.real_indices[idx] for idx in sub_real_idx.tolist()]

            # Lấy mẫu Fake có trọng số
            sub_fake_idx = torch.multinomial(self.fake_weights, num_samples=n_fake, replacement=True)
            batch_fake = [self.fake_indices[idx] for idx in sub_fake_idx.tolist()]

            # Gộp và xáo trộn bên trong từng mini-batch
            batch = batch_real + batch_fake
            perm = torch.randperm(len(batch)).tolist()
            yield [batch[p] for p in perm]

    def __len__(self) -> int:
        return self.total_batches


def run_sampler_unit_test():
    """Unit test kiem chung tinh can bang cua BalancedCrossDomainSampler va BalancedBatchSampler."""
    print("=" * 80)
    print("FATFORMER-XLA: UNIT TEST BALANCED CROSS-DOMAIN SAMPLER (TASK 6.2)")
    print("=" * 80)

    # Tao dataset gia lap voi su mat can bang tram trong:
    # - ProGAN chiem 10.000 anh fake
    # - Midjourney chiem 500 anh fake
    # - Stable Diffusion chiem 500 anh fake
    # - Real ImageNet chiem 5.000 anh
    # - Real FFHQ chiem 1.000 anh
    records = []
    for _ in range(10000):
        records.append({"label": 1, "generator": "progan", "source": "fake"})
    for _ in range(500):
        records.append({"label": 1, "generator": "midjourney", "source": "fake"})
    for _ in range(500):
        records.append({"label": 1, "generator": "stable_diffusion", "source": "fake"})
    for _ in range(5000):
        records.append({"label": 0, "generator": "real", "source": "imagenet"})
    for _ in range(1000):
        records.append({"label": 0, "generator": "real", "source": "ffhq"})

    total_samples = len(records)
    print(f"- Kho du lieu ban dau gom {total_samples:,} mau co thien lech lon:")
    print(f"  • Fake ProGAN: 10,000 | Midjourney: 500 | Stable Diffusion: 500")
    print(f"  • Real ImageNet: 5,000 | FFHQ: 1,000")

    # 1. Kiem thu BalancedCrossDomainSampler (cap do epoch)
    sampler = BalancedCrossDomainSampler(records, batch_size=64, num_samples=12000)
    sampled_indices = list(sampler)

    sampled_labels = [records[i]["label"] for i in sampled_indices]
    sampled_gens = [records[i]["generator"] for i in sampled_indices if records[i]["label"] == 1]
    sampled_srcs = [records[i]["source"] for i in sampled_indices if records[i]["label"] == 0]

    label_counts = Counter(sampled_labels)
    gen_counts = Counter(sampled_gens)
    src_counts = Counter(sampled_srcs)

    real_pct = label_counts[0] / len(sampled_indices) * 100
    fake_pct = label_counts[1] / len(sampled_indices) * 100

    print("\n- Ket qua sau khi lay mau 12,000 luot qua BalancedCrossDomainSampler:")
    print(f"  • Ty le nhan: Real = {real_pct:.1f}% | Fake = {fake_pct:.1f}% (Can bang ly tuong ~50/50)")

    pass_class = abs(real_pct - 50.0) < 3.0
    pass_gen = max(abs(c / len(sampled_gens) * 100 - 33.3) for c in gen_counts.values()) < 5.0
    pass_src = max(abs(c / len(sampled_srcs) * 100 - 50.0) for c in src_counts.values()) < 5.0

    print(f"  [1] Can bang Real/Fake toan cuc (+/-3%): {'PASS' if pass_class else 'FAIL'}")
    print(f"  [2] Can bang Generator Fake (+/-5%): {'PASS' if pass_gen else 'FAIL'}")
    print(f"  [3] Can bang Source Real (+/-5%): {'PASS' if pass_src else 'FAIL'}")

    # 2. Kiem thu BalancedBatchSampler (cap do tung mini-batch)
    print("\n- Kiem thu BalancedBatchSampler (kiem tra ty le tren tung mini-batch):")
    batch_sampler = BalancedBatchSampler(records, batch_size=8, num_batches=100)
    all_batch_50_50 = True
    for b_idx, batch in enumerate(batch_sampler):
        b_labels = [records[i]["label"] for i in batch]
        n_r = sum(1 for l in b_labels if l == 0)
        n_f = sum(1 for l in b_labels if l == 1)
        if n_r != 4 or n_f != 4:
            all_batch_50_50 = False
            break

    print(f"  [4] Bao dam 100% mini-batch (batch_size=8) co dung 4 Real va 4 Fake: {'PASS' if all_batch_50_50 else 'FAIL'}")

    all_pass = pass_class and pass_gen and pass_src and all_batch_50_50
    print(f"\n=> KET LUAN: {'UNIT TEST PASS 100%' if all_pass else 'UNIT TEST FAILED'}")
    print("=" * 80)
    return all_pass


if __name__ == "__main__":
    run_sampler_unit_test()
