import os
from typing import List, Optional, Tuple, Union, Any, Dict
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from .transforms import get_eval_transforms, get_train_transforms


import numpy as np


GAN_SUBSETS = [
    "progan", "stylegan", "stylegan2", "biggan", 
    "cyclegan", "stargan", "gaugan", "deepfake"
]

DIFFUSION_SUBSETS = [
    "guided", "ldm_200", "ldm_200_cfg", "ldm_100", 
    "glide_50_27", "glide_100_10", "glide_100_27", "dalle",
    "pndm", "vqdiffusion"
]

ALL_TEST_SUBSETS = GAN_SUBSETS + DIFFUSION_SUBSETS


def _get_balanced_subset_indices(ds, max_samples: int, seed: int = 42) -> List[int]:
    """
    Trích xuất chỉ số cân bằng nhãn (real=0, fake=1) phục vụ Fast-Eval với seed cố định.
    """
    targets = None
    if hasattr(ds, "targets"):
        targets = np.array(ds.targets)
    elif isinstance(ds, ConcatDataset):
        child_targets = []
        for child in ds.datasets:
            if hasattr(child, "targets"):
                child_targets.append(np.array(child.targets))
        if child_targets:
            targets = np.concatenate(child_targets)

    if targets is not None and len(targets) == len(ds):
        idx_real = np.where(targets == 0)[0]
        idx_fake = np.where(targets == 1)[0]
        samples_per_class = max_samples // 2

        rng = np.random.RandomState(seed)
        n_real = min(samples_per_class, len(idx_real))
        n_fake = min(samples_per_class, len(idx_fake))

        sel_real = rng.choice(idx_real, n_real, replace=False) if len(idx_real) > 0 else np.array([], dtype=int)
        sel_fake = rng.choice(idx_fake, n_fake, replace=False) if len(idx_fake) > 0 else np.array([], dtype=int)

        selected_idx = np.concatenate([sel_real, sel_fake])
        selected_idx.sort()
        return selected_idx.tolist()
    else:
        # Fallback lấy mẫu ngẫu nhiên có kiểm soát seed
        rng = np.random.RandomState(seed)
        n_sample = min(max_samples, len(ds))
        indices = rng.choice(len(ds), n_sample, replace=False)
        indices.sort()
        return indices.tolist()


class DatasetCreator:
    """
    Quản lý việc tạo DataLoader và Dataset cho cả pha Huấn luyện và Đánh giá.
    Hỗ trợ cả tập dữ liệu đầy đủ lẫn chế độ lấy mẫu con (subsampling) phục vụ Fast-Eval.
    """
    def __init__(
        self, 
        dataset_path: str,
        img_resolution: int = 256,
        crop_resolution: int = 224,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        self.dataset_path = dataset_path
        self.img_resolution = img_resolution
        self.crop_resolution = crop_resolution
        self.batch_size = batch_size
        self.num_workers = num_workers

    def build_eval_dataset(
        self, 
        selected_subsets: Union[str, List[str]] = "all",
        degradation: Optional[dict] = None,
        max_samples_per_subset: Optional[int] = None
    ) -> Tuple[List[Dataset], List[str]]:
        """
        Khởi tạo danh sách các Dataset cho các tập test được chỉ định.
        
        Args:
            selected_subsets: "all", "gans", "diffusion", hoặc danh sách cụ thể.
            degradation: Dict chỉ định kiểu suy thoái (JPEG, Blur, v.v.).
            max_samples_per_subset: Giới hạn số lượng mẫu mỗi subset (phục vụ Fast-Eval).
        """
        if selected_subsets == "all":
            subsets = ALL_TEST_SUBSETS
        elif selected_subsets == "gans":
            subsets = GAN_SUBSETS
        elif selected_subsets == "diffusion":
            subsets = DIFFUSION_SUBSETS
        elif isinstance(selected_subsets, (list, tuple)):
            subsets = list(selected_subsets)
        else:
            subsets = [selected_subsets]

        transform = get_eval_transforms(
            img_resolution=self.img_resolution,
            crop_resolution=self.crop_resolution,
            degradation=degradation
        )

        sub_datasets = []
        valid_subset_names = []

        for subset_name in subsets:
            # Thử tìm theo cấu trúc linh hoạt (hỗ trợ cả trường hợp giải nén có tiền tố diffusion_test/)
            candidate_paths = [
                os.path.join(self.dataset_path, "test", subset_name),
                os.path.join(self.dataset_path, subset_name),
                os.path.join(self.dataset_path, "diffusion_test", subset_name),
                os.path.join(self.dataset_path, "diffusion", subset_name),
                os.path.join(self.dataset_path, "test_diffusion", subset_name),
                os.path.join(self.dataset_path, "gans", subset_name),
                os.path.join(self.dataset_path, "test_gans", subset_name),
                os.path.join(self.dataset_path, "test_clean", subset_name),
            ]
            subset_path = None
            for p in candidate_paths:
                if os.path.exists(p) and os.path.isdir(p):
                    subset_path = p
                    break
            
            if subset_path is None:
                print(f"[CẢNH BÁO] Không tìm thấy thư mục dữ liệu cho subset: '{subset_name}' tại {os.path.join(self.dataset_path, subset_name)}")
                continue

            sub_dir_contents = os.listdir(subset_path)
            
            # Cấu trúc 1: Thư mục chứa trực tiếp 0_real và 1_fake
            if "0_real" in sub_dir_contents and "1_fake" in sub_dir_contents:
                ds = ImageFolder(subset_path, transform=transform)
            else:
                # Cấu trúc 2: Thư mục chứa nhiều lớp con (multi-class), mỗi lớp có 0_real / 1_fake
                child_datasets = []
                for sub_class in sorted(sub_dir_contents):
                    sub_class_path = os.path.join(subset_path, sub_class)
                    if os.path.isdir(sub_class_path):
                        child_datasets.append(ImageFolder(sub_class_path, transform=transform))
                if not child_datasets:
                    continue
                ds = ConcatDataset(child_datasets)

            # Lấy mẫu con cân bằng nếu có yêu cầu (Fast-Eval: 250 real + 250 fake, seed=42)
            if max_samples_per_subset is not None and len(ds) > max_samples_per_subset:
                indices = _get_balanced_subset_indices(ds, max_samples=max_samples_per_subset, seed=42)
                ds = Subset(ds, indices)

            sub_datasets.append(ds)
            valid_subset_names.append(subset_name)

        return sub_datasets, valid_subset_names

    def build_train_dataset(
        self,
        train_folder_name: str = "train",
        use_dual_stream: bool = True,
        scheduler: Optional[Any] = None
    ) -> Dataset:
        """
        Khởi tạo Dataset huấn luyện (ProGAN 4-class hoặc tập tùy chọn).
        Hỗ trợ gắn CurriculumDegradationScheduler theo 3 giai đoạn epoch.
        """
        train_path = os.path.join(self.dataset_path, train_folder_name)
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu huấn luyện tại: {train_path}")

        transform = get_train_transforms(
            img_resolution=self.img_resolution,
            crop_resolution=self.crop_resolution,
            use_dual_stream=use_dual_stream,
            scheduler=scheduler
        )

        sub_dir_contents = os.listdir(train_path)
        if "0_real" in sub_dir_contents and "1_fake" in sub_dir_contents:
            return ImageFolder(train_path, transform=transform)

        # Multi-class folder
        child_datasets = []
        for sub_class in sorted(sub_dir_contents):
            sub_class_path = os.path.join(train_path, sub_class)
            if os.path.isdir(sub_class_path):
                child_datasets.append(ImageFolder(sub_class_path, transform=transform))
        
        return ConcatDataset(child_datasets)


def collect_sub_imagefolders(root_dir: str, transform=None) -> List[ImageFolder]:
    """
    Quét và gom toàn bộ các thư mục có định dạng ImageFolder (chứa 0_real và 1_fake)
    từ thư mục gốc root_dir. Hỗ trợ cả cấu trúc 1 cấp lẫn đa cấp phân lớp.
    """
    if not root_dir or not os.path.exists(root_dir):
        return []

    try:
        top_contents = os.listdir(root_dir)
    except Exception:
        return []

    if "0_real" in top_contents and "1_fake" in top_contents:
        return [ImageFolder(root_dir, transform=transform)]

    image_folders = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if "0_real" in dirnames and "1_fake" in dirnames:
            image_folders.append(ImageFolder(dirpath, transform=transform))

    return image_folders


def build_multi_domain_train_dataloader(
    progan_dir: Optional[str] = None,
    staging_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    progan_ratio: float = 0.85,
    staging_ratio: float = 0.15,
    transform: Optional[Any] = None,
    pin_memory: bool = True
) -> Tuple[Optional[DataLoader], int, int]:
    """
    Khởi tạo DataLoader huấn luyện cân bằng đa miền (Multi-Domain Balanced DataLoader)
    cho Phương Án A: Kết hợp kho ProGAN (144.024 ảnh) và Diffusion Staging (3.600 ảnh)
    sử dụng WeightedRandomSampler.

    Args:
        progan_dir: Đường dẫn thư mục dữ liệu ProGAN.
        staging_dir: Đường dẫn thư mục dữ liệu Diffusion Staging.
        batch_size: Kích thước batch.
        num_workers: Số luồng nạp dữ liệu.
        progan_ratio: Tỷ lệ phân bổ mẫu ProGAN mong muốn trong mỗi batch (mặc định 0.85).
        staging_ratio: Tỷ lệ phân bổ mẫu Staging mong muốn trong mỗi batch (mặc định 0.15).
        transform: Bộ tiền xử lý ảnh (Dual-Stream Augmentation).
        pin_memory: Ghim bộ nhớ CUDA.

    Returns:
        (DataLoader, n_progan, n_staging)
    """
    progan_sub_ds = collect_sub_imagefolders(progan_dir, transform=transform) if progan_dir else []
    staging_sub_ds = collect_sub_imagefolders(staging_dir, transform=transform) if staging_dir else []

    progan_ds = ConcatDataset(progan_sub_ds) if len(progan_sub_ds) > 1 else (progan_sub_ds[0] if len(progan_sub_ds) == 1 else None)
    staging_ds = ConcatDataset(staging_sub_ds) if len(staging_sub_ds) > 1 else (staging_sub_ds[0] if len(staging_sub_ds) == 1 else None)

    n_progan = len(progan_ds) if progan_ds is not None else 0
    n_staging = len(staging_ds) if staging_ds is not None else 0

    if n_progan == 0 and n_staging == 0:
        return None, 0, 0

    if n_progan > 0 and n_staging > 0:
        combined_ds = ConcatDataset([progan_ds, staging_ds])
        total_ratio = progan_ratio + staging_ratio
        p1 = progan_ratio / total_ratio
        p2 = staging_ratio / total_ratio

        w1 = p1 / n_progan
        w2 = p2 / n_staging
        sample_weights = torch.DoubleTensor([w1] * n_progan + [w2] * n_staging)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(combined_ds),
            replacement=True
        )

        loader = DataLoader(
            combined_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        print(f"[DATASET] Kích hoạt Balanced DataLoader:")
        print(f"  • ProGAN: {n_progan:,} ảnh (tỷ lệ mục tiêu: {p1*100:.1f}%)")
        print(f"  • Staging: {n_staging:,} ảnh (tỷ lệ mục tiêu: {p2*100:.1f}%)")
        print(f"  • Tổng mẫu khả dụng: {len(combined_ds):,} ảnh | Batch: {batch_size}")
        return loader, n_progan, n_staging

    single_ds = progan_ds if n_progan > 0 else staging_ds
    loader = DataLoader(
        single_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    print(f"[DATASET] Chỉ tìm thấy 1 nguồn dữ liệu ({n_progan + n_staging:,} ảnh). Dùng Shuffle DataLoader thông thường.")
    return loader, n_progan, n_staging


def build_val_dataloader(
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[Any] = None
) -> Optional[DataLoader]:
    """
    Khởi tạo DataLoader validation độc lập (vd: progan_val 8.000 ảnh).
    """
    if not val_dir or not os.path.exists(val_dir):
        return None

    if transform is None:
        transform = get_eval_transforms()

    sub_ds = collect_sub_imagefolders(val_dir, transform=transform)
    if not sub_ds:
        return None

    val_ds = ConcatDataset(sub_ds) if len(sub_ds) > 1 else sub_ds[0]
    print(f"[DATASET] Khởi tạo Validation DataLoader thành công: {len(val_ds):,} ảnh từ {val_dir}")
    return DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
