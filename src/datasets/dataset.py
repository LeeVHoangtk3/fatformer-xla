import os
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
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
            # Thử tìm theo cấu trúc: dataset_path/test/subset_name hoặc dataset_path/subset_name
            subset_path = os.path.join(self.dataset_path, "test", subset_name)
            if not os.path.exists(subset_path):
                subset_path = os.path.join(self.dataset_path, subset_name)
            
            if not os.path.exists(subset_path):
                print(f"[CẢNH BÁO] Không tìm thấy thư mục dữ liệu cho subset: '{subset_name}' tại {subset_path}")
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
        use_dual_stream: bool = True
    ) -> Dataset:
        """
        Khởi tạo Dataset huấn luyện (ProGAN 4-class hoặc tập tùy chọn).
        """
        train_path = os.path.join(self.dataset_path, train_folder_name)
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu huấn luyện tại: {train_path}")

        transform = get_train_transforms(
            img_resolution=self.img_resolution,
            crop_resolution=self.crop_resolution,
            use_dual_stream=use_dual_stream
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
