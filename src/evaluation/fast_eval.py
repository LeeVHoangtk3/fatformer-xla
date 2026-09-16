import os
import time
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from .metrics import compute_binary_metrics, format_evaluation_summary
from ..datasets.dataset import DatasetCreator, ALL_TEST_SUBSETS


@torch.no_grad()
def evaluate_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    print_freq: int = 20
) -> Dict[str, float]:
    """
    Chạy suy luận trên 1 DataLoader cụ thể và tính toán metrics.
    """
    model.eval()
    y_true = []
    y_score = []

    total_batches = len(dataloader)
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        # outputs shape: (B, 2), lấy xác suất lớp 1 (fake)
        scores = outputs.softmax(dim=1)[:, 1].cpu().tolist()
        y_score.extend(scores)
        y_true.extend(labels.tolist())

        if (batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == total_batches:
            print(f"  [Tiến độ: {batch_idx + 1}/{total_batches} batches] Đã xử lý {len(y_true)} ảnh...")

    return compute_binary_metrics(y_true, y_score)


def run_benchmark(
    model: nn.Module,
    dataset_path: str,
    device: torch.device,
    selected_subsets: Optional[List[str]] = None,
    degradation: Optional[dict] = None,
    max_samples_per_subset: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict[str, Dict[str, float]]:
    """
    Chạy đánh giá benchmark trên toàn bộ các tập test được chỉ định.
    Hỗ trợ Fast-Eval thông qua max_samples_per_subset (ví dụ 500 ảnh/tập).
    """
    model = model.to(device)
    model.eval()

    creator = DatasetCreator(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers
    )

    subsets_to_test = selected_subsets or ALL_TEST_SUBSETS
    datasets, subset_names = creator.build_eval_dataset(
        selected_subsets=subsets_to_test,
        degradation=degradation,
        max_samples_per_subset=max_samples_per_subset
    )

    if not datasets:
        print(f"[CẢNH BÁO] Không tải được tập test nào từ đường dẫn: {dataset_path}")
        return {}

    mode_str = f"Fast-Eval ({max_samples_per_subset} ảnh/subset)" if max_samples_per_subset else "Full-Eval"
    deg_str = f"Suy thoái: {degradation}" if degradation else "Dữ liệu Clean (nguyên bản)"
    print("\n" + "=" * 80)
    print(f"BẮT ĐẦU ĐÁNH GIÁ: Chế độ {mode_str} | {deg_str}")
    print(f"Tổng số tập test hợp lệ: {len(subset_names)}")
    print("=" * 80)

    results_by_subset = {}
    start_time = time.time()

    for idx, (ds, name) in enumerate(zip(datasets, subset_names)):
        print(f"\n[{idx + 1}/{len(subset_names)}] Đang đánh giá subset: '{name}' (Tổng: {len(ds)} ảnh)...")
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=SequentialSampler(ds),
            num_workers=num_workers,
            pin_memory=(device.type == "cuda")
        )
        t0 = time.time()
        metrics = evaluate_dataloader(model, loader, device)
        elapsed = time.time() - t0
        results_by_subset[name] = metrics
        print(f"  -> Hoàn thành '{name}' trong {elapsed:.1f}s | ACC: {metrics['acc']:.2f}% | AP: {metrics['ap']:.2f}%")

    total_time = time.time() - start_time
    summary_table, mean_metrics = format_evaluation_summary(results_by_subset)
    print("\n" + summary_table)
    print(f"Tổng thời gian đánh giá: {total_time:.1f} giây ({total_time / 60:.2f} phút)\n")

    return results_by_subset
