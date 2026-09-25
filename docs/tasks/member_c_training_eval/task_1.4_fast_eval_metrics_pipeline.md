# TASK 1.4: XÂY DỰNG PIPELINE ĐÁNH GIÁ NHANH FAST-EVAL (< 8 PHÚT)

- **Mã nhiệm vụ**: `Task 1.4`
- **Người phụ trách**: **Thành viên C** *(Training & Evaluation Lead)*
- **Môi trường thực hiện**: 🖥️ **Local** (phát triển script) + ☁️ **Colab T4** (kiểm thử hiệu năng)
- **Thời hạn hoàn thành**: Tuần 1 (Ngày 7)
- **Độ ưu tiên**: 🔴 Khẩn cấp (Công cụ giám sát validation quan trọng nhất)
- **Đầu vào (Input)**:
  - Công thức tính toán các chỉ số: **Accuracy (ACC)**, **Average Precision (AP)**, **ROC-AUC**.
  - Script đánh giá gốc từ CVPR 2024.
- **Đầu ra (Output)**:
  - Script [`tools/fast_eval.py`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/tools/fast_eval.py).
  - Khả năng đánh giá 500 ảnh ngẫu nhiên trên mỗi tập test trong thời gian < 8 phút trên GPU T4.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Giải quyết bài toán chi phí tài nguyên: Đánh giá toàn bộ 18 tập test paper (~100.000 ảnh) mất từ 2 đến 3 giờ và đốt nhiều Compute Units.
2. Xây dựng cơ chế **Fast-Eval**: Lấy mẫu ngẫu nhiên đại diện 500 ảnh/tập test (250 real + 250 fake với seed cố định `seed=42`), hoàn thành đánh giá toàn diện 18 tập test trong thời gian dưới **8 phút** trên GPU T4 rẻ tiền (~0.2 CU).
3. Đóng vai trò là công cụ giám sát validation nhanh sau mỗi epoch huấn luyện ở Tuần 4.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Cài đặt `tools/fast_eval.py`:
```python
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

def fast_evaluate(model, dataset_root, samples_per_class=250, batch_size=32, device="cuda"):
    """Đánh giá nhanh trên subset 500 ảnh với seed cố định."""
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    dataset = datasets.ImageFolder(dataset_root, transform=transform)
    
    # Lấy mẫu cân bằng
    targets = np.array(dataset.targets)
    idx_real = np.where(targets == 0)[0]
    idx_fake = np.where(targets == 1)[0]
    
    np.random.seed(42)
    selected_idx = np.concatenate([
        np.random.choice(idx_real, min(samples_per_class, len(idx_real)), replace=False),
        np.random.choice(idx_fake, min(samples_per_class, len(idx_fake)), replace=False)
    ])
    
    subset = Subset(dataset, selected_idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Fast-Eval {os.path.basename(dataset_root)}", leave=False):
            imgs = imgs.to(device)
            # Forward pass (Visual branch)
            logits_v, _ = model(imgs)
            probs = torch.softmax(logits_v, dim=-1)[:, 1].cpu().numpy()
            
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds >= 0.5) * 100
    ap = average_precision_score(all_labels, all_preds) * 100
    auc = roc_auc_score(all_labels, all_preds) * 100
    
    print(f"[{os.path.basename(dataset_root)}] -> ACC: {acc:.2f}%, AP: {ap:.2f}%, AUC: {auc:.2f}%")
    return {"acc": acc, "ap": ap, "auc": auc}
```

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Script `tools/fast_eval.py` chạy độc lập, in bảng số liệu chuẩn hóa.
- [ ] Thời gian đo đạc trên 1 tập test < 30 giây trên GPU T4.
- [ ] Sẵn sàng để thực thi đánh giá mốc sàn Baseline ở Tuần 2 (Task 2.2 và Task 2.3).
