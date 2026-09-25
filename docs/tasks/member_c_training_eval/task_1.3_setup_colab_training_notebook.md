# TASK 1.3: THIẾT LẬP NOTEBOOK HUẤN LUYỆN TRÊN COLAB PRO

- **Mã nhiệm vụ**: `Task 1.3`
- **Người phụ trách**: **Thành viên C** *(Training & Evaluation Lead)*
- **Môi trường thực hiện**: ☁️ **Colab T4** (tiết kiệm CU tối đa cho việc dựng khung)
- **Thời hạn hoàn thành**: Tuần 1 (Ngày 6)
- **Độ ưu tiên**: 🔴 Cực kỳ khẩn cấp (Bộ khung huấn luyện chính của dự án)
- **Đầu vào (Input)**:
  - Module mô hình `src/models/` từ Thành viên B.
  - Tệp Dummy Dataset `dummy_data.tar` từ Thành viên A.
  - Thư viện PyTorch, torchvision, AMP FP16.
- **Đầu ra (Output)**:
  - Notebook [`notebooks/train.ipynb`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/notebooks/train.ipynb) chạy thông suốt 1 step forward-backward trên GPU T4.
  - Nhật ký đo đạc thời gian 1 step và mức tiêu thụ VRAM.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Xây dựng môi trường huấn luyện chuẩn hóa trên Google Colab Pro tích hợp:
   - Tự động mount Google Drive 5TB trực tiếp: `DRIVE_HUB = "/content/drive/MyDrive/FatFormer_Hub"`.
   - Giải nén dataset siêu tốc sang SSD máy ảo Colab (`/content/dataset_local/`).
   - Tối ưu hóa tính toán bộ nhớ với **Automatic Mixed Precision (AMP FP16)** và **Gradient Accumulation** (batch size 32, accumulation 2 $\rightarrow$ Effective batch size 64).
2. Chạy thử nghiệm thành công 1 step forward-backward với dummy data, bảo đảm không lỗi CUDA và giải phóng bộ nhớ sạch sẽ.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Cấu trúc mã nguồn trong `notebooks/train.ipynb`:
```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# 1. Thiết bị và cấu hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Đang sử dụng thiết bị: {torch.cuda.get_device_name(0)}")

# 2. Khởi tạo mô hình và loss
from src.models.fatformer import build_fatformer
from src.training.loss import DualStreamFocalLoss

model = build_fatformer(clip_path="ViT-L-14.pt").to(device)
criterion = DualStreamFocalLoss(alpha=0.25, gamma=2.0)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scaler = GradScaler()

# 3. Dummy Step forward-backward
inputs = torch.randn(16, 3, 224, 224, device=device)
targets = torch.randint(0, 2, (16,), device=device)

optimizer.zero_grad()
with autocast():
    logits_v, logits_a = model(inputs)
    loss = criterion(logits_v, logits_a, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

print(f"[✓] Chạy thành công 1 step! Loss: {loss.item():.4f}")
print(f"[✓] VRAM sử dụng: {torch.cuda.max_memory_allocated() / (1024**2):.1f} MB")
```

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Notebook `notebooks/train.ipynb` chạy trên Colab T4 không lỗi, VRAM tiêu tốn < 8GB.
- [ ] Kiểm tra autograd: Loss giảm, tham số adapter được cập nhật gradient.
