# Hướng Dẫn Thực Thi Trên Google Colab Pro (GPU A100)

Tài liệu này hướng dẫn cách kết nối và chạy trọn gói hệ thống **FatFormer-XLA** trên tài khoản Google AI Pro với GPU A100 và Google Drive 5TB trực tiếp (Mô hình Hợp Nhất).

---

## 1. Cài Đặt Môi Trường Trên Colab (Chạy trong 1 Cell)

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Tạo thư mục làm việc và giải nén dữ liệu từ Drive về SSD local của Colab
!mkdir -p /content/dataset_local
!cp /content/drive/MyDrive/FatFormer_Hub/datasets/*.tar /content/
!tar -xf /content/progan_train.tar -C /content/dataset_local/
!tar -xf /content/test_benchmark.tar -C /content/dataset_local/

# 3. Cài đặt các thư viện bổ sung (nếu cần)
!pip install regex ftfy scikit-learn
```

---

## 2. Các Lệnh Thực Thi Tiêu Chuẩn

### A. Kiểm thử nạp Checkpoint tác giả (Smoke Test)
```bash
python tools/test_src_load.py
```
> Kết quả kỳ vọng: Pass 100% cả Baseline Model (`strict=True`) và Mô hình cải tiến SRM + Gating.

### B. Đánh giá Baseline trên tập Clean (Tái hiện kết quả bài báo)
```bash
python -m src.main \
    --eval \
    --fast_eval \
    --max_samples 500 \
    --backbone "CLIP:ViT-L/14" \
    --num_vit_adapter 3 \
    --pretrained_model "fatformer_4class_ckpt.pth" \
    --dataset_path "/content/dataset_local" \
    --batchsize 32
```

### C. Đánh giá Baseline trên tập Degraded (Chứng minh sự suy giảm hiệu năng)
```bash
# Kiểm thử nén JPEG Q=50
python -m src.main \
    --eval \
    --fast_eval \
    --max_samples 500 \
    --backbone "CLIP:ViT-L/14" \
    --num_vit_adapter 3 \
    --pretrained_model "fatformer_4class_ckpt.pth" \
    --dataset_path "/content/dataset_local" \
    --degradation_type "jpeg" \
    --jpeg_quality 50 \
    --batchsize 32
```

### D. Fine-Tuning Nâng Cấp: SRM 3-Kernels + Gating + Dual-Stream
```bash
python -m src.main \
    --train \
    --epochs 20 \
    --lr 4e-4 \
    --batchsize 64 \
    --use_srm \
    --use_gating \
    --backbone "CLIP:ViT-L/14" \
    --num_vit_adapter 3 \
    --pretrained_model "fatformer_4class_ckpt.pth" \
    --dataset_path "/content/dataset_local" \
    --checkpoint_dir "checkpoints" \
    --drive_backup_dir "/content/drive/MyDrive/FatFormer_Hub/checkpoints"
```
