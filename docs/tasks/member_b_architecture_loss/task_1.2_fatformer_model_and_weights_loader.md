# TASK 1.2: KIỂM CHỨNG MÃ NGUỒN FATFORMER & TRÌNH NẠP KHỚP 1.116 TENSORS

- **Mã nhiệm vụ**: `Task 1.2`
- **Người phụ trách**: **Thành viên B** *(Architecture & Loss Lead)*
- **Môi trường thực hiện**: 🖥️ **Local** / ☁️ **Colab T4**
- **Thời hạn hoàn thành**: Tuần 1 (Ngày 5)
- **Độ ưu tiên**: 🔴 Cực kỳ khẩn cấp (Nền tảng kiến trúc cho toàn bộ dự án)
- **Đầu vào (Input)**:
  - Mã nguồn mô hình tại `src/models/` (`fatformer.py`, `clip_models.py`, `srm.py`, `gating.py`).
  - Checkpoint tác giả gốc `fatformer_4class_ckpt.pth` (3.7 GB).
  - Backbone CLIP ViT `ViT-L-14.pt` (TorchScript JIT archive).
- **Đầu ra (Output)**:
  - Script kiểm thử [`tools/test_src_load.py`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/tools/test_src_load.py).
  - Biên bản kiểm thử xác nhận pass `strict=True` 1.116/1.116 tensor, 0 missing keys, 0 unexpected keys.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Thẩm định tính toàn vẹn 100% của kiến trúc FatFormer trong `src/models/`: Khối Haar 2D DWT, tầng tích chập FAA, và khối căn chỉnh ngôn ngữ LGA.
2. Xác nhận cơ chế nạp trọng số với `strict=True` khớp chuẩn xác 1.116 tensors và 933.16M tham số của tác giả gốc CVPR 2024.
3. Giải tỏa áp lực kỹ thuật cho Tuần 1, cung cấp module mô hình sẵn sàng cho Thành viên C tích hợp vào notebook huấn luyện.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Bước 1: 4 Quy Tắc Kỹ Thuật Bắt Buộc
1. **Lỗi chính tả bắt buộc trong LGA**: Trong class `LanguageGuidedAlignment`, module enhancer bắt buộc phải đặt tên thuộc tính là `self.patch_basaed_enhancer` (có chữ `a` thừa) để khớp chính xác với state_dict checkpoint.
2. **Nạp Backbone TorchScript**: Tệp `ViT-L-14.pt` là TorchScript JIT archive, sử dụng `torch.jit.load()` hoặc `torch.load(..., weights_only=False)` để phân tách các tầng ViT.
3. **Bọc Checkpoint**: Checkpoint của tác giả bọc trọng số trong khóa `ckpt['model']`.

### Bước 2: Chạy Kiểm Thử Khớp Trọng Số
Chạy script thẩm định:
```bash
python tools/test_src_load.py \
    --checkpoint fatformer_4class_ckpt.pth \
    --clip_path ViT-L-14.pt
```

Đoạn code thẩm định cốt lõi:
```python
import os
import sys
import torch

# Đảm bảo Python nhận diện được thư mục gốc của dự án
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import build_model

def verify_loader(ckpt_path="fatformer_4class_ckpt.pth", clip_path="ViT-L-14.pt"):
    class Args:
        backbone = "CLIP:ViT-L/14"
        clip_path = os.path.join(project_root, clip_path)
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

    print("[*] Khởi tạo kiến trúc FatFormer...")
    model = build_model(Args())
    
    ckpt_full_path = os.path.join(project_root, ckpt_path)
    print(f"[*] Nạp checkpoint từ {ckpt_full_path}...")
    ckpt = torch.load(ckpt_full_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    
    # Nạp nghiêm ngặt
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    
    print(f"[+] Missing keys: {len(missing)}")
    print(f"[+] Unexpected keys: {len(unexpected)}")
    
    if len(missing) == 0 and len(unexpected) == 0:
        print("[✓] TUYỆT VỜI: Khớp 100% (1.116/1.116 tensors) pass strict=True!")
    else:
        print("[!] Cảnh báo: Vẫn còn tensor chưa khớp!")

if __name__ == "__main__":
    verify_loader()
```

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Script `tools/test_src_load.py` chạy trên Local/Colab in ra thông báo pass 100% không cảnh báo lỗi.
- [ ] Bàn giao module cho Thành viên C tích hợp vào `notebooks/train.ipynb`.
