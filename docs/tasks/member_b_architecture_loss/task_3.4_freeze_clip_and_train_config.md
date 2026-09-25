# TASK 3.4: CẤU HÌNH ĐÓNG BĂNG 94.1% CLIP VIT & FILE THIẾT LẬP HUẤN LUYỆN

- **Mã nhiệm vụ**: `Task 3.4`
- **Người phụ trách**: **Thành viên B** *(Architecture & Loss Lead)*
- **Môi trường thực hiện**: 🖥️ **Local**
- **Thời hạn hoàn thành**: Tuần 3 (Ngày 20)
- **Độ ưu tiên**: 🔴 Khẩn cấp (Bảo vệ bộ trọng số khổng lồ CLIP không bị phá hủy)
- **Đầu vào (Input)**:
  - Mô hình FatFormer-XLA hoàn chỉnh từ Task 3.3.
- **Đầu ra (Output)**:
  - File cấu hình [`configs/train_config.yaml`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/configs/train_config.yaml).
  - Hàm kiểm tra tham số huấn luyện `count_trainable_parameters(model)`.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Thực thi nguyên tắc đóng băng tham số (Parameter-Efficient Fine-Tuning - PEFT):
   - Đóng băng **94.1% tham số** (toàn bộ các lớp Transformer Attention, MLP của CLIP ViT-L/14 và text encoder).
   - Chỉ mở khóa **5.9% tham số** (khoảng ~55M tham số) gồm: FAA Adapters, tầng chiếu SRM projection, mạng MLP gating $\lambda(x)$, khối LGA và prompt tuning vector.
2. Thiết lập file cấu hình chuẩn hóa `configs/train_config.yaml` định nghĩa toàn bộ siêu tham số của 3 giai đoạn Curriculum.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Cài đặt logic đóng băng và kiểm toán tham số:
```python
def freeze_clip_backbone(model):
    """Đóng băng 94.1% CLIP ViT-L/14, chỉ mở 5.9% tham số thích ứng."""
    # 1. Đóng băng toàn bộ mô hình trước
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Mở khóa có chọn lọc các module thích ứng
    trainable_modules = [
        model.visual_adapter,      # FAA adapter
        model.srm_proj,            # SRM projection
        model.gating_network,      # Dynamic Gating λ(x)
        model.lga_enhancer,        # LGA module
        model.soft_prompts         # Prompt tuning
    ]
    
    for module in trainable_modules:
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True
                
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"[*] Tổng tham số: {total_params / 1e6:.2f}M")
    print(f"[*] Tham số trainable: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    print(f"[*] Tham số frozen: {frozen_params / 1e6:.2f}M ({frozen_params / total_params * 100:.2f}%)")
    
    assert trainable_params / total_params < 0.08, "Lỗi: Số lượng tham số mở khóa vượt quá 8%!"
```

### File cấu hình `configs/train_config.yaml`:
```yaml
model:
  backbone: "ViT-L-14.pt"
  checkpoint: "fatformer_4class_ckpt.pth"
  use_srm: true
  use_gating: true

training:
  total_epochs: 8
  batch_size: 32
  grad_accumulation_steps: 2  # Effective batch size = 64
  amp_fp16: true
  optimizer:
    name: "AdamW"
    lr_adapter: 1.0e-4
    lr_gating: 1.0e-3
    weight_decay: 1.0e-2

curriculum:
  phase1:
    epochs: [1, 2]
    jpeg_range: [70, 90]
    blur_range: [0.5, 1.0]
    downsample: false
  phase2:
    epochs: [3, 5]
    jpeg_range: [45, 70]
    blur_range: [1.0, 1.5]
    downsample_size: 160
  phase3:
    epochs: [6, 8]
    jpeg_range: [30, 50]
    blur_range: [1.5, 2.0]
    downsample_size: 112
    lr_multiplier: 0.5
```

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Chạy hàm `freeze_clip_backbone()` xác nhận chính xác tỷ lệ trainable nằm trong khoảng $5.5\% \sim 6.0\%$.
- [ ] Tệp `configs/train_config.yaml` được commit lên Git, sẵn sàng cho Thành viên C nạp vào pipeline huấn luyện.
