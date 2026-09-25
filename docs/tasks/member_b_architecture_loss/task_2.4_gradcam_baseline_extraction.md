# TASK 2.4: TRÍCH XUẤT GRAD-CAM BAN ĐẦU ĐỐI CHỨNG MỐC SÀN

- **Mã nhiệm vụ**: `Task 2.4`
- **Người phụ trách**: **Thành viên B** *(Architecture & Loss Lead)*
- **Môi trường thực hiện**: 🖥️ **Local** / ☁️ **Colab T4**
- **Thời hạn hoàn thành**: Tuần 2 (Ngày 12)
- **Độ ưu tiên**: 🟠 Cao (Bằng chứng trực quan mốc sàn cho tính giải thích XAI)
- **Đầu vào (Input)**:
  - Checkpoint mô hình gốc `fatformer_4class_ckpt.pth`.
  - 5 cặp ảnh mẫu (5 ảnh Real + 5 ảnh Fake từ các nguồn ProGAN, Midjourney, SD) ở 2 trạng thái: Clean và Degraded ($Q=30$).
- **Đầu ra (Output)**:
  - Script [`tools/visualize_cam.py`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/tools/visualize_cam.py).
  - Bộ 10 ảnh Grad-CAM ban đầu đối chứng mốc sàn lưu trong `docs/assets/gradcam_baseline/`.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Hiện thực hóa kỹ thuật **Grad-CAM (Gradient-weighted Class Activation Mapping)** trên nhánh Visual Transformer của FatFormer.
2. Trích xuất bản đồ kích hoạt nhiệt tại tầng `visual.transformer.resblocks[-1]` để quan sát xem mạng đang chú ý vào vùng tạo tác sinh ảnh vi mô hay bị phân tâm bởi các cạnh biên khối nén JPEG.
3. Tạo cơ sở so sánh đối đầu với phiên bản cải tiến FatFormer-XLA ở Tuần 5.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Cài đặt `tools/visualize_cam.py`:
```python
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

class FatFormerGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook gradient và activation
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class=1):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[:, target_class].sum()
        loss.backward()
        
        # Tính trọng số alpha bằng GAP của gradient
        grads = self.gradients # [seq_len, batch, dim]
        acts = self.activations
        
        weights = torch.mean(grads, dim=0, keepdim=True)
        cam = torch.sum(weights * acts, dim=-1)
        cam = torch.relu(cam)
        
        # Reshape về kích thước spatial 2D (16x16 cho patch size 14 trên 224x224)
        cam = cam[1:].view(16, 16).detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
```

Chạy trích xuất Grad-CAM:
```bash
python tools/visualize_cam.py \
    --checkpoint fatformer_4class_ckpt.pth \
    --input_dir datasets/sample_pairs/ \
    --output_dir docs/assets/gradcam_baseline/
```

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Xuất đủ 10 ảnh Grad-CAM nhiệt phân giải cao phủ lên ảnh gốc.
- [ ] Quan sát thấy hiện tượng: Khi ảnh bị nén $Q=30$, vùng kích hoạt của mô hình gốc bị phân rã, bám theo các góc cạnh khối vuông JPEG $8 \times 8$ thay vì đặc trưng khuôn mặt/chủ thể.
- [ ] Bàn giao ảnh vào Báo cáo Chương 1–2.
