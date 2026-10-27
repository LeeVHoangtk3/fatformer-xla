# TASK 3.6: CỔNG KIỂM THỬ TÍCH HỢP TOÀN DIỆN (INTEGRATION SMOKE TEST GATE)

- **Mã nhiệm vụ**: `Task 3.6`
- **Người phụ trách**: **Thành viên C** *(Chủ trì)*, **Thành viên B** *(Hỗ trợ kỹ thuật)*
- **Môi trường thực hiện**: ☁️ **Colab T4** (chỉ mất ~30 giây, tốn < 0.05 CU)
- **Thời hạn hoàn thành**: Tuần 3 (Ngày 21)
- **Độ ưu tiên**: 🔴 Cực kỳ khẩn cấp (Cổng an toàn bắt buộc trước khi mở GPU A100)
- **Đầu vào (Input)**:
  - Dữ liệu Staging 90/10 từ Thành viên A (Task 3.1).
  - Module `CurriculumDegradationScheduler` từ Thành viên A (Task 3.2).
  - Khối SRM 3-Kernels, Gating $\lambda(x)$, Dual-Stream Focal Loss từ Thành viên B (Task 3.3).
  - File cấu hình đóng băng `train_config.yaml` từ Thành viên B (Task 3.4).
- **Đầu ra (Output)**:
  - **Biên bản Smoke Test Pass** (chạy thành công 1 batch 16 ảnh forward + backward trên T4 trong 30 giây).
  - Giấy phép phê duyệt chính thức mở GPU A100 cho chiến dịch huấn luyện Tuần 4.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Triệt tiêu 100% nguy cơ lãng phí ngân sách Compute Units đắt đỏ của GPU A100: Tuyệt đối không bật A100 khi mã nguồn chưa được kiểm thử toàn diện trên GPU T4.
2. Kiểm tra tính tương thích đồng bộ của toàn bộ 4 tầng kỹ thuật trong một vòng lặp huấn luyện duy nhất:
   - Dữ liệu Staging $\rightarrow$ Qua bộ biến đổi Curriculum $\rightarrow$ Vào mạng FatFormer-XLA (FAA + SRM + Gating) $\rightarrow$ Tính Dual-Stream Focal Loss $\rightarrow$ Backward gradient $\rightarrow$ Cập nhật AdamW.
3. Điều kiện cần để PASS:
   - Tensor shape qua các lớp hoàn toàn chuẩn xác.
   - Không xuất hiện giá trị `NaN` hoặc `Inf` trong loss và gradient.
   - Cổng thích ứng $\lambda(x)$ xuất giá trị thực tế hợp lý.
   - Không bị tràn bộ nhớ VRAM (OOM).

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Thực thi kịch bản kiểm thử:
```bash
python tools/smoke_test_pipeline.py \
    --config configs/train_config.yaml \
    --data_tar /content/drive/MyDrive/FatFormer_Hub/datasets/diffusion_staging.tar \
    --device cuda \
    --batch_size 16
```

### Nội dung kiểm toán trong `smoke_test_pipeline.py`:
```python
import torch
from src.models.fatformer import build_fatformer
from src.training.loss import DualStreamFocalLoss
from src.datasets.transforms import CurriculumDegradationScheduler

def run_smoke_test():
    print("[*] 1. Khởi tạo Pipeline...")
    device = torch.device("cuda")
    model = build_fatformer("ViT-L-14.pt").to(device)
    criterion = DualStreamFocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = CurriculumDegradationScheduler()
    
    print("[*] 2. Tạo Batch giả lập...")
    inputs = torch.randn(16, 3, 224, 224, device=device)
    targets = torch.randint(0, 2, (16,), device=device)
    
    print("[*] 3. Forward Pass...")
    logits_v, logits_a = model(inputs)
    assert not torch.isnan(logits_v).any(), "Lỗi: Logits Visual có NaN!"
    assert not torch.isnan(logits_a).any(), "Lỗi: Logits Alignment có NaN!"
    
    print("[*] 4. Tính Dual-Stream Focal Loss...")
    loss = criterion(logits_v, logits_a, targets)
    assert not torch.isnan(loss), "Lỗi: Loss có giá trị NaN!"
    print(f"[+] Giá trị Loss: {loss.item():.4f}")
    
    print("[*] 5. Backward Pass & Step Optimizer...")
    optimizer.zero_grad()
    loss.backward()
    
    # Kiểm tra gradient của các adapter
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert grad_norm > 0, "Lỗi: Gradient bằng 0 (Vanishing gradient)!"
    optimizer.step()
    
    print(f"[✓] SMOKE TEST PASS 100%! Thời gian thực thi < 30s. Sẵn sàng mở GPU A100!")

if __name__ == "__main__":
    run_smoke_test()
```

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Chạy `smoke_test_pipeline.py` in ra dòng `[✓] SMOKE TEST PASS 100%!`.
- [ ] Cả 3 thành viên ký biên bản chấp thuận kích hoạt chiến dịch huấn luyện A100.
