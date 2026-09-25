# TASK 3.5: CÀI ĐẶT CƠ CHẾ AUTO-SAVE CHECKPOINT & RESUME PHIÊN COLAB

- **Mã nhiệm vụ**: `Task 3.5`
- **Người phụ trách**: **Thành viên C** *(Training & Evaluation Lead)*
- **Môi trường thực hiện**: ☁️ **Colab**
- **Thời hạn hoàn thành**: Tuần 3 (Ngày 20)
- **Độ ưu tiên**: 🔴 Cực kỳ khẩn cấp (Bảo vệ tài nguyên tính toán và công sức huấn luyện)
- **Đầu vào (Input)**:
  - Thư mục Google Drive 5TB `FatFormer_Hub/checkpoints/`.
  - Logic PyTorch lưu và khôi phục `state_dict`, optimizer, scheduler, scaler.
- **Đầu ra (Output)**:
  - Module tự động lưu checkpoint sau mỗi epoch trong `train.ipynb`.
  - Tham số dòng lệnh `--resume` cho phép tiếp tục huấn luyện liền mạch khi phiên Colab bị disconnect.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Triệt tiêu 100% rủi ro mất mát dữ liệu do Google Colab tự động ngắt kết nối (*timeout disconnect* hoặc *preemption*): Tuyệt đối không lưu file `.pth` duy nhất trên ổ SSD `/content/`.
2. Sau khi kết thúc mỗi epoch huấn luyện, hệ thống tự động:
   - Lưu đầy đủ: Trọng số mô hình (`model.state_dict()`), trạng thái bộ tối ưu (`optimizer.state_dict()`), trạng thái bộ lập lịch (`scheduler.state_dict()`), bộ cân chỉnh gradient (`scaler.state_dict()`) và số `epoch` hiện tại.
   - Ghi thẳng sang thư mục Google Drive 5TB: `/content/drive/MyDrive/FatFormer_Hub/checkpoints/checkpoint_epoch_{epoch}.pth`.
3. Kiểm thử tính năng khôi phục huấn luyện (`resume`): Khi phiên bị ngắt ở giữa chừng, chỉ cần gọi lại lệnh với cờ `--resume` thì mô hình sẽ tiếp tục học từ epoch tiếp theo mà không phải train lại từ đầu.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Cài đặt hàm Save và Load Checkpoint trong `train.ipynb`:
```python
import os
import torch

DRIVE_CKPT_DIR = "/content/drive/MyDrive/FatFormer_Hub/checkpoints"

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, filename=None):
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    if filename is None:
        filename = f"fatformer_epoch_{epoch}.pth"
    save_path = os.path.join(DRIVE_CKPT_DIR, filename)
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
    }
    torch.save(state, save_path)
    print(f"[✓] Đã lưu Checkpoint Epoch {epoch} an toàn sang Drive 5TB: {save_path}")

def load_checkpoint(resume_path, model, optimizer=None, scheduler=None, scaler=None):
    assert os.path.exists(resume_path), f"Không tìm thấy checkpoint tại {resume_path}"
    print(f"[*] Đang khôi phục huấn luyện từ {resume_path}...")
    checkpoint = torch.load(resume_path, map_location="cpu")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    start_epoch = checkpoint['epoch'] + 1
    print(f"[✓] Khôi phục thành công! Bắt đầu huấn luyện từ Epoch {start_epoch}")
    return start_epoch
```

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Chạy thử nghiệm mô phỏng ngắt kết nối sau epoch 1: Gọi `load_checkpoint()` nạp lại thành công và tiếp tục train epoch 2.
- [ ] Xác nhận file `.pth` xuất hiện đầy đủ trong thư mục `FatFormer_Hub/checkpoints/` trên Drive 5TB.
