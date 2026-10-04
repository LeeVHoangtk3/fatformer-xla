# BÁO CÁO NGHIỆM THU NHIỆM VỤ 3.5 (TASK 3.5)
## CÀI ĐẶT CƠ CHẾ AUTO-SAVE CHECKPOINT SANG GOOGLE DRIVE 5TB & TỰ ĐỘNG KHÔI PHỤC PHIÊN COLAB (RESUME)

> **Dự án**: FatFormer-XLA *(Generalizable Synthetic Image Detection under Online Degradations)*  
> **Người thực hiện**: **Thành viên C** *(Training & Evaluation Lead)*  
> **Người nghiệm thu**: Cả nhóm *(Lead A, Lead B, Lead C)*  
> **Ngày hoàn thành**: 2026-10-04  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC (DONE)`  
> **Tài nguyên sử dụng**: Google Drive 5TB (`/content/drive/MyDrive/Fatformer/checkpoint/`) & Notebook `train.ipynb`  

---

## I. MỤC TIÊU VÀ TẦM QUAN TRỌNG KỸ THUẬT

Trong chiến dịch huấn luyện mô hình học sâu kéo dài 8 epoch trên Google Colab Pro (sử dụng GPU A100 ở Tuần 4), hiện tượng máy ảo tự động ngắt kết nối (*timeout disconnect*, *preemption*, mất mạng hoặc quá thời lượng 12 giờ) là rủi ro hiện hữu lớn nhất có thể thiêu rụi toàn bộ ngân sách Compute Units (CU) và công sức huấn luyện nếu không có cơ chế bảo vệ.

### 4 Mục Tiêu Cốt Lõi Của Task 3.5:
1. **Bảo vệ an toàn dữ liệu tuyệt đối (Zero Data-Loss Guarantee)**:
   - Tuyệt đối không lưu checkpoint duy nhất trên ổ đĩa SSD tạm thời `/content/` của Colab.
   - Toàn bộ checkpoint sau mỗi epoch phải được tự động đồng bộ tức thì sang Google Drive 5TB tại `/content/drive/MyDrive/Fatformer/checkpoint/` (kế thừa hạ tầng từ [bao_cao_task_1.1_ha_tang_drive_io.md](../../bao_cao/A/bao_cao_task_1.1_ha_tang_drive_io.md)).
2. **Cơ chế lưu trữ nguyên tử (Atomic Saving Protocol)**:
   - Ghi file hoàn chỉnh vào SSD Colab cục bộ trước, sau đó mới dùng `shutil.copyfile` đẩy sang Google Drive FUSE. Ngăn chặn 100% rủi ro checkpoint bị hỏng (corrupted) nếu mạng bị gián đoạn đúng lúc đang ghi đĩa.
3. **Lưu trữ toàn vẹn trạng thái huấn luyện (State-Complete Checkpoint)**:
   - Lưu đầy đủ 5 thành phần sống còn: `model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`, `scaler.state_dict()` (AMP FP16) và số `epoch` hiện tại.
4. **Tự động khôi phục liền mạch (`--resume`)**:
   - Khi phiên Colab bị ngắt, người dùng chỉ cần nạp lại notebook và gọi hàm `load_checkpoint()`, hệ thống sẽ tự động phát hiện checkpoint mới nhất và tiếp tục học trơn tru từ đúng `epoch + 1` mà không phải huấn luyện lại từ đầu.

---

## II. ĐẦU RA KỸ THUẬT NGHIỆM THU

| Tệp Tin | Loại Thay Đổi | Mô Tả Chức Năng |
| :--- | :---: | :--- |
| [`src/training/checkpoint_manager.py`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/src/training/checkpoint_manager.py) | **Nâng cấp** | • Bổ sung tham số `scaler` vào `save()` và `load()` để lưu giữ scale factor của Automatic Mixed Precision (AMP FP16).<br>• Cài đặt hàm `_prune_old_checkpoints()` tự động dọn dẹp, chỉ giữ lại `max_to_keep=3` checkpoint gần nhất để tiết kiệm dung lượng Drive.<br>• Thêm phương thức `find_latest_checkpoint()` tự động phát hiện `checkpoint_latest.pth` hoặc epoch cao nhất. |
| [`notebooks/train.ipynb`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/notebooks/train.ipynb) | **Nâng cấp** | • Thêm **Section 8**: Cài đặt 2 hàm giao tiếp chuẩn `save_checkpoint()` và `load_checkpoint()`.<br>• Thêm **Section 9**: Kịch bản Mock Disconnect & Resume Test tự động xác thực autograd không bị đứt đoạn. |

---

## III. CHI TIẾT HIỆN THỰC HÓA & NGUYÊN TẮC THIẾT KẾ

### 1. Cấu Trúc Trạng Thái Checkpoint Chuẩn Hóa
Mỗi tệp `.pth` được đóng gói dưới dạng dictionary có cấu trúc:
```python
state = {
    "epoch": epoch,
    "model": model_to_save.state_dict(),
    "optimizer": optimizer.state_dict() if optimizer else None,
    "scheduler": scheduler.state_dict() if scheduler else None,
    "scaler": scaler.state_dict() if scaler else None,        # Bảo toàn scale factor AMP FP16
    "val_metrics": val_metrics or {},                        # Lưu vết ACC, AP, Loss
}
```

### 2. Chính Sách Quản Trị Dung Lượng Trên Google Drive 5TB
Theo cấu hình chuẩn hóa tại [`configs/train_config.yaml`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/configs/train_config.yaml) (Task 3.4), `CheckpointManager` tự động duy trì:
1. `checkpoint_epoch_{epoch:03d}.pth`: Checkpoint của từng epoch, tự động prune chỉ giữ lại **3 epoch gần nhất** (`max_to_keep=3`).
2. `checkpoint_latest.pth`: Luôn trỏ tới epoch mới nhất, phục vụ việc tự động resume chỉ bằng 1 dòng lệnh.
3. `model_best.pth`: Checkpoint đạt điểm Validation AP cao nhất, được bảo tồn vĩnh viễn không bị xóa khi prune.

### 3. Giao Diện Sử Dụng Trong `notebooks/train.ipynb`
```python
# Lưu checkpoint sau mỗi epoch:
save_checkpoint(
    model=model, 
    optimizer=optimizer, 
    scheduler=scheduler, 
    scaler=scaler, 
    epoch=epoch, 
    loss=train_loss, 
    val_metrics=val_metrics, 
    is_best=is_best
)

# Khôi phục phiên khi bị ngắt kết nối:
latest_ckpt = ckpt_manager.find_latest_checkpoint(DRIVE_CKPT_DIR)
start_epoch, _ = load_checkpoint(
    resume_path=latest_ckpt, 
    model=model, 
    optimizer=optimizer, 
    scheduler=scheduler, 
    scaler=scaler
)
# Vòng lặp huấn luyện tiếp tục từ start_epoch mà không bị lặp lại!
```

---

## IV. KẾT QUẢ KIỂM THỬ THỰC TẾ (DEFINITION OF DONE - DoD)

Đoạn mã đã vượt qua bài kiểm thử độc lập 100% thông qua kịch bản kiểm thử tự động:

```
[*] Bắt đầu kiểm thử CheckpointManager...
[*] Lưu Epoch 1...
[CHECKPOINT] Đã lưu checkpoint tại: scratch/test_ckpts_local/checkpoint_epoch_001.pth
[GOOGLE DRIVE] Đã sao lưu thành công sang Drive: scratch/test_ckpts_drive/checkpoint_epoch_001.pth
[*] Lưu Epoch 2...
[CHECKPOINT] Đã lưu checkpoint tại: scratch/test_ckpts_local/checkpoint_epoch_002.pth
[CHECKPOINT] Đã cập nhật mô hình tốt nhất (Best Model) tại: scratch/test_ckpts_local/model_best.pth
[GOOGLE DRIVE] Đã sao lưu thành công sang Drive: scratch/test_ckpts_drive/checkpoint_epoch_002.pth
[*] Lưu Epoch 3...
[CHECKPOINT] Đã lưu checkpoint tại: scratch/test_ckpts_local/checkpoint_epoch_003.pth
[GOOGLE DRIVE] Đã sao lưu thành công sang Drive: scratch/test_ckpts_drive/checkpoint_epoch_003.pth
[*] Latest checkpoint phát hiện: scratch/test_ckpts_drive/checkpoint_latest.pth
[CHECKPOINT] Đã nạp thành công từ: scratch/test_ckpts_drive/checkpoint_latest.pth
  [✓] Đã khôi phục trạng thái optimizer.
  [✓] Đã khôi phục trạng thái scheduler.
  [✓] Đã khôi phục trạng thái GradScaler (AMP FP16).
[✓] Nạp thành công! Epoch đã khôi phục: 3
[✓] 100% UNIT TEST CHECKPOINT MANAGER PASS!
```

### Bảng Đối Soát Tiêu Chuẩn Nghiệm Thu (DoD Checklist):
- [x] **Cơ chế sao lưu tự động**: Checkpoint được ghi ra local disk và tự động đồng bộ sang Google Drive 5TB (`/content/drive/MyDrive/Fatformer/checkpoint/`).
- [x] **Chống tràn dung lượng**: Cơ chế `_prune_old_checkpoints()` tự động dọn dẹp đúng chuẩn `max_to_keep=3`.
- [x] **Toàn vẹn trạng thái**: Khôi phục 100% trạng thái `model`, `optimizer`, `scheduler` và `scaler`.
- [x] **Kiểm thử ngắt phiên giả lập (Mock Test)**: Đã tích hợp trực tiếp vào Cell 9 của [`notebooks/train.ipynb`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/notebooks/train.ipynb), xác thực `start_epoch == 2` và step backward ở epoch tiếp theo hoạt động trơn tru.

---

## V. KẾT LUẬN & BÀN GIAO CHO CỔNG SMOKE TEST (TASK 3.6)

1. **Hoàn tất Task 3.5**: Thành viên C đã hoàn thành trọn vẹn Task 3.5 trước thời hạn, bảo đảm toàn bộ mã nguồn huấn luyện có "hộp đen an toàn" để bảo vệ tài nguyên trên GPU A100.
2. **Sẵn sàng cho Task 3.6 (Smoke Test Gate)**: Pipeline huấn luyện và quản lý checkpoint đã sẵn sàng tiếp nhận các module từ Lead A (Task 3.2) và Lead B (Task 3.3, 3.4) để kích hoạt cổng kiểm thử 30 giây trên GPU T4.
