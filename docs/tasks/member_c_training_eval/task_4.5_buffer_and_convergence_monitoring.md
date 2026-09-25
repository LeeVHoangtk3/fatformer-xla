# TASK 4.5: QUẢN LÝ KHOẢNG ĐỆM DỰ PHÒNG 2 NGÀY & KIỂM SOÁT HỘI TỤ 4 CHECKPOINTS

- **Mã nhiệm vụ**: `Task 4.5`
- **Người phụ trách**: **Thành viên C** *(Chủ trì)*, **Cả 3 thành viên**
- **Môi trường thực hiện**: ☁️ **Colab** / 🖥️ **Họp nhóm**
- **Thời hạn hoàn thành**: Tuần 4 (Ngày 27–28)
- **Độ ưu tiên**: 🟠 Cao (Khoảng đệm an toàn phòng chống rủi ro tiến độ)
- **Đầu vào (Input)**:
  - Trạng thái hội tụ và file trọng số của đủ 4 checkpoint đối chứng:
    1. Checkpoint 1: `baseline.pth` (Mô hình gốc tải sẵn).
    2. Checkpoint 2: `fatformer_aug_only.pth` (từ Task 4.3 của Thành viên A).
    3. Checkpoint 3: `fatformer_srm_only.pth` (từ Task 4.4 của Thành viên B).
    4. Checkpoint 4: `fatformer_srm_robust_final.pth` (từ Task 4.2 của Thành viên C).
- **Đầu ra (Output)**:
  - Cả 4 checkpoint đều ở trạng thái toàn vẹn, hội tụ chuẩn xác và sẵn sàng cho tuần đánh giá Benchmark cuối cùng.

---

## 1. MỤC TIÊU KỸ THUẬT
1. **Khoảng đệm dự phòng (Buffer Slot)**: Dành riêng 2 ngày đệm cuối Tuần 4 để xử lý các sự cố kỹ thuật ngoài ý muốn (mô hình bị phân kỳ, rớt mạng Colab, thiếu Compute Units hoặc cần tinh chỉnh Learning Rate).
2. Kiểm toán toàn diện 4 checkpoint phục vụ cho bài toán triệt tiêu (Ablation Study) của Tuần 5:
   - Xác nhận mỗi checkpoint đại diện cho đúng một cụm can thiệp kiến trúc / dữ liệu riêng biệt.
   - Thử nghiệm nạp nhanh cả 4 checkpoint trên GPU T4 để đảm bảo không file nào bị lỗi Corrupted weights.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Kịch bản kiểm toán 4 Checkpoints:
```python
import os
import torch
from src.models.fatformer import build_fatformer

DRIVE_CKPT_DIR = "/content/drive/MyDrive/FatFormer_Hub/checkpoints"

checkpoints = {
    "1. Baseline gốc": "fatformer_4class_ckpt.pth",
    "2. Aug-only (Ablation 2)": "fatformer_aug_only.pth",
    "3. SRM-only (Ablation 3)": "fatformer_srm_only.pth",
    "4. FatFormer-XLA Đầy đủ": "fatformer_srm_robust_final.pth"
}

def audit_all_checkpoints():
    print("[*] BẮT ĐẦU KIỂM TOÁN 4 CHECKPOINTS TRÊN DRIVE 5TB...")
    all_pass = True
    for name, filename in checkpoints.items():
        path = os.path.join(DRIVE_CKPT_DIR, filename)
        if not os.path.exists(path):
            print(f"[X] THIẾU: Không tìm thấy {name} tại {path}!")
            all_pass = False
            continue
            
        try:
            ckpt = torch.load(path, map_location="cpu")
            size_mb = os.path.getsize(path) / (1024**2)
            print(f"[✓] PASS: {name} | Dung lượng: {size_mb:.1f} MB | Trạng thái: HỢP LỆ")
        except Exception as e:
            print(f"[X] LỖI: {name} bị hỏng trọng số: {e}")
            all_pass = False
            
    if all_pass:
        print("[✓] CHÚC MỪNG: Cả 4 checkpoint đã sẵn sàng 100% cho Tuần 5!")
    else:
        print("[!] CẢNH BÁO: Cần kích hoạt khoảng đệm 2 ngày để train bù checkpoint bị thiếu/lỗi!")

if __name__ == "__main__":
    audit_all_checkpoints()
```

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Chạy script kiểm toán in ra `[✓] CHÚC MỪNG: Cả 4 checkpoint đã sẵn sàng 100% cho Tuần 5!`.
- [ ] Không còn bất kỳ công việc huấn luyện nặng nào bị tồn đọng sang Tuần 5.
