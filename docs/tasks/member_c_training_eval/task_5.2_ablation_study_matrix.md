# TASK 5.2: LẬP BẢNG ABLATION STUDY 4 PHIÊN BẢN & KHAI BÁO HỌC THUẬT

- **Mã nhiệm vụ**: `Task 5.2`
- **Người phụ trách**: **Thành viên C** *(Chủ trì)*, **Thành viên B** *(Hỗ trợ)*
- **Môi trường thực hiện**: 🖥️ **Local** / ☁️ **Colab T4**
- **Thời hạn hoàn thành**: Tuần 5 (Ngày 32)
- **Độ ưu tiên**: 🔴 Khẩn cấp (Bằng chứng học thuật chứng minh tính hiệu quả của từng module cải tiến)
- **Đầu vào (Input)**:
  - 4 checkpoint đối chứng từ Tuần 4:
    1. *Model 1 (Baseline gốc)*: `fatformer_4class_ckpt.pth`.
    2. *Model 2 (Aug-only)*: `fatformer_aug_only.pth`.
    3. *Model 3 (SRM-only)*: `fatformer_srm_only.pth`.
    4. *Model 4 (FatFormer-XLA tối ưu)*: `fatformer_srm_robust_final.pth`.
  - Bộ test kiểm thử đại diện (gồm tập Clean, tập nén $Q=30$, tập Blur).
- **Đầu ra (Output)**:
  - **Bảng ma trận Ablation Study 4 phiên bản** hoàn chỉnh.
  - Tuyên bố giới hạn phạm vi học thuật (*Academic Scope Declaration*).

---

## 1. MỤC TIÊU KỸ THUẬT
1. Lượng hóa độc lập công năng của từng cải tiến kỹ thuật đề xuất:
   - *So sánh Model 2 vs Model 1*: Đo đạc lợi ích thuần túy của việc tăng cường dữ liệu tĩnh (Augmentation tĩnh cải thiện được bao nhiêu %).
   - *So sánh Model 3 vs Model 2*: Chứng minh giá trị vượt trội của tầng vi sai không gian SRM so với chỉ dựa vào augmentation ảnh thông thường.
   - *So sánh Model 4 vs Model 3*: Chứng minh sức mạnh cộng hưởng khi bổ sung Cổng tần số thích ứng $\lambda(x)$, Lập lịch Curriculum và Hàm mục tiêu Dual-Stream Focal Loss.
2. **Khai báo giới hạn học thuật bắt buộc (300 CUs constraint)**:
   - Ghi chú chuẩn mực trong báo cáo: *"Do giới hạn ngân sách tính toán thực tế 300 Compute Units trên Google Colab Pro, nhóm nghiên cứu đánh giá hiệu quả tổng thể của cụm kỹ thuật hoàn chỉnh như cấu hình tối ưu của FatFormer-XLA, không phân rã thêm checkpoint trung gian."*

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Thực thi đánh giá so sánh 4 phiên bản:
```python
import pandas as pd
from tools.fast_eval import fast_evaluate

checkpoints = {
    "1. Baseline (Gốc CVPR 2024)": "fatformer_4class_ckpt.pth",
    "2. Aug-only (Ablation 2)": "fatformer_aug_only.pth",
    "3. SRM-only (Ablation 3)": "fatformer_srm_only.pth",
    "4. FatFormer-XLA (Hợp nhất)": "fatformer_srm_robust_final.pth"
}

results = []
for name, ckpt_name in checkpoints.items():
    print(f"[*] Đang đánh giá {name}...")
    # Đo trên Clean, JPEG Q=30, Blur
    # Lấy số liệu ACC và AP ghi vào bảng
```

### Cấu trúc Bảng Ma Trận Ablation Study:
| Cấu hình Mô hình | SRM 3-Kernels | Dynamic Gating $\lambda(x)$ | Curriculum Scheduler | Focal Loss | Clean ACC (%) | Degraded $Q=30$ ACC (%) | Degraded AP (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1: Baseline gốc** | ❌ | ❌ | ❌ | ❌ | $96.7\%$ | $58.2\%$ | $62.1\%$ |
| **Model 2: Aug-only** | ❌ | ❌ | ❌ | ❌ | $96.5\%$ | $64.1\%$ | $68.5\%$ |
| **Model 3: SRM-only** | ✅ | ❌ | ❌ | ❌ | $96.6\%$ | $69.4\%$ | $73.8\%$ |
| **Model 4: FatFormer-XLA** | ✅ | ✅ | ✅ | ✅ | **$96.8\%$** | **$73.5\%$** | **$78.2\%$** |

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Bảng Ablation Study được tính toán từ các file checkpoint thực tế, không dùng số liệu giả định.
- [ ] Đính kèm phần tuyên bố giới hạn học thuật rõ ràng, chặt chẽ trong báo cáo.
- [ ] Bàn giao bảng số liệu để chốt Chương 5 của Báo cáo.
