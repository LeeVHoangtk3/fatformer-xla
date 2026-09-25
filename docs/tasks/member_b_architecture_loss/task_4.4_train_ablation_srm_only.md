# TASK 4.4: HUẤN LUYỆN CHECKPOINT ABLATION 3 (SRM-ONLY)

- **Mã nhiệm vụ**: `Task 4.4`
- **Người phụ trách**: **Thành viên B** *(Architecture & Loss Lead)*
- **Môi trường thực hiện**: ☁️ **Colab L4 / A100** (~4–5 Compute Units)
- **Thời hạn hoàn thành**: Tuần 4 (Ngày 26)
- **Độ ưu tiên**: 🟠 Cao (Tạo mẫu đối chứng cho giá trị của nhánh SRM 3-Kernels)
- **Đầu vào (Input)**:
  - Tập dữ liệu ProGAN gốc 4-class (`progan_train.tar`).
  - Mô hình FatFormer có tích hợp khối `SpatialResidualBlock` (SRM 3-Kernels), nhưng **không có Gating $\lambda(x)$, không dùng Curriculum và dùng hàm Cross-Entropy tiêu chuẩn**.
- **Đầu ra (Output)**:
  - Checkpoint đối chứng `fatformer_srm_only.pth` lưu trên Google Drive 5TB `FatFormer_Hub/checkpoints/`.
  - Log loss và validation metrics qua 5 epoch.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Thực hiện thí nghiệm triệt tiêu module (*Ablation Study*): Cô lập hoàn toàn công năng của khối trích xuất vết dư vi sai không gian SRM 3-Kernels để chứng minh trên số liệu thực nghiệm: *"Nếu chỉ thêm SRM mà không có cổng tần số động và không có Curriculum thì độ bền nén cải thiện được bao nhiêu %?"*.
2. Huấn luyện 5 epoch trên GPU L4 hoặc A100 trong khoảng thời gian ngắn (~4–5 CUs), bảo đảm tính đa dạng của bộ 4 checkpoint đối chứng.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Chạy Huấn Luyện Checkpoint Ablation 3:
```bash
python train.py \
    --data_tar /content/drive/MyDrive/FatFormer_Hub/datasets/progan_train.tar \
    --use_srm True \
    --use_gating False \
    --use_curriculum False \
    --loss_type ce \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4 \
    --output_dir /content/drive/MyDrive/FatFormer_Hub/checkpoints/ \
    --checkpoint_name fatformer_srm_only.pth
```

### Giám Sát và Lưu Trữ Trọng Số:
- Kiểm tra tính ổn định của gradient: Đảm bảo lớp tích chập chiếu `srm_proj` học trơn tru, không gặp hiện tượng vanishing gradient.
- Sao lưu file `.pth` trực tiếp sang Drive 5TB sau khi hoàn tất epoch 5.

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Huấn luyện hoàn tất 5 epoch, loss giảm đều đặn.
- [ ] Checkpoint `fatformer_srm_only.pth` lưu trên Drive 5TB toàn vẹn.
- [ ] Bàn giao checkpoint cho Thành viên C lập ma trận so sánh tại Task 5.2.
