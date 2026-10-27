# TASK 4.3: HUẤN LUYỆN CHECKPOINT ABLATION 2 (AUGMENTATION TĨNH)

- **Mã nhiệm vụ**: `Task 4.3`
- **Người phụ trách**: **Thành viên A** *(Data & Infra Lead)*
- **Môi trường thực hiện**: ☁️ **Colab T4 / L4** (~3–4 Compute Units)
- **Thời hạn hoàn thành**: Tuần 4 (Ngày 25)
- **Độ ưu tiên**: 🟠 Cao (Tạo mẫu đối chứng Ablation Study cho toàn dự án)
- **Đầu vào (Input)**:
  - Dữ liệu ProGAN gốc 4-class (`progan_train.tar`).
  - Mô hình FatFormer gốc (FAA + LGA, **không có SRM 3-Kernels, không có Gating $\lambda(x)$**).
  - Hàm mục tiêu Cross-Entropy Loss tiêu chuẩn.
  - Augmentation tĩnh ngẫu nhiên (Uniform Random JPEG $Q \in [40, 95]$).
- **Đầu ra (Output)**:
  - Checkpoint đối chứng `fatformer_aug_only.pth` lưu trên Google Drive 5TB `FatFormer_Hub/checkpoints/`.
  - Log loss quá trình huấn luyện 5 epoch.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Xây dựng mẫu đối chứng khoa học: Trả lời câu hỏi phản biện: *"Liệu chỉ cần áp dụng Data Augmentation tĩnh ngẫu nhiên thông thường thì mô hình đã đủ bền vững chưa, hay bắt buộc phải có kiến trúc vi sai SRM và Cổng tần số thích ứng?"*.
2. Huấn luyện 5 epoch trên GPU T4 hoặc L4 tiết kiệm tài nguyên (~3–4 CUs), chia sẻ gánh nặng tính toán với Thành viên C.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Bước 1: Cấu hình Notebook Huấn Luyện Ablation 2
Sử dụng script huấn luyện với cờ cấu hình tắt SRM và Gating:
```bash
python train.py \
    --data_tar /content/drive/MyDrive/FatFormer_Hub/datasets/progan_train.tar \
    --use_srm False \
    --use_gating False \
    --use_curriculum False \
    --static_aug True \
    --loss_type ce \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4 \
    --output_dir /content/drive/MyDrive/FatFormer_Hub/checkpoints/ \
    --checkpoint_name fatformer_aug_only.pth
```

### Bước 2: Theo Dõi Hội Tụ & Sao Lưu Trọng Số
- Ghi nhận đường cong Loss (CE) sau mỗi epoch.
- Xác nhận file checkpoint `fatformer_aug_only.pth` được lưu toàn vẹn trên Drive 5TB.

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Huấn luyện hoàn tất 5 epoch mà không gặp lỗi tràn VRAM (OOM) trên GPU T4/L4.
- [ ] Checkpoint `fatformer_aug_only.pth` có dung lượng ~3.7 GB (khớp kích thước trọng số đầy đủ).
- [ ] Bàn giao checkpoint cho Thành viên C lập bảng Ablation Study ở Tuần 5 (Task 5.2).
