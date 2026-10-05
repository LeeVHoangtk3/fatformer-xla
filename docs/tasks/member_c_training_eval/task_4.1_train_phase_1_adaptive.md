# TASK 4.1: CHIẾN DỊCH HUẤN LUYỆN GIAI ĐOẠN 1 & 2 TRÊN GPU A100

- **Mã nhiệm vụ**: `Task 4.1`
- **Người phụ trách**: **Thành viên C** *(Training & Evaluation Lead)*
- **Môi trường thực hiện**: ☁️ **Colab A100 SXM4 (40GB)** (~15–18 Compute Units)
- **Thời hạn hoàn thành**: Tuần 4 (Ngày 25)
- **Độ ưu tiên**: 🔴 Cực kỳ khẩn cấp (Trụ cột huấn luyện chính của toàn đồ án)
- **Đầu vào (Input)**:
  - Pipeline đã vượt qua Smoke Test (Task 3.6).
  - Dữ liệu `diffusion_staging.tar` (90% ProGAN + 10% GenImage catalyst) giải nén trên SSD NVMe Colab.
  - Giáo trình Curriculum:
    - *Giai đoạn 1 (Epoch 1–2)*: Nén nhẹ $Q \in [70, 90]$, $\sigma \in [0.5, 1.0]$.
    - *Giai đoạn 2 (Epoch 3–5)*: Nén vừa $Q \in [45, 70]$, $\sigma \in [1.0, 1.5]$, Down-Up $224 \rightarrow 160 \rightarrow 224$.
  - Hàm mục tiêu `DualStreamFocalLoss` ($\gamma=2.0, \alpha=0.25$).
- **Đầu ra (Output)**:
  - Checkpoint trung gian `fatformer_srm_phase2.pth` lưu trên Google Drive 5TB `FatFormer_Hub/checkpoints/`.
  - Log loss và kết quả Fast-Eval validation sau mỗi epoch (Epoch 1 đến 5).

---

## 1. MỤC TIÊU KỸ THUẬT
1. Thực thi huấn luyện 5 epoch đầu tiên trên GPU cao cấp NVIDIA A100 SXM4 (40GB VRAM):
   - Tốc độ huấn luyện kỳ vọng: ~10–12 phút/epoch với dữ liệu Staging ~36.000 ảnh.
   - Batch size 32, Gradient Accumulation 2 (Effective batch size = 64).
   - Learning rate: $10^{-4}$ cho adapter, $10^{-3}$ cho gating network.
2. Mục tiêu hội tụ qua 2 giai đoạn:
   - **Giai đoạn 1 (Epoch 1–2 - Ổn định)**: Khởi tạo êm đềm các trọng số của lớp chiếu SRM và mạng gating mà không làm tổn thương các biểu diễn FAA gốc.
   - **Giai đoạn 2 (Epoch 3–5 - Chuyển tiếp thích ứng)**: Cổng $\lambda(x)$ bắt đầu học hạ giá trị khi gặp các ảnh có nhiễu khối JPEG ($Q \le 60$), chuyển dịch trọng tâm sang biểu diễn vi sai không gian SRM.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Cách 1 (Khuyến nghị tối ưu - One-Shot 8-Epoch Campaign):
Với môi trường GPU cao cấp (8.9 CU/giờ), toàn bộ 8 epoch của mô hình chính chỉ mất **~32–35 phút**. Hệ thống cho phép chạy liền mạch 8 epoch qua notebook [`notebooks/task/task_4.1/task_4.1_train_phase_1.ipynb`](../../../notebooks/task/task_4.1/task_4.1_train_phase_1.ipynb):
```bash
python train.py \
    --data_tar /content/drive/MyDrive/Fatformer/datasets/diffusion_staging.tar \
    --epochs 8 \
    --batch_size 32 \
    --grad_accum 2 \
    --lr 1e-4 \
    --lr_gating 1e-3 \
    --use_srm True \
    --use_gating True \
    --use_curriculum True \
    --loss_type focal \
    --output_dir /content/drive/MyDrive/Fatformer/checkpoint \
    --checkpoint_name fatformer_srm_robust_final.pth
```
*Lưu ý*: Script `train.py` tự động phát hiện và trích xuất checkpoint Epoch 5 thành `fatformer_srm_phase2.pth` lưu trực tiếp trên Drive 5TB để phục vụ nghiệm thu độc lập cho Task 4.1.

### Cách 2 (Dự phòng ngắt quãng - 5 Epochs riêng lẻ):
Nếu muốn kiểm tra điểm dừng tại Epoch 5 trước khi chạy tiếp Giai đoạn 3:
```bash
python train.py \
    --data_tar /content/drive/MyDrive/Fatformer/datasets/diffusion_staging.tar \
    --epochs 5 \
    --batch_size 32 \
    --grad_accum 2 \
    --lr 1e-4 \
    --lr_gating 1e-3 \
    --use_srm True \
    --use_gating True \
    --use_curriculum True \
    --loss_type focal \
    --output_dir /content/drive/MyDrive/Fatformer/checkpoint \
    --checkpoint_name fatformer_srm_phase2.pth
```

### Quy trình theo dõi sau mỗi epoch:
1. Ghi nhận `Loss_total`, `Loss_visual`, `Loss_align` qua thanh tiến trình tqdm.
2. Kiểm tra các file `fatformer_epoch_{epoch}.pth` tự động đồng bộ sang Google Drive 5TB `/content/drive/MyDrive/Fatformer/checkpoint/`.

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Hoàn tất 5 epoch đầu tiên mà không gặp sự cố crash/OOM, tổng thời gian ~20–25 phút trên GPU cao cấp (hoặc nằm trong phiên chạy 8 epoch liên tục).
- [ ] Loss giảm đều đặn, không có biểu hiện phân kỳ hay sốc gradient.
- [ ] Checkpoint `fatformer_srm_phase2.pth` (hoặc `fatformer_epoch_5.pth`) có sẵn và toàn vẹn trên Google Drive 5TB, sẵn sàng chuyển tiếp sang Giai đoạn 3 (Task 4.2).
