# TASK 4.2: CHIẾN DỊCH HUẤN LUYỆN GIAI ĐOẠN 3 (EXTREME HARDENING)

- **Mã nhiệm vụ**: `Task 4.2`
- **Người phụ trách**: **Thành viên C** *(Training & Evaluation Lead)*
- **Môi trường thực hiện**: ☁️ **Colab A100 SXM4 (40GB)** (~10–12 Compute Units)
- **Thời hạn hoàn thành**: Tuần 4 (Ngày 26)
- **Độ ưu tiên**: 🔴 Cực kỳ khẩn cấp (Tạo ra checkpoint hoàn thiện cuối cùng)
- **Đầu vào (Input)**:
  - Checkpoint trung gian `fatformer_srm_phase2.pth` từ Task 4.1.
  - Giáo trình nén sâu Giai đoạn 3 (Epoch 6–8): Nén rất nặng $Q \in [30, 50]$, $\sigma \in [1.5, 2.0]$, Down-Up $224 \rightarrow 112 \rightarrow 224$.
  - Learning rate hạ thấp xuống $5 \times 10^{-5}$ để tinh chỉnh mịn (fine-tuning).
- **Đầu ra (Output)**:
  - **Checkpoint hoàn thiện tối ưu `fatformer_srm_robust_final.pth`** (Mô hình 4 - Đầy đủ S1+S2+S3) lưu trên Google Drive 5TB `FatFormer_Hub/checkpoints/`.
  - Toàn bộ đồ thị hàm mất mát và đường cong hội tụ qua 8 epoch.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Thực thi giai đoạn thử thách khắc nghiệt nhất của giáo trình Curriculum: Đưa mô hình vào dải nén sâu $Q \in [30, 50]$ và thu nhỏ $112 \times 112$.
2. Cơ chế tôi luyện (Hardening Mechanism):
   - Nhánh DWT tần số cao bị phá hủy nặng nề, mạng Gating $\lambda(x)$ chủ động đóng dải tần xuống $\approx 0.05$.
   - Nhánh SRM 3-Kernels trở thành nguồn tín hiệu chủ đạo, học cách trích xuất vết nứt vi sai điểm ảnh và dấu vết nội suy bilinear/bicubic còn sót lại dưới lớp nén JPEG dày đặc.
3. Hoàn tất toàn bộ 8 epoch của mô hình đề xuất FatFormer-XLA.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Lệnh tiếp tục huấn luyện Giai đoạn 3:
```bash
python train.py \
    --config configs/train_config.yaml \
    --data_tar /content/dataset_local/diffusion_staging.tar \
    --resume /content/drive/MyDrive/FatFormer_Hub/checkpoints/fatformer_srm_phase2.pth \
    --epochs 8 \
    --start_epoch 6 \
    --lr_adapter 5e-5 \
    --lr_gating 5e-4 \
    --output_dir /content/drive/MyDrive/FatFormer_Hub/checkpoints/ \
    --checkpoint_name fatformer_srm_robust_final.pth
```

### Đánh giá nhanh mốc sau Epoch 8:
- Chạy `fast_eval.py` trên tập $Q=30$: Xác nhận Accuracy đã tăng vọt lên mức kỳ vọng $+10\% \sim +15\%$ so với mốc sàn Baseline.
- Kiểm tra tính toàn vẹn của file `fatformer_srm_robust_final.pth` trên Drive 5TB.

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Huấn luyện hoàn tất 8 epoch thành công trên GPU A100.
- [ ] Checkpoint cuối cùng `fatformer_srm_robust_final.pth` được lưu an toàn trên Drive 5TB.
- [ ] Bàn giao checkpoint cho toàn nhóm để bước sang Tuần 5 đánh giá Full-Benchmark và xuất Grad-CAM XAI.
