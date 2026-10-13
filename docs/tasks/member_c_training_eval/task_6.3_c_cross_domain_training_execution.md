# TASK 6.3-C: THỰC THI CHIẾN DỊCH HUẤN LUYỆN THÍCH ỨNG ĐA MIỀN TRÊN GPU

- **Mã nhiệm vụ**: `Task 6.3-C` *(Milestone 6 - Tuần 6)*
- **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**
- **Người phối hợp**: **Thành viên A (Data Lead)**, **Thành viên B (Architecture Lead)**
- **Môi trường thực thi**: 🚀 **Kaggle Server (RTX 6000 Blackwell 95GB VRAM)** / ☁️ **Colab A100**
- **Độ ưu tiên**: 🔴 Khẩn cấp (Thực thi huấn luyện cốt lõi Milestone 6)
- **Trạng thái**: `[ ] Sẵn sàng triển khai`

---

## 1. Mục Tiêu & Cơ Sở Khoa Học

Sau khi hoàn thành chuẩn bị kho dữ liệu tại Task 6.2 và vượt qua Smoke Gate với hàm mất mát Task 6.3-B, Task 6.3-C là bước triển khai huấn luyện trên máy chủ GPU để tạo ra các checkpoint ứng viên cho ma trận Ablation Study (Task 6.5).

Mục tiêu chính:
1. Huấn luyện các nhánh mô hình thích ứng với điều kiện đối chứng cố định tuyệt đối (cùng random seed 42, cùng data manifest, cùng checkpoint khởi tạo, cùng learning rate schedule).
2. Tối ưu hóa việc sử dụng đĩa lưu trữ: Áp dụng cơ chế xoay vòng checkpoint, loại bỏ việc lưu checkpoint từng epoch nhằm bảo vệ hạn ngạch bộ nhớ đĩa Drive 5TB và SSD.
3. Tạo ra 3 ứng viên cốt lõi: **M3** (+ Data đa miền), **M4** (+ Distillation & Consistency Loss), **M5** (+ LoRA ViT Adapter) để chuyển tiếp sang Task 6.5 và Task 6.4.

---

## 2. Cấu Hình Thực Nghiệm & Kiểm Soát Biến Số Nghiêm Ngặt

Để bảo đảm tính khoa học của phân tích cắt bỏ thành phần (Ablation Study), mọi nhánh huấn luyện phải cố định các thông số nền tảng:

| Thông Số Kỹ Thuật | Giá Trị Cố Định Toàn Bộ Nhánh | Mục Đích Kiểm Soát |
| :--- | :--- | :--- |
| **Random Seed** | `seed = 42` (torch, numpy, random, cuda) | Triệt tiêu hoàn toàn tính ngẫu nhiên |
| **Khởi tạo trọng số** | `fatformer_plan_a_best_weights.pth` | Kế thừa trạng thái đã hội tụ của SRM và Gating |
| **Mô hình Teacher** | `fatformer_4class_ckpt.pth` (đóng băng 100%) | Giữ nguyên mỏ neo tri thức ảnh sạch |
| **Dữ liệu huấn luyện** | `train_crossdomain` (theo `data_manifest.json`) | Không thay đổi danh sách tệp giữa các lần chạy |
| **Bộ lấy mẫu** | `BalancedCrossDomainSampler` | Đảm bảo tỷ lệ bộ sinh đồng đều |
| **Batch Size** | 64 (RTX 6000) hoặc 32 (A100) | Giữ nguyên kích thước lô |
| **Learning Rate** | $1 \times 10^{-5}$ với Cosine Annealing (5 epochs) | Lập lịch suy giảm đồng bộ |

---

## 3. Chính Sách Quản Lý Đĩa Lưu Trữ (Disk Storage Policy)

Tuyệt đối không lưu file `checkpoint_epoch_{E}.pth` ở mỗi epoch để tránh tràn dung lượng. Quy tắc lưu trữ đĩa được chuẩn hóa như sau:
1. `checkpoint_latest.pth`: Luôn ghi đè ở mỗi epoch, chứa đầy đủ model state, optimizer state, scheduler state và GradScaler để phục vụ khôi phục phiên (`--resume`).
2. `crossdomain_model_best.pth`: Chỉ cập nhật khi điểm composite validation trên `val_crossdomain` vượt qua mốc cao trước đó.
3. Sau khi hoàn thành toàn bộ chiến dịch và tuyển chọn mô hình tại Task 6.4, mới thực hiện trích xuất tệp gọn nhẹ `fatformer_crossdomain_weights_only.pth` (chỉ chứa `state_dict` mô hình, dung lượng ~2.3 GB) để lưu trữ lâu dài.

---

## 4. Quy Trình Vận Hành

### Bước 1: Kiểm định Cổng Smoke Gate M6
- Chạy script kiểm thử tích hợp `tools/test_smoke_m6.py` trên 1 batch dữ liệu. Xác nhận 100% pass mới bắt đầu huấn luyện chính thức.

### Bước 2: Huấn luyện tuần tự các nhánh ứng viên
Chạy script huấn luyện chuẩn hóa cho từng nhánh:
- **Nhánh M3 (Chỉ thêm Data đa miền)**:
  ```bash
  python src/training/train_crossdomain.py \
      --config configs/train_m3_data_only.yaml \
      --seed 42 --epochs 5 --output_dir checkpoints/m3/
  ```
- **Nhánh M4 (Data đa miền + Distillation & Consistency Loss)**:
  ```bash
  python src/training/train_crossdomain.py \
      --config configs/train_m4_distill.yaml \
      --seed 42 --epochs 5 --alpha_kd 0.4 --beta_cons 0.2 --output_dir checkpoints/m4/
  ```
- **Nhánh M5 (M4 + LoRA ViT Adapter)**:
  ```bash
  python src/training/train_crossdomain.py \
      --config configs/train_m5_lora.yaml \
      --seed 42 --epochs 5 --enable_lora --output_dir checkpoints/m5/
  ```

### Bước 3: Thu thập nhật ký hội tụ
- Ghi nhận chi tiết loss train, CE loss, KD loss, Consistency loss và validation metrics qua từng epoch vào `DATASET/logs/m6_training_history.csv`.

---

## 5. Tiêu Chuẩn Nghiệm Thu (Definition of Done)

- [ ] Vượt qua 100% Cổng Smoke Gate Milestone 6 trước khi khởi động huấn luyện.
- [ ] Hoàn thành huấn luyện các nhánh ứng viên với điều kiện seed 42 và manifest cố định.
- [ ] Tuân thủ nghiêm ngặt chính sách đĩa lưu trữ: Không lưu dư thừa checkpoint từng epoch.
- [ ] Checkpoint tốt của từng nhánh và tệp nhật ký `m6_training_history.csv` được lưu an toàn trên Google Drive 5TB.
- [ ] Bàn giao toàn bộ kết quả cho Task 6.5 tiến hành lập ma trận Ablation Study.
- [ ] Biên soạn báo cáo nghiệm thu tại [`docs/bao_cao/C/bao_cao_task_6.3_c_cross_domain_training_execution.md`](../../bao_cao/C/bao_cao_task_6.3_c_cross_domain_training_execution.md).
