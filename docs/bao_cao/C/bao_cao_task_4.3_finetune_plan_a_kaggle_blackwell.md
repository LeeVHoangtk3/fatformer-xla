# BÁO CÁO NGHIỆM THU KỸ THUẬT: TASK 4.3
## CHIẾN DỊCH HUẤN LUYỆN FINETUNE PHƯƠNG ÁN A TRÊN MÁY CHỦ KAGGLE OFFLINE (GPU NVIDIA RTX PRO 6000 BLACKWELL SERVER EDITION)

> **Người thực hiện**: Thành viên C (Training & Evaluation Lead)  
> **Người phối hợp**: Thành viên A (Data & Infra Lead), Thành viên B (Architecture & Loss Lead)  
> **Thời gian thực hiện**: 09/10/2026 – 10/10/2026  
> **Môi trường thực thi**: Kaggle Competition Server (Offline 100%, Internet: Disabled)  
> **Hạ tầng phần cứng**: GPU **NVIDIA RTX PRO 6000 Blackwell Server Edition (94.97 GB VRAM)**  
> **Cấu hình huấn luyện**: Effective Batch Size = 64 (Accumulation = 1), AMP FP16, AdamW (lr=1e-4), DualStreamCELoss(label_smoothing=0.1)  
> **Tài nguyên dữ liệu**: 147.624 ảnh huấn luyện (144.024 ProGAN 85% + 3.600 Staging GenImage 15%) & 8.000 ảnh Validation (`progan_val`)  
> **Notebook thực thi**: [`notebooks/task/task_4.3/task_4.3_train_plan_a_kaggle_blackwell.ipynb`](../../../notebooks/task/task_4.3/task_4.3_train_plan_a_kaggle_blackwell.ipynb)  
> **Hiện vật bàn giao**: Checkpoint trung gian `fatformer_plan_a_final.pth` (3.28 GB, lưu từ Epoch 3 của quá trình huấn luyện bị gián đoạn ở Epoch 4 do đầy bộ nhớ đĩa) & Nhật ký lịch sử [`DATASET/logs/train_val_history.csv`](../../../DATASET/logs/train_val_history.csv)  

---

### I. MỤC TIÊU & BỐI CẢNH TRIỂN KHAI

Sau khi kiểm toán nguyên nhân sụt giảm độ chính xác ở mô hình cũ (do chỉ nạp 3.600 ảnh staging, bỏ quên 144k ảnh ProGAN và Focal Loss đè bẹp logit), nhóm nghiên cứu đã kích hoạt **Chiến Dịch Huấn Luyện Finetune Phương Án A** với các mục tiêu:
1. **Cải thiện độ chính xác trên ảnh sạch**: Giảm thiểu hiện tượng quên tri thức (Catastrophic Forgetting), kéo Clean Validation ACC trên tập kiểm định nội bộ lên mức $\ge 99.0\%$.
2. **Cân bằng tỷ lệ đa miền**: Áp dụng `WeightedRandomSampler` bốc mẫu cân bằng 85% ProGAN + 15% Diffusion Staging vào Batch Size 64.
3. **Bảo toàn mỏ neo chống suy thoái**: Khóa `clean_prob = 0.50` xuyên suốt các epoch kết hợp `CurriculumDegradationScheduler`.
4. **Ổn định hóa Gradient**: Sử dụng `DualStreamCELoss(label_smoothing=0.1)` thay thế Focal Loss $\gamma=2.0$.
5. **Lưu trữ checkpoint đánh giá**: Thu nhận checkpoint có độ chính xác cao trên tập kiểm định (`model_best.pth` tại Epoch 3) phục vụ phân tích thực nghiệm và đánh giá bước đầu.

---

### II. THIẾT LẬP KỸ THUẬT & DỮ LIỆU ĐẦU VÀO

1. **Chuẩn hóa hạ tầng Kaggle Offline (100% không Internet)**:
   - Toàn bộ mã nguồn được nạp qua dataset `fatformer-code`.
   - Toàn bộ 17.75 GB dữ liệu và trọng số pretrained được tải lên Kaggle Dataset `hoangleev/fatformer-assets-pack`.
   - Kaggle tự động giải nén dữ liệu lên NVMe SSD tại `/kaggle/input/datasets/hoangleev/fatformer-assets-pack/`, giảm thiểu độ trễ I/O.
   - Xử lý BPE Tokenizer offline bằng file `bpe_simple_vocab_16e6.txt.gz` nội bộ.
2. **Phân bổ tham số mô hình trong lượt chạy này**:
   - Tổng tham số mô hình: **578,441,298** (~578.44 M).
   - Tham số đóng băng: **427,616,513** (73.93%).
   - Tham số tối ưu hóa trong lần chạy này: **150,824,785** (26.07%).
   - *Ghi chú kỹ thuật về PEFT*: Do danh sách lọc tham số thời điểm này chưa bao gồm điều kiện loại trừ các khối FFN trong ViT transformer blocks, tỷ lệ trainable thực tế của lần chạy đạt 26.07%. Thiết kế chuẩn hóa sau kiểm toán đo đạc trên `ViT-L-14.pt` là 15,408,457 tham số (2.66%).
3. **Cơ cấu phân phối dữ liệu huấn luyện**:
   - Dữ liệu ProGAN (4-class: car, cat, chair, horse): **144.024 ảnh** (85.0%).
   - Dữ liệu Staging (SD v1.5 + Midjourney v5): **3.600 ảnh** (15.0%).
   - Tổng số mẫu: **147.624 ảnh** $\rightarrow$ 2.306 batches/epoch (Batch size = 64).
   - Tập kiểm định (Validation Set): **8.000 ảnh** từ `progan_val`.

---

### III. SỐ LIỆU ĐO ĐẠC THỰC TẾ & KẾT QUẢ ĐẠT ĐƯỢC

Quá trình huấn luyện thực tế trên GPU NVIDIA RTX PRO 6000 Blackwell Server Edition ghi nhận các thông số:
- **Tốc độ xử lý**: Đạt trung bình **75.5 – 76.8 ảnh/giây** (~1.19 batch/giây).
- **Thời gian mỗi epoch**: Hoàn thành trong khoảng **1.921s – 1.955s (~32.0 – 32.5 phút/epoch)**.
- **Tổng thời gian huấn luyện 4 Epochs**: **7.769,53 giây (~2 giờ 09 phút)**.

#### Bảng Ghi Nhận Lịch Sử Huấn Luyện & Kiểm Định Thực Tế (Trích xuất từ `train_val_history.csv`):

| Epoch | Curriculum Stage | Train Loss | Val ACC (%) | Val AP (%) | Val AUC (%) | Val Real ACC (%) | Val Fake ACC (%) | Thời Gian (s) | Trạng Thái Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Giai đoạn 1 ($Q \in [70, 90]$) | 0.555561 | 99.67% | 99.97% | 99.95% | 100.00% | 99.35% | 1947.38s | `checkpoint_epoch_001.pth` (3.28 GB) |
| **2** | Giai đoạn 1 ($Q \in [70, 90]$) | 0.451662 | 99.83% | 100.00% | 100.00% | 99.95% | 99.70% | 1945.57s | `checkpoint_epoch_002.pth` (3.28 GB) |
| **3** | Giai đoạn 2 ($Q \in [45, 70]$) | 0.455155 | **99.84%** | **100.00%** | **100.00%** | **99.92%** | **99.75%** | **1921.38s** | **`model_best.pth` (3.28 GB)** (Lưu trữ làm checkpoint đánh giá) |
| **4** | Giai đoạn 2 ($Q \in [45, 70]$) | 0.450965 | 99.78% | 99.99% | 99.99% | 99.85% | 99.70% | 1955.20s | Quá trình lưu bị lỗi tràn đĩa 19.5GB |

---

### IV. PHÂN TÍCH & ĐÁNH GIÁ KỸ THUẬT

1. **Độ chính xác trên tập kiểm định nội bộ `progan_val`**:
   - Độ chính xác Validation trên 8.000 ảnh `progan_val` đạt **99.84%** tại Epoch 3, với AP đạt **100.00%** và AUC đạt **100.00%**.
   - Độ chính xác trên ảnh Real đạt **99.92%** và ảnh Fake đạt **99.75%**.
   - *Lưu ý đánh giá*: Kết quả này phản ánh độ chính xác trên tập ảnh sạch ProGAN cùng phân phối với dữ liệu huấn luyện, chưa phản ánh độ bền khi gặp các mức nén sâu và các miền sinh ảnh chưa biết.
2. **Tính ổn định của hàm mục tiêu `DualStreamCELoss(label_smoothing=0.1)`**:
   - Loss hội tụ từ 1.6080 xuống 0.4509 mà không có hiện tượng nổ gradient hay sụp đổ logit về sát 0 như Focal Loss cũ.
   - Nhãn làm mịn (Label Smoothing 0.1) hỗ trợ kiểm soát overconfidence khi tiếp xúc với các mẫu ảnh nén biến dạng.
3. **Sự cố tràn đĩa cứng Kaggle ở Epoch 4 & Tình trạng gián đoạn**:
   - *Hiện tượng*: Tại thời điểm kết thúc Epoch 4, hàm `torch.save` gặp ngoại lệ: `RuntimeError: basic_ios::clear: iostream error` / `unexpected pos 3348013248 vs 3348013136`.
   - *Nguyên nhân kỹ thuật*: Dung lượng ổ đĩa `/kaggle/working` bị giới hạn cứng ở mức **19.5 GiB**. Mỗi full checkpoint (chứa cả weights 1.1 GB và trạng thái bộ nhớ AdamW optimizer 2.18 GB) có kích thước **~3.28 GB**. Khi tích lũy đồng thời 4 checkpoint epoch, `checkpoint_latest.pth` và `model_best.pth`, tổng dung lượng vượt ngưỡng 19.5 GiB.
   - *Hệ quả đối với giáo trình*: Kế hoạch huấn luyện 5 epoch đã bị gián đoạn ngoài ý muốn. **Phase 3 (Epoch 5 - Hardening với $Q \in [30, 50]$, learning rate $\times 0.5$) chưa từng được thực hiện trong lần chạy này**.
   - *Xử lý hiện trường*:
     1. Dọn dẹp các tệp tạm và lưu giữ lại checkpoint Epoch 3: `checkpoint_epoch_003.pth` (3.28 GB) và `model_best.pth` (3.28 GB).
     2. Đổi tên thành `fatformer_plan_a_final.pth` (3.28 GB) để làm tệp trọng số chuyển tiếp cho đợt kiểm thử sơ bộ.
     3. Sửa mã nguồn [`src/training/checkpoint_manager.py`](../../../src/training/checkpoint_manager.py) trên local repo để tự động dọn dẹp các checkpoint cũ trước khi ghi mới.
4. **Đánh giá về điểm dừng và hệ quả đối với Task 5.1**:
   - Việc dừng lại ở Epoch 3 là tình huống gián đoạn do sự cố hạ tầng lưu trữ, không phải là điểm dừng kết thúc toàn bộ giáo trình.
   - Do chưa trải qua Phase 3 (nén sâu $Q \in [30, 50]$) và tỷ lệ trainable trong lần chạy này ở mức 26.07%, checkpoint Epoch 3 chưa có khả năng kháng nén sâu đầy đủ.
   - Đây là nguyên nhân trực tiếp dẫn đến kết quả trong Task 5.1 sau đó: Fake ACC tại $Q=30$ chỉ đạt 44.84% (hơn 55% ảnh giả bị bỏ sót), và các tập như StyleGAN2 (6%) hay VQ-Diffusion (4.4%) bị sụt giảm sâu.

---

### V. KẾT LUẬN & ĐỀ XUẤT HÀNH ĐỘNG

1. **Kết luận**:
   - Quá trình huấn luyện Phương Án A cung cấp checkpoint trung gian tại Epoch 3 (`fatformer_plan_a_final.pth`) cho thấy sự phục hồi độ chính xác trên tập ảnh sạch nội bộ so với mô hình cũ bị sụp đổ.
   - Lần chạy này chưa hoàn thành 5 epoch theo thiết kế và chưa bước vào Phase 3 hardening do sự cố tràn đĩa tại Epoch 4.
2. **Danh mục tài sản lưu trữ**:
   - [x] **Trọng số mô hình trung gian**: `fatformer_plan_a_final.pth` (3.28 GB), lưu từ Epoch 3.
   - [x] **Nhật ký huấn luyện**: [`DATASET/logs/train_val_history.csv`](../../../DATASET/logs/train_val_history.csv) ghi nhận số liệu 4 Epochs.
   - [x] **Notebook thực nghiệm**: [`notebooks/task/task_4.3/task_4.3_train_plan_a_kaggle_blackwell.ipynb`](../../../notebooks/task/task_4.3/task_4.3_train_plan_a_kaggle_blackwell.ipynb) lưu giữ log thực thi trên GPU Blackwell.
3. **Đề xuất hành động tiếp theo**:
   - Sử dụng checkpoint Epoch 3 để tiến hành đo đạc tham chiếu sơ bộ trên 18 tập test hai chiều (Task 5.1).
   - Chuẩn bị kế hoạch huấn luyện lại đủ 5 epoch với cấu hình PEFT 2.66% và hoàn thành đầy đủ Phase 3 hardening sau khi cơ chế dọn dẹp checkpoint đã được sửa đổi.

---
*Báo cáo được lập bởi Lead C, lưu trữ tại `docs/bao_cao/C/bao_cao_task_4.3_finetune_plan_a_kaggle_blackwell.md`.*
