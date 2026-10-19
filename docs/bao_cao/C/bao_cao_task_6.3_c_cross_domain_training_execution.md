# BÁO CÁO NGHIỆM THU KỸ THUẬT: TASK 6.3-C
## KIỂM THỬ HẠ TẦNG & QUY TRÌNH HUẤN LUYỆN THÍCH ỨNG ĐA MIỀN TRÊN GPU CLOUD (MILESTONE 6)

> **Mã nhiệm vụ**: `Task 6.3-C` *(Milestone 6 — Tuần 6)*  
> **Người chủ trì**: **Thành viên C (Training & Evaluation Lead)**  
> **Người phối hợp**: **Thành viên A (Data & Infra Lead)**, **Thành viên B (Architecture & Loss Lead)**  
> **Thời gian hoàn thành hạ tầng**: 19/10/2026  
> **Môi trường thực thi**: 🚀 **Kaggle Server (NVIDIA RTX PRO 6000 Blackwell Server Edition — 94.97 GB VRAM, PyTorch 2.11.0+cu128)**  
> **Trạng thái**: `[~] ĐÃ THÔNG SUỐT HẠ TẦNG KỸ THUẬT & SẴN SÀNG HUẤN LUYỆN DỮ LIỆU ĐẦY ĐỦ`  
> **Tài liệu nhiệm vụ gốc**: [`docs/tasks/member_c_training_eval/task_6.3_c_cross_domain_training_execution.md`](../../tasks/member_c_training_eval/task_6.3_c_cross_domain_training_execution.md)  
> **Kịch bản thực thi**: [`notebooks/task/task_6.3/task_6.3_c_train_crossdomain_kaggle.ipynb`](../../../notebooks/task/task_6.3/task_6.3_c_train_crossdomain_kaggle.ipynb) & [`src/training/train_crossdomain.py`](../../../src/training/train_crossdomain.py)

---

### I. MỤC TIÊU & BỐI CẢNH THỰC NGHIỆM

Sau khi hoàn thành chuẩn bị hạ tầng dữ liệu tại Task 6.2 và vượt qua [Cổng Kiểm Thử Tích Hợp Smoke Gate M6](bao_cao_smoke_gate_m6.md), **Task 6.3-C** triển khai quy trình huấn luyện trên máy chủ GPU hiệu năng cao nhằm xây dựng các mô hình ứng viên đa miền phục vụ ma trận nghiên cứu cắt bỏ thành phần (Ablation Study tại Task 6.5) và tuyển chọn mô hình tại Task 6.4.

Giai đoạn đầu của Task 6.3-C tập trung vào **Kiểm định và Thông suốt Hạ tầng Kỹ thuật (Technical Pipeline Staging)**:
1. Xác minh khả năng vận hành tuần tự 3 nhánh kiến trúc (M3, M4, M5) trên phần cứng máy chủ GPU thế hệ mới (RTX PRO 6000 Blackwell 95GB VRAM).
2. Tích hợp thực tế hàm mất mát đa nhiệm triệt tiêu phân kỳ tri thức ($L_{\text{KD}}$) và đồng bộ biểu diễn không gian ($L_{\text{cons}}$) do Lead B thiết kế (Task 6.3-B).
3. Thiết lập và kiểm chứng cơ chế quản lý đĩa lưu trữ, giải quyết triệt để bài toán tràn đĩa 20 GB trên nền tảng Kaggle.
4. Chuẩn hóa bộ điều hợp tham số thấp LoRA (`ViTLoRAAdapter`) trực tiếp trên các tầng MLP của backbone thị giác ViT-L/14.

---

### II. KẾ THỪA THÔNG SỐ KỸ THUẬT (TUÂN THỦ NGUYÊN TẮC 4)

Báo cáo nghiệm thu kế thừa chuẩn xác các giá trị thông số đã được thẩm định từ các giai đoạn tiền đề:
- **Từ Task 4.3 ([Báo cáo Task 4.3](bao_cao_task_4.3_finetune_plan_a_kaggle_blackwell.md) - Lead C)**:
  - Checkpoint khởi tạo Student: [`fatformer_plan_a_best_weights.pth`](../../../DATASET/checkpoints/fatformer_plan_a_best_weights.pth) (2,207.50 MB, hội tụ đầy đủ SRM 3-Kernels và Dynamic Frequency Gating).
- **Từ Task 1.2 ([Báo cáo Task 1.2](../B/bao_cao_task_1.2_fatformer_model_and_weights_loader.md) - Lead B)**:
  - Checkpoint mỏ neo Teacher: [`fatformer_4class_ckpt.pth`](../../../DATASET/pretrained/fatformer_4class_ckpt.pth) (Đóng băng hoàn toàn tri thức ảnh sạch, `requires_grad=False`).
  - Backbone thị giác: [`ViT-L-14.pt`](../../../DATASET/pretrained/ViT-L-14.pt).
- **Từ Task 6.2 ([Báo cáo Task 6.2](../A/bao_cao_task_6.2_cross_domain_data_infra.md) - Lead A)**:
  - Tệp kê khai phân chia tập dữ liệu mô phỏng đối chứng: [`DATASET/manifests/data_manifest.json`](../../../DATASET/manifests/data_manifest.json).
  - Bộ lấy mẫu cân bằng xác suất: [`BalancedCrossDomainSampler`](../../../src/datasets/samplers.py).
- **Từ Task 6.3-B ([Báo cáo Task 6.3-B](../B/bao_cao_task_6.3_b_distillation_and_consistency_loss.md) - Lead B)**:
  - Hàm mất mát triệt tiêu phân kỳ: $\tau = 2.0$, trọng số $\alpha_{\text{kd}} = 0.4$.
  - Hàm mất mát đồng bộ biểu diễn: $\beta_{\text{cons}} = 0.2$.
  - Cấu hình điều hợp LoRA: rank $r = 8$, scale $\alpha = 16.0$, số khối áp dụng: 2 khối cuối.

---

### III. CẤU HÌNH THỰC THI THỰC TẾ

| Thông Số Điều Khiển | Giá Trị Thực Tế | Ý Nghĩa Kiểm Soát Kỹ Thuật |
| :--- | :--- | :--- |
| **Hạt giống ngẫu nhiên (Seed)** | `seed = 42` | Cố định ngẫu nhiên trong khởi tạo và chọn mẫu |
| **Bộ tối ưu hóa** | `AdamW(lr=1e-5, weight_decay=1e-4)` | Tốc độ học nhỏ bảo toàn đặc trưng nền tảng |
| **Lập lịch học tập** | `CosineAnnealingLR(T_max=5, eta_min=1e-6)` | Suy giảm độ dài bước học mịn màng qua từng epoch |
| **Kích thước lô (Batch Size)** | `batch_size = 8`, `grad_accum_steps = 4` | Kích thước lô hiệu dụng $8 \times 4 = 32$ mẫu/lần cập nhật |
| **Số chu kỳ (Epochs)** | `epochs = 5` | Chu kỳ ngắn kiểm định khả năng hội tụ |
| **Cơ chế đóng băng PEFT** | Đóng băng 97.34% tham số (563.03 M), mở khóa 2.66% (15.41 M) | Giữ vững trần độ chính xác Baseline ảnh sạch |

---

### IV. GHI NHẬN KẾT QUẢ KIỂM THỬ HẠ TẦNG (PIPELINE STAGING)

Số liệu trích xuất từ bản ghi thực tế [`DATASET/logs/m6_training_history.csv`](../../../DATASET/logs/m6_training_history.csv):

```
branch,epoch,train_loss,val_macro_acc,val_composite,time_sec
M3,1,1.8116,47.50,47.50,3.5
M3,2,1.8416,47.50,47.50,1.0
M3,3,1.5926,47.50,47.50,1.0
M3,4,1.4979,50.00,50.00,1.0
M3,5,1.6723,50.00,50.00,1.0
M4,1,1.8149,47.50,47.50,3.6
M4,2,1.6334,47.50,47.50,2.5
M4,3,1.5022,47.50,47.50,2.4
M4,4,1.8190,52.50,52.50,2.4
M4,5,1.4678,55.00,55.00,2.4
M5,1,1.8149,47.50,47.50,3.4
M5,2,1.6329,47.50,47.50,2.4
M5,3,1.5017,47.50,47.50,2.4
M5,4,1.8182,52.50,52.50,2.4
M5,5,1.4675,55.00,55.00,2.4
```

#### Bảng Tổng Hợp Kiểm Thử 3 Nhánh:

| Nhánh Huấn Luyện | Thành Phần Cấu Thành | Train Loss Cuối | Val Macro ACC | Val Composite | Thời Gian Chạy | Trọng Số Lưu Trữ |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **M3 (Data Only)** | Cross-Entropy + Dữ liệu đối chứng | 1.6723 | 50.00% | 50.00% | 46.21s | `m3_crossdomain_model_best.pth` (2,325.18 MB) |
| **M4 (Distill + Cons)** | M3 + Chưng cất Teacher + Ràng buộc Logits | 1.4678 | 55.00% | 55.00% | 45.73s | `m4_crossdomain_model_best.pth` (2,325.18 MB) |
| **M5 (LoRA Adapter)** | M4 + Bộ điều hợp LoRA ViT 2 khối cuối | 1.4675 | 55.00% | 55.00% | 42.57s | `m5_crossdomain_model_best.pth` (2,325.18 MB) |

---

### V. ĐÁNH GIÁ KỸ THUẬT & ĐÍNH CHÍNH MINH BẠCH

1. **Minh bạch hóa bản chất dữ liệu trong phiên chạy ban đầu**:
   - Tệp kê khai `data_manifest.json` trong phiên chạy ban đầu là bản mẫu mô phỏng (dummy template) với 60 mẫu train và 40 mẫu val. Do chưa đính kèm kho dữ liệu ảnh thật trên Kaggle, cơ chế dự phòng an toàn tại [src/training/train_crossdomain.py](file:///d:/GIT%20REPO/.nam4/fatformer-xla/src/training/train_crossdomain.py#L133-L134) đã sinh các mảng ảnh màu RGB trong RAM để duy trì luồng chạy.
   - Vì kích thước tập thẩm định chỉ có 40 mẫu giả lập, kết quả độ chính xác dao động quanh mức 47.5% – 55.0% (mức xác suất ngẫu nhiên). Nhóm kỹ thuật ghi nhận đây là kết quả kiểm thử tính thông suốt của pipeline, **không dùng để kết luận khoa học về việc mô hình đã vượt trội trên tập dữ liệu thực tế**.

2. **Chuẩn hóa thành công module LoRA trên backbone CLIP thật**:
   - Trong phiên chạy đầu, `ViTLoRAAdapter` chưa kích hoạt trọn vẹn do `nn.MultiheadAttention` của CLIP không có các tầng `q_proj`/`v_proj` độc lập.
   - Module đã được tái cấu trúc chuẩn xác: Chèn `LoRALinear` vào các tầng chiếu MLP (`c_fc` và `c_proj`) của 2 khối Transformer cuối.
   - **Kết quả kiểm thử trên mô hình thật**:
     - Số lượng tầng adapter: 4 tầng.
     - Số tham số có thể tối ưu của LoRA: **163,840 tham số**.
     - Cả 8 tham số của LoRA (gồm $A$ và $B$) đều nhận gradient thông suốt 100% qua các bước tối ưu.

3. **Khắc phục triệt để nguy cơ tràn đĩa cứng trên Kaggle**:
   - Đã loại bỏ các tệp trùng lặp `checkpoint_latest.pth`, chỉ lưu trữ 3 mô hình đại diện và tệp nhật ký CSV ra thư mục Output. Dung lượng đĩa được duy trì an toàn ở mức **~6.9 GB**, giải quyết triệt để lỗi tràn đĩa `[Errno 28]`.

---

### VI. TIÊU CHUẨN HOÀN THÀNH HẠ TẦNG (DEFINITION OF DONE)

| Tiêu Chuẩn Nghiệm Thu | Trạng Thái | Minh Chứng Kỹ Thuật |
| :--- | :---: | :--- |
| **Vượt qua Cổng Smoke Gate Milestone 6** | `[x] Đạt` | [Báo cáo Smoke Gate M6](bao_cao_smoke_gate_m6.md) pass 100% 6/6 tiêu chí. |
| **Thông suốt quy trình autograd 3 nhánh M3, M4, M5** | `[x] Đạt` | Cả 3 nhánh hoàn thành 5 epochs không phát sinh lỗi mã nguồn. |
| **Tuân thủ chính sách đĩa lưu trữ an toàn** | `[x] Đạt` | Không vượt quá giới hạn 20 GB của Kaggle, lưu Output an toàn. |
| **Kiểm chứng LoRA Adapter trên mô hình thật** | `[x] Đạt` | Đã kiểm thử unit test thành công, 163,840 tham số LoRA nhận gradient. |
| **Huấn luyện trên kho dữ liệu thực tế đầy đủ** | `[ ] Chờ thực hiện` | Đang chuẩn bị tệp manifest từ kho ảnh thật 3.600 ảnh (Bước 2). |

---

### VII. KẾ HOẠCH BÀN GIAO & BƯỚC TIẾP THEO

1. **Chuẩn bị kho dữ liệu thật (Lead A & C phối hợp)**: Trích xuất tập dữ liệu từ `diffusion_staging.tar` (3.600 ảnh) và `progan_train.tar` để tạo `data_manifest.json` chứa dữ liệu thực nghiệm thật sự.
2. **Huấn luyện chính thức quy mô đầy đủ**: Đưa mã nguồn đã chuẩn hóa và gói dữ liệu thật lên GPU Kaggle / Colab để hoàn thành các checkpoint ứng viên có giá trị khoa học cho Task 6.4 và Task 6.5.
