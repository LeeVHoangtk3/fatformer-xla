# BÁO CÁO NGHIỆM THU KỸ THUẬT: TASK 6.3-C
## CHIẾN DỊCH HUẤN LUYỆN THÍCH ỨNG ĐA MIỀN TRÊN MÁY CHỦ GPU CLOUD (MILESTONE 6)

> **Mã nhiệm vụ**: `Task 6.3-C` *(Milestone 6 — Tuần 6)*  
> **Người chủ trì**: **Thành viên C (Training & Evaluation Lead)**  
> **Người phối hợp**: **Thành viên A (Data & Infra Lead)**, **Thành viên B (Architecture & Loss Lead)**  
> **Thời gian hoàn thành**: 20/10/2026  
> **Môi trường thực thi**: 🚀 **Kaggle Server (NVIDIA RTX PRO 6000 Blackwell Server Edition — 94.97 GB VRAM, PyTorch 2.11.0+cu128)**  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH TOÀN DIỆN VÀ NGHIỆM THU THỰC TẾ TRÊN 100% ẢNH THẬT`  
> **Tài liệu nhiệm vụ gốc**: [`docs/tasks/member_c_training_eval/task_6.3_c_cross_domain_training_execution.md`](../../tasks/member_c_training_eval/task_6.3_c_cross_domain_training_execution.md)  
> **Kịch bản thực thi**: [`notebooks/task/task_6.3/task_6.3_c_train_crossdomain_kaggle.ipynb`](../../../notebooks/task/task_6.3/task_6.3_c_train_crossdomain_kaggle.ipynb) & [`src/training/train_crossdomain.py`](../../../src/training/train_crossdomain.py)

---

### I. MỤC TIÊU & BỐI CẢNH THỰC NGHIỆM

Sau khi hoàn thành chuẩn bị hạ tầng dữ liệu tại Task 6.2 và vượt qua [Cổng Kiểm Thử Tích Hợp Smoke Gate M6](bao_cao_smoke_gate_m6.md), **Task 6.3-C** triển khai chiến dịch huấn luyện thực tế trên máy chủ GPU hiệu năng cao nhằm xây dựng các mô hình ứng viên đa miền phục vụ ma trận nghiên cứu cắt bỏ thành phần (Ablation Study tại Task 6.5) và tuyển chọn mô hình tại Task 6.4.

Chiến dịch Task 6.3-C tập trung hoàn thành các mục tiêu khoa học:
1. Huấn luyện thực tế tuần tự 3 nhánh kiến trúc ứng viên trên 100% kho ảnh đối chứng (3.600 ảnh thật):
   - **M3 (Data Only)**: Mở rộng phân phối dữ liệu đa bộ sinh kết hợp Cross-Entropy Loss (label smoothing = 0.05).
   - **M4 (Data + Distill + Cons)**: Tích hợp hàm mất mát đa nhiệm triệt tiêu phân kỳ tri thức ($L_{\text{KD}}$) và đồng bộ biểu diễn không gian ($L_{\text{cons}}$) do Lead B thiết kế (Task 6.3-B).
   - **M5 (M4 + LoRA Adapter)**: Tích hợp bộ điều hợp tham số thấp LoRA (`ViTLoRAAdapter`) trực tiếp trên các tầng MLP của 2 block cuối CLIP ViT-L/14.
2. Kiểm chứng thực nghiệm tính ổn định của gradient, khả năng triệt tiêu hiện tượng quên tri thức ảnh sạch, và đo lường độ chính xác phân loại trên tập kiểm định đa miền.
3. Thu hồi và lưu trữ an toàn toàn bộ các bộ trọng số ứng viên và nhật ký hội tụ chuẩn mực phục vụ đánh giá downstream.

---

### II. KẾ THỪA THÔNG SỐ KỸ THUẬT (TUÂN THỦ NGUYÊN TẮC 4)

Báo cáo nghiệm thu kế thừa chuẩn xác các giá trị thông số đã được thẩm định từ các giai đoạn tiền đề:
- **Từ Task 4.3 ([Báo cáo Task 4.3](bao_cao_task_4.3_finetune_plan_a_kaggle_blackwell.md) - Lead C)**:
  - Checkpoint khởi tạo Student: [`fatformer_plan_a_best_weights.pth`](../../../DATASET/checkpoints/fatformer_plan_a_best_weights.pth) (2.207,50 MB, hội tụ đầy đủ SRM 3-Kernels và Dynamic Frequency Gating).
- **Từ Task 1.2 ([Báo cáo Task 1.2](../B/bao_cao_task_1.2_fatformer_model_and_weights_loader.md) - Lead B)**:
  - Checkpoint mỏ neo Teacher: [`fatformer_4class_ckpt.pth`](../../../DATASET/pretrained/fatformer_4class_ckpt.pth) (Đóng băng hoàn toàn tri thức ảnh sạch, `requires_grad=False`).
  - Backbone thị giác: [`ViT-L-14.pt`](../../../DATASET/pretrained/ViT-L-14.pt).
- **Từ Task 6.2 ([Báo cáo Task 6.2](../A/bao_cao_task_6.2_cross_domain_data_infra.md) - Lead A)**:
  - Kho dữ liệu ảnh đối chứng đa miền: `diffusion_staging.tar` gồm 3.600 ảnh (Stable Diffusion v1.5, Midjourney, StyleGAN, ProGAN, DALL-E).
  - Tệp kê khai phân chia tập dữ liệu: [`DATASET/manifests/data_manifest.json`](../../../DATASET/manifests/data_manifest.json) với 3.060 ảnh huấn luyện và 540 ảnh kiểm định (85/15 cân bằng Real/Fake).
  - Bộ lấy mẫu cân bằng xác suất: [`BalancedBatchSampler`](../../../src/datasets/samplers.py).
- **Từ Task 6.3-B ([Báo cáo Task 6.3-B](../B/bao_cao_task_6.3_b_distillation_and_consistency_loss.md) - Lead B)**:
  - Hàm mất mát triệt tiêu phân kỳ: $\tau = 2.0$, trọng số $\alpha_{\text{kd}} = 0.4$.
  - Hàm mất mát đồng bộ biểu diễn: $\beta_{\text{cons}} = 0.2$.
  - Cấu hình điều hợp LoRA: rank $r = 8$, scale $\alpha = 16.0$, số khối áp dụng: 2 khối Transformer cuối.

---

### III. CẤU HÌNH THỰC THI THỰC TẾ

| Thông Số Điều Khiển | Giá Trị Thực Tế | Ý Nghĩa Kiểm Soát Kỹ Thuật |
| :--- | :--- | :--- |
| **Phần cứng GPU** | NVIDIA RTX PRO 6000 Blackwell Server Edition | 94.97 GB VRAM, PyTorch 2.11.0+cu128 |
| **Hạt giống ngẫu nhiên (Seed)** | `seed = 42` | Cố định ngẫu nhiên trong khởi tạo và chọn mẫu |
| **Bộ tối ưu hóa** | `AdamW(lr=1e-5, weight_decay=1e-4)` | Tốc độ học nhỏ bảo toàn đặc trưng nền tảng |
| **Lập lịch học tập** | `CosineAnnealingLR(T_max=5, eta_min=1e-6)` | Suy giảm độ dài bước học mịn màng qua từng epoch |
| **Kích thước lô (Batch Size)** | `batch_size = 8`, `grad_accum_steps = 4` | Kích thước lô hiệu dụng $8 \times 4 = 32$ mẫu/lần cập nhật |
| **Số chu kỳ (Epochs)** | `epochs = 5` mỗi nhánh (Tổng cộng 15 epochs) | Chu kỳ chuẩn hóa thẩm định độ hội tụ |
| **Cơ chế đóng băng PEFT** | Đóng băng 97.31% – 97.34% tham số (563.03 M), mở khóa 2.66% – 2.69% | Giữ vững trần độ chính xác Baseline ảnh sạch |
| **Kỹ thuật tối ưu bộ nhớ** | Teacher FP16 Half Precision + `autocast` + CUDA Cache Purge | Triệt tiêu nguy cơ tràn VRAM |

---

### IV. GHI NHẬN KẾT QUẢ THỰC NGHIỆM TRÊN 100% ẢNH THẬT

Dữ liệu nghiệm thu trích xuất từ bản ghi thực tế [`DATASET/logs/m6_training_history.csv`](../../../DATASET/logs/m6_training_history.csv):

```csv
branch,epoch,train_loss,val_macro_acc,val_composite,time_sec
M3,1,0.2639,94.63,94.63,225.6
M3,2,0.2228,93.70,93.70,123.2
M3,3,0.2329,94.44,94.44,99.1
M3,4,0.2225,93.89,93.89,89.6
M3,5,0.2068,94.44,94.44,87.2
M4,1,0.6084,92.22,92.22,150.3
M4,2,0.4267,93.15,93.15,146.8
M4,3,0.3978,94.26,94.26,147.7
M4,4,0.4009,93.33,93.33,146.9
M4,5,0.3703,92.59,92.59,148.6
M5,1,0.6013,93.15,93.15,148.8
M5,2,0.4169,93.70,93.70,148.2
M5,3,0.3892,94.07,94.07,149.2
M5,4,0.3903,93.89,93.89,148.2
M5,5,0.3584,94.07,94.07,148.2
```

#### 1. Bảng Tổng Hợp So Sánh 3 Nhánh Ứng Viên:

| Nhánh Huấn Luyện | Thành Phần Cấu Thành | Train Loss Cuối | Val Composite Đỉnh | Val Composite Cuối | Thời Gian Chạy | Trọng Số Lưu Trữ Vượt Trội |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **M3 (Data Only)** | Cross-Entropy + Dữ liệu đối chứng | **0.2068** | **94.63%** *(Epoch 1)* | **94.44%** | 655.51s *(10.9 phút)* | `m3/crossdomain_model_best.pth` *(2.44 GB)* |
| **M4 (Distill + Cons)** | M3 + Chưng cất Teacher + Ràng buộc Logits | **0.3703** | **94.26%** *(Epoch 3)* | **92.59%** | 788.47s *(13.1 phút)* | `m4/crossdomain_model_best.pth` *(2.44 GB)* |
| **M5 (LoRA Adapter)** | M4 + Bộ điều hợp LoRA ViT 2 khối cuối | **0.3584** | **94.07%** *(Epoch 3 & 5)* | **94.07%** | 758.61s *(12.6 phút)* | `m5/crossdomain_model_best.pth` *(2.44 GB)* |

#### 2. Biểu Đồ Trực Quan Hóa Tiến Trình Hội Tụ:

![Đồ thị hội tụ huấn luyện Milestone 6](m6_training_convergence.png)

---

### V. PHÂN TÍCH & ĐÁNH GIÁ CHUYÊN SÂU

1. **Hiệu quả phân loại trên tập dữ liệu đối chứng**:
   - Khác với phiên chạy kiểm thử ban đầu (với dữ liệu mẫu mô phỏng 60 ảnh cho độ chính xác quanh 50%), phiên chạy trên **3.600 ảnh đối chứng thực tế** cho thấy mô hình thích ứng tốt trên phân phối 5 bộ sinh đã nạp (Stable Diffusion v1.5, Midjourney, StyleGAN, ProGAN, DALL-E) với độ chính xác thẩm định định kỳ đạt quanh **94.0% – 94.6%**.
   - Việc bổ sung dữ liệu sinh từ các bộ khuếch tán giúp mô hình mở rộng trường nhận diện sang các dạng giả mạo thế hệ mới so với mốc xuất phát ban đầu (73.25% trên `val_crossdomain`).

2. **Động lực hội tụ của hàm mất mát Chưng cất & Đồng bộ (M4 & M5)**:
   - Ở nhánh M4 và M5, giá trị hàm mất mát tổng hợp ban đầu ghi nhận cao hơn nhánh M3 ($0.6084$ và $0.6013$ so với $0.2639$) do gánh thêm 2 số hạng điều chuẩn: Hàm mất mát chưng cất tri thức ($L_{\text{KD}}$) và hàm mất mát đồng bộ biểu diễn trên cặp ảnh sạch - suy biến ($L_{\text{cons}}$).
   - Qua 5 chu kỳ huấn luyện, hàm mất mát giảm đều đặn về $0.3703$ (M4) và $0.3584$ (M5), với độ chính xác thẩm định định kỳ đạt quanh mức **94.07% – 94.26%**. Quá trình tối ưu diễn ra ổn định, không xuất hiện hiện tượng bùng nổ gradient hay phân kỳ số học.

3. **Vận hành thực tế của bộ điều hợp LoRA ViT (M5)**:
   - Module `ViTLoRAAdapter` với 163.840 tham số bổ sung tại 4 tầng chiếu MLP của 2 khối Transformer cuối CLIP ViT-L/14 hoạt động trơn tru cùng cơ chế PEFT.
   - Dung lượng lưu trữ của checkpoint M5 ghi nhận **2.440.110.477 bytes** (tăng thêm đúng 1.97 MB tương ứng với các ma trận trọng số $A$ và $B$ của LoRA so với M3/M4). Độ chính xác thẩm định tại Epoch 3 và Epoch 5 duy trì ở mức **94.07%**.

4. **Hạn Chế Thực Nghiệm & Cảnh Báo Khoa Học (Experimental Limitations & Nuances)**:
   - **Biên độ sai số thống kê của tập thẩm định (Sample Size & Noise)**: Tập kiểm định định kỳ gồm 540 ảnh (270 ảnh thật, 270 ảnh giả). Với cỡ mẫu này, khoảng tin cậy 95% có biên độ sai số vào khoảng $\pm 1.0\%$. Mức chênh lệch điểm số giữa M3 (94.63%), M4 (94.26%) và M5 (94.07%) chỉ tương ứng với sự khác biệt từ 2 đến 3 mẫu ảnh trong tổng số 540 ảnh. Do đó, các sai khác nhỏ này chưa đủ cơ sở để khẳng định sự vượt trội dứt khoát giữa các nhánh kiến trúc khi chỉ dựa vào điểm thẩm định định kỳ này.
   - **Bản chất tập kiểm định (In-Distribution Validation)**: Tập kiểm định 540 ảnh được trích xuất ngẫu nhiên theo tỷ lệ 85/15 từ cùng một kho 3.600 ảnh đối chứng thuộc 5 bộ sinh. Vì vậy, đây là phép đo nội phân phối (in-distribution) nhằm kiểm tra khả năng khớp dữ liệu đối chứng, chưa phải là bài kiểm tra tổng quát hóa trên các bộ sinh hoàn toàn xa lạ (unseen out-of-distribution).
   - **Hiện tượng chọn điểm lưu trữ checkpoint (Best Checkpoint Selection)**: Nhánh M3 đạt điểm đỉnh 94.63% tại Epoch 1 và giảm nhẹ về 94.44% ở Epoch 5. Việc điểm số đạt mốc cao ngay từ Epoch 1 có thể chứa đựng yếu tố ngẫu nhiên cục bộ của quá trình lấy mẫu.
   - **Chưa phản ánh toàn diện các tiêu chí của Gate 4 điều kiện**: Vòng lặp thẩm định định kỳ của Task 6.3 đo đạc độ chính xác phân loại cơ bản (`val_macro_acc`), chưa đo mức độ duy trì tri thức ảnh gốc (Clean ACC, Clean Fake ACC) và khả năng chịu nén JPEG sâu (Q30 Macro ACC, Q30 Fake ACC) cũng như độ tin cậy xác suất (ECE).
   - **Quy mô phiên chạy đơn lẻ (Single Seed Run)**: Chiến dịch thực hiện với `seed = 42`. Do đó, vai trò độc lập của từng thành phần (Dữ liệu đối chứng, $L_{\text{KD}}$, $L_{\text{cons}}$, LoRA) sẽ được bóc tách định lượng thận trọng tại Task 6.5 thay vì vội vàng kết luận tại giai đoạn này.

5. **Phân Định Trách Nhiệm Thẩm Định Độc Lập Sang Task 6.4**:
   - Toàn bộ 5 mô hình ứng viên (M1 Plan A, M2 Baseline 4-class, M3 Data Only, M4 Distill+Cons, M5 LoRA) sẽ được đưa vào quy trình kiểm thử và sàng lọc khách quan tại **Task 6.4 (Calibration & Model Selection)** thông qua công cụ `tools/calibrate_and_select.py`.
   - Quyết định tuyển chọn mô hình đại diện chính thức sẽ căn cứ vào việc vượt qua Cổng Gate 4 điều kiện khắt khe (bao gồm cả ảnh sạch lẫn ảnh nén JPEG Q=30) cùng điểm ECE sau khi hiệu chuẩn nhiệt độ, bảo đảm tính công tâm và minh bạch thực nghiệm.

---

### VI. TIÊU CHUẨN HOÀN THÀNH NHIỆM VỤ (DEFINITION OF DONE)

| Tiêu Chuẩn Nghiệm Thu | Trạng Thái | Minh Chứng Kỹ Thuật |
| :--- | :---: | :--- |
| **Vượt qua Cổng Smoke Gate Milestone 6** | `[x] Đạt` | [Báo cáo Smoke Gate M6](bao_cao_smoke_gate_m6.md) pass 100% 6/6 tiêu chí. |
| **Huấn luyện thực tế trên kho dữ liệu 3.600 ảnh** | `[x] Đạt` | Sử dụng 3.060 ảnh train / 540 ảnh val từ `diffusion_staging`. |
| **Thông suốt quy trình autograd 3 nhánh M3, M4, M5** | `[x] Đạt` | Cả 3 nhánh hoàn thành 5 epochs không phát sinh lỗi mã nguồn. |
| **Lưu trữ đầy đủ các checkpoint ứng viên** | `[x] Đạt` | Cả 3 tệp `crossdomain_model_best.pth` (2.44 GB mỗi tệp) đã kiểm tra toàn vẹn. |
| **Xuất nhật ký CSV và biểu đồ trực quan hóa** | `[x] Đạt` | File `m6_training_history.csv` và ảnh `m6_training_convergence.png` đầy đủ. |
| **Không vi phạm ràng buộc bảo vệ 18 tập test Task 5.1** | `[x] Đạt` | 18 tập test benchmark hoàn toàn bị cô lập, không rò rỉ dữ liệu. |

---

### VII. KẾ HOẠCH BÀN GIAO & TRIỂN KHAI TIẾP THEO

Toàn bộ tài nguyên của Task 6.3-C đã sẵn sàng chuyển giao cho các nhiệm vụ tiếp theo trong lộ trình Milestone 6:
1. **Chuyển giao cho Task 6.4 (Hiệu chuẩn nhiệt độ & Tuyển chọn mô hình)**:
   - Cung cấp 3 ứng viên `M3`, `M4`, `M5` cùng 2 mỏ neo `M1` (Plan A) và `M2` (Baseline 4-class) để chạy script tuyển chọn [`tools/calibrate_and_select.py`](../../../tools/calibrate_and_select.py) trên tập kiểm định đa miền theo Gate 4 điều kiện.
2. **Chuyển giao cho Task 6.5 (Ablation Study M1–M5)**:
   - Sử dụng các checkpoint và nhật ký đã hoàn thiện để lập bảng ma trận đối chứng cắt bỏ thành phần, phân tích định lượng đóng góp của từng module (Data, Distillation, Consistency, LoRA).
