# TASK 6.3-B: THIẾT KẾ HÀM MẤT MÁT DISTILLATION ĐA MIỀN & TĂNG CƯỜNG DỮ LIỆU GHÉP CẶP

- **Mã nhiệm vụ**: `Task 6.3-B` *(Milestone 6 - Tuần 6)*
- **Người thực hiện**: **Thành viên B (Architecture & Loss Lead)**
- **Người phối hợp**: **Thành viên C (Training Lead)**, **Thành viên A (Data Lead)**
- **Môi trường thực thi**: 🖥️ **Local / Cloud**
- **Độ ưu tiên**: 🔴 Khẩn cấp (Hợp phần kiến trúc & loss cho Task 6.3)
- **Trạng thái**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC` ([Báo cáo](../../bao_cao/B/bao_cao_task_6.3_b_distillation_and_consistency_loss.md))

---

## 1. Mục Tiêu & Cơ Sở Khoa Học

Tại đợt finetune Plan A, hàm mất mát Cross-Entropy tiêu chuẩn giúp tối ưu tốt trên miền dữ liệu ProGAN nhưng chưa có cơ chế giữ neo tri thức tổng quát vốn có của checkpoint gốc, dẫn đến hiện tượng trôi dạt biểu diễn (representation drift) trên các bộ sinh chưa thấy.

Mục tiêu của Task 6.3-B:
1. Thiết kế hàm mất mát chưng cất tri thức `CrossDomainDistillationLoss` dùng checkpoint baseline gốc làm mô hình mẫu (teacher) trên nhánh ảnh sạch nhằm bảo vệ độ chính xác ảnh sạch.
2. Xây dựng pipeline tăng cường dữ liệu ghép cặp `PairedCleanDegradedAugmentation` và hàm mất mát đồng bộ `PairedConsistencyLoss` ràng buộc tính ổn định của logit giữa ảnh sạch và ảnh bị nén suy biến.
3. Chuẩn bị module LoRA thích ứng có kiểm soát (Low-Rank Adaptation, $r=8, \alpha=16$) trên các khối Transformer cuối của backbone ViT-L/14 để phục vụ nhánh thử nghiệm M5 tại Task 6.5.
4. Thiết lập kịch bản kiểm thử tích hợp (Smoke Gate M6) phối hợp với Lead C trước khi bước vào huấn luyện lớn.

---

## 2. Thiết Kế Kỹ Thuật Chi Tiết

### 2.1. Hàm Chưng Cất Tri Thức Triệt Tiêu Phân Kỳ KL (`CrossDomainDistillationLoss`)
- Trên các mẫu ảnh sạch ($x \in \mathcal{D}_{clean}$), tính toán hàm mất mát chưng cất tri thức theo phân kỳ Kullback-Leibler:
  $$\mathcal{L}_{KD} = \tau^2 \cdot D_{KL}\left(\text{Softmax}\left(\frac{z_{student}}{\tau}\right) \;\parallel\; \text{Softmax}\left(\frac{z_{teacher}}{\tau}\right)\right)$$
- Trong đó:
  - $z_{student}$ là logit dự đoán của mô hình đang huấn luyện.
  - $z_{teacher}$ là logit từ checkpoint baseline gốc đã đóng băng (không cập nhật gradient).
  - Nhiệt độ làm mịn phân phối: $\tau = 2.0$.
  - Trọng số hàm mất mát: $\alpha_{KD} \in [0.3, 0.5]$.

### 2.2. Pipeline Tăng Cường Ghép Cặp & Hàm Mất Mát Đồng Bộ (`PairedConsistencyLoss`)
- Xây dựng `PairedCleanDegradedAugmentation` trong `src/datasets/transforms.py`:
  - Mỗi mẫu ảnh $x$ được chuyển thành cặp $(x_{clean}, x_{degraded})$ với cùng nhãn nhãn phân loại.
  - $x_{clean}$: chỉ áp dụng chuẩn hóa và resize cơ bản.
  - $x_{degraded}$: áp dụng nén JPEG ngẫu nhiên ($Q \in [30, 70]$) hoặc Gaussian Blur ($\sigma \in [1.0, 2.0]$).
- Ràng buộc tương thích logit giữa hai nhánh:
  $$\mathcal{L}_{cons} = \frac{1}{2} \| z(x_{clean}) - z(x_{degraded}) \|_2^2$$
- Giúp mô hình bảo toàn quyết định phân loại dù cấu trúc vi mô bị biến dạng bởi ma trận lượng tử hóa khối $8 \times 8$.

### 2.3. Hàm Mất Mát Toàn Phần Đa Mục Tiêu
$$\mathcal{L}_{total} = \mathcal{L}_{CE}(y, \hat{y}) + \alpha_{KD} \cdot \mathcal{L}_{KD} + \beta_{cons} \cdot \mathcal{L}_{cons}$$
- Trong đó $\mathcal{L}_{CE}$ áp dụng kỹ thuật làm mịn nhãn (`label_smoothing=0.05` đến `0.10`) nhằm tránh cực đoan hóa logit.

### 2.4. Module LoRA Có Kiểm Soát Cho ViT Backbone
- Bổ sung adapter tham số thấp vào các ma trận chiếu $W_q, W_v$ của 2 khối Transformer cuối cùng của ViT-L/14:
  $$W' = W + \frac{\alpha}{r} (B \cdot A), \quad A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{d \times r}, r=8$$
- Cơ chế này chỉ kích hoạt cho nhánh M5 trong Task 6.5 nhằm đo lường khả năng mở rộng tham số ViT mà không gây phá vỡ đặc trưng gốc.

---

## 3. Cổng Kiểm Thử Tích Hợp (Smoke Gate Milestone 6)

Trước khi chuyển giao mã nguồn sang Task 6.3-C để chạy huấn luyện GPU quy mô lớn, Lead B phối hợp với Lead C thực thi script `tools/test_smoke_m6.py` trên 1 batch dữ liệu để xác nhận 5 tiêu chí:
1. `PairedCleanDegradedAugmentation` trả về đúng cặp tensor kích thước chuẩn $(B, 3, 224, 224)$.
2. Teacher model nạp checkpoint gốc ở chế độ `eval()`, toàn bộ tham số có `requires_grad = False`.
3. `BalancedCrossDomainSampler` cấp phát đủ đại diện các bộ sinh trong batch.
4. Forward và backward pass diễn ra ổn định, loss tổng hợp hữu hạn, không có giá trị NaN hoặc Inf.
5. Kiểm tra lưu và nạp trạng thái optimizer / model không bị lệch tensor shape.

---

## 4. Tiêu Chuẩn Nghiệm Thu (Definition of Done)

- [x] Module `src/training/loss.py` hoàn thiện đầy đủ `CrossDomainDistillationLoss` và `PairedConsistencyLoss`.
- [x] Module `src/datasets/transforms.py` cập nhật pipeline `PairedCleanDegradedAugmentation`.
- [x] Module LoRA được tích hợp trong `src/models/lora_adapter.py` với cờ cấu hình bật/tắt minh bạch.
- [x] Vượt qua Cổng Smoke Gate Milestone 6 (100% pass trên 1 mini-batch trong 0.40s).
- [x] Biên soạn báo cáo nghiệm thu tại [`docs/bao_cao/B/bao_cao_task_6.3_b_distillation_and_consistency_loss.md`](../../bao_cao/B/bao_cao_task_6.3_b_distillation_and_consistency_loss.md).
