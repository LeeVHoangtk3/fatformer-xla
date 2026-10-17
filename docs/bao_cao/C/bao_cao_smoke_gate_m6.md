# BÁO CÁO NGHIỆM THU KỸ THUẬT: SMOKE GATE MILESTONE 6
## CỔNG KIỂM THỬ TÍCH HỢP TOÀN DIỆN KHỚP NỐI ĐA MIỀN (CROSS-DOMAIN INTEGRATION SMOKE GATE)

> **Mã mốc nhiệm vụ**: `Smoke Gate M6` *(Milestone 6)*  
> **Người chủ trì**: **Thành viên C (Training & Evaluation Lead)**  
> **Người phối hợp**: **Thành viên B (Architecture Lead)**, **Thành viên A (Data & Infra Lead)**  
> **Thời gian hoàn thành**: 17/10/2026  
> **Môi trường thực thi & nghiệm thu**: CPU Local & GPU Cloud  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC (PASS 100% 6/6 TIÊU CHÍ DoD)`  
> **Quyết định điểm dừng (Gate Verdict)**: **PHÊ DUYỆT CHÍNH THỨC CẤP PHÁT GPU CHO CHIẾN DỊCH HUẤN LUYỆN TASK 6.3-C**  
> **Công cụ kiểm thử**: [`tools/test_smoke_m6.py`](../../../tools/test_smoke_m6.py)

---

### I. MỤC TIÊU & TIÊU CHUẨN ĐIỂM DỪNG (SMOKE GATE CRITERIA)

Theo lộ trình tại [Kế Hoạch Các Cột Mốc & Điểm Dừng](../../plan/ke_hoach_cac_cot_moc_va_diem_dung.md) và [Lộ Trình Milestone 6](../C/milestone_6_cross_domain_hardening_roadmap.md), **Smoke Gate M6** là chốt chặn kỹ thuật bắt buộc trước khi cấp phát tài nguyên GPU lớn cho Task 6.3-C.

Mục tiêu cốt lõi:
1. **Khớp nối liên hoàn 3 tầng kỹ thuật**:
   - **Tầng Dữ liệu (Lead A - Task 6.2)**: Module [`BalancedCrossDomainSampler`](../../../src/datasets/samplers.py) cân bằng 50/50 Real/Fake và chia đều xác suất giữa các bộ sinh.
   - **Tầng Biến đổi & Kiến trúc (Lead B - Task 6.3-B)**: Module biến đổi cặp ảnh [`PairedCleanDegradedAugmentation`](../../../src/datasets/transforms.py), bộ điều hợp LoRA [`ViTLoRAAdapter`](../../../src/models/lora_adapter.py) và mô hình Teacher mỏ neo đóng băng.
   - **Tầng Tổn thất Đa nhiệm (Lead B & C)**: Hàm mục tiêu [`CrossDomainTotalLoss`](../../../src/training/loss.py) kết hợp Cross-Entropy, triệt tiêu phân kỳ KL Distillation ($\alpha_{\text{kd}}=0,4$) và ràng buộc đồng bộ phân phối Logits Consistency ($\beta_{\text{cons}}=0,2$).
2. **Tiêu chuẩn Định lượng Hoàn thành (DoD Checklist)**:
   - [x] Tiêu chí 1: `PairedCleanDegradedAugmentation` sinh cặp tensor chuẩn $(B, 3, 224, 224)$.
   - [x] Tiêu chí 2: Teacher model ở chế độ `eval()`, toàn bộ tham số `requires_grad=False`.
   - [x] Tiêu chí 3: `CrossDomainDistillationLoss` ($L_{\text{KD}}$) hữu hạn, không xuất hiện NaN/Inf.
   - [x] Tiêu chí 4: `PairedConsistencyLoss` ($L_{\text{cons}}$) hữu hạn, không xuất hiện NaN/Inf.
   - [x] Tiêu chí 5: `CrossDomainTotalLoss` thực hiện Forward + Backward thông suốt, gradient truyền qua student, teacher giữ nguyên không có gradient.
   - [x] Tiêu chí 6: `BalancedCrossDomainSampler` tích hợp mượt mà với `DataLoader`, nạp batch cân bằng và chạy autograd 1 batch thành công.
   - [x] Tiêu chí BONUS: Khối `LoRALinear` có $\Delta W = 0$ tại thời điểm khởi tạo khi ma trận $B=0$.

---

### II. KẾ THỪA THÔNG SỐ KỸ THUẬT (TUÂN THỦ NGUYÊN TẮC 4)

Báo cáo nghiệm thu kế thừa đầy đủ các kết quả đã được xác thực:
- **Từ Task 6.2 ([Báo cáo Task 6.2](../A/bao_cao_task_6.2_cross_domain_data_infra.md) - Lead A)**:
  - Bộ lấy mẫu `BalancedCrossDomainSampler` với trọng số nghịch đảo tần suất $w_g = \frac{1}{N_g}$.
  - Tệp kê khai nguồn gốc [`DATASET/manifests/data_manifest.json`](../../../DATASET/manifests/data_manifest.json).
  - Bảng mốc sàn tham chiếu [`DATASET/logs/val_crossdomain_baseline_metrics.json`](../../../DATASET/logs/val_crossdomain_baseline_metrics.json).
- **Từ Task 6.3-B ([Báo cáo Task 6.3-B](../B/bao_cao_task_6.3_b_distillation_and_consistency_loss.md) - Lead B)**:
  - Hàm tổn thất triệt tiêu phân kỳ KL: $\tau = 2,0$, $\alpha_{\text{kd}} = 0,4$.
  - Hàm tổn thất đồng bộ logits: $\beta_{\text{cons}} = 0,2$.
  - Cấu hình điều hợp LoRA: rank $r=8$, scale $\alpha=16,0$.
- **Từ Task 1.2 ([Báo cáo Task 1.2](../B/bao_cao_task_1.2_fatformer_model_and_weights_loader.md) - Lead B)**:
  - Checkpoint baseline gốc `fatformer_4class_ckpt.pth` làm mỏ neo Teacher đóng băng tri thức.

---

### III. KẾT QUẢ THỰC NGHIỆM ĐO ĐẠC THỰC TẾ

Toàn bộ chu trình kiểm thử được thực thi tự động qua lệnh:
```bash
python tools/test_smoke_m6.py --device cpu
```

#### Bảng Tổng Hợp Kết Quả 6 Tiêu Chí:

| # | Tiêu Chí Kiểm Tra | Giá Trị Thực Tế Đo Được | Ngưỡng Yêu Cầu | Đánh Giá |
| :-: | :--- | :--- | :--- | :---: |
| **1** | **Paired Augmentation Shape** | $x_{\text{clean}}=(8, 3, 224, 224)$, $x_{\text{deg}}=(8, 3, 224, 224)$ | Khớp kích thước ảnh đầu vào | **PASS** |
| **2** | **Teacher Freeze Status** | $\text{trainable}=0$, $\text{frozen}=301.058$, $\text{eval}=\text{True}$ | Đóng băng 100% trọng số | **PASS** |
| **3** | **Distillation Loss ($L_{\text{KD}}$)** | $L_{\text{KD}} = 0,009487$ ($\tau=2,0, \text{clean\_mask}=4/8$) | Hữu hạn, không NaN/Inf | **PASS** |
| **4** | **Consistency Loss ($L_{\text{cons}}$)** | $L_{\text{cons}} = 0,000015$ ($\beta=0,2$) | Hữu hạn, không NaN/Inf | **PASS** |
| **5** | **Total Loss & Autograd** | $L_{\text{total}} = 0,8148$ ($L_{\text{ce}}=0,8053, L_{\text{kd}}=0,0095$)<br>Gradient Student $> 0$, Teacher Gradient = None | Không có lỗi autograd | **PASS** |
| **6** | **Sampler & DataLoader Hook** | Lấy mẫu mini-batch 8 mẫu cân bằng thành công<br>$L_{\text{total}} = 0,8016$, Backward thông suốt | Tích hợp không lỗi | **PASS** |
| **+** | **LoRA Init Contract** | $\Delta W = 0$ tại $B=0$ ($\text{allclose}=\text{True}$)<br>Tham số: $\text{lora\_A}=8.192, \text{lora\_B}=8.192$ | Giữ nguyên trọng số gốc | **PASS** |

- **Tổng thời gian thực thi**: **0,15 giây** (Cực nhanh, an toàn tài nguyên).
- **Trạng thái cổng**: **PASS 100% (6/6 Tiêu chí + 1 Tiêu chí BONUS LoRA)**.

---

### IV. KẾT LUẬN & PHÊ DUYỆT BÀN GIAO CHO TASK 6.3-C

1. Toàn bộ hạ tầng dữ liệu của Thành viên A và khối kiến trúc tổn thất của Thành viên B đã ăn khớp hoàn chỉnh, không có bất kỳ xung đột nào về kích thước tensor hay dòng chảy gradient.
2. Cổng kiểm thử **Smoke Gate M6** chính thức được phê duyệt thông qua.
3. Dự án đủ điều kiện an toàn 100% để chuyển giao sang **Task 6.3-C: Huấn luyện thích ứng chống lệch phân phối trên GPU** (do Thành viên C chủ trì).
