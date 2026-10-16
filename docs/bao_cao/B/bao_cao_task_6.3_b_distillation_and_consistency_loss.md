# BÁO CÁO NGHIỆM THU NHIỆM VỤ 6.3-B (TASK 6.3-B)
## THIẾT KẾ HÀM MẤT MÁT DISTILLATION ĐA MIỀN & TĂNG CƯỜNG DỮ LIỆU GHÉP CẶP

> **Người thực hiện**: **Thành viên B** *(Architecture & Loss Lead)*  
> **Người phối hợp**: **Thành viên C** *(Training Lead — sử dụng tại Task 6.3-C)*  
> **Ngày hoàn thành**: 2026-10-16  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC (PASS 100% TIÊU CHUẨN DoD)`  
> **Milestone**: Milestone 6 — Cross-Domain Robustness Hardening  

---

## I. MỤC TIÊU & CƠ SỞ KHOA HỌC

**Vấn đề tiếp nhận từ Task 6.1** (Phân tích phân kỳ phân phối):  
Mô hình FatFormer-XLA sau finetune Plan A trên ProGAN đạt Acc phòng lab cao nhưng xuất hiện hiện tượng **representation drift** — sụt giảm độ chính xác trên các generator chưa thấy (SD v1.5, Midjourney v5) và ảnh bị nén JPEG Q=30 thực tế.

**Giải pháp Task 6.3-B triển khai 3 cơ chế bảo vệ**:

1. **Chưng cất tri thức có chọn lọc** (`CrossDomainDistillationLoss`): Dùng checkpoint baseline gốc CVPR 2024 làm neo teacher trên batch ảnh sạch, ngăn logit student trôi dạt xa khỏi vùng quyết định tổng quát ban đầu.

2. **Ràng buộc đồng bộ logit** (`PairedConsistencyLoss`): Ép mô hình duy trì quyết định phân loại nhất quán giữa cùng một ảnh ở dạng sạch và dạng bị nén/blur, giải quyết đặc trưng bị phá hủy bởi ma trận lượng tử hóa khối 8×8.

3. **Adapter tham số thấp** (`ViTLoRAAdapter`): Module LoRA $r=8, \alpha=16$ để thử nghiệm mở rộng ViT-L/14 tại nhánh M5 (Task 6.5) mà không phá vỡ đặc trưng đóng băng.

---

## II. KẾ THỪA THÔNG SỐ TỪ CÁC TASK TIỀN ĐỀ (RULE 4)

| Thông số | Giá trị thực tế | Nguồn kế thừa |
|:---|:---|:---|
| Teacher model | Checkpoint gốc CVPR 2024 (`fatformer_4class_ckpt.pth`) | [Task 1.2](../../tasks/member_b_architecture_loss/task_1.2_fatformer_model_and_weights_loader.md) |
| Temperature KD | τ = 2.0 | Task 6.3-B spec |
| α_KD | 0.4 | Task 6.3-B spec |
| β_cons | 0.2 | Task 6.3-B spec |
| JPEG range suy biến | Q ∈ [30, 70] | [Báo cáo Task 6.1](../B/bao_cao_task_6.1_distribution_shift_analysis.md) (nén thực tế mạng xã hội) |
| Gaussian Blur σ | [1.0, 2.0] | Task 6.3-B spec |
| Label Smoothing | 0.05 | Phòng chống over-confidence M6 |
| CLIP input size | (3, 224, 224) | [Báo cáo Task 3.3](bao_cao_task_3.3_srm_gating_focal_loss.md) |
| LoRA rank r | 8 | Task 6.3-B spec (Hu et al., 2021) |
| LoRA alpha α | 16.0 (scaling = 2.0) | Task 6.3-B spec |

---

## III. ĐẦU RA KỸ THUẬT ĐÃ HOÀN THIỆN

### 3.1. Module `src/training/loss.py` — 3 Class mới (backward-compatible)

| Class | Công thức | Mục đích |
|:---|:---|:---|
| `CrossDomainDistillationLoss` | $\mathcal{L}_{KD} = \tau^2 \cdot D_{KL}(\text{Softmax}(z_s/\tau) \| \text{Softmax}(z_t/\tau))$ | Neo logit student vào teacher trên ảnh sạch |
| `PairedConsistencyLoss` | $\mathcal{L}_{cons} = \frac{\beta}{2} \cdot \|z_{clean} - z_{deg}\|_2^2$ | Đồng bộ quyết định clean↔degraded |
| `CrossDomainTotalLoss` | $\mathcal{L}_{total} = \mathcal{L}_{CE} + \alpha_{KD}\mathcal{L}_{KD} + \beta_{cons}\mathcal{L}_{cons}$ | Interface tổng hợp cho Lead C |

**Thiết kế an toàn**:
- `CrossDomainDistillationLoss.forward()` gọi `.detach()` trên `teacher_logits` tự động — không rủi ro gradient chảy ngược vào teacher.
- Xử lý `clean_mask=None` → áp dụng toàn batch; `clean_mask` toàn False → trả về `0.0` tensor, không crash.
- `CrossDomainTotalLoss.forward()` trả về `dict` thay vì scalar → Lead C có thể log từng thành phần riêng.

### 3.2. Module `src/datasets/transforms.py` — `PairedCleanDegradedAugmentation`

Pipeline xuất cặp `(x_clean, x_degraded)` cùng kích thước `(3, 224, 224)`:
- **x_clean**: Resize(256) → CenterCrop(224) → ToTensor → Normalize
- **x_degraded**: Resize(256) → CenterCrop(224) → Suy biến ngẫu nhiên → ToTensor → Normalize

Phân phối suy biến: JPEG Q∈[30,70] (50%) | Gaussian Blur σ∈[1.0,2.0] (30%) | Down-Up 0.5× (20%)

### 3.3. Module `src/models/lora_adapter.py` *(file mới)*

| Class | Mô tả |
|:---|:---|
| `LoRALinear` | Thay thế `nn.Linear` với delta $\Delta W = \frac{\alpha}{r} BA$. Khởi tạo B=0 → delta=0 ban đầu |
| `ViTLoRAAdapter` | Bộ điều khiển chèn LoRALinear vào $W_q, W_v$ của N khối cuối ViT |

**Cờ bật/tắt**: `enabled=False` (mặc định) → không ảnh hưởng pipeline M4/M6 đang chạy.  
Kích hoạt tại Task 6.5: `config = {"enabled": True, "rank": 8, "alpha": 16, "num_layers": 2}`

### 3.4. Script `tools/test_smoke_m6.py`

Kiểm thử độc lập 5 tiêu chí, chạy **CPU-only** không cần checkpoint thực:
```
python tools/test_smoke_m6.py --device cpu --batch_size 8
```

---

## IV. KẾT QUẢ SMOKE GATE MILESTONE 6

Script `tools/test_smoke_m6.py` thực thi trên CPU local, batch_size=8:

| # | Tiêu chí DoD | Kết quả | Ghi chú (số liệu thực tế) |
|:---:|:---|:---:|:---|
| 1 | `PairedCleanDegradedAugmentation` → tensor (8, 3, 224, 224) hợp lệ | ✅ **PASS** | `x_clean=(8,3,224,224)`, `x_degraded=(8,3,224,224)`, không NaN |
| 2 | Teacher model: `eval()` & `requires_grad=False` hoàn toàn | ✅ **PASS** | `trainable_params=0`, `frozen_params=301,058` |
| 3 | `CrossDomainDistillationLoss` → $L_{KD}$ hữu hạn, không NaN/Inf | ✅ **PASS** | `L_KD = 0.000302` (τ=2.0, α_kd=0.4, clean_mask=4/8 mẫu) |
| 4 | `PairedConsistencyLoss` → $L_{cons}$ hữu hạn, không NaN/Inf | ✅ **PASS** | `L_cons = 0.000013` (β_cons=0.2, shape=(8, 2)) |
| 5 | `CrossDomainTotalLoss` forward+backward, gradient thông suốt, teacher frozen | ✅ **PASS** | `loss_total=0.6921`, `loss_ce=0.6917`, `loss_kd=0.0003`, `loss_cons=0.0000`; Teacher grad=None ✓ |
| BONUS | `LoRALinear`: delta=0 tại init (B=0 contract), lora_A=8192, lora_B=8192 params | ✅ **PASS** | `scaling = α/r = 16/8 = 2.0` ✓ |

> **Thời gian thực thi thực tế**: ⚡ **0.13 giây** trên CPU local (batch_size=8, Python 3.14)  
> **Kết luận Smoke Gate**: ✅ **PASS 100% — 5/5 tiêu chí + BONUS — Đủ điều kiện chuyển giao Task 6.3-C**


---

## V. ĐỐI SOÁT TIÊU CHUẨN HOÀN THÀNH (DoD CHECKLIST)

- [x] Module `src/training/loss.py` hoàn thiện đầy đủ `CrossDomainDistillationLoss` và `PairedConsistencyLoss`.
- [x] Module `src/datasets/transforms.py` cập nhật pipeline `PairedCleanDegradedAugmentation`.
- [x] Module LoRA tích hợp tại `src/models/lora_adapter.py` với cờ `enabled` bật/tắt minh bạch.
- [x] Vượt qua Cổng Smoke Gate Milestone 6 (PASS 100% trên 1 mini-batch CPU).
- [x] Báo cáo nghiệm thu tại `docs/bao_cao/B/bao_cao_task_6.3_b_distillation_and_consistency_loss.md`.

---

## VI. GHI CHÚ BÀN GIAO CHO LEAD C (Task 6.3-C)

### Interface tích hợp `CrossDomainTotalLoss` vào trainer M6:
```python
from src.training.loss import CrossDomainTotalLoss
from src.datasets.transforms import PairedCleanDegradedAugmentation

# Khởi tạo
criterion = CrossDomainTotalLoss(label_smoothing=0.05, temperature=2.0, alpha_kd=0.4, beta_cons=0.2)
aug       = PairedCleanDegradedAugmentation()

# Teacher đóng băng (checkpoint gốc)
teacher.eval()
for p in teacher.parameters(): p.requires_grad = False

# Trong training loop:
x_clean, x_degraded = aug(pil_image)          # (3,224,224) mỗi cái
# ... stack thành batch → (B, 3, 224, 224)

with torch.no_grad():
    teacher_logits = teacher(x_clean_batch)

student_logits    = student(x_clean_batch)
logits_degraded   = student(x_degraded_batch)
clean_mask        = (targets == 0)             # 0 = real ảnh sạch

loss_dict = criterion(
    logits          = student_logits,
    targets         = targets,
    teacher_logits  = teacher_logits,
    logits_degraded = logits_degraded,
    clean_mask      = clean_mask,
)
loss_dict["loss_total"].backward()
# Log: loss_dict["loss_ce"], loss_dict["loss_kd"], loss_dict["loss_cons"]
```

### Khởi tạo LoRA cho Task 6.5:
```python
from src.models.lora_adapter import ViTLoRAAdapter

lora_config = {"enabled": True, "rank": 8, "alpha": 16, "num_layers": 2}
adapter = ViTLoRAAdapter(vit_backbone, config=lora_config)
# Chỉ train adapter params:
optimizer = torch.optim.AdamW(adapter.trainable_parameters(), lr=1e-4)
print(adapter.summary())
# → ViTLoRAAdapter | enabled=True | rank=8 | alpha=16.0 | num_layers=2 | adapter_layers=4 | trainable_params=32,768
```
