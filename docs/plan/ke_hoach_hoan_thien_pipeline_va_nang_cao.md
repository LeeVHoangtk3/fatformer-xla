# KẾ HOẠCH HOÀN THIỆN PIPELINE VÀ NÂNG CAO ĐỘ CHÍNH XÁC
## FatFormer-XLA — Từ Checkpoint Trung Gian Đến Run Đáng Tin Cậy

> **Tác giả**: Thành viên C (Training & Evaluation Lead)  
> **Ngày lập**: 2026-10-12  
> **Phạm vi**: Tài liệu kế hoạch kỹ thuật cho giai đoạn sau khi hoàn thành kiểm toán pipeline

---

## 0. BỐI CẢNH VÀ LÝ DO

Các số liệu trong Task 5.1 (Clean ACC 90.19%, Q=30 Overall ACC 71.03%, Fake ACC 44.84%) **không thể được dùng làm kết luận cuối** vì ba lý do kỹ thuật cụ thể:

1. **Checkpoint chưa hoàn thành giáo trình**: Lần chạy Task 4.3 dừng ở Epoch 4 do đầy đĩa 19.5GB Kaggle. Phase 3 Hardening (Epoch 5, $Q \in [30, 50]$, lr × 0.5) chưa bao giờ được thực hiện.
2. **Pipeline benchmark cũ không nhất quán**: Nhánh Degraded đã từng nạp trọng số với `strict=False`, có thể dẫn đến một phần tham số không khớp bị bỏ qua thầm lặng.
3. **Tỷ lệ tham số trainable trong run cũ là 26.07%**: Whitelist chưa loại trừ các khối FFN nội bộ của ViT. Số liệu thực tế sau khi sửa whitelist là 2.66% (15.41M / 578.44M tham số).

Do đó, ưu tiên số một là **hoàn thiện tính đúng đắn của pipeline, chạy lại đủ 5 epoch, và benchmark lại từ đầu** trước khi đưa ra kết luận hay mở rộng kiến trúc.

---

## 1. LỖI PIPELINE CÒN TỒN ĐỌNG (Cần sửa trước khi chạy lại)

### 1.1 `np.mean` dùng mà không có `import numpy`
**Vị trí**: [`src/training/trainer.py`, dòng 169–170](../../../src/training/trainer.py)  
**Triệu chứng**: `NameError: name 'np' is not defined` khi validation dict đa miền được truyền vào, tức là ngay từ Epoch 1 khi dùng multi-domain val loader.  
**Sửa**: Thêm `import numpy as np` vào phần đầu `trainer.py`.

```python
# Thêm vào đầu file src/training/trainer.py
import numpy as np
```

**Mức độ**: Blocker — run sẽ crash ngay lần validation đa miền đầu tiên.

---

### 1.2 Resume không khôi phục `scheduler.state_dict()` và `scaler.state_dict()`
**Vị trí**: [`train.py`, dòng 517–527](../../../train.py)  
**Triệu chứng**: Sau khi resume từ checkpoint, lịch học tốc (learning rate schedule) bắt đầu lại từ đầu thay vì tiếp tục từ epoch đã lưu. Nếu dùng AMP, scaler cũng bị reset về trạng thái ban đầu.  
**Tác động thực tế**: Khi resume từ Epoch 3 (ví dụ để chạy tiếp Epoch 4–5), learning rate sẽ bị tính sai — không ảnh hưởng khi run liên tục từ đầu trong một session duy nhất.  
**Sửa**:

```python
# Trong train.py, đoạn resume — bổ sung sau khi load model/optimizer:
if resumed_meta:
    if scheduler is not None and "scheduler_state" in resumed_meta:
        scheduler.load_state_dict(resumed_meta["scheduler_state"])
        print("  [✓] Khôi phục scheduler state.")
    if scaler is not None and "scaler_state" in resumed_meta:
        scaler.load_state_dict(resumed_meta["scaler_state"])
        print("  [✓] Khôi phục AMP scaler state.")

# Trong checkpoint_manager.py, đoạn save — bổ sung:
if scheduler is not None:
    checkpoint["scheduler_state"] = scheduler.state_dict()
if scaler is not None:
    checkpoint["scaler_state"] = scaler.state_dict()
```

**Mức độ**: Quan trọng khi resume giữa session (Colab/Kaggle ngắt kết nối).

---

### 1.3 Whitelist PEFT trong YAML không được truyền vào `freeze_clip_backbone()`
**Vị trí**: [`train.py`, dòng 450–455](../../../train.py), [`src/training/freeze_utils.py`, dòng 82](../../../src/training/freeze_utils.py)  
**Triệu chứng**: Mặc dù `configs/train_config.yaml` có khai báo `freeze.trainable_modules`, danh sách này không được đọc và truyền vào `freeze_clip_backbone()`. Hàm này đang dùng hardcode fallback nội bộ.  
**Sửa**:

```python
# Trong load_config_and_override_args, thêm:
f_cfg = cfg.get("freeze", {})
args.trainable_modules = f_cfg.get("trainable_modules", None)   # list hoặc None
args.trainable_params  = f_cfg.get("trainable_params", None)    # list param cụ thể

# Trong main(), đổi lời gọi freeze:
stats = freeze_clip_backbone(
    model,
    trainable_module_prefixes=getattr(args, "trainable_modules", None),  # từ YAML
    target_trainable_ratio_min=getattr(args, "target_trainable_ratio_min", 0.025),
    target_trainable_ratio_max=getattr(args, "target_trainable_ratio_max", 0.035),
    verbose=True
)
```

**Mức độ**: Quan trọng để YAML thực sự là source of truth cho PEFT.

---

### 1.4 `lr_prompt`, `keep_last_n`, `resume_from`, `num_classes`, `num_context_embedding` bị bỏ qua
**Vị trí**: [`train.py`, dòng 400–406](../../../train.py), [`load_config_and_override_args`](../../../train.py)

| Tham số YAML | Trạng thái hiện tại | Hành động |
|---|---|---|
| `model.num_classes` | Đọc vào `args.num_classes` (dòng 168) nhưng `model_args.num_classes` hardcode = 2 (dòng 402) | Sửa dòng 402 dùng `getattr(args, "num_classes", 2)` |
| `model.num_context_embedding` | Đọc vào args (dòng 169) nhưng `model_args.num_context_embedding` hardcode = 8 (dòng 403) | Sửa tương tự |
| `optimizer.lr_prompt` | Không có trong optimizer param groups | Thêm nhóm param cho soft prompt ctx |
| `checkpoint.keep_last_n` | `max_to_keep=3` hardcode (dòng 512) | Đọc từ YAML `c_cfg.get("keep_last_n", 3)` |
| `checkpoint.resume_from` | Không được đọc từ YAML (chỉ đọc từ `--resume` CLI) | Thêm fallback đọc `c_cfg.get("resume_from")` nếu không có CLI flag |

**Mức độ**: Nhỏ — không gây crash, nhưng YAML không hoàn toàn là source of truth.

---

### 1.5 Benchmark phải lưu config hash + seed cố định
Mỗi lần chạy benchmark cần ghi lại:
- Đường dẫn + MD5/SHA256 của checkpoint.
- Seed ngẫu nhiên (`torch.manual_seed`, `np.random.seed`).
- Phiên bản cấu hình YAML (commit hash hoặc nội dung hash).

Đây là điều kiện tối thiểu để kết quả có thể tái lập và đối chiếu học thuật.

---

## 2. KẾ HOẠCH RUN LẠI PLAN A ĐẦY ĐỦ 5 EPOCH

### Thứ tự thực hiện:
1. **Sửa 3 lỗi blocker/important** theo mục 1.1–1.3 ở trên.
2. **Chạy dry-run** `python train.py --epochs 1 --config configs/train_config.yaml` trên local (với model fallback).
3. **Đẩy lên Kaggle** và chạy đủ 5 epoch liên tục, không resume giữa chừng nếu có thể.
4. **Sau khi xong**: Benchmark lại trên 18 tập test với `strict=True`, ghi đầy đủ config hash, seed và log missing/unexpected keys.
5. **So sánh**: Các chỉ số từ run mới sẽ là số liệu chính thức thay thế tham chiếu tạm thời trong Task 5.1.

### Ước tính thời gian (GPU RTX PRO 6000 Blackwell ~95GB VRAM):
- 5 epoch × ~32 phút/epoch ≈ **2 giờ 40 phút** huấn luyện.
- Benchmark 18 tập × 7 trạng thái ≈ **~200 phút** (dự kiến giảm khi dùng batch inference lớn hơn trên Blackwell).

---

## 3. HƯỚNG NÂNG CAO SAU KHI CÓ RUN ĐẦY ĐỦ (Theo thứ tự ưu tiên)

### Ưu tiên 1 — Fine-tune theo Hard Negative, tập trung Q=30

**Vấn đề**: Fake ACC tại Q=30 là 44.84% (>55% ảnh giả bỏ lọt). Các trường hợp đặc biệt nghiêm trọng: StyleGAN2 (6%), VQ-Diffusion (4.4%), họ GLIDE (~20–24%), Guided (26%).

**Hướng kỹ thuật**:
- Tăng tỷ trọng mẫu Fake bị nén nặng (Q=30–50) trong WeightedRandomSampler.
- Thử `asymmetric CE` hoặc Focal Loss $\gamma=1.5$ với $\alpha_{fake} > 0.5$ cho Phase 3.
- Hiệu chỉnh threshold phân loại (0.5 → ~0.35–0.40) trên validation độc lập sau khi train xong.

**Mục tiêu thực tế**: Cải thiện Fake ACC Q=30 lên 55–65% mà Real ACC không giảm quá 3 điểm.

---

### Ưu tiên 2 — Consistency Distillation: Giải quyết đồng thời Robustness và Forgetting

**Vấn đề**: Clean ACC đang thiếu 6.26 điểm so với baseline gốc (90.19% vs 96.45%). Tăng cường corruption đơn thuần sẽ không giải quyết được forgetting.

**Hướng kỹ thuật**:

$$\mathcal{L}_{total} = \mathcal{L}_{CE}(clean) + \mathcal{L}_{CE}(degraded) + \lambda_{KL} \cdot \text{KL}(f_{clean} \| f_{degraded}) + \lambda_{dist} \cdot \text{KL}(f_{clean} \| f_{baseline})$$

- Với mỗi ảnh, tạo cặp `(clean, degraded)` từ cùng nguồn.
- $\text{KL}(f_{clean} \| f_{degraded})$: buộc logit nhất quán giữa hai phiên bản.
- $\text{KL}(f_{clean} \| f_{baseline})$: soft-distillation từ checkpoint baseline gốc — giữ kiến thức nhận diện ảnh sạch.

**Điều kiện tiên quyết**: Cần giữ checkpoint baseline gốc CVPR 2024 (`fatformer_4class_ckpt.pth`) ở chế độ evaluation-only, không tối ưu hóa.

---

### Ưu tiên 3 — Cân Bằng Lại Sampler Theo Generator

**Vấn đề hiện tại**: Với tỷ lệ 15% staging trong ~147k lượt mẫu/epoch, mỗi ảnh trong 3.6k staging được lặp lại xấp xỉ 6 lần/epoch. Điều này khiến mô hình nhớ thuộc một số ít ảnh staging, trong khi ProGAN trung bình chưa đến 1 lần/epoch.

**Hướng kỹ thuật**:
- Bổ sung ảnh StyleGAN2, VQ-Diffusion, GLIDE, Guided và Deepfake vào tập staging. Ưu tiên các generator đang có Fake ACC thấp nhất trong benchmark.
- Áp dụng **stratified sampler theo generator × class × degradation level**: mỗi bucket đóng góp tương đương số lượng mẫu mỗi epoch.
- Đặt giới hạn max oversample ratio cho một nguồn đơn lẻ: ví dụ không quá 3× so với tần suất tự nhiên.

---

### Ưu tiên 4 — Mở Rộng Corruption Thực Tế

**Vấn đề**: Curriculum hiện tại chỉ chọn 1 loại corruption mỗi lần. Ảnh thực tế trên mạng xã hội thường qua nhiều bước xử lý liên tiếp.

**Hướng kỹ thuật**:
- **Corruption hỗn hợp**: Resize → JPEG → Gaussian Blur, hoặc JPEG → WebP → JPEG.
- **Chroma subsampling**: Mô phỏng chuyển đổi màu khi chia sẻ qua WhatsApp/Telegram.
- **Crop-reencode**: Cắt nhỏ rồi lưu lại, bắt chước hành vi screenshot.
- **Dải quality có nhiều mẫu tập trung quanh Q=30–50**: Thay phân phối đều bằng phân phối có trọng số cao hơn ở vùng khó.
- **Test suite "unseen corruption"**: Một tập biến thể chưa dùng trong huấn luyện, để kiểm tra tổng quát hóa.

---

### Ưu tiên 5 — PEFT Có Kiểm Soát (Chỉ sau khi pipeline ổn định)

**Nguyên tắc**: Không thay đổi kiến trúc khi pipeline chưa ổn định và dataset vẫn thiên mạnh về ProGAN.

**Lộ trình ablation từng bước** (mỗi bước là một run riêng biệt):

| Bước | Thành phần mở thêm | Điều kiện chuyển sang bước tiếp |
|---|---|---|
| Baseline | Chỉ soft prompt `ctx` + gating | Clean ACC ổn định |
| +SRM | SRM projection + gating | Clean không giảm >1% |
| +LoRA | LoRA rank-4 tại 2 block ViT cuối | Clean không giảm >1% |
| +LayerNorm | Mở LayerNorm của 4 block ViT cuối | Chỉ khi dataset đa dạng đủ |

**Không nên**: Progressive unfreeze toàn backbone khi tập dữ liệu vẫn chủ yếu là ProGAN.

---

## 4. TIÊU CHUẨN KẾT THÚC (Definition of Done)

| Tiêu chí | Mức tối thiểu cần đạt | Ghi chú |
|---|---|---|
| Clean Overall ACC | ≥ 94.0% | Phục hồi về gần baseline gốc 96.45% |
| Q=30 Overall ACC | ≥ 75.0% | Tăng từ 71.03% |
| Q=30 Fake ACC | ≥ 55.0% | Tăng từ 44.84%; >55% là tỷ lệ bỏ lọt chấp nhận được |
| StyleGAN2 Fake ACC | ≥ 30.0% | Hiện 6% — đây là trường hợp nghiêm trọng nhất |
| VQ-Diffusion Fake ACC | ≥ 30.0% | Hiện 4.4% |
| Pipeline tái lập | Benchmark chạy lại cho cùng kết quả ±0.1% | Seed cố định + config hash |
| Checkpoint audit | Missing/unexpected keys = 0 hoặc ghi log rõ | `strict=True` toàn bộ |

---

## 5. MỐI QUAN HỆ GIỮA CÁC HƯỚNG

```
Pipeline đúng (Mục 1-2)
        │
        ▼
Run Plan A đủ 5 epoch (Phase 3 hardening)
        │
        ├──► Hard Negative + Asymmetric Loss (Ưu tiên 1)
        │           giải quyết Fake ACC Q=30
        │
        ├──► Consistency Distillation (Ưu tiên 2)
        │           giải quyết Clean ACC forgetting
        │
        ├──► Sampler đa generator (Ưu tiên 3)
        │           cải thiện tổng quát hóa đa miền
        │
        ├──► Mixed Corruption (Ưu tiên 4)
        │           cải thiện robustness thực tế
        │
        └──► PEFT ablation (Ưu tiên 5)
                    chỉ khi 3 hướng trên đã ổn định
```

---

## 6. KHUYẾN NGHỊ THỰC TẾ

1. **Sửa 3 lỗi blocker trước** (1.1 np.mean import, 1.2 resume scheduler/scaler, 1.3 PEFT whitelist từ YAML).
2. **Chạy lại Plan A đủ 5 epoch** với PEFT 2.66% và Phase 3 hardening đầy đủ.
3. **Benchmark lại toàn bộ** trên pipeline `strict=True` với seed cố định và config hash.
4. **Đánh giá kết quả thực tế mới** trước khi quyết định hướng tiếp theo.
5. **Hướng có xác suất cải thiện cao nhất**: Hard Negative + Consistency Distillation đồng thời (Ưu tiên 1+2), vì giải quyết song song hai vấn đề robustness và forgetting mà không cần thêm dữ liệu mới.

---

*Tài liệu này được lập trên cơ sở kiểm toán code tĩnh. Các số liệu benchmark trong Task 5.1 hiện tại đóng vai trò tham chiếu trung gian và cần được đo đạc lại sau khi hoàn thiện pipeline.*

*Lưu trữ tại: `docs/plan/ke_hoach_hoan_thien_pipeline_va_nang_cao.md`*
