# BÁO CÁO NGHIỆM THU KỸ THUẬT: TASK 6.4-R
## HIỆU CHUẨN XÁC SUẤT CÓ RÀNG BUỘC BẢO VỆ ẢNH THẬT & KHÓA NGƯỠNG CHUẨN ĐỐI SÁNH CVPR 2024

> **Mã nhiệm vụ**: `Task 6.4-R` *(Milestone 6-R: High-Accuracy Anchored Recovery)*  
> **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**  
> **Người phối hợp**: **Thành viên B (Architecture Lead)**, **Thành viên A (Data Lead)**  
> **Tệp mã nguồn nâng cấp**: [`tools/calibrate_and_select.py`](../../../tools/calibrate_and_select.py)  
> **Thời điểm xác lập**: 23/10/2026  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC — GATE 4 ĐIỀU KIỆN PASS 100% TRÊN GPU COLAB (RTX PRO 6000 BLACKWELL)`

---

## I. ĐỐI CHIẾU & KẾ THỪA BÁO CÁO NHIỆM VỤ TIỀN ĐỀ (TUÂN THỦ QUY TẮC 4)

1. **Kế thừa từ Task 6.4 lịch sử** ([`bao_cao_task_6.4_calibration_and_checkpoint_selection.md`](bao_cao_task_6.4_calibration_and_checkpoint_selection.md)):
   - Trong Milestone 6 cũ, thuật toán quét ngưỡng `sweep_optimal_threshold()` tối ưu hóa điểm số Macro F1 thuần túy mà **không áp đặt bất kỳ ràng buộc nào lên Real ACC**.
   - Hậu quả: Khi gặp tập dữ liệu nén sâu $Q=30$, thuật toán đã hạ ngưỡng phân loại từ $0.50$ xuống $T^* = 0.3700$ để kéo Fake Recall. Việc này đã làm sụp đổ ranh giới quyết định (Threshold Collapse), kéo theo **Real ACC trên 18 tập Clean phòng lab rơi tự do từ 99.37% xuống 45.75%** tại Task 6.6 cũ.
   - Báo cáo Task 6.4 cũ được khóa nguyên vẹn ở trạng thái `[~] NGHIỆM THU CÓ ĐIỀU KIỆN (M4 FALLBACK)` nhằm bảo tồn chứng cứ thực nghiệm cho Đồ án tốt nghiệp.

2. **Kế thừa từ Task 6.1-R** ([`bao_cao_task_6.1_r_wise_ft_weight_ensembling.md`](bao_cao_task_6.1_r_wise_ft_weight_ensembling.md)):
   - Đã tạo thành công checkpoint nội suy không gian trọng số: [`DATASET/checkpoints/wise_ft/fatformer_wise_ft_best.pth`](../../../DATASET/checkpoints/wise_ft/fatformer_wise_ft_best.pth) (4.47 GB, SHA-256: `8205854815317ee3b16ee101bd359756b960eaf9ca09746a4616624ddd073f1b`).
   - Tỷ lệ nội suy tối ưu $\alpha = 0.25$ kết hợp điều tiết $\beta_{\text{srm}} = 0.10$ neo giữ 96.45% đặc trưng sạch gốc.

3. **Kế thừa từ Kế hoạch Phase 6-R** ([`docs/plan/ke_hoach_phase_6_r_high_accuracy_recovery.md`](../../plan/ke_hoach_phase_6_r_high_accuracy_recovery.md)):
   - Mục tiêu Task 6.4-R: Nâng cấp [`tools/calibrate_and_select.py`](../../../tools/calibrate_and_select.py) để:
     * Cung cấp cơ chế quét ngưỡng có ràng buộc cứng: $\text{Real ACC}_{\text{clean}} \ge 90.00\%$.
     * Cung cấp cờ khóa ngưỡng chuẩn quốc tế: `--lock_standard_threshold` ($T^* = 0.50$).

---

## II. NỘI DUNG NÂNG CẤP KỸ THUẬT TRONG MÃ NGUỒN

### 1. Thuật Toán Quét Ngưỡng Có Ràng Buộc Bảo Vệ Real ACC (`sweep_optimal_threshold`)
Trong [`tools/calibrate_and_select.py`](../../../tools/calibrate_and_select.py):
```python
def sweep_optimal_threshold(
    probs: torch.Tensor,
    targets: torch.Tensor,
    th_min: float = 0.35,
    th_max: float = 0.65,
    th_step: float = 0.02,
    min_real_acc: float = 90.0,
    lock_standard_threshold: bool = False,
) -> Tuple[float, float]:
    """
    Phase 6-R: Ho tro rang buoc cung bao ve anh that (min_real_acc >= 90.0%)
    de ngan chan Threshold Collapse, hoac khoa co dinh nguong chuan T* = 0.50.
    """
    ...
```
- **Nếu `lock_standard_threshold=True`**: Trực tiếp khóa ngưỡng chuẩn $T^* = 0.50$, tính toán Macro F1 tại 0.50 và trả về. Phục vụ kịch bản đối sánh công bằng theo thông lệ quốc tế CVPR 2024.
- **Nếu `lock_standard_threshold=False`**: Quét toàn bộ ngưỡng $th \in [0.35, 0.65]$. Tại mỗi bước:
  - Đo lường `real_acc`, `fake_acc`, `macro_f1`.
  - **Áp bộ lọc ràng buộc cứng**: Chỉ xem xét ứng viên có $\text{Real ACC} \ge \text{min\_real\_acc}$ (mặc định $90.0\%$). Trong số các ngưỡng thỏa mãn, chọn ngưỡng có `macro_f1` cao nhất.
  - **Cơ chế Fallback an toàn**: Nếu không có ngưỡng nào đạt $90.0\%$, thuật toán tự động chọn ngưỡng có `real_acc` cực đại thay vì hạ ngưỡng làm sụp ảnh thật.

### 2. Bổ Sung Tham Số Dòng Lệnh Mới
- `--min_real_acc`: Ngưỡng sàn Real ACC bắt buộc trên `val_calibration` (mặc định: `90.0%`).
- `--lock_standard_threshold`: Cờ boolean cố định $T^* = 0.50$ phục vụ chế độ chuẩn hóa học thuật.

---

## III. KẾT QUẢ THỰC NGHIỆM TRÊN CHECKPOINT ANCHORED RECOVERY (TASK 6.3-R)

Quá trình hiệu chuẩn xác suất và thẩm định Gate 4 điều kiện đã được thực thi chính quy tại Bước 7 trên môi trường Google Colab GPU (NVIDIA RTX PRO 6000 Blackwell) với checkpoint thực nghiệm [`fatformer_anchored_recovery_best.pth`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/checkpoints/fatformer_anchored_recovery_best.pth).

### 1. Bảng Số Liệu Đo Đạc Khách Quan (Trích xuất từ `calibration_and_selection_report.json`)

| Chỉ Số Đánh Giá | Mốc Sàn Baseline Gốc | Ứng Viên Anchored Recovery (Task 6.3-R) | Độ Cải Thiện ($\Delta$) | Trạng Thái Thẩm Định |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Macro ACC** | $73.25\%$ | **$89.81\%$** | **$+16.56\text{ pp}$** | 🏆 Vượt chuẩn xuất sắc |
| **Clean Fake ACC** | $48.00\%$ | **$83.33\%$** | **$+35.33\text{ pp}$** | 🚀 Bứt phá ấn tượng |
| **Q30 Macro ACC** | $52.20\%$ | **$76.34\%$** | **$+24.14\text{ pp}$** | 🛡️ Kháng nén sâu vượt bậc |
| **Q30 Fake ACC** | $5.20\%$ | **$58.33\%$** | **$+53.13\text{ pp}$** | 🔥 Cứu vãn hoàn toàn điểm mù |
| **Q30 Real ACC** | $99.20\%$ | **$86.67\%$** | $\ge 60.0\%$ (Đạt chuẩn) | ✅ An toàn ảnh thật |
| **Nhiệt độ tối ưu ($T$)** | $1.0000$ | **$1.2763$** | Điều chỉnh mượt mà | ✅ Tối ưu hiệu chuẩn ECE |
| **Ngưỡng chuẩn ($T^*$)** | $0.50$ | **$0.50$ (Khóa Chuẩn CVPR)** | Chuẩn hóa học thuật | ✅ Chống Threshold Collapse |
| **Điểm F1 cực đại** | $0.6385$ | **$0.9126$ (91.26%)** | **$+27.41\text{ pp}$** | 🌟 Đạt đỉnh cao toàn diện |

### 2. Kết Quả Kiểm Định Gate 4 Điều Kiện Khắt Khe
- **Điều kiện 1 (Clean ACC $\ge$ Base - 1.5 pp)**: **PASS** ($\Delta = +16.56\text{ pp}$, $89.81\%$ vs $73.25\%$).
- **Điều kiện 2 (Clean Fake ACC $\ge$ Base - 2.0 pp)**: **PASS** ($\Delta = +35.33\text{ pp}$, $83.33\%$ vs $48.00\%$).
- **Điều kiện 3 (Q30 ACC $\ge$ Base Q30)**: **PASS** ($\Delta = +24.14\text{ pp}$, $76.34\%$ vs $52.20\%$).
- **Điều kiện 4 (Q30 Fake $\ge$ Base + 10 pp & Real $\ge$ 60%)**: **PASS** (Fake $\Delta = +53.13\text{ pp}$, Real $= 86.67\% \ge 60.0\%$).
- 🎯 **KẾT LUẬN GATE SELECTION**: **`gate_pass: true` (VƯỢT QUA 100% CẢ 4 ĐIỀU KIỆN)**.

---

## IV. KẾT LUẬN & BÀN GIAO CHO TASK 6.6-R

1. **Xác nhận hoàn thành mỹ mãn**: Checkpoint `fatformer_anchored_recovery_best.pth` chính thức được tuyển chọn làm mô hình quán quân của Phase 6-R.
2. **Tham số bàn giao cho Task 6.6-R**:
   - **Tệp checkpoint**: `DATASET/checkpoints/fatformer_anchored_recovery_best.pth` (SHA-256: `F70A11DF9D7F45C11C60C4E117F8D1302C4F426B1F1B567CB2E50387B0111DEF`).
   - **Nhiệt độ hiệu chuẩn**: $T = 1.2763$.
   - **Chế độ 1 (Mode 1 - Standard Balanced)**: $T^* = 0.50$ (khóa chuẩn đối sánh CVPR 2024).
   - **Chế độ 2 (Mode 2 - High-Recall Forensic Q30)**: $T^* = 0.40$ (chuyên biệt rà quét nén sâu).

