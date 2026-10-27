# BÁO CÁO NGHIỆM THU NHIỆM VỤ 7.1 (TASK 7.1)
## KIỂM TOÁN CẤU TRÚC CHECKPOINT & ĐO LẠI GEN 1 (WiSE-FT CŨ: $n=3$ vs $n=8$)

> **Mã nhiệm vụ**: `Task 7.1` *(Phase 7 — D1–D2)*  
> **Người thực hiện**: **Thành viên C** *(Training & Evaluation Lead)*  
> **Người phối hợp**: **Thành viên B** *(Architecture Lead)*  
> **Cổng kiểm soát chất lượng**: **Cổng G7.0 (Toàn vẹn đo lường)**  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH — NGHIỆM THU PASS G7.0`  
> **Hiện vật bàn giao**:
> - Công cụ audit tương thích: [`tools/audit_checkpoint_compat.py`](../../../tools/audit_checkpoint_compat.py)
> - Dữ liệu audit 4 checkpoint: [`logs/phase7/checkpoint_structure_audit.json`](../../../logs/phase7/checkpoint_structure_audit.json)
> - Kịch bản đo lại Gen 1: [`tools/eval_gen1_reinterpretation.py`](../../../tools/eval_gen1_reinterpretation.py)
> - Dữ liệu đo lại Gen 1 ($n=3$ vs $n=8$): [`logs/phase7/wise_ft_legacy_reinterpretation.json`](../../../logs/phase7/wise_ft_legacy_reinterpretation.json)
> - Biểu đồ trực quan đối chiếu: [`logs/phase7/gen1_n3_vs_n8.png`](../../../logs/phase7/gen1_n3_vs_n8.png)

---

## 1. Kết Quả Kiểm Toán Cấu Trúc 4 Checkpoints

Thực thi bằng công cụ độc lập trên CPU (`tools/audit_checkpoint_compat.py`), giải mã chính xác số lượng keys và topology khối adapter của 4 checkpoint mục tiêu:

| Checkpoint | Số tensor keys | Khối adapter phát hiện | $n$ khả biểu | Layout $n$ | Partial load? | SRM per block? | Gating? | Trạng thái layout |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** (`fatformer_4class_ckpt.pth`) | **1.116** | `{7, 15, 23}` | `[3]` | `3` | **KHÔNG** | Không (66 tensors) | Không | **Hợp lệ chuẩn** |
| **M4** (`crossdomain_model_best_m4.pth`) | **1.684** | `{2, 5, 8, 11, 14, 17, 20, 23}` | `[8]` | `8` | **KHÔNG** | Có (94 tensors) | Có | **Hợp lệ chuẩn** |
| **Gen 1 (WiSE-FT)** (`fatformer_wise_ft_best.pth`) | **1.816** | **`{2, 5, 7, 8, 11, 14, 15, 17, 20, 23}`** | **`[]`** (Rỗng) | **None** | **CÓ (Bắt buộc)** | Có (Hỗn hợp 66 & 94) | Có | **`invalid_layout`** |
| **Anchored** (`fatformer_anchored_recovery_best.pth`) | **1.200** | `{7, 15, 23}` | `[3]` | `3` | **KHÔNG** | Có (94 tensors) | Có | **Hợp lệ chuẩn** |

---

## 2. Trả Lời 4 Câu Hỏi Bắt Buộc Theo Khế Ước Nghiệm Thu (§5.4)

### Câu hỏi 1: Cấu trúc — Gen 1 có bao nhiêu khối adapter, có biểu diễn được không?
* **Số khối:** Gen 1 chứa đúng **10 khối adapter** `{2, 5, 7, 8, 11, 14, 15, 17, 20, 23}` (là phép hợp giữa 3 khối của Baseline và 8 khối của M4, với khối 23 trùng nhau).
* **Cấu trúc tensor:** Khối `{7, 15}` chỉ chứa **66 tensors** (không có SRM), trong khi các khối còn lại chứa **94 tensors** (có SRM).
* **Tính khả biểu:** Không tồn tại bất kỳ số nguyên $n \in \{3, 4, 6, 8, 12, 24\}$ nào có thể sinh ra tập chỉ số trên theo công thức kiến trúc $(i+1) \cdot (24/n) - 1$.
* **Kết luận:** Gen 1 là một checkpoint dị tật kiến trúc (**chimeric checkpoint**), không thể biểu diễn nguyên vẹn trong bất kỳ mô hình FatFormer hợp lệ nào (`partial=true`, `layout.n=null`).

### Câu hỏi 2: Số đo — Layout $n=8$ có cao hơn $n=3$ có ý nghĩa hay không?
Thực nghiệm đo đạc độc lập trên cùng 2.000 ảnh Clean của CommunityForensics-Eval (CFE) với $T=1.0$, threshold=$0.50$, batch\_size=$32$ cho kết quả:

| Chỉ số đo đạc | Layout $n=3$ (Nạp 3 khối `{7,15,23}`, Bỏ 7 khối) | Layout $n=8$ (Nạp 8 khối `{2..23}`, Bỏ 2 khối `{7,15}`) | Chênh lệch ($\Delta = n_8 - n_3$) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy (ACC)** | **64.75%** | **52.05%** | **-12.70 pp** |
| **Real Accuracy** | **94.00%** | **100.00%** | **+6.00 pp** |
| **Fake Accuracy** | **35.50%** | **4.10%** | **-31.40 pp (Sụp đổ)** |
| **ROC-AUC** | **76.29%** | **57.07%** | **-19.22 pp** |
| **Average Precision (AP)** | **78.24%** | **60.41%** | **-17.83 pp** |

$\rightarrow$ **Trả lời:** **KHÔNG**. Layout $n=8$ không những không cao hơn mà còn **suy giảm nghiêm trọng** Fake ACC (-31.40 pp), đẩy mô hình vào trạng thái tê liệt hoàn toàn năng lực phát hiện ảnh giả (đoán Real 100%, Fake chỉ 4.10%). Ngược lại, layout $n=3$ có Fake ACC đạt 35.50% chỉ vì nó đã loại bỏ 7 khối pha trộn và khôi phục lại gần như nguyên vẹn trạng thái của mô hình Baseline gốc (Fake ACC Baseline = 35.30%).

### Câu hỏi 3: Phán quyết nguyên nhân — `artifact_loader`, `different_basin` hay `inconclusive`?
* **Phán quyết:** **`different_basin` kết hợp `invalid_layout`**.
* **Giải thích cơ chế:**  
  Giả thuyết `artifact_loader` (cho rằng loader cũ ép $n=3$ làm hỏng Gen 1) đã bị **bác bỏ hoàn toàn**. Thực tế chứng minh chính cấu hình $n=8$ mới là thủ phạm tạo ra kết quả sụp đổ 4.10% ở Phase 6-R. Nguyên nhân gốc rễ là việc nội suy trọng số WiSE-FT giữa hai không gian tham số không tương thích về mặt cấu trúc (Baseline $n=3$ và M4 $n=8$) đã phá vỡ không gian đặc trưng của bộ mã hóa thị giác, khiến mô hình trượt vào một hố cực tiểu suy biến (**degenerate basin**).

### Câu hỏi 4: Hệ quả học thuật — Gen 1 có được dùng không và số liệu nào bị vô hiệu?
1. **Cô lập vĩnh viễn:** Checkpoint `fatformer_wise_ft_best.pth` bị gắn nhãn bất biến **`invalid_layout`**, tuyệt đối **CẤM** đưa vào bất kỳ báo cáo kết quả hay sản phẩm triển khai nào.
2. **Vô hiệu hóa toàn bộ số liệu Phase 6-R:**
   - Số liệu `52.05%` (Micro ACC), `4.10%` (Fake ACC), và `96.58%` gắn với Gen 1 trong các báo cáo cũ chính thức bị hủy bỏ giá trị tham chiếu.
3. **Mở khóa Cổng G7.0 & Bàn giao:**
   - Toàn bộ kết quả kiểm toán này mở khóa thành công Cổng **G7.0**.
   - Bàn giao script cho Thành viên B để triển khai `canonical_loader` ở Task 7.2.
   - Định hướng rõ ràng cho Thành viên C ở Task 7.5 (WiSE-FT Gen 2): Chỉ thực hiện nội suy trọng số trên các checkpoint có **CÙNG CẤU TRÚC ADAPTER ĐỒNG NHẤT** (cùng $n=8$).

---

## 3. Kết Luận Nghiệm Thu

Task 7.1 đã hoàn thành trọn vẹn 100% mục tiêu, làm sáng tỏ bí ẩn kỹ thuật lớn nhất của Phase 6-R bằng số liệu đo đạc thực chứng khách quan. Cổng **G7.0 (Toàn vẹn đo lường)** chính thức được xác nhận **PASS**.

