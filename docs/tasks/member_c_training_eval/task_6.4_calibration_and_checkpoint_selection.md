# TASK 6.4: HIỆU CHUẨN XÁC SUẤT, TỐI ƯU HÓA NGƯỠNG & TUYỂN CHỌN CHECKPOINT THEO GATE 4 ĐIỀU KIỆN

- **Mã nhiệm vụ**: `Task 6.4` *(Milestone 6 - Tuần 6)*
- **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**
- **Người phối hợp**: **Thành viên B (Architecture Lead)**, **Thành viên A (Data Lead)**
- **Vị trí trong luồng quy trình**: **Thực thi sau khi hoàn thành ma trận Ablation Study tại Task 6.5**
- **Môi trường thực thi**: ☁️ **Colab GPU (T4 / L4)** / 🖥️ **Local**
- **Độ ưu tiên**: 🔴 Khẩn cấp (Cổng kiểm định chất lượng trước khi bàn giao cho Task 6.6)
- **Trạng thái**: `[~] NGHIỆM THU CÓ ĐIỀU KIỆN (FAIL GATE Q30 / CHỌN ỨNG VIÊN FALLBACK M4)`

---

## 1. Mục Tiêu & Cơ Sở Khoa Học

Sau khi hoàn thành ma trận đo lường Ablation Study cho các nhánh M3, M4, M5 tại Task 6.5, Task 6.4 tiếp nhận toàn bộ các checkpoint ứng viên để tiến hành hiệu chuẩn xác suất và tuyển chọn công bằng.

Mục tiêu chính:
1. Phân chia tập validation thành 2 phân vùng độc lập: `val_calibration` (để fit nhiệt độ và quét ngưỡng) và `val_selection` (để chấm điểm xếp hạng), tránh hoàn toàn hiện tượng tối ưu hóa và chấm điểm trên cùng một mẫu dữ liệu.
2. Hiệu chuẩn độ tin cậy bằng Temperature Scaling và xác định ngưỡng cắt phân loại $T^*$.
3. Tuyển chọn checkpoint đại diện dựa trên **Gate 4 điều kiện ràng buộc so sánh delta với mốc sàn baseline trên cùng tập validation** (kế thừa từ Task 6.2).
4. Đóng băng trọng số và xuất tệp rút gọn `fatformer_crossdomain_weights_only.pth`.

---

## 2. Quy Trình Hiệu Chuẩn Hai Phân Vùng Độc Lập

### 2.1. Phân Vùng 1: `val_calibration` (Tối ưu hóa tham số)
- Sử dụng 50% dữ liệu validation để:
  - Tối ưu hóa tham số nhiệt độ $T > 0$ bằng cực tiểu hóa Negative Log-Likelihood:
    $$\hat{p}_i = \sigma\left(\frac{z_i}{T}\right)$$
  - Quét ngưỡng phân loại $T_{th} \in [0.35, 0.65]$ với bước nhảy $\Delta = 0.02$ nhằm cân bằng Macro F1 giữa ảnh thật và ảnh giả.
  - Cố định cặp tham số $(T, T^*)$ cho từng ứng viên M3, M4, M5.

### 2.2. Phân Vùng 2: `val_selection` (Đánh giá xếp hạng khách quan)
- Nạp cặp tham số $(T, T^*)$ đã cố định để đánh giá độc lập trên 50% dữ liệu còn lại.
- Tuyệt đối không tinh chỉnh lại $T$ hay $T^*$ trên phân vùng này để bảo đảm tính khách quan của bảng xếp hạng.

---

## 3. Bộ Tiêu Chuẩn Tuyển Chọn: Gate 4 Điều Kiện So Sánh Tương Đối

Mọi ứng viên phải được đối chiếu trực tiếp với mốc sàn baseline gốc đã đo đạc tại Task 6.2 trên cùng tập validation (`val_crossdomain_baseline_metrics.json`):

| Tiêu Chí | Chỉ Số Ràng Buộc So Với Baseline Val | Ý Nghĩa Kỹ Thuật |
| :---: | :--- | :--- |
| **Điều kiện 1** | $\text{Clean Macro ACC} \ge \text{Base}_{\text{clean}} - 1.5\text{ pp}$ | Giới hạn độ sụt giảm trên ảnh sạch không quá 1.5 điểm phần trăm |
| **Điều kiện 2** | $\text{Clean Fake ACC} \ge \text{Base}_{\text{fake\_clean}} - 2.0\text{ pp}$ | Bảo đảm độ nhạy phát hiện ảnh giả clean không bị suy giảm |
| **Điều kiện 3** | $\text{Q30 Overall ACC} \ge \text{Base}_{\text{q30}}$ | Độ chính xác tổng thể nén sâu ngang bằng hoặc vượt baseline val |
| **Điều kiện 4** | $\text{Q30 Fake ACC} \ge \text{Base}_{\text{fake\_q30}} + 10.0\text{ pp}$<br>đồng thời $\text{Q30 Real ACC} \ge 60.0\%$ | Nâng cao rõ rệt khả năng bắt ảnh giả nén mà không làm sụp ảnh thật |

---

## 4. Tiêu Chuẩn Nghiệm Thu (Definition of Done)

- [x] Thực hiện phân tách rõ ràng giữa `val_calibration` và `val_selection` ([`val_split_indices.json`](../../../DATASET/logs/val_split_indices.json)).
- [x] Bảng xếp hạng chi tiết toàn bộ các ứng viên M1–M5 sau khi hiệu chuẩn trên `val_selection` ([`calibration_and_selection_report.json`](../../../DATASET/logs/calibration_and_selection_report.json)).
- [!] Xác định rõ checkpoint vượt qua toàn bộ 4 điều kiện của Gate: Không có nhánh nào đạt 4/4 Gate; kích hoạt quy chế chọn Ứng viên Dự Phòng (Fallback Candidate).
- [x] Đóng băng trọng số và xuất tệp rút gọn `fatformer_crossdomain_weights_only.pth` (~2.3 GB), xác thực mã băm SHA-256 bit-perfect khớp 100% `model_state` M4 ([`weights_only_verification_report.json`](../../../DATASET/logs/weights_only_verification_report.json)).
- [x] Bàn giao ứng viên dự phòng M4 và cặp tham số $(T=1.0811, T^*=0.37)$ cho Task 6.6 thực hiện kiểm chuẩn cuối cùng.
- [x] Biên soạn báo cáo nghiệm thu tại [`docs/bao_cao/C/bao_cao_task_6.4_calibration_and_checkpoint_selection.md`](../../bao_cao/C/bao_cao_task_6.4_calibration_and_checkpoint_selection.md).
