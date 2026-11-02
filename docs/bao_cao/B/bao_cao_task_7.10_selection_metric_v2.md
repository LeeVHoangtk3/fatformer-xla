# BÁO CÁO NGHIỆM THU NHIỆM VỤ 7.10 (TASK 7.10)
## SELECTION METRIC v2: WORST-GROUP + RÀNG BUỘC REAL ACC

> **Mã nhiệm vụ**: `Task 7.10` *(Phase 7 — D5–D6)*  
> **Người thực hiện**: **Thành viên B** *(Architecture & Loss Lead)*  
> **Người phối hợp**: **Thành viên C** *(Training Lead — sử dụng hàm chọn checkpoint trong Task 7.11)*  
> **Cổng kiểm soát chất lượng**: — *(nạp đầu vào cho Cổng **G7.2**)*  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH`  
> **Môi trường xác minh**: Local CPU (Python 3.14 / pytest 9.0.3) — 2026-11-03  
> **Hiện vật bàn giao**:
> - Module metric: [`src/training/metrics.py`](../../../src/training/metrics.py) (`group_recalls`, `worst_group_score`)
> - Module trainer nâng cấp: [`src/training/trainer.py`](../../../src/training/trainer.py) (tích hợp selection metric v2, log `val_groups.csv`)
> - Module lưu checkpoint: [`src/training/checkpoint_manager.py`](../../../src/training/checkpoint_manager.py) (phân tách `best.pth` và `best_effort.pth`)
> - Bộ test kiểm thử: [`tests/test_selection_metric.py`](../../../tests/test_selection_metric.py) (5/5 unit tests pass)
> - Báo cáo này: `docs/bao_cao/B/bao_cao_task_7.10_selection_metric_v2.md`

---

## 1. Bối Cảnh & Vấn Đề Giải Quyết

**Kế thừa từ Task 7.1 & 7.6**:
- Phân tích từ Task 7.1 ([`bao_cao_task_7.1_checkpoint_audit.md`](../C/bao_cao_task_7.1_checkpoint_audit.md)) chỉ ra checkpoint WiSE-FT Gen 1 đạt điểm composite trung bình cao giả tạo nhưng Fake ACC thực tế bị sụp đổ còn **4.10%**.
- Nguyên nhân cốt lõi trong logic chọn cũ (`trainer.py` dòng 204–231):
  1. **Trung bình che lấp nhóm yếu nhất**: Phép lấy trung bình (Clean 40%, Robust 40%, Diff 20%) cho phép mô hình hy sinh Real ACC hoặc Fake ACC ở một miền suy biến mà điểm tổng hợp vẫn cao.
  2. **Trộn AP vào quyết định checkpoint**: AP đo toàn dải ngưỡng phân loại thay vì ngưỡng vận hành thực chiến.
  3. **Thiếu cơ chế bảo vệ Real**: Không có ràng buộc đảm bảo Real ACC đạt chuẩn an toàn.

---

## 2. Thiết Kế & Hiện Thực Kỹ Thuật

### 2.1. Không gian đánh giá phân tầng 4 nhóm
$$g \in \{\text{clean}, \text{Q30}\} \times \{\text{real}, \text{fake}\}$$

Hàm `group_recalls(y_true, y_pred, condition)` tính toán recall độc lập:
- `clean_r`: Recall ảnh Real ở trạng thái Clean
- `clean_f`: Recall ảnh Fake ở trạng thái Clean
- `q30_r`: Recall ảnh Real bị nén JPEG Q=30
- `q30_f`: Recall ảnh Fake bị nén JPEG Q=30

### 2.2. Điểm chọn Worst-Group Score
$$\text{worst\_group} = \min(\text{clean\_r}, \text{clean\_f}, \text{q30\_r}, \text{q30\_f})$$
- Nếu phát hiện nhóm rỗng (thiếu dữ liệu kiểm thử), hàm `worst_group_score(recalls, nan_policy="raise")` lập tức fail-loud bằng ngoại lệ `ValueError`, ngăn chặn tuyệt đối việc ngụy tạo điểm số.

### 2.3. Cơ chế phân tách Checkpoint Kép
Mô hình loại bỏ hoàn toàn việc quyết định lưu dựa trên composite score cũ:
1. **Checkpoint chính thức (`best.pth`)**:
   - Điều kiện: $\text{worst\_group} > \text{best\_worst\_group}$ **VÀ** $\text{clean\_r} \ge 90.0\%$.
2. **Checkpoint nỗ lực cao nhất (`best_effort.pth`)**:
   - Điều kiện: $\text{worst\_group} > \text{best\_effort\_worst\_group}$ nhưng $\text{clean\_r} < 90.0\%$.
   - Giữ lại phục vụ nghiên cứu và phân tích phân kỳ, tuyệt đối không dùng làm nghiệm thu đích.

### 2.4. Xuất nhật ký 6 cột `val_groups.csv`
Mỗi epoch tự động xuất các trường:
`epoch, clean_r, clean_f, q30_r, q30_f, worst, best_effort`

---

## 3. Kết Quả Kiểm Thử (DoD Verification)

Đã chạy toàn bộ 5/5 test cases trong [`tests/test_selection_metric.py`](../../../tests/test_selection_metric.py) trên local:

| Test Case | Mục tiêu kiểm chứng | Trạng thái |
| :---: | :--- | :---: |
| **T1** | Mảng giả 4 nhóm đã biết $\rightarrow$ `worst` đúng giá trị min | ✅ PASS |
| **T2** | Nhóm rỗng $\rightarrow$ `raise ValueError` (không che giấu lỗi bằng 0) | ✅ PASS |
| **T3** | `worst` tăng nhưng `clean_r = 0.85` $\rightarrow$ lưu `best_effort`, từ chối `best.pth` | ✅ PASS |
| **T4** | `worst` tăng và `clean_r = 0.92` $\rightarrow$ lưu chính thức `best.pth` | ✅ PASS |
| **T5** | Loader đơn (không dict) $\rightarrow$ fallback an toàn, không crash | ✅ PASS |

---

## 4. Bàn Giao & Kết Nối Giai Đoạn Tiếp Theo

- **Đầu ra bàn giao**: Đã hoàn tất toàn bộ logic chọn và lưu checkpoint cho vòng huấn luyện Phase 7.
- **Kết nối Task 7.11 (Thành viên C)**: Script huấn luyện `train_crossdomain.py` / `trainer.py` khi chạy trên Kaggle/Colab GPU sẽ tự động áp dụng luật worst-group mới để chọn checkpoint, bảo vệ hoàn toàn Real ACC không bị suy giảm.

