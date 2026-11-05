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

Đã chạy toàn bộ **9/9 test cases** (5 unit tests + 4 smoke integration tests) trên local CPU:

### 3.1. Unit Tests (`tests/test_selection_metric.py`)
| Test Case | Mục tiêu kiểm chứng | Trạng thái |
| :---: | :--- | :---: |
| **T1** | Mảng giả 4 nhóm đã biết $\rightarrow$ `worst` đúng giá trị min | ✅ PASS |
| **T2** | Nhóm rỗng $\rightarrow$ `raise ValueError` (không che giấu lỗi bằng 0) | ✅ PASS |
| **T3** | `worst` tăng nhưng `clean_r = 0.85` $\rightarrow$ lưu `best_effort`, từ chối `best.pth` | ✅ PASS |
| **T4** | `worst` tăng và `clean_r = 0.92` $\rightarrow$ lưu chính thức `best.pth` | ✅ PASS |
| **T5** | Loader đơn (không dict) $\rightarrow$ fallback an toàn, không crash | ✅ PASS |

### 3.2. Integration Tests (`tests/test_smoke_selection_v2_integration.py`)
| Test Case | Mục tiêu kiểm chứng | Trạng thái |
| :---: | :--- | :---: |
| **S1** | `test_val_q30_cache_and_worst_group_calculation`: Tự động sinh nhánh Q30 trong RAM và tính chuẩn 4 nhóm recall | ✅ PASS |
| **S2** | `test_dual_checkpoint_saving_best_and_best_effort`: Cơ chế phân tách lưu `best.pth` vs `best_effort.pth` | ✅ PASS |
| **S3** | `test_val_groups_csv_logging`: Ghi xuất nhật ký 7 cột `val_groups.csv` chuẩn xác | ✅ PASS |
| **S4** | `test_config_parsing_real_guard_threshold`: Nạp đúng cấu hình `real_guard_threshold: 0.90` từ file YAML Phase 7 | ✅ PASS |

---

## 4. Tích Hợp Đồng Bộ Vào Pipeline Huấn Luyện (`train_crossdomain.py`)

1. **Bộ nhớ đệm Q30 trong RAM (`val_q30_cache`)**:
   - Tự động lấy tối đa 300 mẫu phân tầng (150 Real / 150 Fake, `seed=42`) từ tập validation.
   - Nén JPEG $Q=30$ một lần duy nhất lúc khởi tạo và lưu trong RAM, loại bỏ hoàn toàn hiện tượng bottleneck CPU ở mỗi epoch.
2. **Quy tắc chọn Checkpoint Kép**:
   - `clean_r >= 0.90` (hoặc `--real_guard_threshold`) $\rightarrow$ Lưu `fatformer_anchored_v2_best.pth`.
   - `clean_r < 0.90` nhưng `worst_group` tăng $\rightarrow$ Lưu `fatformer_anchored_v2_best_effort.pth`.
3. **Nhật ký tiến trình 7 cột**:
   - Tự động ghi xuất file `val_groups.csv` với các cột: `epoch,clean_r,clean_f,q30_r,q30_f,worst,best_effort`.
   - Tự động đồng bộ sang Google Drive `/content/drive/MyDrive/Fatformer/logs` khi phát hiện môi trường Colab.

---

## 5. Bàn Giao & Mở Đường Cho Task 7.11 (Thành viên C)

- **Đầu ra bàn giao**: Hoàn tất $100\%$ logic chọn và lưu checkpoint cho toàn bộ pipeline huấn luyện.
- **Sẵn sàng cho Task 7.11**: Cả 4 nhiệm vụ tiền đề của Phase 7:
  + **Task 7.7**: Kho dữ liệu v2 (`data_manifest_v2.json`, `train_crossdomain_v2.tar` 7.78 GB trên Drive).
  + **Task 7.8**: Degradation Pipeline v2 (`degradation_v2.yaml`, WebP + Composite, benchmark overhead $-14.69\% \le 15\%$).
  + **Task 7.9**: Loss v2 (`CrossDomainTotalLoss` với `RealGuardLoss` + `FeatureConsistencyLoss`).
  + **Task 7.10**: Selection Metric v2 (Worst-Group + Checkpoint Kép + In-memory Q30 cache + `val_groups.csv`).
  Đều đã được tích hợp đồng bộ và kiểm thử khép kín thành công $100\%$, sẵn sàng thực thi huấn luyện **Anchored Recovery v2** tại Task 7.11 để chinh phục Cổng G7.2!

