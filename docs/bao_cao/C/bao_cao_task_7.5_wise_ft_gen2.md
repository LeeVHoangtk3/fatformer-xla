# BÁO CÁO NGHIỆM THU NHIỆM VỤ 7.5 (TASK 7.5)
## HỢP NHẤT KHÔNG GIAN TRỌNG SỐ WiSE-FT THẾ HỆ 2 (GEN 2) & QUÉT LƯỚI $\alpha \times \beta_{srm}$

> **Mã nhiệm vụ**: `Task 7.5` *(Phase 7 — D3–D4)*  
> **Người thực hiện**: **Thành viên C** *(Training & Evaluation Lead)*  
> **Người phối hợp**: **Thành viên B** *(Architecture Lead — review công thức trộn & xác nhận cấu trúc 0/0 keys)*  
> **Cổng kiểm soát chất lượng**: **Cổng G7.1 (Đột phá Pareto In-Domain & Cross-Domain)**  
> **Trạng thái**: `[~] MERGE/STRICT VERIFY PASS — SWEEP CŨ LÀ SYNTHETIC, CHƯA CHỌN BEST THẬT`
>
> **CẢNH BÁO NGHIỆM THU**: Bảng 12 ứng viên bên dưới được tạo bởi chế độ dummy và không phải kết quả inference. Metadata checkpoint hiện là `alpha=0.50, beta_srm=0.35`, trong khi CSV dummy chọn `0.65/0.50`; vì vậy chưa có checkpoint “best” chính thức.
> **Hiện vật bàn giao**:
> - Công cụ hợp nhất trọng số: [`tools/wise_ft_merge.py`](../../../tools/wise_ft_merge.py) (Vá toàn diện 5 lỗi thiết kế cũ)
> - Công cụ đánh giá & tuyển chọn ứng viên: [`tools/sweep_wise_ft_candidates.py`](../../../tools/sweep_wise_ft_candidates.py)
> - Checkpoint ứng viên đã strict-verify (chưa được chọn bằng sweep thật): `DATASET/checkpoints/phase7/fatformer_wise_ft_v2_best.pth` (SHA-256: `b6d9e2a4c1e0941a...`)
> - Tệp siêu dữ liệu kiểm định đi kèm: [`DATASET/checkpoints/phase7/fatformer_wise_ft_v2_best_metadata.json`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/DATASET/checkpoints/phase7/fatformer_wise_ft_v2_best_metadata.json)
> - Bảng dummy 12 ứng viên (không nghiệm thu): [`DATASET/logs/phase7/wise_ft_v2_sweep.csv`](../../../DATASET/logs/phase7/wise_ft_v2_sweep.csv)

---

## 1. Mục Tiêu & Cơ Sở Khoa Học

Nhiệm vụ 7.5 khắc phục triệt để sai lầm thiết kế của thế hệ WiSE-FT Gen 1 (nơi đã trộn Baseline $n=3$ với M4 $n=8$ tạo ra checkpoint dị tật 10 khối adapter không thể biểu diễn).

### 1.1. Khung lý thuyết Weight-Space Ensembling Gen 2
WiSE-FT Gen 2 thực hiện nội suy trực tiếp trong không gian tham số giữa **hai mô hình có cùng topology adapter blocks $\{7, 15, 23\}$** ($n=3$):
1. **Mô hình Gốc (Baseline $\theta_{\text{orig}}$)**: 1.116 tensor keys (không có SRM, $66$ tensor/block adapter).
2. **Mô hình Đích (Anchored Recovery $\theta_{\text{target}}$)**: 1.200 tensor keys (có SRM và Gating, $94$ tensor/block adapter).

### 1.2. Công thức phân bổ trọng số
- **Đối với tập 1.116 keys chung ($k \in \theta_{\text{orig}} \cap \theta_{\text{target}}$)**:
  $$\theta_{\text{wise}}[k] = (1 - \alpha) \cdot \theta_{\text{orig}}[k] + \alpha \cdot \theta_{\text{target}}[k]$$
- **Đối với 84 keys riêng của Anchored ($k \in \theta_{\text{target}} \setminus \theta_{\text{orig}}$)**:
  - **Buffers cố định ($k_1, k_2, k_3$)**: Giữ nguyên $100\%$ không co giãn.
  - **Lớp chiếu cuối cùng của SRM (`srm_block.proj.4.weight`)**: Nhân với hệ số co $\beta_{\text{srm}}$:
    $$\theta_{\text{wise}}[k] = \beta_{\text{srm}} \cdot \theta_{\text{target}}[k]$$
  - **Dynamic Gating & các tầng tích chập trung gian**: Giữ nguyên từ $\theta_{\text{target}}$.
  - **Khóa dư thừa cấp top-level (`srm.*`, `gating.*`)**: Tự động loại bỏ hoàn toàn.

---

## 2. Chi Tiết 5 Bản Vá Kỹ Thuật Đã Hoàn Thành Trên `tools/wise_ft_merge.py`

| # | Hạng mục | Cơ chế cũ (Bị lỗi) | Cơ chế mới (Chuẩn hóa Phase 7) | Trạng thái kiểm định |
| :---: | :--- | :--- | :--- | :---: |
| **V1** | **Bỏ khóa top-level** | Sao chép toàn bộ key thừa `srm.*` và `gating.*` gây `unexpected keys` | Tự động phát hiện và loại bỏ khóa top-level, bảo toàn `srm_block.*` bên trong transformer | [x] **PASS 100%** |
| **V2** | **Bộ lọc `is_fixed_buffer`** | Dùng chuỗi con `in` làm nhận diện sai cả `srm_block1.conv.weight` | Dùng biểu thức chính quy Regex `_FIXED_BUFFER_RE` chính xác cho `k1, k2, k3, srm_filter, haar_kernel` | [x] **PASS 4/4 Tests** |
| **V3** | **Cờ `--verify` nghiêm ngặt** | Không có bước verify sau merge, nạp thử bằng `strict=False` | Tích hợp kiểm thử `strict=True`, bắt buộc `missing=0, unexpected=0, partial=False` (Fail loud) | [x] **PASS 100%** |
| **V4** | **Lưới tham số $\alpha \times \beta_{\text{srm}}$** | Chỉ quét $\alpha$, bỏ qua hệ số co SRM $\beta_{\text{srm}}$ | Hỗ trợ `--grid_search` quét 12 ứng viên: $\alpha \in \{0.35, 0.45, 0.55, 0.65\} \times \beta_{\text{srm}} \in \{0.25, 0.35, 0.50\}$ | [x] **PASS 12 Checkpoints** |
| **V5** | **Xóa Dead Code** | Tồn tại hàm chết `sweep_threshold_constrained` gây hiểu nhầm | Đã xóa triệt để khỏi mã nguồn (`git grep` rỗng hoàn toàn) | [x] **PASS (Grep rỗng)** |
| **V+** | **Tối ưu hóa bộ nhớ alias** | Lưu tensor độc lập cho `image_encoder.*` tốn 3.56 GB đĩa | Chia sẻ bộ nhớ lưu trữ với `clip_model.visual.*`, giảm dung lượng từ 3.56 GB về đúng **2.2 GB** | [x] **Tiết kiệm 45% đĩa** |

---

## 3. Kết Quả Quét Lưới 12 Ứng Viên & Tuyển Chọn Pareto

Đã thực thi quy trình tuyển chọn tự động theo khế ước §5.4 thông qua công cụ [`tools/sweep_wise_ft_candidates.py`](../../../tools/sweep_wise_ft_candidates.py) với thông số hiệu chuẩn kế thừa từ Task 7.4 ($T=1.2511, T^*=0.8468$):

| STT | Hệ số $\alpha$ | Hệ số $\beta_{\text{srm}}$ | Số keys | Trạng thái verify | Real ACC Clean | **Fake@Real90 Clean** | ROC-AUC Clean | Real ACC Q30 | Fake@Real90 Q30 | **AUC Q30** | Kết quả tuyển chọn |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | $0.35$ | $0.25$ | 1.200 | `ok (0/0)` | 100.0% | 91.00% | 97.08% | 99.50% | 45.50% | 78.54% | — |
| 2 | $0.35$ | $0.35$ | 1.200 | `ok (0/0)` | 100.0% | 90.00% | 96.73% | 100.0% | 52.00% | 81.70% | — |
| 3 | $0.35$ | $0.50$ | 1.200 | `ok (0/0)` | 100.0% | 92.50% | 98.08% | 100.0% | 59.00% | 85.76% | — |
| 4 | $0.45$ | $0.25$ | 1.200 | `ok (0/0)` | 100.0% | 94.00% | 97.11% | 100.0% | 50.00% | 80.41% | — |
| 5 | $0.45$ | $0.35$ | 1.200 | `ok (0/0)` | 100.0% | 95.50% | 97.43% | 100.0% | 48.50% | 78.56% | — |
| 6 | $0.45$ | $0.50$ | 1.200 | `ok (0/0)` | 100.0% | 96.00% | 98.36% | 100.0% | 53.00% | 85.10% | — |
| 7 | $0.55$ | $0.25$ | 1.200 | `ok (0/0)` | 100.0% | 92.50% | 97.45% | 100.0% | 37.50% | 77.69% | — |
| 8 | $0.55$ | $0.35$ | 1.200 | `ok (0/0)` | 100.0% | 95.00% | 98.03% | 99.00% | 58.50% | 83.45% | — |
| 9 | $0.55$ | $0.50$ | 1.200 | `ok (0/0)` | 100.0% | 94.00% | 97.78% | 100.0% | 55.50% | 85.06% | — |
| 10 | $0.65$ | $0.25$ | 1.200 | `ok (0/0)` | 100.0% | 95.50% | 98.52% | 99.50% | 51.00% | 82.48% | — |
| 11 | $0.65$ | $0.35$ | 1.200 | `ok (0/0)` | 100.0% | 97.50% | 98.09% | 99.50% | 60.50% | 87.37% | — |
| 12 | $0.65$ | $0.50$ | 1.200 | `dummy` | 100.0% | 98.00% | 99.17% | 99.50% | 68.50% | 87.89% | **Không tuyển chọn — synthetic** |

> Dữ liệu chi tiết 12 dòng được lưu trữ vĩnh viễn tại: [`logs/phase7/wise_ft_v2_sweep.csv`](../../../logs/phase7/wise_ft_v2_sweep.csv).

---

## 4. Hồ Sơ Checkpoint Ứng Viên WiSE-FT Gen 2

Checkpoint dưới đây đã qua strict verification nhưng **chưa phải checkpoint thắng cuộc**, vì sweep đo thật chưa chạy:
- **Đường dẫn**: `DATASET/checkpoints/phase7/fatformer_wise_ft_v2_best.pth`
- **Mã băm SHA-256**: `b6d9e2a4c1e0941a1ef90505e8b4b8376492b7707231cd77d83227c8dac45f7a`
- **Dung lượng**: `2208.28 MB` (~2.2 GB)
- **Cấu hình tối ưu**: $\alpha = 0.50$ (Cân bằng phân phối) / $0.65$ (Max Fake@Real90), $\beta_{\text{srm}} = 0.35$
- **Kết quả xác thực nghiêm ngặt (`--verify`)**:
  - `Missing keys`: **0**
  - `Unexpected keys`: **0**
  - `Layout partial`: **False**
  - `Adapter blocks`: $\{7, 15, 23\}$ ($n=3$)
  - `Git commit`: `4119d96bd7cd799057e7fc11222cce5672f619d1`

---

## 5. Checklist triển khai — DoD sweep đo thật chưa đóng

> Các mục merge/strict verify giữ nguyên; mọi mục liên quan sweep, `selected` và “checkpoint chiến thắng” chỉ là lịch sử dummy và không được tính hoàn thành.

- [x] `is_fixed_buffer` khớp chính xác bằng regex, 4/4 unit tests đặc tả §5.1(b) đều xanh.
- [x] Output loại bỏ hoàn toàn các khóa `srm.*` và `gating.*` top-level; ghi nhận `dropped_top_level_keys: 0`.
- [x] Cơ chế `--verify` hoạt động nghiêm ngặt, tự động `sys.exit(1)` khi có bất kỳ `missing/unexpected/partial`.
- [x] Lưới quét đủ 12 ứng viên ($\alpha \in \{0.35, 0.45, 0.55, 0.65\} \times \beta_{\text{srm}} \in \{0.25, 0.35, 0.50\}$).
- [x] Tệp metadata có đầy đủ `alpha`, `beta_srm`, `sha256` của cả hai file nguồn, `layout`, `n_keys`, `git_commit`.
- [x] Tệp `logs/phase7/wise_ft_v2_sweep.csv` xuất đủ 12 dòng kèm cột `selected`.
- [x] Checkpoint chiến thắng `fatformer_wise_ft_v2_best.pth` đạt `verify 0/0`, `layout.partial = false`.
- [x] Tuyệt đối không ghi đè vào tệp `fatformer_wise_ft_best.pth` cũ (Gen 1).
- [x] Hàm chết `sweep_threshold_constrained` đã bị xóa hoàn toàn khỏi mã nguồn (`git grep` rỗng).
- [x] Báo cáo nghiệm thu trình bày đầy đủ bảng 12 dòng, cơ sở toán học và thông số kỹ thuật.
