# KẾ HOẠCH TRIỂN KHAI ĐỒ ÁN: TỐI ƯU HÓA ĐỘ BỀN VỮNG PHÁP Y CHO FATFORMER TRÊN GOOGLE COLAB PRO
> **Đề tài**: Nâng cao độ bền phát hiện ảnh AI (Generalizable Synthetic Image Detection) trước biến dạng nén mạng xã hội (JPEG, Blur) và mở rộng nhận diện mô hình Diffusion thế hệ mới.  
> **Kiến trúc đề xuất**: **FatFormer + SRM 3-Kernels + Dynamic Frequency Gating $\lambda(x)$** (Phương án 2).  
> **Hạ tầng thực thi**: Google Colab Pro (GPU A100 / T4, 300 Compute Units) kết hợp Google Drive 5TB nhóm (Phương án A - Shared Shortcut).  
> **Nhân sự**: Nhóm 3 thành viên (Lead A, Lead B, Lead C).  
> **Thời gian thực hiện**: 5 tuần.

---

## I. PHÂN CÔNG VAI TRÒ & TRÁCH NHIỆM KỸ THUẬT (CORE ROLES)

Để tránh tình trạng "nghẽn phụ thuộc" (Pipeline Blocking), mỗi thành viên nắm giữ một trụ cột độc lập và phối hợp qua giao diện mã nguồn chuẩn:

| Thành viên | Trụ cột kỹ thuật | Trách nhiệm cốt lõi | Tiêu chí bàn giao (Deliverables) |
| :--- | :--- | :--- | :--- |
| **Thành viên A**<br>*(Data & Infra Lead)* | **Hạ tầng I/O, Dữ liệu & Pipeline Tăng Cường** | • Cấu hình thư mục dùng chung từ Google Drive 5TB sang Colab Pro.<br>• Viết script đóng gói `.tar` và hàm `extract_to_local()` giải nén vào SSD `/content/` Colab.<br>• Thu thập tập ProGAN 4-class, GenImage subset (Midjourney, SD v1.5), và tạo 3 biến thể suy biến (JPEG, Blur, Down-Up).<br>• Xây dựng module Data Augmentation 2 luồng (Clean vs Degraded). | • Thư mục Drive 5TB chuẩn hóa.<br>• Module `datasets/transforms.py`.<br>• Script `data/make_degraded.py`.<br>• Tốc độ DataLoader đạt tối đa, 0 nghẽn I/O. |
| **Thành viên B**<br>*(Architecture & Loss Lead)* | **Kiến trúc Mô hình, SRM 3-Kernels & Cổng Gating $\lambda(x)$** | • Xây dựng mã nguồn mô hình tại `models/` nạp đủ 1,116 tensors (xử lý typo `self.patch_basaed_enhancer`).<br>• Hiện thực hóa khối `SpatialResidualBlock` chứa **3 bộ lọc SRM kinh điển** (Đạo hàm bậc 1, bậc 2, và Square $5 \times 5$).<br>• Thiết kế mạng `DynamicFrequencyGating` dự đoán hệ số thích ứng $\lambda(x)$ dựa trên mức suy biến của ảnh.<br>• Thiết lập cơ chế đóng băng 94.1% CLIP, chỉ mở khóa 5.9% Adapter. | • Module `models/fatformer.py`, `models/srm.py`, `models/gating.py`.<br>• Script `test_model_shapes.py` pass 100% tensors.<br>• Kiểm soát hàm mất mát Augmented Contrastive Loss. |
| **Thành viên C**<br>*(Training & Evaluation Lead)* | **Huấn luyện Colab Pro, Benchmark & Trực Quan Hóa XAI** | • Quản lý tài nguyên 300 Compute Units, thiết lập Notebook Colab Pro chạy GPU A100 với AMP FP16 + Gradient Accumulation.<br>• Viết script `fast_eval.py` (đánh giá nhanh trên subset 500 ảnh) và `full_eval.py` (đánh giá đủ 18 tập test).<br>• Giám sát quá trình huấn luyện 2 Pha (Adaptive $\rightarrow$ Hard Degradation), tự động backup checkpoint sau mỗi epoch.<br>• Xuất ảnh đối chứng Grad-CAM và tổng hợp báo cáo đồ án. | • Notebooks Colab hoàn chỉnh (`train.ipynb`, `eval.ipynb`).<br>• Checkpoint `fatformer_srm_robust.pth`.<br>• Bảng số liệu Benchmark đa chiều & Ablation Study.<br>• Báo cáo kỹ thuật (PDF) + Slide thuyết trình. |

---

## II. QUY CHUẨN KỸ THUẬT HẠ TẦNG (INFRASTRUCTURE & DRIVE PROTOCOL)

### 1. Cơ Chế Kết Nối 2 Tài Khoản (Phương Án A — Google Drive Shortcut)
* **Tài khoản A (Chính — 5TB Drive)**: Đóng vai trò **Data Hub & Model Registry**.
  1. Tạo thư mục gốc `FatFormer_Hub/`.
  2. Bấm **Chia sẻ (Share)** $\rightarrow$ Nhập email của Tài khoản Colab Pro $\rightarrow$ Phân quyền **Người chỉnh sửa (Editor)**.
* **Tài khoản B (Phụ — Colab Pro 300 CU + 15GB Drive)**: Đóng vai trò **Compute Engine**.
  1. Vào Google Drive của Tài khoản B $\rightarrow$ Chọn mục **"Được chia sẻ với tôi" (Shared with me)**.
  2. Click chuột phải vào `FatFormer_Hub` $\rightarrow$ Chọn **"Thêm lối tắt vào Drive" (Add shortcut to Drive)** $\rightarrow$ Lưu vào `MyDrive`.
  3. Trên Colab Pro:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     DRIVE_DIR = "/content/drive/MyDrive/FatFormer_Hub"
     ```
* **Lợi ích cốt lõi**: Tiêu thụ **0 byte** trên tài khoản 15GB, tận dụng trọn vẹn 5TB của tài khoản chính; đọc ghi checkpoint trực tiếp mượt mà.

### 2. Quy Tắc I/O Chống Nghẽn Bắt Buộc (SSD Local Extraction)
* **Tuyệt đối không** đọc từng ảnh trực tiếp từ đường dẫn `/content/drive/MyDrive/...` (sẽ gây sập I/O Colab).
* **Quy trình chuẩn**:
  1. Dữ liệu trên Drive luôn được nén thành file `.tar` (ví dụ: `genimage_subset.tar`, `test_degraded.tar`).
  2. Đầu notebook, script copy file `.tar` về `/content/` của máy ảo Colab:
     ```bash
     cp /content/drive/MyDrive/FatFormer_Hub/datasets/progan_train.tar /content/
     tar -xf /content/progan_train.tar -C /content/dataset_local/
     ```
  3. DataLoader chỉ đọc dữ liệu từ `/content/dataset_local/` (tốc độ SSD NVMe của Colab tăng gấp 10 lần).

---

## III. NÂNG CẤP KIẾN TRÚC: PHƯƠNG ÁN 2 (SRM 3-KERNELS + FREQUENCY GATING)

Nhằm nâng tầm đồ án đạt mức **Xử lý ảnh chuyên sâu**, nhóm hiện thực hóa 2 cải tiến kiến trúc:

### 1. Khối Lọc Vết Dư Không Gian SRM (`SpatialResidualBlock`)
Trước khi đưa đặc trưng vào nhánh tích chập của FAA, áp dụng 3 bộ lọc vi sai pháp y cố định (`requires_grad = False`):
* **Kernel 1 ($1^{st}$ order)**: Bắt vết nứt vi sai điểm ảnh liền kề do nội suy:
  $$K_1 = \begin{bmatrix} 0 & 0 & 0 \\ -1 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$
* **Kernel 2 ($2^{nd}$ order - Laplacian)**: Bắt độ cong vi sai điểm ảnh:
  $$K_2 = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$
* **Kernel 3 (Square $5 \times 5$)**: Bắt vết nội suy song tuyến tính (Bilinear/Bicubic):
  $$K_3 = \frac{1}{12} \begin{bmatrix} -1 & 2 & -2 & 2 & -1 \\ 2 & -6 & 8 & -6 & 2 \\ -2 & 8 & -12 & 8 & -2 \\ 2 & -6 & 8 & -6 & 2 \\ -1 & 2 & -2 & 2 & -1 \end{bmatrix}$$
* *Tác dụng*: Triệt tiêu hoàn toàn thông tin ngữ nghĩa (màu sắc con người, cảnh vật), chỉ giữ lại dấu vết nhiễu vi mô của quá trình sinh ảnh.

### 2. Cổng Thích Ứng Tần Số Động (`DynamicFrequencyGating` $\lambda(x)$)
* Thay thế hệ số $\lambda$ cố định trong bài báo gốc bằng hàm thích ứng phụ thuộc mức suy biến ảnh:
  $$\lambda(x) = \text{Sigmoid}\Big(\mathbf{W}_2 \cdot \text{GELU}\big(\mathbf{W}_1 \cdot \text{GAP}(x) + \mathbf{b}_1\big) + b_2\Big) \times 2.0$$
* *Cơ chế hoạt động*:
  * **Ảnh sạch (Clean)**: Cổng giữ $\lambda(x) \approx 1.0 \sim 1.5$ (khai thác trọn vẹn dải tần DWT).
  * **Ảnh nén sâu ($Q \le 50$) hoặc mờ (Blur)**: Tần số cao bị phá hủy thành nhiễu khối, cổng tự động hạ $\lambda(x) \rightarrow 0.05 \sim 0.1$ (đóng cổng tần số, ngăn "nhiễu độc" xâm nhập mô hình, chuyển ưu tiên 95% sang nhánh không gian SRM).

---

## IV. LỘ TRÌNH 5 TUẦN CHI TIẾT (5-WEEK SPRINT PLAN)

### 🗓️ TUẦN 1: THIẾT LẬP HẠ TẦNG, SANITY CHECK & TẠO PIPELINE DUMMY
*Mục tiêu: Hoàn tất kết nối Drive 5TB sang Colab Pro, xây dựng mã nguồn mô hình nạp khớp 100% weights và tạo pipeline dummy để 3 thành viên làm việc song song.*

| Mã việc | Nhiệm vụ kỹ thuật | Người phụ trách | Sản phẩm bàn giao (Deliverables) |
| :---: | :--- | :---: | :--- |
| **1.1** | Thiết lập thư mục `FatFormer_Hub` trên Drive 5TB, chia sẻ shortcut sang tài khoản Colab Pro. Tạo Dummy Dataset (20 ảnh) để test code. | **A** | Thư mục Drive hoạt động, notebook `00_setup_env.ipynb`. |
| **1.2** | Viết module mô hình trong thư mục `models/` (`fatformer.py`, `faa.py`, `lga.py`, `srm.py`, `gating.py`). Xử lý typo `self.patch_basaed_enhancer`. Chạy `tools/check_weights.py` nạp pass `strict=True` trên CPU/T4. | **B** | Mã nguồn mô hình hoàn chỉnh, script `verify_model.py` pass 1,116 tensors. |
| **1.3** | Thiết lập notebook huấn luyện `train.ipynb` trên Colab Pro (GPU A100) với AMP FP16 và Gradient Accumulation. Chạy thử 1 step dummy forward-backward-optimizer. | **C** | Notebook `train.ipynb` chạy mượt mà trên A100, đo thời gian 1 step. |
| **1.4** | Xây dựng script `fast_eval.py`: Lấy ngẫu nhiên 500 ảnh/tập test để đánh giá nhanh trong < 8 phút thay vì chạy cả ngày. | **C** | Script `fast_eval.py` kèm các hàm đo ACC, AP, AUC. |

---

### 🗓️ TUẦN 2: XÂY DỰNG BỘ TEST DEGRADED & THIẾT LẬP BASELINE GỐC
*Mục tiêu: Đo đạc số liệu của mô hình gốc trên cả ảnh Clean và ảnh Degraded để làm mốc đối chứng khoa học.*

| Mã việc | Nhiệm vụ kỹ thuật | Người phụ trách | Sản phẩm bàn giao (Deliverables) |
| :---: | :--- | :---: | :--- |
| **2.1** | Viết script `data/make_degraded.py` sinh 3 biến thể suy biến: (1) JPEG ($Q \in \{30, 50, 70\}$), (2) Gaussian Blur ($\sigma \in \{1.0, 2.0\}$), (3) Down-Up ($224 \rightarrow 112 \rightarrow 224$). Đóng gói thành `test_degraded.tar`. | **A** | Tệp dữ liệu kiểm thử suy biến chuẩn hóa trên Drive 5TB. |
| **2.2** | Chạy `fast_eval.py` và `full_eval.py` với Checkpoint gốc trên tập **Clean** (8 GANs + 6 Diffusion). | **C** | Bảng số liệu Baseline Clean (tái lập GANs $\approx 98.4\%$, Diffusion $\approx 95.0\%$). |
| **2.3** | Chạy benchmark Checkpoint gốc trên tập **Degraded** ($Q=30, 50, 70$). Ghi nhận mức độ sụt giảm hiệu năng nghiêm trọng của mô hình gốc. | **C** | Bảng số liệu Baseline Degraded (chứng minh tử huyệt của FatFormer gốc khi gặp nén). |
| **2.4** | Trích xuất Grad-CAM ban đầu trên 5 cặp ảnh Real/Fake ở cả hai trạng thái Clean và Degraded. | **B** | Script `visualize_cam.py` + Bộ ảnh Grad-CAM đối chứng ban đầu. |
| **2.5** | **Họp nhóm Milestone 1**: Chốt bảng số liệu Baseline, xác định mục tiêu bù đắp sụt giảm hiệu năng cho giai đoạn finetune. | **Cả 3** | Biên bản kỹ thuật Milestone 1. |

---

### 🗓️ TUẦN 3: TỔNG HỢP DỮ LIỆU ĐA NGUỒN & NÂNG CẤP KIẾN TRÚC MÔ HÌNH
*Mục tiêu: Đưa dữ liệu sinh hiện đại vào tập train và tích hợp hoàn chỉnh SRM 3-Kernels + Cổng Gating.*

| Mã việc | Nhiệm vụ kỹ thuật | Người phụ trách | Sản phẩm bàn giao (Deliverables) |
| :---: | :--- | :---: | :--- |
| **3.1** | Tải và đóng gói tập con **GenImage** (Midjourney + SD v1.5, khoảng 25k ảnh) đưa lên Drive 5TB, kết hợp với ProGAN 4-class gốc tạo tập train hỗn hợp `train_mixed.tar`. | **A** | Bộ dữ liệu huấn luyện hỗn hợp chuẩn hóa trên Drive. |
| **3.2** | Xây dựng pipeline Data Augmentation 2 luồng trong `datasets/transforms.py`: 30% ảnh Clean, 70% ảnh áp dụng ngẫu nhiên JPEG ($Q \in [40, 95]$) và Gaussian Blur. | **A** | Module `datasets/transforms.py` chống học vẹt nén. |
| **3.3** | Tích hợp module `SpatialResidualBlock` (3 kernels) và `DynamicFrequencyGating` vào class `FatFormer` trong `models/fatformer.py`. | **B** | Kiến trúc hoàn thiện của Phương Án 2, sẵn sàng nhận gradient. |
| **3.4** | Thiết lập cấu hình đóng băng 94.1% CLIP ViT-L/14, chỉ mở khóa 5.9% tham số (FAA, SRM projection, Gating, LGA, soft prompts). | **B** | File cấu hình `train_config.yaml`. |
| **3.5** | Thiết lập hook tự động lưu checkpoint `.pth` sau mỗi epoch về Drive 5TB và cơ chế `--resume` phòng ngừa Colab ngắt kết nối. | **C** | Checkpoint auto-saving pipeline hoàn thiện. |

---

### 🗓️ TUẦN 4: CHIẾN DỊCH HUẤN LUYỆN 2 PHA TRÊN A100 (COLAB PRO)
*Mục tiêu: Huấn luyện mô hình đạt trạng thái hội tụ tối ưu, vừa bền trước nén vừa không tụt hiệu năng ảnh Clean.*

| Mã việc | Nhiệm vụ kỹ thuật | Người phụ trách | Sản phẩm bàn giao (Deliverables) |
| :---: | :--- | :---: | :--- |
| **4.1** | **Huấn luyện Pha 1 (Adaptive Generalization)**: Chạy 4 epoch trên GPU A100 (tiêu tốn ~35 CU). Batch size = 32, Accumulation = 2 (Effective = 64). LR adapter = $2 \times 10^{-4}$, LR gating = $10^{-3}$. Áp dụng biến dạng nén nhẹ. | **B + C** | Checkpoint `fatformer_srm_phase1.pth` + Log hội tụ Loss. |
| **4.2** | **Huấn luyện Pha 2 (Hard Degradation Hardening)**: Chạy 4 epoch tiếp theo trên A100 (~35 CU). Tăng tỷ lệ nén sâu ($Q \in [30, 60]$), giảm LR xuống $5 \times 10^{-5}$ để tinh chỉnh mịn. | **B + C** | Checkpoint hoàn thiện `fatformer_srm_robust_final.pth`. |
| **4.3** | Giám sát và vẽ biểu đồ phân bố giá trị đầu ra của cổng thích ứng $\lambda(x)$ trên các dải chất lượng $Q=30, 50, 70, 100$. | **B** | Biểu đồ chứng minh cơ chế tự động đóng cổng tần số khi ảnh nén sâu. |
| **4.4** | Chạy `fast_eval.py` sau mỗi epoch để phát hiện sớm overfitting. | **C** | Biểu đồ Validation ACC/AP theo từng epoch. |

---

### 🗓️ TUẦN 5: BENCHMARK ĐA CHIỀU, XAI GRAD-CAM & ĐÓNG GÓI ĐỒ ÁN
*Mục tiêu: Chạy kiểm thử toàn diện, hoàn thiện bảng so sánh Ablation Study đạt điểm 10 và đóng gói sản phẩm.*

| Mã việc | Nhiệm vụ kỹ thuật | Người phụ trách | Sản phẩm bàn giao (Deliverables) |
| :---: | :--- | :---: | :--- |
| **5.1** | Chạy **Full-Benchmark** trên toàn bộ 18 tập test ở cả 2 trạng thái Clean và Degraded ($Q \in \{30, 50, 70\}$). | **C** | Bảng ma trận kết quả tổng hợp cuối cùng. |
| **5.2** | **Lập bảng Ablation Study 4 phiên bản**: (1) Baseline gốc, (2) Augmentation thuần, (3) Thêm SRM 3 Kernels, (4) Mô hình hoàn chỉnh (+ Dynamic Gating). | **B + C** | Bảng phân tích triệt tiêu chứng minh giá trị từng module cải tiến. |
| **5.3** | Xuất bộ ảnh trực quan hóa **Grad-CAM so sánh đối đầu**: Chứng minh mô hình cải tiến tập trung chuẩn xác vào dị thường tạo tác AI thay vì viền khối nén JPEG. | **C** | Bộ ảnh XAI độ phân giải cao đưa vào báo cáo. |
| **5.4** | Soạn thảo Báo cáo kỹ thuật đồ án (PDF chuẩn IEEE/CVPR) và Slide thuyết trình. | **Cả 3** | Báo cáo đồ án hoàn chỉnh + Slide bảo vệ. |
| **5.5** | Xây dựng script `inference_demo.py` hỗ trợ kéo ảnh từ Facebook/Telegram chạy suy luận trực tiếp để demo trước hội đồng. | **A + B** | Ứng dụng Demo suy luận thực chiến. |

---

## V. BẢNG TIÊU CHUẨN NGHIỆM THU ĐỒ ÁN (DONE CRITERIA)

Dự án được đánh giá là hoàn thành xuất sắc khi đạt đủ 7 tiêu chuẩn sau:
- [ ] Khắc phục 100% nghẽn I/O qua cơ chế giải nén SSD Colab và liên kết Drive 5TB không tốn dung lượng tài khoản phụ.
- [ ] Bảng số liệu Baseline gốc tái lập trên cả 2 miền Clean và Degraded ($Q=30, 50, 70$).
- [ ] Hai checkpoint chất lượng cao: `fatformer_srm_phase1.pth` và `fatformer_srm_robust_final.pth`.
- [ ] Bảng số liệu đối chứng chứng minh độ bền tăng vượt bậc ($> 15–20\%$ trên ảnh nén nặng $Q=30$) trong khi độ chính xác trên ảnh Clean không bị sụt giảm quá $0.5\%$.
- [ ] Bảng Ablation Study định lượng rõ rệt công năng của SRM 3-Kernels và Dynamic Frequency Gating $\lambda(x)$.
- [ ] Bộ ảnh trực quan hóa Grad-CAM phân giải cao minh chứng tính giải thích được (XAI).
- [ ] Kho mã nguồn sạch sẽ, kèm script `inference_demo.py` và tài liệu hướng dẫn tái lập kết quả từ đầu (`README.md`).

---
*Kế hoạch này là tài liệu kỹ thuật chuẩn của nhóm, phân công rõ trách nhiệm và lộ trình thực thi theo từng mốc thời gian.*