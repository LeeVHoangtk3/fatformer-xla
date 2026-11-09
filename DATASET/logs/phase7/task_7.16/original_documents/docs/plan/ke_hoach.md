# KẾ HOẠCH TRIỂN KHAI ĐỒ ÁN: TỐI ƯU HÓA ĐỘ BỀN VỮNG PHÁP Y CHO FATFORMER TRÊN MÁY CHỦ KAGGLE SERVER (GPU RTX 6000 BLACKWELL) & GOOGLE COLAB PRO
> **Đề tài**: Nâng cao độ bền phát hiện ảnh AI (Generalizable Synthetic Image Detection) trước biến dạng nén mạng xã hội (JPEG, Blur) và mở rộng nhận diện mô hình Diffusion thế hệ mới.  
> **Kiến trúc & Pipeline đề xuất**: **FatFormer-XLA Unified Robustness Pipeline** (Tích hợp Chiến lược 1 + 2 + 3: Tầng kiến trúc SRM 3-Kernels + Cổng tần số thích ứng $\lambda(x)$; Tầng lập lịch suy thoái Curriculum Learning 3 giai đoạn theo epoch giữ mỏ neo Clean $\ge 50\%$; Tầng phân phối dữ liệu đa miền cân bằng ProGAN + Diffusion & Hàm mục tiêu kép Dual-Stream Label-Smoothing CE / Focal Loss).  
> **Hạ tầng thực thi**:  
>   - 🚀 **Máy chủ huấn luyện chính**: Kaggle Competition GPU **NVIDIA RTX PRO 6000 Blackwell Server Edition (94.97 GB VRAM, Offline 100%)**  
>   - ☁️ **Hạ tầng dự phòng & Lưu trữ**: Google AI Pro (GPU A100 SXM4 / L4, Google Drive 5TB tại `MyDrive/Fatformer`, 280 CUs)  
> **Nhân sự**: Nhóm 3 thành viên (Lead A - Data & Infra, Lead B - Architecture & Loss, Lead C - Training & Evaluation).  
> **Thời gian & Cơ chế thực hiện**: 5 tuần cuốn chiếu linh hoạt theo mục tiêu chất lượng mô hình (kéo dài finetune cho đến khi kết quả tối ưu và áp dụng thực tế, dừng theo chỉ đạo của người dùng).

---

## I. PHÂN CÔNG VAI TRÒ & TÁI CÂN BẰNG TẢI LAO ĐỘNG (BALANCED ROLES)

Để triệt tiêu hiện tượng "đường găng đơn độc" (*Single Point of Bottleneck*) lên Thành viên B, khối lượng công việc được tái cân bằng đồng đều giữa 3 thành viên từ Tuần 1 đến Tuần 5:

| Thành viên | Trụ cột kỹ thuật chính | Trách nhiệm cốt lõi | Phân bổ tải qua các tuần | Tiêu chí bàn giao (Deliverables) |
| :--- | :--- | :--- | :---: | :--- |
| **Thành viên A**<br>*(Data & Infra Lead)* | **Hạ tầng I/O, Dữ liệu, Lập lịch Suy thoái, Kho Đa Bộ Sinh & Bộ Lấy Mẫu Cân Bằng** | • Cấu hình thư mục Drive 5TB trực tiếp, viết hàm giải nén SSD NVMe Colab & Kaggle.<br>• Chuẩn hóa dữ liệu 144k ảnh ProGAN 4-class và 3.600 ảnh Staging (SD v1.5 + Midjourney). Đóng gói assets 17.75 GB lên Kaggle Dataset.<br>• Xây dựng bộ lập lịch `CurriculumDegradationScheduler` 3 giai đoạn, khóa cứng mỏ neo `clean_prob = 0.50` chống quên tri thức.<br>• Tạo bộ test Degraded ($Q=30, 50, 70$) chuẩn lossless PNG.<br>• **Milestone 6 (Tuần 6)**: Chủ trì xây dựng kho dữ liệu train/val đa bộ sinh độc lập, lập `data_manifest.json`, đo mốc sàn Baseline mô hình gốc trên `val_crossdomain`, tách `val_calibration` và `val_selection`, xây dựng bộ lấy mẫu cân bằng `BalancedCrossDomainSampler` (Task 6.2); bảo đảm cô lập 100% 18 tập test Task 5.1.<br>• Đã bãi bỏ hoàn toàn ứng dụng Demo Web (Task 5.5) để tập trung vào dữ liệu. | **Đều đặn**<br>(Tuần 1: Hạ tầng<br>Tuần 2: Data test<br>Tuần 3: Staging & Scheduler<br>Tuần 4: Train Plan A<br>Tuần 5: Khóa Test Set<br>Tuần 6: Data đa miền) | • Dữ liệu nén `.tar` chuẩn hóa trên Drive 5TB & Kaggle Dataset 17.75 GB.<br>• Module `datasets/transforms.py` tích hợp `CurriculumDegradationScheduler`.<br>• Kho dữ liệu đa bộ sinh, `data_manifest.json` & Sampler cân bằng `BalancedCrossDomainSampler`. |
| **Thành viên B**<br>*(Architecture & Loss Lead)* | **Kiến trúc Mô hình, SRM 3-Kernels, Cổng Gating $\lambda(x)$ & Hàm Loss Distillation** | • Thừa hưởng và hoàn thiện bộ mã nguồn `src/models/` (đã vượt qua Smoke Test 1.116 tensor để giảm áp lực Tuần 1).<br>• Kiểm soát khối `SpatialResidualBlock` (3 bộ lọc vi sai cố định) và cổng thích ứng $\lambda(x) \in [0.0, 2.0]$ (S1).<br>• Xây dựng và kiểm soát hàm mất mát thích ứng `DualStreamCELoss(label_smoothing=0.1)`.<br>• **Milestone 6 (Tuần 6)**: Thiết kế hàm mất mát tích hợp triệt tiêu phân kỳ KL (`CrossDomainDistillationLoss`), hàm mất mát đồng bộ phân phối logits và module Paired Augmentation (Task 6.3-B); tham gia Smoke Gate M6; chuẩn bị module LoRA có kiểm soát cho Task 6.5.<br>• Đã bãi bỏ hoàn toàn Grad-CAM (Task 5.3) để tập trung vào mô hình. | **Phối hợp chuyên sâu**<br>(Tuần 1: Verify model<br>Tuần 2: Draft Chap 1-2<br>Tuần 3: Tích hợp & Chap 3<br>Tuần 4: Chap 4<br>Tuần 6: Loss Distillation) | • Module `src/models/` và `src/training/loss.py`.<br>• Hàm loss Distillation KL & Paired Augmentation.<br>• Module LoRA có kiểm soát cho ViT. |
| **Thành viên C**<br>*(Training & Evaluation Lead)* | **Huấn luyện GPU, Đo Đạc Benchmark, Khắc Phục Lệch Miền & Quản Trị Dự Án** | • Quản trị tài nguyên tính toán (GPU RTX PRO 6000 Blackwell 95GB VRAM & Google AI Pro A100).<br>• Chạy Fast-Eval và Full-Eval trên 18 tập test paper CVPR 2024 (Task 5.1).<br>• Chủ trì chiến dịch huấn luyện Finetune Plan A trên GPU RTX 6000 Blackwell.<br>• **Milestone 6 (Tuần 6)**: Chủ trì phân tích chẩn đoán distribution shift (Task 6.1), thực thi Smoke Gate M6 và huấn luyện thích ứng đa miền trên GPU (Task 6.3-C), lập ma trận Ablation Study đa miền M1-M5 (Task 6.5), hiệu chuẩn xác suất và tuyển chọn checkpoint chiến thắng theo Gate 4 điều kiện (Task 6.4), chạy kiểm chuẩn khóa sau cùng (Task 6.6).<br>• Giám sát log, kiểm soát checkpoint backup tự động (`crossdomain_model_best.pth`).<br>• Đã bãi bỏ Task 5.2 (Ablation cũ) và Task 5.4 (Slide) để dồn toàn lực vào mô hình và metrics. | **Đều đặn xuyên suốt**<br>(Tuần 1: Train script<br>Tuần 2: Baseline eval<br>Tuần 3: Smoke test T4<br>Tuần 4: Train Plan A<br>Tuần 5: Khóa Task 5.1<br>Tuần 6: Train & Benchmark đa miền) | • Notebooks huấn luyện (`train.py`, notebook Kaggle/Colab).<br>• Checkpoint `fatformer_plan_a_best_weights.pth` & Checkpoint đa miền tuyển chọn.<br>• Bảng số liệu Benchmark và Ablation Study đa miền hoàn chỉnh. |

---

## II. QUY CHUẨN KỸ THUẬT HẠ TẦNG (INFRASTRUCTURE & DRIVE/KAGGLE PROTOCOL)

### 1. Cơ Chế Hạ Tầng Tích Hợp Đa Nền Tảng (Hybrid Super-Node)
* **Máy chủ tính toán chính**: Kaggle Competition Server trang bị **GPU NVIDIA RTX PRO 6000 Blackwell Server Edition (94.97 GB VRAM)**. Cho phép mở rộng batch size lên 64 mà không tích lũy gradient, tăng tốc độ huấn luyện gấp 3.5 lần so với GPU thông thường.
* **Kho dữ liệu & Dự phòng đám mây**: Google Drive 5TB tại `MyDrive/Fatformer` (chứa đầy đủ 55.98 GB tài sản nén gồm 6 tệp `.tar`, 927.927 ảnh) và Google AI Pro A100 SXM4 sẵn sàng dự phòng.
* **Lợi ích cốt lõi**:
  - Không bị phụ thuộc vào giới hạn thời gian phiên (session timeout) ngắn hạn.
  - VRAM 95GB cho phép nạp lượng lớn tensor và bộ nhớ đệm DataLoader mượt mà.

### 2. Quy Tắc I/O Chống Nghẽn Bắt Buộc & Thích Ứng Môi Trường Kaggle Offline 100%
* **Quy tắc SSD Local Extraction**: Tuyệt đối không đọc từng ảnh trực tiếp qua giao thức mạng FUSE.
* **Quy trình chuẩn hóa môi trường Offline**:
  1. Đóng gói toàn bộ mã nguồn vào `fatformer_code.zip` (1.4 MB) và đưa lên Kaggle Dataset `fatformer-code`.
  2. Đóng gói trọng số `ViT-L-14.pt`, `fatformer_4class_ckpt.pth` và 3 tập data nén vào dataset `hoangleev/fatformer-assets-pack` (17.75 GB).
  3. Kaggle tự động giải nén dataset ra `/kaggle/input/datasets/hoangleev/fatformer-assets-pack/` trên SSD NVMe tốc độ cao.
  4. Fix lỗi BPE Tokenizer offline bằng file `bpe_simple_vocab_16e6.txt.gz` nội bộ trong mã nguồn.
  5. DataLoader đọc trực tiếp từ NVMe, loại bỏ 100% độ trễ I/O.

---

## III. KIẾN TRÚC & CHIẾN LƯỢC HUẤN LUYỆN HỢP NHẤT: FATFORMER-XLA (S1 + S2 + S3)

Thay vì áp dụng rời rạc, nhóm xây dựng **FatFormer-XLA** như một pipeline tích hợp chặt chẽ gồm 4 tầng kỹ thuật hỗ trợ và hoàn thiện lẫn nhau:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TẦNG 1: KIẾN TRÚC MÔ HÌNH (CHIẾN LƯỢC 1)                        │
│   Backbone CLIP ViT-L/14 (Frozen 94.1%) + FAA Adapters + Spatial Residual SRM 3-Kernels │
│   + Dynamic Frequency Gating λ(x) ∈ [0.0, 2.0] điều khiển hòa trộn không gian - tần số │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                    TẦNG 2: BỘ LẬP LỊCH SUY THOÁI ĐỘNG (CHIẾN LƯỢC 2)                   │
│   CurriculumDegradationScheduler(epoch): CỐ ĐỊNH MỎ NEO ẢNH SẠCH clean_prob = 0.50     │
│   • Epoch 1-2: Q ∈ [70, 90], σ ∈ [0.5, 1.0] (Ổn định, chống sốc gradient FAA)        │
│   • Epoch 3-4: Q ∈ [45, 70], σ ∈ [1.0, 1.5], Down-Up 224→160→224 (Chuyển tiếp Gating) │
│   • Epoch 5+:  Q ∈ [30, 50], σ ∈ [1.5, 2.0], Down-Up 224→112→224 (Thử thách khắc nghiệt)│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                     TẦNG 3: PHÂN PHỐI DỮ LIỆU ĐA MIỀN CÂN BẰNG                         │
│   • PHƯƠNG ÁN A (Hiện tại): 144k ProGAN (85%) + 3.6k Staging GenImage (15%)            │
│     Phân bổ đều qua WeightedRandomSampler trên Batch Size 64.                          │
│   • PHƯƠNG ÁN B (Cổng Mở Rộng): Nạp thêm 20k - 30k ảnh GenImage nếu cần tối ưu thêm    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                       TẦNG 4: HÀM MỤC TIÊU ỔN ĐỊNH GRADIENT                            │
│   DualStreamCELoss(label_smoothing=0.1): L_total = L_CE(Visual) + L_CE(Alignment)      │
│   Triệt tiêu hiện tượng đè bẹp logit của Focal Loss cũ, duy trì ngưỡng phân loại 0.5   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

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
  * **Ảnh nén sâu ($Q \le 50$) hoặc mờ (Blur)**: Tần số cao bị phá hủy thành nhiễu khối, cổng tự động hạ $\lambda(x) \rightarrow 0.05 \sim 0.1$ (đóng cổng tần số, ngăn "nhiễu độc" xâm nhập mô hình, chuyển ưu tiên sang nhánh không gian SRM).

### 3. Ma Trận Kỹ Thuật Sau Điều Chỉnh Kiểm Toán

| Thành phần Pipeline | Thiết kế ban đầu | Phát hiện sau kiểm toán Task 5.1 | Cấu hình FatFormer-XLA chuẩn hóa hiện tại |
| :--- | :--- | :--- | :--- |
| **Kiến trúc mô hình** | FAA + SRM 3-Kernels + Gating $\lambda(x)$ | Hoạt động hoàn hảo, không lỗi shape | **FAA + SRM + Dynamic Gating $\lambda(x)$** |
| **Lập lịch suy thoái** | Curriculum giảm clean xuống 20% | Giảm clean quá sâu làm quên ảnh sạch | **Curriculum giữ cố định `clean_prob = 0.50`** |
| **Dữ liệu huấn luyện** | Chỉ dùng 3.600 ảnh Staging (bỏ quên 144k ProGAN) | Mẫu quá ít gây sụp đổ overfit | **Phương Án A: 144k ProGAN (85%) + 3.6k Staging (15%)** (Phương Án B: +20k-30k GenImage) |
| **Hàm mất mát** | `DualStreamFocalLoss` ($\gamma=2.0$) | $\gamma=2.0$ đè bẹp logit về sát 0 | **`DualStreamCELoss(label_smoothing=0.1)`** |
| **Hạ tầng huấn luyện** | Colab Pro GPU A100 | Hết hạn session, giới hạn CUs | **Kaggle Server GPU RTX 6000 Blackwell (95GB VRAM)** |

---

## IV. LỘ TRÌNH 5 TUẦN TRIỂN KHAI & TIẾN ĐỘ THỰC TẾ (MỞ RỘNG: PHASE 7)

```mermaid
gantt
    title TIẾN ĐỘ THỰC TẾ DỰ ÁN FATFORMER-XLA
    dateFormat  YYYY-MM-DD
    section Tuần 1: Hạ tầng & Verify [HOÀN THÀNH]
    Drive 5TB Setup & IO (A)          :done, t1_a, 2026-09-21, 3d
    Verify model strict pass 100% (B) :done, t1_b, 2026-09-21, 4d
    Notebook train/eval test (C)      :done, t1_c, 2026-09-23, 4d
    section Tuần 2: Baseline & Báo Cáo [HOÀN THÀNH]
    Tạo test_degraded.tar lossless (A):done, t2_a, 2026-09-28, 3d
    Chạy Baseline Clean & Degraded (C):done, t2_c, 2026-09-29, 4d
    Viết Báo cáo Chương 1 & 2 (B)    :done, t2_b, 2026-09-28, 5d
    section Tuần 3: Tích Hợp & Smoke Gate [HOÀN THÀNH]
    Staging 3.600 ảnh GenImage (A)   :done, t3_a, 2026-10-05, 3d
    Curriculum Scheduler mỏ neo 50% (A):done, t3_s, 2026-10-06, 3d
    Tích hợp SRM + Gating + Loss (B) :done, t3_b, 2026-10-05, 4d
    Integration Smoke Test Gate (C)  :done, t3_c, 2026-10-08, 2d
    Viết Báo cáo Chương 3 (B)        :done, t3_w, 2026-10-07, 4d
    section Tuần 4: Finetune RTX 6000 [HOÀN THÀNH]
    Setup Kaggle Offline & Plan A (C):done, t4_train, 2026-10-08, 3d
    Viết Báo cáo Chương 4 (B)        :done, t4_w, 2026-10-10, 2d
    section Tuần 5: Benchmark Task 5.1 [HOÀN THÀNH]
    Full Benchmark 18 tập test (C)   :done, t5_c, 2026-10-11, 2d
    Khóa Test Set độc lập Task 5.1   :done, t5_lock, 2026-10-12, 1d
    section Tuần 6: Milestone 6 Đa Miền [HIỆN TẠI]
    Task 6.1 Phân tích shift (C)     :active, t6_1, 2026-10-13, 1d
    Task 6.2 Data đa miền (A)        :t6_2, 2026-10-13, 2d
    Task 6.3-B Loss Distill (B)      :t6_3b, 2026-10-13, 2d
    Smoke Gate M6 (B/C)              :t6_smoke, 2026-10-14, 1d
    Task 6.3-C Train đa miền (C)     :t6_3c, 2026-10-14, 2d
    Task 6.5 Ablation đa miền (C/B)  :t6_5, 2026-10-15, 2d
    Task 6.4 Hiệu chuẩn & Chọn lọc(C):t6_4, 2026-10-16, 1d
    Task 6.6 Final Benchmark (C)     :t6_6, 2026-10-17, 1d
    section Tuần 7-8: Phase 7 [SẴN SÀNG]
    G7.0 Toàn vẹn đo lường (B/C)     :t7_g0, 2026-10-27, 2d
    WiSE-FT Gen 2 + Calib (C)        :t7_g1, 2026-10-29, 2d
    Kho dữ liệu v2 + Degradation (A) :t7_data, 2026-10-27, 4d
    Loss v2 + Selection v2 (B)       :t7_loss, 2026-10-30, 3d
    Train Anchored v2 + Ablation (C) :t7_g2, 2026-11-03, 4d
    Router + Unified (B/C)           :t7_g3, 2026-11-06, 4d
    Locked Benchmark + Docs (Cả 3)   :t7_close, 2026-11-10, 3d
```

---

### 🗓️ TUẦN 1: THIẾT LẬP HẠ TẦNG, VERIFY MÃ NGUỒN & PIPELINE DUMMY `[x] ĐÃ HOÀN THÀNH 100%`
*Mục tiêu: Kích hoạt hạ tầng kết nối Drive 5TB - Colab, xác nhận mã nguồn `src/` nạp khớp 100% weights và thông suốt pipeline dummy.*

| Mã việc | Nhiệm vụ kỹ thuật | Phụ trách | Môi trường | Đầu ra (Deliverables) | Trạng thái |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **1.1** | Khởi tạo thư mục gốc `Fatformer` trên Drive 5TB; đóng gói Dummy Dataset và viết hàm giải nén SSD NVMe. | **A** | 🖥️ **Local** + ☁️ **Colab** | Thư mục `Fatformer/` + 36.45 GB dữ liệu ([Báo cáo Task 1.1](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/A/bao_cao_task_1.1_ha_tang_drive_io.md)) | `[x] Hoàn thành` |
| **1.2** | Kiểm chứng mã nguồn `src/models/`, chạy script test nạp khớp 100% weights gốc (`strict=True`). | **B** | 🖥️ **Local** / ☁️ **Colab T4** | Báo cáo pass `strict=True` 1.116 tensor ([Báo cáo Task 1.2](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/B/bao_cao_task_1.2_fatformer_model_and_weights_loader.md)) | `[x] Hoàn thành` |
| **1.3** | Thiết lập notebook huấn luyện `train.ipynb` với AMP FP16 và Gradient Accumulation; chạy 1 step dummy. | **C** | ☁️ **Colab T4** | Notebook huấn luyện hoàn chỉnh ([Báo cáo Task 1.3](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/C/bao_cao_task_1.3_colab_train_pipeline.md)) | `[x] Hoàn thành` |
| **1.4** | Xây dựng script `fast_eval.py`: Lấy ngẫu nhiên 500 ảnh/tập test để đánh giá nhanh trong < 8 phút. | **C** | 🖥️ **Local** + ☁️ **Colab T4** | Script `tools/fast_eval.py` đo ACC, AP, AUC ([Báo cáo Task 1.4](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/C/bao_cao_task_1.4_pipeline_fast_eval.md)) | `[x] Hoàn thành` |

---

### 🗓️ TUẦN 2: XÂY DỰNG BỘ TEST DEGRADED, THIẾT LẬP BASELINE & VIẾT CHƯƠNG 1-2 `[x] ĐÃ HOÀN THÀNH 100%`
*Mục tiêu: Xác lập mốc sàn khoa học trước khi can thiệp mô hình; khởi động viết báo cáo từ sớm.*

| Mã việc | Nhiệm vụ kỹ thuật | Phụ trách | Môi trường | Đầu ra (Deliverables) | Trạng thái |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **2.1** | Viết script `tools/make_degraded.py` sinh 3 biến thể suy biến (JPEG, Blur, Down-Up). Sửa lỗi nén kép PNG lossless. | **A** | 🖥️ **Local** + ☁️ **Colab** | Script `make_degraded.py` + `test_degraded.tar` ([Báo cáo Task 2.1](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/A/bao_cao_task_2.1_make_degraded_dataset.md)) | `[x] Hoàn thành` |
| **2.2** | Chạy benchmark Checkpoint gốc trên 18 tập **Clean** (8 GANs + 10 Diffusion). | **C** | ☁️ **Colab T4/A100** | Bảng số liệu Clean (GANs 98.4%, Diffusion 95.0%) ([Báo cáo Task 2.2](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/C/bao_cao_task_2.2_baseline_clean_benchmark.md)) | `[x] Hoàn thành` |
| **2.3** | Chạy benchmark Checkpoint gốc trên tập **Degraded** ($Q=30, 50, 70$), ghi nhận sụt giảm nghiêm trọng. | **C** | ☁️ **Colab T4** | Bảng số liệu Degraded mốc sàn ($Q=30$ rớt sâu) ([Báo cáo Task 2.3](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/C/bao_cao_task_2.3_baseline_degraded_benchmark.md)) | `[x] Hoàn thành` |
| **2.4** | Trích xuất Grad-CAM ban đầu trên 5 cặp ảnh Clean vs Degraded ($Q=30$). | **B** | 🖥️ **Local** / ☁️ **Colab T4** | Bộ 10 ảnh Grad-CAM đối chứng ban đầu ([Báo cáo Task 2.4](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/B/bao_cao_task_2.4_gradcam_baseline_extraction.md)) | `[x] Hoàn thành` |
| **2.5** | **Họp nhóm Milestone 1**: Chốt bảng số liệu Baseline, chính thức lượng hóa chỉ tiêu cải thiện. | **Cả 3** | 🖥️ **Họp nhóm** | Đóng băng mốc sàn Clean ACC = 96.45% | `[x] Hoàn thành` |
| **2.6** | Soạn thảo **Chương 1 (Giới thiệu bài toán)** và **Chương 2 (Related Work)**. | **B (+ C review)** | 🖥️ **Local** | Bản thảo Chương 1 & Chương 2 hoàn chỉnh ([Báo cáo Task 2.6](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/B/bao_cao_task_2.6_draft_report_chap_1_2.md)) | `[x] Hoàn thành` |

---

### 🗓️ TUẦN 3: DỮ LIỆU STAGING, CURRICULUM SCHEDULER, SRM + GATING + LOSS & SMOKE TEST `[x] ĐÃ HOÀN THÀNH 100%`
*Mục tiêu: Đưa dữ liệu sinh hiện đại vào tập train, tích hợp bộ lập lịch Curriculum, hoàn thiện SRM + Gating và vượt qua cổng kiểm thử an toàn.*

| Mã việc | Nhiệm vụ kỹ thuật | Phụ trách | Môi trường | Đầu ra (Deliverables) | Trạng thái |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **3.1** | Chuẩn hóa **3.600 ảnh GenImage** (SD v1.5 + Midjourney) tạo `diffusion_staging.tar`. | **A** | ☁️ **Colab** | Tệp `diffusion_staging.tar` trên Drive 5TB | `[x] Hoàn thành` |
| **3.2** | Xây dựng module `CurriculumDegradationScheduler` trong `src/datasets/transforms.py`. | **A** | 🖥️ **Local** | Module lập lịch 3 giai đoạn theo epoch | `[x] Hoàn thành` |
| **3.3** | Hoàn thiện `SpatialResidualBlock`, `DynamicFrequencyGating` và `DualStreamCELoss`/Focal Loss. | **B** | 🖥️ **Local** + ☁️ **Colab** | Code `srm.py`, `gating.py`, `loss.py` sẵn sàng | `[x] Hoàn thành` |
| **3.4** | Thiết lập cấu hình đóng băng 94.1% CLIP ViT-L/14, chỉ mở khóa 5.9% tham số. | **B** | 🖥️ **Local** | File cấu hình `configs/train_config.yaml` | `[x] Hoàn thành` |
| **3.5** | Thiết lập hook tự động lưu checkpoint `.pth` sau mỗi epoch và cơ chế `--resume`. | **C** | ☁️ **Colab** | Pipeline auto-save & resume ([Báo cáo Task 3.5](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/C/bao_cao_task_3.5_checkpoint_autosave_and_resume.md)) | `[x] Hoàn thành` |
| **3.6** | **CỔNG KIỂM THỬ TÍCH HỢP (Smoke Gate)**: Chạy forward + backward pass 1 batch trên GPU T4 (30 giây). | **B + C** | ☁️ **Colab T4** | Pass 100% autograd, gradient không NaN/Inf ([Báo cáo Task 3.6](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/C/bao_cao_task_3.6_integration_smoke_test_gate.md)) | `[x] Hoàn thành` |
| **3.7** | Soạn thảo **Chương 3 (Phương pháp luận & Kiến trúc đề xuất FatFormer-XLA)**. | **B** | 🖥️ **Local** | Bản thảo Chương 3 kèm sơ đồ vector | `[x] Hoàn thành` |

---

### 🗓️ TUẦN 4: KIỂM TOÁN LỖI TASK 5.1 & CHIẾN DỊCH FINETUNE TRÊN GPU RTX 6000 BLACKWELL `[x] ĐÃ HOÀN THÀNH 100%`
*Mục tiêu: Khắc phục triệt để lỗi mô hình cũ ở Task 5.1, triển khai Finetune Phương Án A (144k ProGAN + 3.6k Staging, batch 64) trên máy chủ Kaggle Offline GPU NVIDIA RTX PRO 6000 Blackwell (94.97 GB VRAM) và sẵn sàng kích hoạt Phương Án B nếu cần.*

| Mã việc | Nhiệm vụ kỹ thuật | Phụ trách | Môi trường | Đầu vào (Input) | Đầu ra (Deliverables) | Trạng thái |
| :---: | :--- | :---: | :--- | :--- | :--- | :---: |
| **4.1** | **Kiểm toán lỗi mô hình cũ Task 5.1**: Xác định nguyên nhân Clean ACC rớt 50.31% (do thiếu 144k ProGAN, clean ratio hạ 20%, Focal Loss đè logit). | **Cả nhóm** | 🖥️ **Local / Cloud** | Kết quả Task 5.1 cũ | [Báo cáo Kiểm toán Task 5.1](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/plan/ke_hoach_cac_cot_moc_va_diem_dung.md) | `[x] Hoàn thành` |
| **4.2** | **Đóng gói dữ liệu & Dựng môi trường Kaggle Offline**: Đưa 17.75 GB dữ liệu + weights lên Kaggle Dataset, sửa lỗi offline BPE tokenizer & typing `Optional`. | **A + C** | 🚀 **Kaggle Server** | Assets pack 17.75 GB, zip mã nguồn 1.4 MB | Môi trường thi đấu Offline sẵn sàng | `[x] Hoàn thành` |
| **4.3** | **Chiến dịch Finetune Phương Án A (5 Epochs)**: Khóa cứng `clean_prob=0.50`, tỷ lệ 85% ProGAN (144k) + 15% Staging (3.6k) qua `WeightedRandomSampler`, Batch Size 64, CE Loss. | **C** | 🚀 **RTX 6000 (95GB VRAM)** | Dataset giải nén trên NVMe Kaggle | Checkpoint `fatformer_plan_a_best_weights.pth` (Epoch 5, composite score 83.09) | `[x] Hoàn thành` |
| **4.4** | **Kiểm định vòng lặp (Validation Loop)**: Đo đạc tự động sau mỗi epoch trên 8.000 ảnh `progan_val.tar`, lưu metrics thời gian thực. | **C** | 🚀 **RTX 6000 (95GB VRAM)** | Tập validation 8k ảnh | File log `train_val_history.csv` | `[x] Hoàn thành` |
| **4.5** | **Cổng Đánh Giá Mở Rộng (Expansion Gate)**: Đánh giá checkpoint Phương Án A. Hội tụ ProGAN Clean 97.95%, Q30 61.6%; tuy nhiên ghi nhận distribution shift ngoài miền ProGAN trong Task 5.1 $\rightarrow$ Kích hoạt chuyển tiếp sang **Milestone 6**. | **Cả nhóm** | 🚀 **Kaggle / Local** | Metrics nghiệm thu Phương Án A | Quyết định kích hoạt Milestone 6 đa miền | `[x] Hoàn thành` |
| **4.6** | Soạn thảo **Chương 4 (Thiết lập thực nghiệm & Môi trường huấn luyện)**: mô tả hạ tầng RTX 6000 Blackwell 95GB VRAM, siêu tham số và cơ chế nới rộng dữ liệu. | **B (+ A data)** | 🖥️ **Local** | Thông số huấn luyện thực tế | Bản thảo Chương 4 hoàn chỉnh ([Báo cáo Task 4.6](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/B/bao_cao_task_4.6_draft_report_chap_4.md)) | `[x] Hoàn thành` |

---

### 🗓️ TUẦN 5: ĐO ĐẠC BENCHMARK TASK 5.1 & KHÓA TEST SET ĐỐI CHỨNG `[x] ĐÃ HOÀN THÀNH`
*Mục tiêu: Hoàn tất đo đạc 126 lượt trên 18 tập test hai chiều, đối chiếu toàn diện với bài báo gốc CVPR 2024, ghi nhận hiện tượng over-specialization và khóa cố định 18 tập test làm bài kiểm chuẩn độc lập (Test Set Locked).*

| Mã việc | Nhiệm vụ kỹ thuật | Phụ trách | Môi trường | Đầu vào (Input) | Đầu ra (Deliverables) | Trạng thái |
| :---: | :--- | :---: | :--- | :--- | :--- | :---: |
| **5.1** | Chạy **Full-Benchmark** trên toàn bộ 18 tập test paper chuẩn ở cả 2 trạng thái Clean và Degraded ($Q \in \{30, 50, 70\}$, Blur, Down-Up). | **C** | ☁️ **Colab T4** | Checkpoint `fatformer_plan_a_best_weights.pth` + 18 tập test | Ma trận kết quả chi tiết `final_full_benchmark_matrix.csv` ([Báo cáo Task 5.1](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/C/bao_cao_task_5.1_full_benchmark_18_datasets.md)) | `[x] Hoàn thành` |
| **5.1-L** | **Khóa Test Set Task 5.1 độc lập**: Niêm phong 18 tập dữ liệu, tuyệt đối không dùng để train, tune threshold hay chọn checkpoint. | **Cả 3** | 🖥️ **Local / Cloud** | Ma trận Task 5.1 | Biên bản khóa kiểm chuẩn độc lập | `[x] Hoàn thành` |

---

### 🗓️ TUẦN 6: MILESTONE 6 – KHẮC PHỤC LỆCH PHÂN PHỐI & HUẤN LUYỆN ĐA MIỀN TỔNG QUÁT `[~] HOÀN THÀNH CÓ ĐIỀU KIỆN`
*Mục tiêu: Giải quyết hiện tượng lệch phân phối (distribution shift) bằng kho dữ liệu đa bộ sinh độc lập, bổ sung distillation giữ vững tri thức ảnh sạch, và vượt qua Gate 4 điều kiện ràng buộc.*
*Trạng thái thực tế (27/10/2026): Baseline và Anchored Recovery hợp lệ; **WiSE-FT Gen 1 bị `invalid_layout`** ⇒ Milestone 6 chưa thể đóng, chuyển tiếp sang Phase 7 để tái lập đo lường.*

| Mã việc | Nhiệm vụ kỹ thuật | Phụ trách | Môi trường | Đầu vào (Input) | Đầu ra (Deliverables) | Trạng thái |
| **6.1** | **Phân tích định lượng distribution shift**: Đo lường Fake Recall, Real Recall, AUC và phân tầng lỗi per-generator từ ma trận Task 5.1; không sửa mô hình. | **C** | 🖥️ **Local** | Ma trận Task 5.1 | Báo cáo chẩn đoán lỗi ([Báo cáo Task 6.1](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/C/bao_cao_task_6.1_distribution_shift_analysis.md)) | `[x] Đã hoàn thành` |
| **6.2** | **Xây dựng kho train/val đa bộ sinh độc lập**: Thu thập các họ GAN/Diffusion yếu; lập data manifest; đo Baseline mô hình gốc trên val_crossdomain; tách val_calibration và val_selection; xây dựng `BalancedCrossDomainSampler`. | **A** | ☁️ **Drive / Colab** | Nguồn dữ liệu đa miền độc lập | Kho dữ liệu train và val_crossdomain | `[x] Đã hoàn thành` ([Báo cáo Task 6.2](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/A/bao_cao_task_6.2_cross_domain_data_infra.md)) |
| **6.3-B**| **Thiết kế hàm loss Distillation + Consistency & Paired Aug**: Module loss triệt tiêu phân kỳ KL + Consistency Loss + Paired Augmentation; chuẩn bị module LoRA có kiểm soát cho Task 6.5. | **B** | 🖥️ **Local** | Khung phân tích lỗi Task 6.1 + Baseline mỏ neo | Module loss và pipeline paired augmentation | `[x] Đã hoàn thành` ([Báo cáo Task 6.3-B](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/B/bao_cao_task_6.3_b_distillation_and_consistency_loss.md)) |
| **Smoke Gate M6** | **Cổng kiểm thử khớp nối tích hợp Milestone 6**: Chạy kiểm thử tích hợp 1 batch forward-backward với teacher logit trên GPU trước khi train quy mô lớn. | **B + C** | ☁️ **Colab T4** | Tích hợp data Task 6.2 và loss Task 6.3-B | Biên bản pass Smoke Gate M6 | `[ ] Sẵn sàng` |
| **6.3-C**| **Fine-tuning thích ứng chống domain shift**: Huấn luyện với dữ liệu cân bằng và hàm mất mát tích hợp trên GPU; tối ưu lưu trữ chỉ giữ checkpoint hợp lệ. | **C** | ☁️ **Colab GPU** | Dữ liệu Task 6.2 + Baseline gốc | Checkpoint thích ứng đa miền | `[ ] Đang lập kế hoạch` |
| **6.5** | **Nghiên cứu cắt bỏ thành phần (Ablation đa miền)**: Huấn luyện và đối chứng ma trận M1-M5 (Plan A, +Data, +Distill, +LoRA) trên cùng random seed 42 và data manifest. | **C + B** | 🖥️ **Local / Cloud** | Checkpoint đối chứng đa miền | Ma trận Ablation Study hoàn thiện | `[ ] Đang lập kế hoạch` |
| **6.4** | **Hiệu chuẩn xác suất & Tuyển chọn Checkpoint**: Hiệu chuẩn nhiệt độ $T, T^*$ trên `val_calibration`; tuyển chọn checkpoint chiến thắng trên `val_selection` theo Gate 4 điều kiện. | **C** | ☁️ **Colab T4** | Tập val_crossdomain + Các checkpoint ứng viên | Checkpoint tuyển chọn & Ngưỡng chuẩn | `[ ] Đang lập kế hoạch` |
| **6.6** | **Kiểm chuẩn khóa sau cùng (Final Locked Benchmark)**: Đối chiếu hậu nghiệm trên 18 tập test Task 5.1 và kiểm thử độc lập trên tập mở rộng; xuất metrics và bootstrap 95% CI. | **C** | ☁️ **Colab T4** | Checkpoint tuyển chọn + Tập kiểm thử khóa | Báo cáo kiểm chuẩn chung cuộc | `[ ] Đang lập kế hoạch` |

---

### 🗓️ ĐÃ BÃI BỎ TUẦN 7 (KẾ HOẠCH CŨ): TẬP TRUNG TOÀN DIỆN VÀO MÔ HÌNH & CHỈ SỐ ĐỊNH LƯỢNG
*Theo quyết định điều hành, toàn bộ các nhiệm vụ phụ trợ gồm Task 5.2 (Ablation cũ), Task 5.3 (Grad-CAM XAI), Task 5.5 (Demo App Web) và Task 5.4 (Slide & Báo cáo) đã được bãi bỏ hoàn toàn để tập trung 100% nguồn lực vào chiến dịch Milestone 6 (Task 6.1 đến 6.6).*

---

### 🗓️ TUẦN 7–8: PHASE 7 – TÁI LẬP ĐO LƯỜNG & HỢP NHẤT MÔ HÌNH `[ ] SẴN SÀNG TRIỂN KHAI`
*Mục tiêu: (1) sửa cho đúng thước đo trước khi tin bất kỳ con số nào; (2) thắng nhanh trên không gian trọng số bằng WiSE-FT Gen 2 + hiệu chuẩn (0 giờ GPU train); (3) nâng năng lực thật bằng dữ liệu đa bộ sinh + hàm phạt bảo vệ ảnh thật; (4) đóng gói **một** checkpoint tự định tuyến theo mức suy biến.*

> **Văn kiện gốc (source of truth)**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](ke_hoach_phase_7_dieu_chinh_toan_dien.md) — bản `v3.0` viết lại sau khi kiểm chứng cấu trúc 4 checkpoint bằng PyTorch. Bản `v2.0` chỉ còn giá trị truy vết tại [`archive/`](archive/ke_hoach_phase_7_dieu_chinh_toan_dien_v2.0_2026-10-27.md).
> **Điều kiện tiên quyết**: Milestone 6 đã đóng băng kết quả (Baseline, M4, Anchored hợp lệ; **WiSE-FT Gen 1 bị cách ly** vì không biểu diễn được bằng kiến trúc hiện tại — xem §I.1 của kế hoạch Phase 7).

**Bảng 16 task Phase 7:**

| Mã | Nhiệm vụ | Chủ trì | Hỗ trợ | Ước lượng | Cổng |
| :---: | :--- | :---: | :---: | ---: | :---: |
| 7.1 | Audit cấu trúc 4 checkpoint & đo lại Gen 1 (`n=3` vs `n=8`) | C | B | 4 h | G7.0 |
| 7.2 | `canonical_loader` + adapter index tường minh + test (`strict=True`, cấm fallback) | B | C | 8 h | G7.0 |
| 7.3 | Eval v3: `report_schema v3`, bỏ ngưỡng phụ thuộc batch, thêm `TPR@FPR 1/5/10` + CI | C | B | 6 h | G7.0 |
| 7.4 | Hiệu chuẩn nhiệt độ & ngưỡng có ràng buộc FPR + ECE (split tách rời) | C | — | 4 h | G7.0 |
| 7.5 | WiSE-FT **Gen 2** (Baseline + Anchored, cùng `n=3`) + lưới `α×β_srm` + `--verify` | C | B | 5 h | G7.1 |
| 7.6 | Re-baseline 4 cấu hình trên CFE bằng harness v3 | C | — | 3 h | G7.1 |
| 7.7 | Kho dữ liệu v2: CFE-Small phân tầng + ảnh thật đa nguồn (12–16k, generator-disjoint) | A | C | 14 h | — |
| 7.8 | Degradation pipeline v2: $Q\in[30,85]$, WebP, double-JPEG, ghép 2 suy biến | A | B | 5 h | — |
| 7.9 | Loss v2: `RealGuardLoss` + `FeatureConsistencyLoss` (weight=0 ⇒ backward-compatible) | B | C | 7 h | — |
| 7.10 | Selection v2: worst-group + ràng buộc `Real ACC ≥ 90%` | B | C | 4 h | — |
| 7.11 | Huấn luyện **Anchored Recovery v2** (dữ liệu v2 + loss v2 + worst-group) | C | A+B | 5 h + GPU | G7.2 |
| 7.12 | Ablation mini 4 cấu hình (cùng seed 42) | C+B | — | 4 h + GPU | G7.2 |
| 7.13 | `DegradationRouter` + fusion xác suất đã hiệu chuẩn (fit, không đặt tay) | B | C | 8 h | — |
| 7.14 | Locked Benchmark Phase 7 (18 tập in-dist + CFE + 6 biến thể) | C | A | 5 h | G7.3 |
| 7.15 | Đóng gói `UnifiedFatFormerXLA` + `tools/infer_unified.py` | C | B | 4 h | G7.3 |
| 7.16 | Cập nhật 10 file tài liệu dự án + Chương 5/Hạn chế | Cả 3 | — | 6 h | G7.3 |

**Bốn cổng chặn Phase 7:**

| Cổng | Thời điểm | Điều kiện tối thiểu (Must) |
| :--- | :---: | :--- |
| **G7.0** | Cuối D2 | 4/4 checkpoint nạp `0 missing / 0 unexpected` hoặc bị từ chối có tên; Gen 1 gắn `invalid_layout`; JSON đủ `AUC/AP/TPR@FPR/CI`; không còn `strict=False` trong `tools/` |
| **G7.1** | Cuối D4 | CFE clean: `Fake@Real=90% ≥ 55%`, `AUC ≥ 80%`; CFE Q30: `AUC ≥ 55%`; không tệ hơn Baseline > 5 pp ở bất kỳ subset nào |
| **G7.2** | Cuối D8 | CFE clean `Fake@Real=90% ≥ 65%` + `AUC ≥ 78%`; Q30 `AUC ≥ 60%`; in-dist `Macro ACC ≥ 90%` và `Real ACC ≥ 90%` (cứng); `ECE ≤ 0,08` |
| **G7.3** | Cuối D10 | **Một** checkpoint + `router_params.json`, một `forward()` không cờ chế độ; trong 1,5 pp so với tốt nhất từng điều kiện; thêm < 5 ms/ảnh trên T4 |

> **Điều khoản kết quả âm**: nếu G7.2 trượt, Phase 7 vẫn hợp lệ khi bàn giao checkpoint Pareto tốt nhất + ablation chỉ rõ đòn bẩy thất bại + mục "Hạn chế" viết rõ. Không được nâng số.

---

## V. BẢNG TIÊU CHUẨN NGHIỆM THU ĐỒ ÁN (GATE 4 → CHUYỂN TIẾP SANG CỔNG PHASE 7)

Dự án được đánh giá là hoàn thành chuẩn mực khi đạt đủ các tiêu chuẩn sau:
- [x] Khắc phục 100% nghẽn I/O qua cơ chế NVMe Colab/Kaggle Offline và kiểm toán toàn diện kho dữ liệu Drive 5TB (55.98 GB, 927.927 ảnh).
- [x] Bảng số liệu Baseline gốc tái lập trên cả 2 miền Clean và Degraded ($Q=30, 50, 70$) khớp chuẩn xác với paper gốc CVPR 2024.
- [x] Hoàn tất đo đạc trung thực Task 5.1 và khóa cố định 18 tập test làm Test Set độc lập (Clean Macro ACC đạt 67.34%, Fake ACC tại $Q=30$ tăng gấp 4.76 lần đạt 27.53%).
- [ ] **Thỏa mãn Gate 4 điều kiện tại Milestone 6 (đo theo delta pp so với mốc sàn baseline trên `val_crossdomain`)**:
  1. **Clean Macro ACC**: $\text{ACC}_{\text{clean}} \ge \text{Base}_{\text{clean}} - 1,5\text{ pp}$ (Không suy giảm quá 1,5 điểm phần trăm so với mốc sàn baseline đo trên cùng tập validation đa miền).
  2. **Clean Fake ACC**: $\text{ACC}_{\text{fake\_clean}} \ge \text{Base}_{\text{fake\_clean}} - 2,0\text{ pp}$ (Bảo toàn năng lực phát hiện ảnh giả ở trạng thái sạch, không đánh đổi độ nhạy).
  3. **Q30 Overall ACC**: $\text{ACC}_{\text{q30}} \ge \text{Base}_{\text{q30}}$ (Độ chính xác tổng thể dưới nén sâu $Q=30$ tối thiểu ngang bằng hoặc vượt mốc sàn baseline).
  4. **Q30 Fake ACC & Real Preservation**: $\text{ACC}_{\text{fake\_q30}} \ge \text{Base}_{\text{fake\_q30}} + 10,0\text{ pp}$ và đồng thời $\text{ACC}_{\text{real\_q30}} \ge 60,0\%$ (Nâng mạnh khả năng nhận diện ảnh giả bị nén sâu thêm tối thiểu 10 điểm phần trăm, đồng thời bảo đảm độ chính xác trên ảnh thật không bị sụp đổ).
- [ ] Bảng Ablation Study định lượng rõ rệt công năng của từng module: SRM 3-Kernels, Dynamic Frequency Gating $\lambda(x)$, Distillation Loss và Data đa miền (Task 6.5).

> ⚠️ **Đính chính phạm vi hiệu lực (27/10/2026)**: Các số liệu Milestone 6 từng công bố **chỉ còn hiệu lực cho Baseline và Anchored Recovery**. Checkpoint `fatformer_wise_ft_best.pth` (Gen 1) chứa **10 khối adapter** `{2,5,7,8,11,14,15,17,20,23}` trong khi hàm dựng mô hình chỉ biểu diễn được `n ∈ {3,4,6,8,12,24}` (vị trí cố định `(i+1)·24/n − 1`), nên **không thể nạp đầy đủ**; mọi con số của checkpoint này (kể cả in-dist 96,58% và CFE 52,05%/Fake 4,10%) bị đánh dấu `invalid_layout` cho tới khi đo lại. Vì vậy Gate 4 **không còn là cổng nghiệm thu cuối cùng**; nó được thay bằng chuỗi **G7.0 → G7.3** của Phase 7 (xem bảng cổng ở Tuần 7–8 và [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](ke_hoach_phase_7_dieu_chinh_toan_dien.md) §II.3).

---

## VI. MA TRẬN QUẢN LÝ RỦI RO TIẾN ĐỘ (RISK MANAGEMENT MATRIX)

| Rủi ro tiềm ẩn | Mức độ | Kế hoạch dự phòng & Giảm thiểu rủi ro (Contingency Plan) | Trạng thái kiểm soát |
| :--- | :---: | :--- | :---: |
| **Môi trường thi đấu Offline cấm Internet** | Đã xử lý | Đóng gói mã nguồn `fatformer_code.zip` và tài sản 17.75 GB lên Dataset; tích hợp BPE vocab nội bộ; nạp trọng số qua đường dẫn cục bộ. | **Triệt tiêu 100%** |
| **Quên tri thức ảnh sạch (Catastrophic Forgetting)** | Đã xử lý | Khóa cứng mỏ neo `clean_prob = 0.50` xuyên suốt các epoch; áp dụng triệt tiêu phân kỳ KL từ mô hình gốc trên nhánh ảnh sạch trong Milestone 6. | **Kiểm soát chặt chẽ** |
| **Hàm mất mát đè bẹp logit (Focal Loss collapse)** | Đã xử lý | Thay thế Focal Loss $\gamma=2.0$ bằng `DualStreamCELoss(label_smoothing=0.1)` giúp phân phối gradient mượt mà, bảo vệ ngưỡng phân loại 0.5. | **Triệt tiêu 100%** |
| **Lệch phân phối đa bộ sinh (Over-specialization)** | Đang xử lý | Kích hoạt **Milestone 6**: Thu thập kho train/val đa miền độc lập ngoài 18 tập test, áp dụng `BalancedCrossDomainSampler` và Distillation Loss. | **Đã kích hoạt Milestone 6** |
| **Dồn việc viết báo cáo cuối kỳ** | Triệt tiêu | Áp dụng mô hình **viết cuốn chiếu**: Tuần 1-3 đã hoàn tất Chương 1, 2, 3. Tuần 4 hoàn thành Chương 4. Tuần 7 chỉ cần điền bảng số liệu vào Chương 5. | **Đúng tiến độ** |
| **Checkpoint trộn khác số khối adapter** (mới, 27/10/2026) | **Cao** | Ràng buộc cứng `(i+1)·24/n − 1` khiến file trộn 3-khối + 8-khối tạo ra 10 khối không `n` nào nạp được; loader cũ còn fallback `len>8 → 3` nên **im lặng chọn tập khối tệ nhất**. Đã xảy ra thật với WiSE-FT Gen 1. | **Bắt buộc**: chỉ trộn hai checkpoint **cùng `num_vit_adapter`**; mọi lần nạp phải qua `canonical_loader` với `strict=True` + assert `0/0`; cấm fallback chọn `n` tự động (Task 7.2). | **Đang xử lý (G7.0)** |
| **Kết luận dựa trên ngưỡng phụ thuộc thành phần lô** (mới, 27/10/2026) | **Cao** | `adaptive_thresh` tính trung bình trên chính tập đang chấm ⇒ rò rỉ thông tin, không triển khai được; các con số "unified" chỉ là trộn heuristic. | Ngưỡng cố định fit trên tập hiệu chuẩn tách rời; mọi báo cáo phải có `TPR@FPR` + bootstrap CI; cấm báo cáo "overall accuracy" đơn lẻ (Task 7.3–7.4, Phụ lục A của kế hoạch test CFE). | **Đang xử lý (G7.0)** |

---
*Văn kiện này là bản kế hoạch điều hành chuẩn của đồ án FatFormer-XLA, được cập nhật và đồng bộ theo chiến dịch huấn luyện thực tế và quy chuẩn học thuật nghiêm ngặt.*
