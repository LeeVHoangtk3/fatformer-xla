# KẾ HOẠCH TRIỂN KHAI ĐỒ ÁN: TỐI ƯU HÓA ĐỘ BỀN VỮNG PHÁP Y CHO FATFORMER TRÊN GOOGLE COLAB
> **Chủ đề**: Nâng cao độ bền phát hiện ảnh AI (Generalizable Synthetic Image Detection) trước biến dạng nén mạng xã hội (JPEG, Blur) và mở rộng nhận diện mô hình Diffusion hiện đại[cite: 4, 6].  
> **Kiến trúc cốt lõi**: FatFormer (CVPR 2024: CLIP ViT-L/14 + Forgery-Aware Adapter + Language-Guided Alignment)[cite: 1, 8].  
> **Hạ tầng thực thi**: Google Colab Pro (GPU T4 / A100) kết hợp lưu trữ Google Drive nhóm[cite: 6, 9].  
> **Quy mô nhân sự**: Nhóm 3 thành viên[cite: 6].

---

## I. PHÂN CÔNG VAI TRÒ & TRÁCH NHIỆM CỐT LÕI (CORE ROLES)

| Thành viên | Vai trò kỹ thuật | Trách nhiệm chính | Tiêu chí bàn giao trọng tâm |
| :--- | :--- | :--- | :--- |
| **A — Data & Infra Lead** | Quản lý dữ liệu, tối ưu I/O & Augmentation | Quản lý Google Drive, viết pipeline giải nén SSD Colab, xây dựng pipeline data augmentation 2 luồng (Clean + Degraded)[cite: 4, 6]. | Không nghẽn I/O (I/O Bottleneck), DataLoader nạp batch mượt mà[cite: 5, 6]. |
| **B — Architecture & Loss Lead** | Kiến trúc mô hình, SRM & Tối ưu hóa tham số | Kiểm soát nạp state_dict (xử lý lỗi chính tả biến), cấu hình đóng băng CLIP (94.1%), mở khóa Adapter (5.9%), tích hợp bộ lọc SRM đúng kích thước tensor, giám sát hàm mất mát Augmented Contrastive Loss và hệ số $\lambda$[cite: 2, 7, 8]. | Tensors khớp 100%, không lỗi runtime shape[cite: 2, 6], Loss hội tụ ổn định[cite: 5]. |
| **C — Training & Evaluation Lead** | Huấn luyện AMP FP16, Benchmark & XAI | Thiết lập Notebook Colab, tối ưu bộ nhớ VRAM bằng Gradient Accumulation[cite: 7, 9], chạy benchmark 18 tập test 2 chiều (Clean vs Degraded)[cite: 1, 6], trích xuất Grad-CAM[cite: 3, 6], viết báo cáo[cite: 6]. | Bảng số liệu ACC/AP chuẩn xác[cite: 1, 6], ảnh trực quan hóa Grad-CAM minh bạch[cite: 6, 7]. |

---

## II. LỘ TRÌNH 1: GIAI ĐOẠN ĐẦU — THIẾT LẬP, TỐI ƯU HÓA I/O & TÁI LẬP BASELINE
*Thời lượng: Tuần 1 – Tuần 2 (14 ngày)*  
*Mục tiêu: Thiết lập môi trường sạch, khắc phục triệt để nghẽn I/O Drive, tái lập chính xác số liệu Baseline gốc (~98.4% GANs, ~95.0% Diffusion) và lập mốc đo lường suy biến (Degraded Baseline)[cite: 1, 6].*

### Tuần 1: Setup Hạ Tầng, Giải Nén SSD Cục Bộ & Kiểm Định Checkpoint

| Mã việc | Nội dung công việc chi tiết | Người phụ trách | Sản phẩm bàn giao (Deliverables) |
| :--- | :--- | :---: | :--- |
| **1.1** | Setup Colab Pro, cài đặt PyTorch, `torchvision`, `pytorch_wavelets`[cite: 6, 8]. **Quy tắc I/O:** Viết script tự động sao chép dataset dạng nén (`.tar`/`.zip`) từ Drive sang `/content/` của máy ảo Colab rồi giải nén tại chỗ (tăng tốc độ đọc lên 10 lần so với đọc trực tiếp qua Drive mount)[cite: 6]. | **A** | Notebook `00_env_setup.ipynb` có hàm `extract_to_local()` tối ưu[cite: 6]. |
| **1.2** | Tải 2 file trọng số: `ViT-L-14.pt` (OpenAI TorchScript JIT, 932 MB) và `fatformer_4class_ckpt.pth` (1.97 GB)[cite: 2, 8]. Chạy script `check_weights.py` thẩm định nạp đủ 1,116 tensors và 933.16M tham số[cite: 2, 8]. **Bắt buộc:** Giữ nguyên tên biến `patch_basaed_enhancer` trong `models/lga.py` để tránh lỗi missing key khi load state_dict[cite: 2, 7]. | **B** | Script `load_model_test.py` thực thi pass `strict=True`[cite: 2]. |
| **1.3** | Thu thập và đóng gói dữ liệu kiểm thử: 8 GANs (*ProGAN, StyleGAN2, BigGAN, CycleGAN...*)[cite: 1, 6] và 6 Diffusion (*LDM, Glide, Guided Diffusion, PNDM...*)[cite: 1, 10]. Chuẩn hóa cây thư mục `0_real/` và `1_fake/`[cite: 1, 6]. | **A + C** | Bộ dữ liệu test nén dạng `.tar` lưu trữ sẵn trên Google Drive nhóm[cite: 6]. |
| **1.4** | Đọc hiểu và viết module `benchmark_metrics.py`: cài đặt các hàm đo lường Accuracy (ACC tại ngưỡng 0.5), Average Precision (AP), AUC-ROC và Equal Error Rate (EER)[cite: 5, 6]. | **C** | Module `benchmark_metrics.py` hoàn chỉnh[cite: 5, 6]. |
| **1.5** | Trích xuất công thức tensor: Lập bảng truy vết kích thước tensor (Tensor Shape Tracing) qua các khối Haar DWT 2D, Inter/Intra-band Attention, PBE, TGI để chuẩn bị nội dung cho báo cáo[cite: 7, 8]. | **B** | Bảng Tensor Shape Tracing chi tiết[cite: 7]. |

### Tuần 2: Xây Dựng Bộ Test Degraded & Đo Đạc Baseline 2 Chiều

| Mã việc | Nội dung công việc chi tiết | Người phụ trách | Sản phẩm bàn giao (Deliverables) |
| :--- | :--- | :---: | :--- |
| **2.1** | Viết script tạo dữ liệu suy biến vật lý `make_degraded_testset.py`: Tự động tạo 3 biến thể trên tập test: (1) JPEG ($Q \in \{30, 50, 70\}$)[cite: 6], (2) Gaussian Blur ($\sigma \in \{1.0, 2.0\}$)[cite: 4, 13], (3) Downsample-Upsample ($224 \rightarrow 112 \rightarrow 224$ mô phỏng nén mạng xã hội)[cite: 6]. | **A** | Bộ test set suy biến cục bộ trên ổ đĩa Colab[cite: 6]. |
| **2.2** | Chạy `evaluate.py` với Checkpoint gốc trên toàn bộ tập **Clean** (8 GANs + 6 Diffusion)[cite: 1, 6]. | **C** | Bảng số liệu: Tái lập GANs $\approx 98.4\%$, Diffusion $\approx 95.0\%$[cite: 1]. |
| **2.3** | Chạy `evaluate.py` với Checkpoint gốc trên toàn bộ tập **Degraded** đã tạo ở mục 2.1[cite: 6]. | **C** | Bảng số liệu ghi nhận mức độ sụt giảm hiệu năng của mô hình gốc (Baseline Degraded)[cite: 6]. |
| **2.4** | Xây dựng pipeline trích xuất Grad-CAM và phổ DWT: Xuất bản đồ nhiệt chú ý trên 5 cặp ảnh Real/Fake ở cả hai trạng thái Clean và Degraded[cite: 6, 7]. | **B** | Script `visualize_cam.py` + 10 ảnh Grad-CAM đối chứng ban đầu[cite: 6, 7]. |
| **2.5** | **Họp nhóm Milestone 1**: Phân tích mức độ suy giảm của nhánh Wavelet khi gặp nén JPEG, xác định mức suy biến mục tiêu cần khắc phục và thống nhất phương án finetune[cite: 4, 6]. | **Cả 3** | Biên bản kỹ thuật & Bảng số liệu Baseline tổng hợp[cite: 6]. |

---

## III. LỘ TRÌNH 2: GIAI ĐOẠN FINETUNE — HUẤN LUYỆN NÂNG CẤP & ĐÁNH GIÁ ĐA CHIỀU
*Thời lượng: Tuần 3 – Tuần 5 (21 ngày)*  
*Mục tiêu: Đóng băng CLIP backbone, mở khóa Adapter, tích hợp Data Augmentation 2 luồng và bộ lọc High-Pass SRM, nâng cao độ bền trước nén ảnh mà không làm suy thoái hiệu năng trên ảnh Clean[cite: 4, 6].*

### Tuần 3: Bổ Sung Dữ Liệu Tạo Sinh Hiện Đại & Nâng Cấp Kiến Trúc

| Mã việc | Nội dung công việc chi tiết | Người phụ trách | Sản phẩm bàn giao (Deliverables) |
| :--- | :--- | :---: | :--- |
| **3.1** | **Mở rộng dữ liệu:** Tải bổ sung tập con đại diện từ **GenImage** (Midjourney, Stable Diffusion v1.5, khoảng 20k ảnh) để bổ sung đặc trưng của Diffusion hiện đại vào tập train[cite: 4, 8]. | **A** | Tập dữ liệu train hỗn hợp: ProGAN 4-class + GenImage Subset[cite: 4, 8]. |
| **3.2** | **Xây dựng Data Augmentation 2 luồng:** Cấu hình xác suất: 30% ảnh giữ Clean (để chống sụt giảm hiệu năng trên ảnh gốc), 70% ảnh áp dụng ngẫu nhiên: JPEG ($Q \in [30, 95]$), Gaussian Blur ($\sigma \in [0.0, 3.0]$), và Double-JPEG[cite: 6, 7]. | **A** | Module `datasets/transforms.py` chống hiện tượng "học vẹt nén"[cite: 4, 6]. |
| **3.3** | **Hiện thực hóa nhánh High-Pass SRM:** Thiết kế module lọc vết dư điểm ảnh trước khi vào Conv của FAA[cite: 4, 7]: dùng ma trận Laplacian cố định để ép nhánh không gian học residual noise thay vì phụ thuộc vào màu sắc nội dung[cite: 4, 7]. | **B** | Module `SpatialResidualBlock` tích hợp an toàn trong `faa.py`[cite: 4, 7]. |
| **3.4** | **Cấu hình tham số huấn luyện:** Đóng băng 94.1% CLIP ViT-L/14[cite: 2, 7]. Chỉ mở khóa huấn luyện 5.9% (FAA, LGA, soft prompts, hệ số $\lambda$)[cite: 2, 7]. Thiết lập Optimizer AdamW ($lr = 2 \times 10^{-4}$ cho adapter, $lr = 10^{-4}$ cho prompt context) kết hợp Cosine Annealing[cite: 4, 6]. | **B** | File cấu hình `train_config.yaml`[cite: 6]. |
| **3.5** | Chuẩn bị script đánh giá tự động (validation hook) sau mỗi epoch để phát hiện sớm hiện tượng overfitting[cite: 6]. | **C** | Hook đánh giá tự động trong `train.py`[cite: 6]. |

### Tuần 4: Thực Thi Huấn Luyện Phân Tầng & Giám Sát Hội Tụ

| Mã việc | Nội dung công việc chi tiết | Người phụ trách | Sản phẩm bàn giao (Deliverables) |
| :--- | :--- | :---: | :--- |
| **4.1** | **Huấn luyện Phiên bản 1 (Robust Augment-Only):** Chạy 5–8 epoch trên tập dữ liệu hỗn hợp với pipeline tăng cường 2 luồng[cite: 6]. Bật Mixed Precision AMP FP16 kết hợp Gradient Accumulation (`batch_size = 16`, steps = 4 $\rightarrow$ `effective_batch_size = 64`)[cite: 7, 8]. | **B + C** | Checkpoint `fatformer_robust_v1.pth` + Log hội tụ Loss[cite: 6]. |
| **4.2** | **Huấn luyện Phiên bản 2 (Robust + SRM Filter):** Huấn luyện phiên bản có tích hợp bộ lọc vết dư không gian SRM trong FAA[cite: 4, 6]. | **B + C** | Checkpoint `fatformer_srm_v2.pth` + Log[cite: 6]. |
| **4.3** | Giám sát biến thiên của hệ số $\lambda$ (`freq_scale`): Kiểm chứng sự thích ứng của mô hình khi gặp ảnh nén nặng (hệ số $\lambda$ tự động giảm để chuyển ưu tiên sang nhánh không gian)[cite: 2, 4]. | **B** | Biểu đồ biến thiên giá trị trọng số $\lambda$ qua các epoch[cite: 2]. |
| **4.4** | Thiết lập cơ chế tự động sao lưu checkpoint `.pth` sau mỗi epoch sang Google Drive để phòng ngừa Colab ngắt kết nối session đột ngột[cite: 6, 8]. | **A** | Thư mục checkpoint backup tự động trên Google Drive[cite: 6, 8]. |

### Tuần 5: Đánh Giá Đa Chiều, Trực Quan Hóa XAI & Đóng Gói Đồ Án

| Mã việc | Nội dung công việc chi tiết | Người phụ trách | Sản phẩm bàn giao (Deliverables) |
| :--- | :--- | :---: | :--- |
| **5.1** | Chạy benchmark hoàn chỉnh so sánh 3 mô hình: (1) Baseline gốc, (2) Robust v1, (3) Robust SRM v2 trên toàn bộ 18 tập test ở cả 2 trạng thái: **Clean** và **Degraded ($Q \in \{30, 50, 70\}$)**[cite: 1, 6]. | **C** | Bảng ma trận số liệu tổng hợp cuối cùng[cite: 6]. |
| **5.2** | Phân tích triệt tiêu (Ablation Study): Định lượng mức đóng góp riêng rẽ của kỹ thuật Data Augmentation đa luồng và nhánh lọc vết dư SRM[cite: 1, 6]. | **B + C** | Bảng số liệu Ablation Study hoàn chỉnh cho báo cáo[cite: 6]. |
| **5.3** | Trực quan hóa XAI (Explainable AI): Xuất Grad-CAM so sánh cạnh nhau giữa mô hình gốc và mô hình finetuned khi gặp ảnh nén nặng để chứng minh mô hình mới tập trung đúng vào dị thường tạo tác thay vì viền khối nén (blocking artifacts)[cite: 6, 7]. | **C** | Bộ ảnh so sánh Grad-CAM độ phân giải cao[cite: 6, 7]. |
| **5.4** | Soạn thảo báo cáo kỹ thuật đồ án: Đặt vấn đề $\rightarrow$ Phương pháp thích ứng đa miền $\rightarrow$ Thiết kế chống suy biến $\rightarrow$ Thực nghiệm & Đánh giá định lượng[cite: 6]. | **Cả 3** *(C tổng hợp)* | Báo cáo đồ án hoàn chỉnh (PDF)[cite: 6]. |
| **5.5** | Thiết kế slide thuyết trình, kịch bản trả lời phản biện và chuẩn bị script `inference.py` chạy suy luận live trên ảnh tải từ Facebook/Telegram để demo trực tiếp[cite: 4, 6]. | **Cả 3** | Slide thuyết trình + Demo chạy trực tiếp[cite: 6]. |

---

## IV. BẢNG DỰ PHÒNG RỦI RO KỸ THUẬT TRÊN GOOGLE COLAB

| Rủi ro kỹ thuật | Mức độ | Biện pháp xử lý kỹ thuật |
| :--- | :---: | :--- |
| **Đọc Drive chậm (I/O Bottleneck)** | Rất cao | Tuyệt đối không đọc trực tiếp từng ảnh từ Drive mount. Luôn nén dataset thành 1 file `.tar`, copy về `/content/` của máy ảo Colab rồi mới giải nén ra đĩa SSD nội bộ[cite: 6]. |
| **Tràn RAM / VRAM GPU** | Cao | Đóng băng tuyệt đối 100% ViT và Text Encoder (`requires_grad = False`)[cite: 2, 7]. Bật PyTorch AMP FP16, dùng `batch_size = 16` kết hợp `gradient_accumulation_steps = 4`[cite: 7, 8]. |
| **Tụt hiệu năng trên ảnh Clean** | Trung bình | Giữ tỷ lệ 30% ảnh Clean không nén trong pipeline huấn luyện để neo giữ không gian phân tách gốc, tránh việc mô hình chỉ học phát hiện vết nén JPEG[cite: 6]. |
| **Mất kết nối Colab giữa chừng** | Cao | Trong vòng lặp train, sau mỗi epoch lưu `state_dict` vào local và gọi lệnh copy ngay file `.pth` sang Google Drive[cite: 6, 8]. Hỗ trợ cờ `--resume` nạp lại checkpoint gần nhất[cite: 6]. |
| **Lỗi nạp trọng số missing key** | Trung bình | Kiểm tra biến `patch_basaed_enhancer` trong `models/lga.py` (giữ nguyên lỗi chính tả chữ `a` thừa) để tương thích 100% với checkpoint gốc[cite: 2, 7]. |

---

## V. CHECKLIST TIÊU CHUẨN NGHIỆM THU ĐỒ ÁN (DONE CRITERIA)

- [ ] Bảng số liệu Baseline gốc chạy trên cả 2 tập Clean và Degraded ($Q \in \{30, 50, 70\}$)[cite: 6].
- [ ] Checkpoint đã finetune (`fatformer_robust_v1.pth` và `fatformer_srm_v2.pth`)[cite: 6].
- [ ] Bảng số liệu đối chứng trước và sau finetune, chứng minh độ bền tăng rõ rệt trên ảnh nén/mờ mà không làm sụt giảm tập Clean[cite: 6].
- [ ] Bảng Ablation Study định lượng vai trò của Augmentation và SRM[cite: 1, 6].
- [ ] Bộ ảnh trực quan hóa Grad-CAM so sánh đối đầu[cite: 6, 7].
- [ ] Mã nguồn dự án sạch sẽ, có file `README.md` hướng dẫn chạy lại từ đầu[cite: 6].
- [ ] Báo cáo đồ án hoàn chỉnh + Slide thuyết trình + Demo chạy suy luận ảnh thực tế[cite: 6].