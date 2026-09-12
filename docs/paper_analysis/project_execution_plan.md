# Kế Hoạch Triển Khai Kỹ Thuật Dự Án FatFormer
## Lộ Trình Thực Nghiệm, Pipeline Dữ Liệu, Chiến Lược Finetune & Nâng Cấp Kiến Trúc

> **Mục tiêu kỹ thuật**: Hiện thực hóa mô hình FatFormer, xây dựng pipeline dữ liệu chuẩn hóa, thực thi chiến lược finetune toàn diện trên các bộ dữ liệu sinh ảnh hiện đại, nghiên cứu cải tiến kiến trúc (Wavelet đa cấp, lọc vết dư nhiễu) và kiểm thử benchmark độ bền thực chiến.  
> **Tài liệu lý thuyết nền tảng**: [`docs/paper_analysis/fatformer_review.md`](file:///D:/GIT%20REPO/.nam4/fatformer/docs/paper_analysis/fatformer_review.md)  
> **Checkpoint & Trọng số sẵn có**:
> - CLIP ViT-L/14 Backbone: [`ViT-L-14.pt`](file:///D:/GIT%20REPO/.nam4/fatformer/ViT-L-14.pt) (932 MB)
> - Checkpoint tiền huấn luyện FatFormer: [`fatformer_4class_ckpt.pth`](file:///D:/GIT%20REPO/.nam4/fatformer/fatformer_4class_ckpt.pth) (1.97 GB)

---

## 1. Lộ Trình Triển Khai Kỹ Thuật 5 Giai Đoạn (Technical Roadmap)

```
[ Giai đoạn 1: Môi Trường & Thẩm Định Baseline ]
                       │
                       ▼
[ Giai đoạn 2: Thu Thập Dữ Liệu & Xây Dựng ETL Pipeline ]
                       │
                       ▼
[ Giai đoạn 3: Chiến Lược Finetune Tổng Thể (Adapter -> Partial Backbone) ]
                       │
                       ▼
[ Giai đoạn 4: Nghiên Cứu & Hiện Thực Hóa Nâng Cấp Kiến Trúc ]
                       │
                       ▼
[ Giai đoạn 5: Đánh Giá Benchmark Toàn Diện & Kiểm Thử Độ Bền (Robustness) ]
```

---

### Giai đoạn 1: Thiết Lập Môi Trường & Thẩm Định Trọng Số Baseline
*Mục tiêu*: Xác lập môi trường thực thi ổn định, kiểm tra tính toàn vẹn của 2 file trọng số có sẵn và chạy thử nghiệm suy luận đầu tiên.

1. **Chuẩn hóa môi trường Python/PyTorch**:
   - Python 3.10+, PyTorch $\ge$ 2.0 (tương thích CUDA 11.8 hoặc 12.x).
   - Thư viện xử lý sóng con: `PyWavelets` (`pywt` hoặc `pytorch_wavelets`).
   - Thư viện thị giác & phụ trợ: `torchvision`, `timm`, `ftfy`, `regex`, `scikit-learn`, `albumentations`.
2. **Khảo sát mã nguồn gốc**:
   - Tham chiếu kho mã nguồn chính thức [FatFormer GitHub](https://github.com/Michel-liu/FatFormer).
   - Thiết lập cấu trúc module nạp mô hình (`models/fatformer.py`, `models/faa.py`, `models/lga.py`).
3. **Thực thi script kiểm tra tính toàn vẹn (Sanity Check Script)**:
   - Viết kịch bản `tools/check_weights.py` nạp tệp [`ViT-L-14.pt`](file:///D:/GIT%20REPO/.nam4/fatformer/ViT-L-14.pt) và [`fatformer_4class_ckpt.pth`](file:///D:/GIT%20REPO/.nam4/fatformer/fatformer_4class_ckpt.pth).
   - Kiểm tra `state_dict` keys, khớp nối trọng số CLIP với nhánh Transformer, và các trọng số của FAA/LGA.
4. **Kiểm thử suy luận đơn lẻ (Single-image Inference Benchmark)**:
   - Dựng script `inference.py`: nhận ảnh đầu vào $\rightarrow$ resize 256, crop tâm 224 $\rightarrow$ trích xuất đặc trưng $\rightarrow$ tính xác suất Fake/Real.
   - Đo đạc thông số kỹ thuật cơ sở: Thời gian trễ (latency/image) và lượng VRAM chiếm dụng khi suy luận (ước tính: ~3.5 GB VRAM ở FP16).

---

### Giai đoạn 2: Thu Thập Dữ Liệu & Xây Dựng Pipeline Tiền Xử Lý (ETL)
*Mục tiêu*: Tải các bộ dữ liệu theo chiến lược phân tầng, chuẩn hóa cấu trúc thư mục và xây dựng pipeline DataLoader tối ưu tốc độ.

1. **Chiến lược phân tầng dữ liệu (Data Staging)**:
   - *Tầng 1 (Cơ sở)*: Tập kiểm thử ProGAN 4-class từ CNNDetection để tái lập (replicate) kết quả bài báo.
   - *Tầng 2 (Finetune diện rộng)*: GenImage Benchmark (lựa chọn các generator đại diện: Midjourney, Stable Diffusion v1.5, BigGAN).
   - *Tầng 3 (Chuyên sâu vi mô)*: DRCT-2M (subset SDXL, ControlNet) dùng để tinh chỉnh nhánh tần số DWT.
   - *Tầng 4 (Kiểm thử mù)*: UniversalFakeDetect (DALL-E 2, LDM).
2. **Chuẩn hóa cấu trúc thư mục dữ liệu cục bộ**:
   ```
   data/
   ├── CNNDetection/
   │   └── test/               # ProGAN, StyleGAN, BigGAN, CycleGAN...
   ├── GenImage/
   │   ├── midjourney/         # train / val (real, fake)
   │   ├── stable_diffusion/   # train / val (real, fake)
   │   └── biggan/             # train / val (real, fake)
   └── DRCT-2M/
       └── sdxl/               # train / val (real, fake)
   ```
3. **Xây dựng Data Pipeline & Data Augmentation**:
   - Áp dụng các biến đổi chống học vẹt (Anti-shortcut Augmentations):
     - `RandomResizedCrop(224, scale=(0.8, 1.0))`
     - `RandomHorizontalFlip(p=0.5)`
     - `JPEGCompression(quality_range=(50, 95), p=0.4)` mô phỏng ảnh mạng xã hội.
     - `GaussianBlur(kernel_size=(3, 3), p=0.2)` mô phỏng ảnh bị làm mờ nhẹ.
   - Đảm bảo cơ chế nạp dữ liệu đa luồng (`num_workers=4`, `pin_memory=True`).

---

### Giai đoạn 3: Chiến Lược Finetune Tổng Thể (Fine-tuning Campaign)
*Mục tiêu*: Nâng cao khả năng nhận diện trên các ảnh sinh bởi Diffusion thế hệ mới mà không gây quên lãng thảm họa (Catastrophic Forgetting).

```
   [ Giai đoạn 3.1: Adapter-Only Tuning ]
   - CLIP Backbone: ĐÓNG BĂNG HOÀN TOÀN (Frozen)
   - Trainable: Khối FAA, Khối LGA, Soft Prompts, Scale factor λ
   - Learning rate: 4e-4 (Warmup Cosine) | Epochs: 5 - 10
                       │
                       ▼
   [ Giai đoạn 3.2: Partial Backbone Unfreezing ]
   - Trainable: 2-3 Layers Transformer cuối của ViT-L/14 + FAA + LGA
   - Discriminative LR: Backbone: 1e-6 | Adapters: 5e-5 | Epochs: 3 - 5
```

1. **Giai đoạn 3.1 - Adapter-Only Tuning (Tinh chỉnh thích ứng)**:
   - Đóng băng toàn bộ trọng số gốc của ViT-L/14 Image Encoder và Transformer Text Encoder.
   - Chỉ cập nhật các tham số của FAA (tầng Conv không gian, Attention dải tần DWT, FFN), LGA (PBE, TGI, Soft prompt context) và $\lambda$.
   - *Hàm mất mát*: Augmented Contrastive Loss kết hợp giữa global CLS similarity $S(i)$ và local patch similarity $S'(i)$.
   - *Cấu hình tối ưu*: Optimizer AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-4}$), Learning rate $4 \times 10^{-4}$, Cosine Annealing scheduler.
2. **Giai đoạn 3.2 - Partial Backbone Fine-tuning (Mở băng chọn lọc)**:
   - Mở băng 2-3 tầng Transformer cuối cùng của ViT-L/14 để các đặc trưng ngữ nghĩa mức cao (*high-level semantic representations*) tự đồng bộ tốt hơn với không gian biểu diễn giả mạo.
   - Áp dụng kỹ thuật **Discriminative Learning Rate**: Đặt learning rate cho backbone cực nhỏ ($10^{-6}$) để bảo toàn kiến thức tiền huấn luyện của CLIP, trong khi giữ learning rate của Adapter ở mức $5 \times 10^{-5}$.
3. **Kỹ thuật quản lý bộ nhớ GPU khi huấn luyện**:
   - Bật **Mixed Precision (AMP FP16 hoặc BF16)** giúp giảm 50% VRAM và tăng tốc x2 tốc độ tính toán.
   - Sử dụng **Gradient Accumulation (steps=2 hoặc 4)** nếu kích thước batch bị giới hạn bởi phần cứng (đạt effective batch size = 64).

---

### Giai đoạn 4: Nghiên Cứu & Hiện Thực Hóa Nâng Cấp Kiến Trúc (Model Upgrades)
*Mục tiêu*: Đột phá khỏi giới hạn của bài báo gốc, mở rộng năng lực xử lý dấu vết vi mô của ảnh Diffusion hiện đại.

1. **Nâng cấp 1: Multi-Scale Wavelet Decomposition (DWT Đa Cấp Độ)**:
   - *Vấn đề*: 1-level DWT chỉ phân rã 1 lần, dải tần $LL$ vẫn chứa nhiều năng lượng tần số hỗn hợp.
   - *Giải pháp*: Áp dụng 2-level DWT trên dải tần thấp: $LL_1 \rightarrow \{LL_2, LH_2, HL_2, HH_2\}$. Cơ chế Grouped Attention sẽ học mối tương quan giữa dải tần chi tiết cấp 1 ($HH_1$ - tần số cực cao) và cấp 2 ($HH_2$ - tần số trung cao).
2. **Nâng cấp 2: Spatial Noise Residual Fusion (Bổ sung Nhánh Lọc Vết Dư Nhiễu SRM)**:
   - *Vấn đề*: Nhánh không gian của FAA dùng mạng Conv thông thường dễ bị chi phối bởi nội dung ngữ nghĩa của vật thể thay vì vết nhiễu sinh ảnh.
   - *Giải pháp*: Tích hợp bộ lọc High-Pass SRM (Spatial Rich Model - 3 kernels chuẩn trong pháp y hình ảnh: $1^{st}$ order, $2^{nd}$ order, và Laplacian) vào trước tầng Conv của nhánh không gian, ép mô hình chỉ nhìn vào vết nhiễu nội suy điểm ảnh.
3. **Nâng cấp 3: Dynamic Mixture-of-Adapters (MoA)**:
   - Thiết kế 2 adapter chuyên biệt: *GAN-Adapter* (tối ưu cho checkerboard artifact) và *Diffusion-Adapter* (tối ưu cho gradient mismatch). Một mạng cổng nhẹ (*Gating Network*) tự động tính trọng số pha trộn dựa trên đặc trưng ảnh đầu vào.
4. **Nâng cấp 4: Fine-Grained Semantic Text Prompting**:
   - Nâng cấp nhãn phân loại: thay vì chỉ `"real"` vs `"fake"`, xây dựng từ điển prompt mô tả chi tiết:
     - Thật: `"a pristine camera captured photograph with natural optical sensor noise"`
     - Giả: `"a synthetic image created by generative artificial intelligence with digital artifacts"`

---

### Giai đoạn 5: Đánh Giá Benchmark Toàn Diện & Kiểm Thử Độ Bền (Robustness)
*Mục tiêu*: Định lượng chính xác năng lực tổng quát hóa và độ bền thực chiến của mô hình sau finetune/nâng cấp.

1. **Hệ thống chỉ số đánh giá (Evaluation Metrics)**:
   - **Accuracy (ACC)**: Tỷ lệ dự đoán đúng tại ngưỡng chuẩn 0.5.
   - **Average Precision (AP)**: Đánh giá diện tích dưới đường Precision-Recall (không phụ thuộc việc chọn ngưỡng).
   - **Area Under ROC (AUC-ROC)**: Đo lường độ tách biệt giữa phân bố ảnh thật và ảnh giả.
   - **Equal Error Rate (EER)**: Điểm cân bằng giữa tỷ lệ báo động giả (FPR) và tỷ lệ bỏ sót (FNR).
2. **Kịch bản kiểm thử độ bền trước nén ảnh mạng xã hội (Robustness Stress Testing)**:
   - Đánh giá mô hình trên tập kiểm thử khi bị suy biến qua các mức nén JPEG: Quality = 95, 85, 75, 65, 50.
   - Đánh giá khi bị áp bộ lọc Gaussian Blur ($\sigma = 1, 2, 3$).
   - Vẽ biểu đồ so sánh độ suy giảm hiệu năng giữa: *FatFormer Gốc* vs *FatFormer Finetuned* vs *FatFormer Nâng Cấp*.

---

## 2. Danh Mục Bộ Dữ Liệu Chi Tiết Kèm Nguồn Tải Hợp Lý (Dataset Registry)

Bảng tổng hợp các bộ dữ liệu tiêu chuẩn cho bài toán AIGC Detection, đã được xác minh nguồn tải:

| Bộ Dữ Liệu | Quy Mô & Đặc Trưng | Các Mô Hình Sinh (Generators) | Vai Trò Trong Dự Án | Nguồn Tải Chính Thức & Mirrors |
| :--- | :--- | :--- | :--- | :--- |
| **CNNDetection** *(Wang et al., CVPR 2020)* | ~36k ảnh train, 80k ảnh test | ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, Deepfake | **Baseline Replicate**: Dùng để kiểm thử khớp kết quả với checkpoint [`fatformer_4class_ckpt.pth`](file:///D:/GIT%20REPO/.nam4/fatformer/fatformer_4class_ckpt.pth). | - [GitHub CNNDetection](https://github.com/peterwang512/CNNDetection)<br>- [OneDrive Mirror](https://1drv.ms/u/s!Aqkrc9gPuk8jqaM2khCvAejz_K4Jow?e=e0yeDQ)<br>- [Baidu Mirror (Mã: 9i5w)](https://pan.baidu.com/s/1obzmrCsWvGyUlmH8MkSTLA) |
| **GenImage Benchmark** *(NeurIPS 2023)* | 1,331,167 cặp ảnh thật/giả (kích thước lớn) | Midjourney, Stable Diffusion v1.4, v1.5, BigGAN, ADM, GLIDE, Wukong, VQ-DM | **Trục Xương Sống Finetune**: Cung cấp đa dạng phong cách từ các mô hình sinh mạnh nhất hiện nay. | - [GitHub GenImage](https://github.com/GenImage-Dataset/GenImage)<br>- [Google Drive Tải Trực Tiếp](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS)<br>- [Baidu Yunpan (Mã: ztf1)](https://pan.baidu.com/s/1z-ztf1) |
| **DRCT-2M** *(ICML 2024)* | 2,000,000 ảnh chất lượng cao | 16 biến thể Diffusion: SDXL, SD-Turbo, SD-2.1, ControlNet, PixArt... | **Tinh Chỉnh Sâu Tần Số**: Cung cấp các vết nhiễu vi mô của các kiến trúc Latent Diffusion đời mới. | - [GitHub DRCT](https://github.com/beibuwandeluori/DRCT)<br>- [Hugging Face DRCT Dataset](https://huggingface.co/datasets/beibuwandeluori/DRCT-2M) |
| **UniversalFakeDetect** *(CVPR 2023)* | Tập kiểm thử chuyên biệt (~40k ảnh) | DALL-E 2, Latent Diffusion (LDM), Guided Diffusion, VQ-Diffusion | **Kiểm Thử Unseen**: Đánh giá khả năng tổng quát hóa trên các mô hình thương mại kín nguồn. | - [GitHub UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect) |
| **DIRE Dataset** *(ICCV 2023)* | Tập kiểm thử lỗi tái cấu trúc Diffusion | ADM, LDM, PNDM, Glide | **Kiểm Thử Đối Chứng**: Đánh giá hiệu năng nhận diện vết tái cấu trúc diffusion. | - [GitHub DIRE](https://github.com/ZhendongWang6/DIRE)<br>- [Baidu Netdisk (Mã: dire)](https://pan.baidu.com/s/1zoubPr5n_mGI27En9uyL8Q) |

### Khuyến nghị Chiến lược Tải Dữ Liệu Thực Tế:
- Không cần tải toàn bộ 300GB của GenImage ngay lập tức.
- **Chiến lược tối ưu**:
  1. Tải trước **CNNDetection testset** (~5GB) để viết code test baseline.
  2. Tải 2 thư mục con đại diện nhất của **GenImage** (Midjourney + Stable Diffusion v1.5, khoảng 30GB-40GB) để bắt đầu pipeline finetune.
  3. Mở rộng thêm subset của **DRCT-2M (SDXL)** sau khi pipeline huấn luyện đã chạy ổn định.

---

## 3. Danh Mục Nguồn Tham Chiếu Ngoại Vi (External Codebases & Papers)

### 3.1. Các Kho Mã Nguồn Kỹ Thuật (GitHub Repositories)
1. **FatFormer Official Codebase**:  
   👉 [https://github.com/Michel-liu/FatFormer](https://github.com/Michel-liu/FatFormer)  
   *Công dụng*: Cung cấp cấu trúc class của FAA, LGA, script nạp trọng số ViT-L-14 và cấu hình huấn luyện ProGAN 4-class.
2. **UniversalFakeDetect (UniFD)**:  
   👉 [https://github.com/Yuheng-Li/UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect)  
   *Công dụng*: Cơ sở so sánh baseline SOTA dùng CLIP ViT-L/14 đông cứng + Linear probing.
3. **PyWavelets (Thư viện biến đổi sóng con DWT)**:  
   👉 [https://github.com/PyWavelets/pywt](https://github.com/PyWavelets/pywt) & [PyTorch Wavelets](https://github.com/fbcotter/pytorch_wavelets)  
   *Công dụng*: Cung cấp 2D DWT / IDWT hỗ trợ tính toán trực tiếp trên tensor GPU và backward gradient.
4. **OpenCLIP**:  
   👉 [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)  
   *Công dụng*: Tham chiếu chuẩn về kiến trúc Vision-Language transformer và tokenizer.
5. **Awesome AIGC Image Detection (Kho tổng hợp tài nguyên)**:  
   👉 [https://github.com/graydove/Awesome-AIGC-Image-Detection](https://github.com/graydove/Awesome-AIGC-Image-Detection)  
   *Công dụng*: Cập nhật liên tục các bài báo, mã nguồn đối chứng và liên kết tải dataset mới nhất (2024-2025).

### 3.2. Các Bài Báo Khoa Học Đối Chứng Then Chốt
- **FatFormer**: *Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection* (CVPR 2024) - [arXiv:2312.16649](https://arxiv.org/abs/2312.16649)
- **UniFD**: *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (CVPR 2023) - [arXiv:2302.10174](https://arxiv.org/abs/2302.10174)
- **CNNDetection**: *CNN-generated images are surprisingly easy to spot... for now* (CVPR 2020) - [arXiv:1912.11035](https://arxiv.org/abs/1912.11035)
- **DIRE**: *Diffusion Reconstruction Error for Synthesized Image Detection* (ICCV 2023) - [arXiv:2303.09295](https://arxiv.org/abs/2303.09295)
- **GenImage**: *GenImage: A Hundred-Million-Scale Benchmark for Image Generation Detection* (NeurIPS 2023) - [arXiv:2306.08571](https://arxiv.org/abs/2306.08571)
- **DRCT**: *DRCT: Diffusion Reconstruction with Contrastive Training* (ICML 2024) - [arXiv:2403.03045](https://arxiv.org/abs/2403.03045)

---

## 4. Bảng Đánh Giá Rủi Ro Kỹ Thuật & Giải Pháp Dự Phòng (Risk Matrix)

| Rủi Ro Kỹ Thuật | Mức Độ | Tác Động Dự Kiến | Phương Án Ứng Phó & Khắc Phục |
| :--- | :---: | :--- | :--- |
| **Tràn bộ nhớ GPU (OOM VRAM)** khi train ViT-L/14 | **Cao** | Huấn luyện bị sập giữa chừng, không thể tăng batch size. | 1. Đóng băng 100% CLIP backbone, chỉ tính đạo hàm cho FAA/LGA.<br>2. Bật PyTorch AMP FP16/BF16.<br>3. Áp dụng Gradient Accumulation để giữ batch size nhỏ (8 hoặc 16) nhưng effective batch size lớn (64).<br>4. Bật PyTorch `torch.utils.checkpoint` cho các tầng attention. |
| **Suy giảm độ chính xác khi gặp nén JPEG** | **Trung bình** | Nhánh tần số DWT bị mất đặc trưng khi kiểm thử ảnh mạng xã hội. | 1. Tích hợp nén JPEG ngẫu nhiên vào ngay pipeline DataLoader khi finetune.<br>2. Huấn luyện tham số cân bằng thích nghi $\lambda$ để tự giảm trọng số nhánh tần số khi ảnh bị nén mạnh. |
| **Quá tải dung lượng lưu trữ cục bộ** | **Trung bình** | Ổ cứng bị đầy khi tải các dataset quy mô lớn (GenImage ~300GB, DRCT ~500GB). | 1. Áp dụng chiến lược tải phân tầng (Staged download).<br>2. Chỉ tải các subset đại diện (Midjourney + SD v1.5 ~35GB).<br>3. Nén ảnh về định dạng nén tối ưu trước khi nạp. |
| **Xung đột phiên bản thư viện DWT** | **Thấp** | Lỗi gradient khi backward qua tầng 2D Haar DWT / IDWT. | 1. Triển khai 2D Haar Wavelet thuần bằng PyTorch 2D Conv/ConvTranspose với ma trận trọng số Haar cố định, không phụ thuộc thư viện ngoài C++. |

---

## 5. Tổng Kết & Mốc Hành Động Kế Tiếp

Kế hoạch này vạch ra toàn bộ lộ trình kỹ thuật cho dự án. Bước triển khai cụ thể kế tiếp theo Giai đoạn 1:
1. Viết script kiểm tra khớp trọng số giữa [`ViT-L-14.pt`](file:///D:/GIT%20REPO/.nam4/fatformer/ViT-L-14.pt) và [`fatformer_4class_ckpt.pth`](file:///D:/GIT%20REPO/.nam4/fatformer/fatformer_4class_ckpt.pth).
2. Xây dựng module suy luận cơ bản để kiểm thử với một vài ảnh thực tế.
