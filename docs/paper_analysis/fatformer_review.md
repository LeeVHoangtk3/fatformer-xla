# Đánh Giá Chuyên Sâu Bài Báo FatFormer (CVPR 2024)
## Nền Tảng Kỹ Thuật, Kế Hoạch Finetune Tổng Thể & Lộ Trình Nâng Cấp Mô Hình

> **Tên bài báo gốc**: *Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection*  
> **Tác giả**: Huan Liu, Zichang Tan, Chuangchuang Tan, Yunchao Wei, Yao Zhao, Jingdong Wang (Beijing Jiaotong University & Baidu VIS)  
> **Hội nghị / Nguồn**: CVPR 2024 / [arXiv:2312.16649v1](https://arxiv.org/abs/2312.16649)  
> **Tệp PDF lưu trữ cục bộ**: [`docs/sources/7. 2312.16649v1.pdf`](file:///D:/GIT%20REPO/.nam4/fatformer/docs/sources/7.%202312.16649v1.pdf)  
> **Trọng số liên quan trong dự án**:
> - CLIP ViT-L/14 Backbone: [`ViT-L-14.pt`](file:///D:/GIT%20REPO/.nam4/fatformer/ViT-L-14.pt) (932 MB)
> - Checkpoint tiền huấn luyện FatFormer: [`fatformer_4class_ckpt.pth`](file:///D:/GIT%20REPO/.nam4/fatformer/fatformer_4class_ckpt.pth) (1.97 GB)

---

## 1. Tổng Quan & Bối Cảnh Nghiên Cứu

### 1.1. Bài toán: Khả năng tổng quát hóa trong phát hiện ảnh giả mạo (Generalizable Synthetic Image Detection)
Sự phát triển bùng nổ của các mô hình sinh ảnh (Generative Adversarial Networks - GANs như StyleGAN, ProGAN và Diffusion Models như LDM, SDXL, Midjourney) đặt ra thách thức bảo mật nghiêm trọng. Hầu hết các bộ phát hiện truyền thống gặp phải hiện tượng **suy giảm hiệu năng trầm trọng khi kiểm thử trên các mô hình sinh chưa từng thấy (unseen generators)**.

### 1.2. Hạn chế của mô hình tiền nhiệm (UniFD)
- Phương pháp tiên tiến trước đó là **UniFD** (Ojha et al., CVPR 2023) sử dụng mô hình thị giác - ngôn ngữ **CLIP ViT-L/14** được đóng băng hoàn toàn (*frozen*), chỉ huấn luyện một lớp phân loại tuyến tính (*linear classifier*).
- **Điểm nghẽn của UniFD**: Việc đóng băng hoàn toàn khiến mô hình không thể thích ứng (*lack of forgery adaptation*) với các dấu vết giả mạo đặc thù (vết mờ texture vi mô, sai lệch phổ tần số). Phân bố logit của UniFD bị chồng lấn lớn giữa ảnh thật và ảnh giả trên các mô hình unseen.

### 1.3. Đột phá cốt lõi của FatFormer
FatFormer giải quyết vấn đề bằng nguyên lý **Tương thích Giả mạo (Forgery Adaptation)** thông qua việc giữ lại không gian biểu diễn tổng quát của CLIP nhưng chèn thêm các khối thích ứng siêu nhẹ (Parameter-Efficient Adapters) kết hợp giữa hai miền:
1. **Miền không gian (Spatial Domain)**: Trích xuất các lỗi hiển thị cục bộ (blur textures, color mismatch).
2. **Miền tần số (Frequency Domain)**: Phân tách biến đổi sóng con rời rạc (Discrete Wavelet Transform - DWT) để nắm bắt các biến dạng phổ tần số mà mắt thường không thấy.
3. **Giám sát liên kết ngôn ngữ (Language-Guided Alignment - LGA)**: Sử dụng mục tiêu đối sánh (*contrastive objectives*) thay vì hàm mất mát Cross-Entropy thông thường, tối ưu hóa sự tương đồng giữa patch ảnh và prompt văn bản mềm (*soft prompts*).

---

## 2. Giải Phẫu Chi Tiết Kiến Trúc FatFormer

```
                           +----------------------------------------+
                           |           Input Image (224x224)        |
                           +-------------------+--------------------+
                                               |
                                    [ Patch Embedding ]
                                               |
                     +-------------------------v-------------------------+
                     |    Stage 1..N: Frozen CLIP ViT-L/14 Layers        |
                     +-------------------------+-------------------------+
                                               |
                    +--------------------------v--------------------------+
                    |           FORGERY-AWARE ADAPTER (FAA)               |
                    |                                                     |
                    |  [Spatial Branch]            [Frequency Branch]     |
                    |   Conv -> ReLU -> Conv        2D DWT (LL,LH,HL,HH)  |
                    |                                       |             |
                    |                              Inter-Band Attention   |
                    |                              Intra-Band Attention   |
                    |                                       |             |
                    |                                  IDWT + FFN         |
                    |                                       |             |
                    |       Adapted Feature = Spatial + λ * Frequency     |
                    +--------------------------+--------------------------+
                                               |
                     +-------------------------v-------------------------+
                     |         LANGUAGE-GUIDED ALIGNMENT (LGA)           |
                     |                                                   |
                     |  Patch-Based Enhancer: Condition soft prompts on  |
                     |                        image patch tokens         |
                     |  Text-Guided Interactor: Align patch tokens with  |
                     |                          global text embeddings   |
                     +-------------------------+-------------------------+
                                               |
                     [ Augmented Contrastive Loss (CLS Sim + Patch Sim)  ]
```

### 2.1. Khối Thích Ứng Giả Mạo (Forgery-Aware Adapter - FAA)
FAA được chèn vào giữa các phân đoạn (stages) của Vision Transformer:
- **Nhánh không gian**: Gồm hai tầng Convolution kích thước nhỏ kèm kích hoạt ReLU:
  $$\hat{g}_{img}^{(j)} = \text{Conv}(\text{ReLU}(\text{Conv}(g^{(j)}_{img})))$$
- **Nhánh tần số (DWT + Grouped Attention)**:
  - Khác với FFT hay DCT làm mất thông tin tọa độ không gian (*spatial locality*), FatFormer chọn **2D Haar Discrete Wavelet Transform (DWT)**. DWT phân rã đặc trưng thành 4 dải tần: $LL$ (xấp xỉ tần số thấp), $LH$ (chi tiết ngang tần số cao), $HL$ (chi tiết dọc), và $HH$ (chi tiết chéo).
  - **Grouped Attention**:
    - *Inter-band Attention*: Cho phép các dải tần số khác nhau tương tác, chia sẻ thông tin tương quan chéo giữa tần số thấp và cao.
    - *Intra-band Attention*: Học tương tác không gian cục bộ trong nội bộ từng dải tần số.
  - Tái tạo lại miền không gian thông qua Phép biến đổi sóng con ngược (IDWT) và FFN.
- **Tổng hợp thích nghi**:
  $$\hat{g}^{(j)} = \hat{g}_{img}^{(j)} + \lambda \cdot \hat{g}_{freq}^{(j)}$$
  *(với $\lambda$ là tham số học được, tự động cân bằng tỷ trọng thông tin giữa không gian và tần số).*

### 2.2. Khối Căn Chỉnh Dẫn Hướng Ngôn Ngữ (Language-Guided Alignment - LGA)
- **Patch-based Enhancer (PBE)**: Không sử dụng prompt tĩnh cứng như `"this is a [CLASS] photo"`, FatFormer dùng các vector nhúng ngữ cảnh học được $p_{ctx}$. PBE tính ma trận tương đồng $A_{pbe} = p_{ctx} \cdot (f_{img}^{(1:N)})^T$ để tạo ra prompt thích ứng theo từng patch ảnh:
  $$\hat{p}_{ctx} = \text{softmax}(A_{pbe}) \cdot f_{img}^{(1:N)} + p_{ctx}$$
- **Text-guided Interactor (TGI)**: Căn chỉnh ngược lại từ vector nhúng text $f_{text}$ về các token patch ảnh:
  $$\hat{f}_{img}^{(1:N)} = \text{softmax}(A_{tgi}) \cdot f_{text} + f_{img}^{(1:N)}$$
- **Augmented Contrastive Loss**:
  Kết hợp độ tương đồng toàn cục (CLS token) $S(i)$ và trung bình độ tương đồng cục bộ của các patch $\hat{f}_{img}^{(t)}$:
  $$S'(i) = \frac{1}{N} \sum_{t=1}^N \cos(\hat{f}_{img}^{(t)}, f_{text}^{(i)})$$
  Xác suất dự đoán: $\hat{P}(i) = \text{softmax}\left(\frac{S(i) + S'(i)}{\tau}\right)$.

---

## 3. Đánh Giá Khách Quan: Điểm Mạnh, Điểm Nghẽn & Giới Hạn Thực Tế

### 3.1. Điểm mạnh vượt trội
- **Hiệu năng Zero-shot Generalization ấn tượng**: Dù chỉ huấn luyện trên 4 lớp đối tượng của ProGAN (car, cat, chair, horse), mô hình đạt **98.4% ACC** trên 8 họ GAN khác nhau và **95.0% ACC** trên 10 mô hình Diffusion chưa từng thấy.
- **Tiết kiệm tài nguyên (Parameter-Efficient)**: Giữ đông cứng 95%+ tham số của CLIP ViT-L/14, chỉ huấn luyện các adapter FAA và tham số prompt của LGA, giảm thiểu nguy cơ catastrophic forgetting và overfit.
- **Giữ trọn thông tin không gian trong miền tần số**: DWT khắc phục nhược điểm mất tọa độ của FFT/DCT, giúp cơ chế Attention định vị chính xác vị trí phát sinh artifact.

### 3.2. Điểm nghẽn & Giới hạn kỹ thuật cần khắc phục
1. **Bias dữ liệu huấn luyện cổ điển**: Bộ dữ liệu huấn luyện ProGAN (2018) có cấu trúc artifact dạng lưới (checkerboard artifacts) rất khác so với các mô hình Diffusion hiện đại (SDXL, Midjourney v6, FLUX.1).
2. **Độ phân giải cố định thấp (224x224)**: Quá trình crop/resize về 224x224 làm triệt tiêu các vết nhiễu vi mô tần số cao (*micro-textures*), vốn là dấu vết cốt tử để phát hiện ảnh sinh bởi diffusion chất lượng cao.
3. **Độ bền trước nén ảnh mạng xã hội (Robustness)**: Biến đổi tần số DWT rất nhạy cảm với thuật toán nén mất mát JPEG và bộ lọc Gaussian Blur. Khi ảnh bị nén mạnh khi tải lên Facebook/Telegram/TikTok, độ chính xác của nhánh DWT có thể sụt giảm đáng kể.
4. **DWT 1 cấp độ (Single-level DWT)**: Chỉ phân rã 1 cấp (1-level DWT) có thể chưa đủ chi tiết để tách biệt các tần số siêu cao (vùng biên viền sắc nhọn của chi tiết tóc, mắt).

---

## 4. Kế Hoạch Finetune Tổng Thể (Comprehensive Fine-tuning Strategy)

Để nâng tầm FatFormer đáp ứng tốt dữ liệu ảnh AI thế hệ mới, chiến lược finetune cần được thiết kế theo 4 giai đoạn mạch lạc:

### 4.1. Chiến lược đóng băng và mở băng tham số (Layer-wise Unfreezing)
- **Giai đoạn 1 (Warmup Adapter)**:
  - *Đóng băng hoàn toàn*: CLIP ViT-L/14 Image & Text Encoders.
  - *Chỉ huấn luyện*: Khối FAA (Conv, DWT Inter/Intra Attention), khối LGA (PBE, TGI, Soft Prompts) và scale factor $\lambda$.
  - *Tốc độ học (Learning Rate)*: $10^{-4}$ đến $4 \times 10^{-4}$.
- **Giai đoạn 2 (Partial Backbone Fine-tuning)**:
  - *Mở băng*: 2-3 tầng Transformer cuối cùng của ViT-L/14 Image Encoder.
  - *Tốc độ học*: Sử dụng Discriminative Learning Rate:
    - Backbone ViT layers cuối: $10^{-6} - 5 \times 10^{-6}$ (rất nhỏ để tránh làm vỡ biểu diễn tổng quát).
    - Adapters FAA & LGA: $5 \times 10^{-5}$.

### 4.2. Tuyển chọn tập dữ liệu huấn luyện mục tiêu (Dataset Selection)
Dựa trên tài nguyên được ghi nhận tại [`docs/sources/links.md`](file:///D:/GIT%20REPO/.nam4/fatformer/docs/sources/links.md):

| Tập dữ liệu | Số lượng & Đặc trưng | Mục đích trong Finetune |
| :--- | :--- | :--- |
| **GenImage Benchmark** *(NeurIPS 2023)* | > 1 triệu ảnh, 8 generator (Midjourney, SD v1.4/v1.5, GLIDE, BigGAN, VQ-DM) | **Trục xương sống**: Cung cấp đa dạng phong cách ảnh thực tế và các họ mô hình sinh hiện đại. |
| **DRCT-2M** *(ICML 2024)* | 2 triệu ảnh từ 16 biến thể Diffusion (SDXL, SD-Turbo, ControlNet) | **Tinh chỉnh sâu**: Huấn luyện FAA bắt các vết nhiễu vi mô đặc thù của các kiến trúc Latent Diffusion cỡ lớn. |
| **CommunityForensics** *(CVPR 2025)* | 50.000 ảnh từ 21+ mô hình thương mại kèm prompt nhúng | **Đánh giá thực chiến**: Kiểm thử độ bền và khả năng bắt ảnh sinh thương mại (Midjourney v6, DALL-E 3). |

### 4.3. Chiến lược tăng cường dữ liệu chống suy biến (Robustness Augmentation)
Để mô hình hoạt động bền vững ngoài đời thực, pipeline dữ liệu bắt buộc phải tích hợp:
- **JPEG Compression Augmentation**: Nén ngẫu nhiên với Quality Factor $Q \in [50, 95]$.
- **Gaussian Blur / Noise Injection**: Thêm nhiễu ngẫu nhiên mô phỏng ảnh chụp điện thoại hoặc sensor camera.
- **Random Resized Crop (Scale Jittering)**: Đảm bảo mô hình không học vẹt vị trí patch.

---

## 5. Lộ Trình & Đề Xuất Nâng Cấp Mô Hình (Model Upgrade Roadmap)

```
[ Hiện Tại: FatFormer CVPR 2024 ]
   |
   +--> Nâng Cấp 1: Multi-Scale Wavelet Decomposition (2-Level DWT / Wavelet Packet)
   |
   +--> Nâng Cấp 2: SRM / High-Pass Residual Fusion (Bổ sung nhánh vết dư nhiễu không gian)
   |
   +--> Nâng Cấp 3: Dynamic Mixture of Adapters (MoA) phân tách chuyên biệt GAN vs Diffusion
   |
   +--> Nâng Cấp 4: Đa dạng hóa Prompt Ngữ Nghĩa (Fine-grained Forgery Semantic Alignment)
```

### Đề xuất 1: Multi-Scale Wavelet Decomposition (DWT đa cấp độ)
- **Cơ chế**: Thay vì 1-level DWT chỉ chia làm 4 band, áp dụng 2-level DWT để phân tích sâu dải tần $LL$ thành $\{LL_2, LH_2, HL_2, HH_2\}$.
- **Lợi ích**: Tách bạch được dải tần trung bình (nơi chứa cấu trúc ngữ nghĩa khuôn mặt/vật thể) và dải tần siêu cao (nơi xuất hiện nhiễu sinh ảnh của FLUX và SDXL).

### Đề xuất 2: Bổ sung Nhánh Vết Dư Không Gian (High-Pass Residual / SRM Filter)
- **Cơ chế**: Đặt thêm một bộ lọc Spatial Rich Model (SRM) 3 kênh hoặc Laplacian of Gaussian (LoG) trước khi đưa vào nhánh không gian của FAA.
- **Lợi ích**: Loại bỏ nội dung ngữ nghĩa thông thường của bức ảnh, buộc nhánh không gian phải tập trung 100% vào vết nhiễu nội suy điểm ảnh (*pixel interpolation artifacts*).

### Đề xuất 3: Dynamic Mixture-of-Adapters (MoA) cho từng trường phái sinh ảnh
- **Cơ chế**: Sử dụng một Gating Network nhẹ để tự động phân phối trọng số giữa 2 adapter chuyên biệt:
  - *Adapter chuyên trị GAN*: Ưu tiên các dải tần số đối xứng và vết lưới checkerboard.
  - *Adapter chuyên trị Diffusion*: Ưu tiên độ lệch chuẩn gradient và sự thiếu nhất quán màu sắc vi mô.

### Đề xuất 4: Cải tiến Prompt Ngôn Ngữ Chuyên Sâu (Fine-grained Prompting)
- Thay vì chỉ dùng nhãn đối lập `["real", "fake"]`, mở rộng bộ nhãn giám sát thành:
  - `["authentic photograph with natural sensor noise", "synthetic image generated by artificial intelligence with generative artifacts"]`.
  - Giúp không gian CLIP kích hoạt các đặc trưng ngữ nghĩa tinh vi hơn.

---

## 6. Tổng Kết & Kế Hoạch Hành Động Kế Tiếp

Tài liệu này xác lập nền tảng lý thuyết và định hướng kỹ thuật cho toàn bộ dự án. Các bước hành động cụ thể tiếp theo trong workspace:
1. **Kiểm tra môi trường & Checkpoint**: Viết script kiểm tra khả năng nạp trọng số của [`ViT-L-14.pt`](file:///D:/GIT%20REPO/.nam4/fatformer/ViT-L-14.pt) và [`fatformer_4class_ckpt.pth`](file:///D:/GIT%20REPO/.nam4/fatformer/fatformer_4class_ckpt.pth).
2. **Dựng Pipeline Suy Luận (Inference Pipeline)**: Kiểm thử suy luận trên một số mẫu ảnh thực/giả để đo đạc baseline ban đầu.
3. **Thiết lập Data Loader & Pipeline Finetune**: Xây dựng cấu trúc nạp dữ liệu GenImage / ProGAN theo các đề xuất tại Mục 4.
4. **Hiện thực hóa cải tiến kiến trúc**: Triển khai thử nghiệm Multi-scale DWT trên nhánh FAA.
