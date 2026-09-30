# ĐỒ ÁN TỐT NGHIỆP ĐẠI HỌC
## NÂNG CAO ĐỘ BỀN VỮNG PHÁP Y CHO MÔ HÌNH FATFORMER TRƯỚC BIẾN DẠNG NÉN MẠNG XÃ HỘI (FATFORMER-XLA)

> **Nhóm sinh viên thực hiện**:
> - Thành viên A: *Kỹ sư Dữ liệu & Hạ tầng Lưu trữ (Data & Storage Lead)*
> - Thành viên B: *Kỹ sư Kiến trúc Mô hình & Hàm Mất Mát (Architecture & Loss Lead)*
> - Thành viên C: *Trưởng nhóm Huấn luyện & Đánh giá Thực nghiệm (Training & Evaluation Lead)*  
> **Chuyên ngành**: Khoa học Máy tính / Trí tuệ Nhân tạo  
> **Thời gian thực hiện**: 2026  

---

# MỤC LỤC BẢN THẢO

- [CHƯƠNG 1: GIỚI THIỆU VÀ ĐỘNG LỰC NGHIÊN CỨU](#chương-1-giới-thiệu-và-động-lực-nghiên-cứu)
  - [1.1. Bối cảnh bùng nổ của AI tạo sinh và thách thức Deepfake](#11-bối-cảnh-bùng-nổ-của-ai-tạo-sinh-và-thách-thức-deepfake)
  - [1.2. Thách thức thực tiễn: Sự suy giảm độ bền vững pháp y trên không gian mạng](#12-thách-thức-thực-tiễn-sự-suy-giảm-độ-bền-vững-pháp-y-trên-không-gian-mạng)
  - [1.3. Phân tích lỗ hổng kỹ thuật của mô hình FatFormer (CVPR 2024)](#13-phân-tích-lỗ-hổng-kỹ-thuật-của-mô-hình-fatformer-cvpr-2024)
  - [1.4. Mục tiêu nghiên cứu và Các đóng góp khoa học của đồ án](#14-mục-tiêu-nghiên-cứu-và-các-đóng-góp-khoa-học-của-đồ-án)
  - [1.5. Bố cục của đồ án](#15-bố-cục-của-đồ-án)
- [CHƯƠNG 2: CÁC CÔNG TRÌNH NGHIÊN CỨU LIÊN QUAN](#chương-2-các-công-trình-nghiên-cứu-liên-quan)
  - [2.1. Tiến trình phát triển của các phương pháp phát hiện ảnh tạo bởi AI](#21-tiến-trình-phát-triển-của-các-phương-pháp-phát-hiện-ảnh-tạo-bởi-ai)
  - [2.2. Phương pháp phân tích miền tần số và biến đổi Wavelet trong pháp y số](#22-phương-pháp-phân-tích-miền-tần-số-và-biến-đổi-wavelet-trong-pháp-y-số)
  - [2.3. Kỹ thuật trích xuất vết dư không gian (Spatial Rich Models - SRM)](#23-kỹ-thuật-trích-xuất-vết-dư-không-gian-spatial-rich-models---srm)
  - [2.4. Học tăng cường độ bền vững và chiến lược thích ứng giáo trình](#24-học-tăng-cường-độ-bền-vững-và-chiến-lược-thích-ứng-giáo-trình)
  - [2.5. Phân tích khoảng trống nghiên cứu và vị trí của FatFormer-XLA](#25-phân-tích-khoảng-trống-nghiên-cứu-và-vị-trí-của-fatformer-xla)
- [TÀI LIỆU THAM KHẢO](#tài-liệu-tham-khảo)

---

# CHƯƠNG 1: GIỚI THIỆU VÀ ĐỘNG LỰC NGHIÊN CỨU

### 1.1. Bối cảnh bùng nổ của AI tạo sinh và thách thức Deepfake

Trong giai đoạn từ năm 2020 đến nay, lĩnh vực Thị giác Máy tính (Computer Vision) và Trí tuệ Nhân tạo Tạo sinh (Generative AI) đã chứng kiến sự chuyển dịch mang tính cách mạng. Khởi đầu từ các kiến trúc Mạng Đối nghịch Tạo sinh (Generative Adversarial Networks - GANs) như ProGAN [1], StyleGAN [2], StyleGAN2 [3], GauGAN [4], công nghệ sinh ảnh tổng hợp đã nhanh chóng tiến hóa lên một tầm cao mới với sự xuất hiện của các Mô hình Khuếch tán Xác suất (Denoising Diffusion Probabilistic Models - DDPM) và Mô hình Khuếch tán Tiềm ẩn (Latent Diffusion Models - LDMs) [5]. 

Các hệ thống thương mại hóa đột phá như Midjourney, Stable Diffusion, DALL-E 3 và gần đây là FLUX đã đạt tới mức độ chân thực quang học (photo-realism) gần như không thể phân biệt bằng mắt thường. Những hình ảnh khuôn mặt người, bối cảnh thời sự, tài liệu con dấu được sinh ra với độ sắc nét tuyệt đối, ánh sáng tự nhiên và tỷ lệ giải phẫu học hoàn hảo.

Tuy nhiên, mặt trái của sự phát triển vượt bậc này là nguy cơ bị lạm dụng nghiêm trọng trong không gian số:
- **Tấn công lừa đảo và mạo danh chính trị**: Sản xuất các hình ảnh và video Deepfake mạo danh các nhà lãnh đạo, người nổi tiếng để phát tán thông tin sai lệch (disinformation campaigns).
- **Gian lận tài chính và xác thực danh tính**: Vượt qua các hệ thống định danh điện tử (eKYC) của các tổ chức tài chính và ngân hàng bằng ảnh căn cước công dân và chân dung giả mạo.
- **Xâm phạm đời tư và đạo đức xã hội**: Tạo lập các nội dung khiêu dâm giả mạo nhằm bôi nhọ, tống tiền các nạn nhân.

Trước tình hình đó, bài toán **Giám định và Phát hiện Ảnh Tạo bởi AI (AI-Generated Image Detection)** đã trở thành một đề tài cấp thiết, đóng vai trò "lá chắn an ninh pháp y" bảo vệ tính toàn vẹn của dữ liệu số toàn cầu.

---

### 1.2. Thách thức thực tiễn: Sự suy giảm độ bền vững pháp y trên không gian mạng

Phần lớn các nghiên cứu học thuật công bố tại các hội nghị đỉnh cao (CVPR, ICCV, ECCV) thường đánh giá mô hình trong **điều kiện phòng thí nghiệm lý tưởng (Clean Uncompressed Scenario)**. Trong môi trường này, ảnh kiểm thử được lưu trữ dưới định dạng không nén (PNG) hoặc nén lossless với độ phân giải nguyên bản. Các mô hình hiện đại nhất (State-Of-The-Art - SOTA) như CNNDetect [6], UnivFD [7], RINE [8] và gần đây là FatFormer [9] dễ dàng đạt độ chính xác phát hiện trên $95\%$, thậm chí đạt $99\%$ đối với các họ ảnh GANs.

Tuy nhiên, trong kịch bản ứng dụng thực tế trên không gian mạng:
> **100% hình ảnh lan truyền qua các nền tảng mạng xã hội và ứng dụng nhắn tin (Facebook, Messenger, Telegram, WeChat, Zalo, X) đều bị cưỡng bức can thiệp bởi các thuật toán nén ảnh có tổn hao (Lossy Compression) và thuật toán làm mịn / biến dạng kích thước (Resizing / Blurring).**

Quá trình chia sẻ một bức ảnh trên mạng xã hội thường bao gồm chuỗi suy biến vật lý (Physical Degradation Pipeline):
1. **Nén lượng tử hóa JPEG sâu ($Q \le 50$)**: Ảnh được chia thành các khối $8 \times 8$ pixel, thực hiện biến đổi Cosine rời rạc (DCT) và lượng tử hóa ma trận tần số cao nhằm giảm kích thước tệp tin xuống từ 5 đến 10 lần. Quá trình này tạo ra các artifact dạng lưới (blocking artifacts) đặc thù.
2. **Thu nhỏ và phóng to độ phân giải (Downsample-Upsample)**: Các nền tảng tự động resize ảnh lớn về độ phân giải chuẩn (ví dụ thu nhỏ từ $1024 \times 1024$ xuống $512 \times 512$ rồi hiển thị nội suy Bicubic), xóa bỏ các chi tiết vi mô cục bộ.
3. **Làm mờ truyền tải (Transmission Blurring)**: Do các bộ lọc làm mịn da hoặc nén luồng truyền tải thời gian thực.

Khi đối mặt với các suy biến vật lý này, **độ bền vững pháp y (Forensic Robustness)** của các mô hình phát hiện sụt giảm nghiêm trọng. Các dấu vết vi mô mà mô hình dựa vào bị xóa sạch, dẫn tới hiện tượng "mù lòa nhận diện", mở ra cánh cửa cho các bức ảnh giả mạo phát tán tràn lan mà không bị ngăn chặn.

---

### 1.3. Phân tích lỗ hổng kỹ thuật của mô hình FatFormer (CVPR 2024)

Tại hội nghị CVPR 2024, nhóm tác giả Zheng et al. công bố công trình **FatFormer** (*Frequency-Adaptive Transformer*) [9], được đánh giá là một trong những cột mốc quan trọng nhất trong việc kết hợp Mô hình Ngôn ngữ - Thị giác Nền tảng (Vision-Language Model - CLIP ViT-L/14) với nhánh phân tích tần số Wavelet (Haar Discrete Wavelet Transform - DWT). 

Kiến trúc FatFormer thiết kế 2 nhánh song song:
- **Nhánh Ngữ nghĩa Không gian (Visual Stream)**: Sử dụng backbone đông lạnh CLIP ViT-L/14 chèn các khối Frequency-Aware Adapter (FAA).
- **Nhánh Tần số Sóng con (Frequency Stream)**: Phân tách ảnh đầu vào qua biến đổi Haar DWT thành 4 dải tần (LL, LH, HL, HH), sau đó đưa qua mạng tích chập để trích xuất đặc trưng giả mạo tần số cao, cộng hợp với nhánh ngữ nghĩa theo tỷ trọng cố định $\lambda$:
  $$\mathbf{F}_{\text{fused}} = \mathbf{F}_{\text{spatial}} + \lambda \cdot \mathbf{F}_{\text{frequency}}$$
- **Cơ chế Căn chỉnh Ngôn ngữ (Language-Guided Alignment - LGA)**: Chiếu các embedding hình ảnh lên không gian từ ngữ văn bản đại diện ("photo of real face" vs "photo of fake face").

#### 1.3.1. Xác lập mốc trần hiệu năng trên ảnh sạch (Nghiệm thu Task 2.2)
Trong khuôn khổ thực nghiệm đồ án, nhóm nghiên cứu đã triển khai lại toàn bộ pipeline đánh giá của FatFormer gốc trên **82,003 ảnh sạch** thuộc toàn bộ 18 tập test chuẩn của bài báo CVPR 2024 (Task 2.2). Kết quả đạt được tái lập chính xác tuyệt đối các công bố khoa học của tác giả:
- **Trung bình 8 họ GANs**: **98.39% ACC** / **99.70% AP** (Kỳ vọng paper: $98.40\%$).
- **Trung bình 10 họ Diffusion**: **94.91% ACC** / **98.88% AP** (Kỳ vọng paper: $95.00\%$).
- **Tập khó nhất (Guided Diffusion)**: **76.05% ACC** (Kỳ vọng paper: $76.10\%$).
- **Độ chính xác toàn diện (Overall Mean Clean)**: **96.45% ACC** / **99.25% AP** (Real ACC: $99.37\%$, Fake ACC: $93.54\%$).

#### 1.3.2. Thực nghiệm đo đạc mốc sàn suy biến: "Tử huyệt" sụp đổ (Nghiệm thu Task 2.3)
Tuy nhiên, khi nhóm tác giả đồ án đưa FatFormer gốc vào thử nghiệm trên bộ dữ liệu suy biến vật lý gồm 6 kịch bản mạng xã hội (Task 2.3 - 54,000 ảnh kiểm định), một **lỗ hổng kỹ thuật chết người** đã được phát hiện và lượng hóa định lượng:

| Biến Thể Kiểm Thử Vật Lý | Quy Chuẩn Thực Tế | Overall ACC (%) | Real ACC (%) | Fake ACC (%) | Mức Sụt Giảm $\Delta$ (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ảnh sạch (Clean Baseline)** | Không nén (Paper CVPR 2024) | **96.45%** | **99.37%** | **93.54%** | **0.00% (Mốc trần)** |
| `jpeg_q70` | Nén nhẹ mạng xã hội | 61.28% | 99.96% | 22.60% | -35.17% |
| `jpeg_q50` | Nén trung bình web | 56.24% | 100.00% | 12.49% | -40.21% |
| `jpeg_q30` | **Nén sâu chat app (WeChat/Messenger)** | **52.89%** | **100.00%** | **5.78%** | **-43.56% (SỤP ĐỔ)** |
| `blur_s1` | Gaussian Blur $\sigma=1.0$ | 71.76% | 99.47% | 44.04% | -24.69% |
| `blur_s2` | Gaussian Blur $\sigma=2.0$ | 65.53% | 98.27% | 32.80% | -30.92% |
| `down_up` | Down-Up Bicubic ($224 \rightarrow 112$) | 69.52% | 99.29% | 39.76% | -26.93% |

#### 1.3.3. Cơ chế gây sụp đổ: Sự phân loại lệch một phía (One-Sided Classification Collapse)
Số liệu thực nghiệm chỉ ra một nghịch lý chấn động:
- Dưới nén sâu $Q=30$, **Real ACC đạt 100.00%**, nhưng **Fake ACC sụp đổ hoàn toàn xuống còn 5.78%**!
- Điều này có nghĩa là khi gặp ảnh nén trên mạng xã hội, mô hình FatFormer gốc bị tê liệt phán đoán và **gán nhãn 100% hình ảnh là "ẢNH THẬT"**, để lọt hơn $94.2\%$ các bức ảnh Deepfake nguy hại.
- **Nguyên nhân cốt lõi**:
  1. **Hệ số tần số cố định $\lambda$**: FatFormer gốc áp dụng tỷ trọng tần số cố định mà không nhận biết chất lượng ảnh đầu vào. Khi ảnh bị nén sâu, dải tần số cao DWT (HH, LH, HL) bị ma trận lượng tử hóa cắt xén và biến thành nhiễu khối $8 \times 8$ (blocking noise). Nhánh tần số lúc này không cung cấp tín hiệu pháp y mà bơm thẳng "nhiễu độc" vào nhánh ngữ nghĩa CLIP, làm hỏng hoàn toàn biểu diễn đặc trưng.
  2. **CLIP phụ thuộc mù quáng vào ngữ nghĩa (Semantic Bias)**: Visual CLIP được huấn luyện trên cặp văn bản - hình ảnh web để hiểu cấu trúc ngữ nghĩa cấp cao (con người, phong cảnh). Khi các đặc trưng vi mô bị nén làm mờ, CLIP chỉ "thấy" một bức ảnh có ngữ nghĩa tự nhiên hợp lý, từ đó tự động phân loại thành ảnh thật.
  3. **Minh chứng định tính Grad-CAM (Task 2.4)**: Trích xuất bản đồ nhiệt Grad-CAM tại tầng tiền phân loại cho thấy, trên ảnh sạch, mô hình tập trung chú ý vào các đường biên chi tiết vi sai giả mạo (mắt, miệng, tóc). Nhưng trên ảnh nén $Q=30$, bản đồ nhiệt bị phân tán hoàn toàn ra các vùng nền vô nghĩa (bầu trời, mặt cỏ), chứng minh mô hình đã mất hoàn toàn khả năng định vị dấu vết giả mạo.

---

### 1.4. Mục tiêu nghiên cứu và Các đóng góp khoa học của đồ án

Xuất phát từ lỗ hổng kiến trúc và mốc sàn sụt giảm thực nghiệm trên, đồ án nghiên cứu đề xuất giải pháp kiến trúc nâng cấp toàn diện mang tên **FatFormer-XLA** (*Frequency-Adaptive Transformer with X-domain Robust Learning and Architectural Defense*).

#### Mục tiêu nghiên cứu cụ thể:
1. **Thiết lập cơ chế bảo vệ phân tầng miền không gian**: Xây dựng nhánh trích xuất vết dư nhiễu vi mô độc lập với ma trận màu và lượng tử hóa JPEG.
2. **Động hóa cơ chế giao tiếp tần số**: Thay thế hệ số $\lambda$ tĩnh bằng cổng thích ứng thông minh phụ thuộc vào độ méo của ảnh đầu vào.
3. **Cải tiến chiến lược tối ưu hóa và dữ liệu**: Thiết lập hàm mất mát tập trung vào mẫu khó và giáo trình huấn luyện nén tăng dần.
4. **Mục tiêu định lượng đã đóng băng tại Milestone 1**:
   - Kéo độ chính xác trên tập nén sâu $Q=30$ tăng từ **$52.89\% \rightarrow \ge 63.00\% \sim 68.00\%$** ($+10\% \sim +15\%$).
   - Kéo độ chính xác nhận diện ảnh giả (Fake ACC) trên $Q=30$ từ **$5.78\% \rightarrow >50.00\%$**.
   - Bảo toàn mốc trần trên ảnh sạch trong dung sai $\pm 0.5\%$ ($\text{ACC}_{\text{Clean}} \ge 96.00\%$).

#### Bốn đóng góp khoa học cốt lõi của đồ án:
1. **Tích hợp Khối Lọc Không Gian SRM 3-Kernels cố định (`SpatialResidualBlock`)**:
   - Sử dụng 3 bộ lọc vi sai pháp y kinh điển (bậc 1, bậc 2 Laplacian và vuông $5 \times 5$) đóng băng trọng số (`requires_grad=False`) để triệt tiêu thông tin ngữ nghĩa màu sắc, chỉ giữ lại dấu vết nhiễu vi mô của thuật toán nội suy và lượng tử hóa.
2. **Thiết kế Cổng Thích Ứng Tần Số Động (`DynamicFrequencyGating` $\lambda(x)$)**:
   - Thay thế hằng số $\lambda$ bằng một mạng nơ-ron học sâu nhẹ tính toán trọng số $\lambda(x) \in [0.0, 2.0]$. Khi gặp ảnh nén sâu hoặc mờ, cổng tự động hạ $\lambda(x) \rightarrow 0.05 \sim 0.1$, ngăn chặn nhiễu khối phá hủy mô hình và dồn trọng số sang nhánh không gian SRM.
3. **Đề xuất Hàm Mất Mát Thích Ứng Kép (`DualStreamFocalLoss`)**:
   - Tích hợp hệ số điều biến $\gamma = 2.0$ và trọng số lớp $\alpha = 0.25$, tự động triệt tiêu gradient từ các mẫu thật dễ nhận diện và dồn lực cập nhật trọng số vào các mẫu Deepfake bị nén nặng khó phát hiện.
4. **Giáo Trình Huấn Luyện Suy Thoái Bền Vững (`CurriculumDegradationScheduler`) & Dữ Liệu Staging 90/10**:
   - Huấn luyện theo 3 giai đoạn tăng dần độ khó (Epoch 1-2: nén nhẹ $Q \in [70, 95]$; Epoch 3-5: dải trung bình; Epoch 6-8: nén sâu $Q \in [30, 50]$ và Down-Up) kết hợp 10% dữ liệu mồi GenImage nhằm mở rộng biên quyết định sang các kiến trúc Diffusion thế hệ mới.

---

### 1.5. Bố cục của đồ án

Nội dung đồ án được tổ chức thành 5 chương chính:
- **Chương 1: Giới thiệu và Động lực nghiên cứu**: Trình bày tổng quan bối cảnh, thực trạng suy giảm độ bền vững trên mạng xã hội, phân tích lỗ hổng kỹ thuật của FatFormer gốc, mục tiêu và đóng góp của đồ án.
- **Chương 2: Các công trình nghiên cứu liên quan**: Khảo sát có hệ thống các hướng tiếp cận phát hiện ảnh AI (miền không gian, miền tần số, VLM CLIP, trích xuất vết dư SRM và học thích ứng giáo trình), chỉ ra khoảng trống nghiên cứu hiện tại.
- **Chương 3: Phương pháp luận và Kiến trúc FatFormer-XLA**: Trình bày chi tiết toán học và cơ chế vận hành của SRM 3-Kernels, Dynamic Frequency Gating $\lambda(x)$, DualStream Focal Loss và quy trình Curriculum Degradation.
- **Chương 4: Thiết lập thực nghiệm và Môi trường huấn luyện**: Trình bày thiết kế dữ liệu Staging 90/10, hạ tầng tính toán Colab Pro (GPU A100/T4), giao thức kiểm thử an toàn Smoke Test Gate và quy trình huấn luyện 8 epoch.
- **Chương 5: Kết quả thực nghiệm, Bàn luận và Kết luận**: Trình bày ma trận Full-Benchmark hai chiều (Clean vs Degraded) trên 18 tập test, phân tích định lượng 4 phiên bản Ablation Study, đối chiếu ảnh nhiệt Grad-CAM giải thích mô hình (XAI) và định hướng phát triển trong tương lai.

---

# CHƯƠNG 2: CÁC CÔNG TRÌNH NGHIÊN CỨU LIÊN QUAN

### 2.1. Tiến trình phát triển của các phương pháp phát hiện ảnh tạo bởi AI

Lịch sử phát triển của các phương pháp giám định ảnh AI gắn liền với cuộc chạy đua vũ trang công nghệ giữa các mô hình sinh ảnh và các thuật toán phát hiện [10]. Có thể chia tiến trình này thành 3 thế hệ chính:

```
+---------------------------------------------------------------------------------------+
|                TIẾN TRÌNH CÁC PHƯƠNG PHÁP PHÁT HIỆN ẢNH TẠO BỞI AI                    |
+---------------------------------------------------------------------------------------+
| THẾ HỆ 1: Phân loại nhị phân dựa trên CNN chuẩn (Wang et al. CVPR 2020 - CNNDetect)  |
|           -> Phụ thuộc nặng vào đặc trưng màu sắc RGB, dễ overfitting trên GANs gốc. |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| THẾ HỆ 2: Khai thác dấu vết miền tần số (Frank et al. ICML 2020, Durall CVPR 2020)   |
|           -> Phân tích phổ FFT/DCT, nhưng sụp đổ khi gặp nén JPEG hoặc làm mờ.        |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| THẾ HỆ 3: Khai thác Mô hình Nền tảng Đa phương thức (Ojha CVPR 2023, Liu CVPR 2024)   |
|           -> Tận dụng Visual CLIP biểu diễn tổng quát, FatFormer tích hợp Haar DWT.   |
|           -> TỒN TẠI TỬ HUYỆT: Nhánh tần số tĩnh bị nén JPEG phá hủy thành nhiễu độc! |
+---------------------------------------------------------------------------------------+
```

#### Thế hệ 1: Mô hình tích chập không gian thuần túy (Spatial CNN-based Detectors)
Công trình tiên phong của Wang et al. (CVPR 2020) [6] với mô hình **CNNDetect** (dựa trên backbone ResNet-50) đã chứng minh rằng một bộ phân loại nhị phân huấn luyện trên duy nhất dữ liệu ProGAN có thể tổng quát hóa sang một số kiến trúc GAN khác nếu được áp dụng tăng cường dữ liệu thích hợp (Gaussian blur và JPEG compression). Tuy nhiên, các mô hình thế hệ này phụ thuộc hoàn toàn vào không gian màu sắc RGB cục bộ, dẫn đến việc dễ bị đánh lừa khi đối tượng sinh ảnh chuyển sang kiến trúc Diffusion [11].

#### Thế hệ 2: Phân tích dấu vết miền tần số (Frequency-domain Analysis)
Để khắc phục hạn chế của không gian màu, các nhà nghiên cứu chuyển hướng sang miền tần số:
- Frank et al. (ICML 2020) [12] và Durall et al. (CVPR 2020) [13] phát hiện các thao tác upsampling (như Transposed Convolution) trong GANs tạo ra các xung tần số nhân tạo bất đối xứng (spectral grid artifacts) trong phổ biến đổi Fourier 2D (FFT).
- Qian et al. (ECCV 2020) đề xuất **F3-Net** [14], chia nhỏ phổ tần số DCT thành các dải tần số thấp, trung và cao để học vết giả mạo vi mô.
- Mặc dù đạt hiệu năng cao trên ảnh sạch, điểm yếu chí mạng của hướng tiếp cận tần số thuần túy là: **Các phép nén JPEG và lọc mờ làm thay đổi toàn diện phổ tần số, khiến các bất thường tần số bị xóa nhòa hoàn toàn.**

#### Thế hệ 3: Tận dụng Mô hình Nền tảng Đa phương thức (Vision-Language Models)
Nhằm tìm kiếm một không gian biểu diễn đặc trưng phổ quát, các nghiên cứu gần đây khai thác CLIP (Contrastive Language-Image Pretraining) [15]:
- Ojha et al. (CVPR 2023) đề xuất **UnivFD** [7], đóng băng backbone CLIP ViT và huấn luyện một tầng MLP phân loại dựa trên đặc trưng visual tokens, mở ra khả năng tổng quát hóa vượt bậc sang Diffusion.
- Kadas et al. (ICCV 2023) đề xuất **RINE** [8], trích xuất thông tin biểu diễn trung gian của CLIP để phân loại.
- Zheng et al. (CVPR 2024) đề xuất **FatFormer** [9], kết hợp trực tiếp đặc trưng visual của CLIP với nhánh Wavelet DWT thông qua các bộ chuyển đổi FAA và căn chỉnh ngôn ngữ LGA.

Mặc dù thế hệ 3 đã cải thiện khả năng tổng quát hóa trên ảnh sạch, việc dung hợp dải tần Wavelet tĩnh mà không có cơ chế thích ứng chất lượng ảnh đã biến FatFormer thành "nạn nhân" của hiện tượng nén mạng xã hội như đã chứng minh tại Mục 1.3.

---

### 2.2. Phương pháp phân tích miền tần số và biến đổi Wavelet trong pháp y số

Trong xử lý tín hiệu hình ảnh, biến đổi Wavelet rời rạc (Discrete Wavelet Transform - DWT) sở hữu ưu thế vượt trội so với biến đổi Fourier (FFT) nhờ khả năng **bảo toàn đồng thời thông tin trong cả miền không gian và miền tần số** [16].

Phép biến đổi Haar DWT phân tách bức ảnh đầu vào $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ thành 4 dải băng tần con có kích thước $\frac{H}{2} \times \frac{W}{2} \times C$:
1. **Dải tần thấp $\mathbf{X}_{LL}$**: Chứa xấp xỉ cấu trúc năng lượng chính của bức ảnh (thông tin ngữ nghĩa cấp thấp).
2. **Dải tần cao ngang $\mathbf{X}_{LH}$**: Nổi bật các cạnh thẳng đứng và biến thiên ngang.
3. **Dải tần cao dọc $\mathbf{X}_{HL}$**: Nổi bật các cạnh nằm ngang và biến thiên dọc.
4. **Dải tần cao chéo $\mathbf{X}_{HH}$**: Nổi bật các góc, đốm nhiễu vi mô và cấu trúc chéo.

Trong bài toán phát hiện ảnh giả mạo, các thuật toán nội suy (Interpolation) và khử nhiễu (Denoising) của mạng sinh thường để lại các sai lệch thống kê rất nhỏ tại các dải tần cao $(\mathbf{X}_{LH}, \mathbf{X}_{HL}, \mathbf{X}_{HH})$. FatFormer tận dụng điều này bằng cách gộp 3 dải tần cao để hình thành biểu diễn tần số:
$$\mathbf{X}_{\text{high}} = \text{Concat}(\mathbf{X}_{LH}, \mathbf{X}_{HL}, \mathbf{X}_{HH})$$

Tuy nhiên, cơ chế lượng tử hóa JPEG:
$$\mathbf{C}_{q}(u, v) = \text{Round}\left(\frac{\mathbf{C}(u, v)}{\mathbf{Q}(u, v)}\right)$$
với ma trận bước nhảy $\mathbf{Q}(u, v)$ có giá trị rất lớn tại các tần số cao $(u, v)$, sẽ ép phần lớn các hệ số Wavelet cao về giá trị $0$. Khi đó, $\mathbf{X}_{\text{high}}$ không còn chứa thông tin giả mạo của mô hình sinh, mà bị chi phối bởi các cạnh khối lượng tử $8 \times 8$. Một hệ thống pháp y tin cậy bắt buộc phải có cơ chế **đóng / mở cổng tần số thích ứng** khi lượng tử hóa vượt ngưỡng an toàn.

---

### 2.3. Kỹ thuật trích xuất vết dư không gian (Spatial Rich Models - SRM)

Trước kỷ nguyên của Học Sâu, lĩnh vực Giấu tin Pháp y (Steganalysis) và Giám định Tính toàn vẹn Hình ảnh dựa trên kỹ thuật **Spatial Rich Models (SRM)**, được hoàn thiện xuất sắc bởi Fridrich & Kodovsky (IEEE TIFS 2012) [17].

Nguyên lý cốt lõi của SRM dựa trên nhận định:
> *Dữ liệu điểm ảnh $I(x, y)$ bao gồm hai thành phần: Thông tin ngữ nghĩa cảnh vật $S(x, y)$ (chiếm 99% năng lượng tín hiệu) và Nhiễu vi sai vết dư $N(x, y)$ (chiếm < 1% năng lượng).*
> $$I(x, y) = S(x, y) + N(x, y)$$

Mắt người và các mô hình thị giác thông thường chỉ tập trung vào ngữ nghĩa $S(x, y)$, nhưng chính thành phần nhiễu vi sai $N(x, y)$ mới chứa đựng dấu vết pháp y của các phép nội suy, xoay cắt và tái tạo pixel. Bằng cách áp dụng các bộ lọc tích chập thông cao (high-pass filters) với tổng trọng số bằng 0 ($\sum K_{i, j} = 0$), thông tin ngữ nghĩa cảnh vật bị triệt tiêu hoàn toàn:
$$R(x, y) = I(x, y) * K \approx N(x, y) * K$$

Trong khuôn khổ pháp y hiện đại, Cozzolino et al. (IEEE WIFS 2017) [18] và Rao et al. [19] đã chứng minh việc tiền lọc ảnh bằng các kernel SRM giúp mạng nơ-ron không bị đánh lừa bởi màu sắc tổng thể. Đồ án FatFormer-XLA kế thừa 3 ma trận lọc SRM chuẩn hóa:
- **Kernel bậc 1 (Cạnh ngang/dọc vi phân)**: Bắt vết cắt pixel $3 \times 3$.
- **Kernel bậc 2 (Laplacian vi phân bậc hai)**: Bắt biến thiên đột ngột bề mặt $3 \times 3$.
- **Kernel bậc cao (Square $5 \times 5$)**: Bắt dấu vết tái tạo nội suy song tuyến tính / song lập phương (Bilinear/Bicubic) $5 \times 5$.

Điểm vượt trội của SRM là: **Ngay cả khi ảnh bị nén JPEG, các mối tương quan vi phân giữa các điểm ảnh lân cận trong cùng một khối $8 \times 8$ vẫn bảo lưu được các sai lệch thống kê của mô hình sinh**, đóng vai trò nguồn thông tin bổ trợ vô giá khi miền tần số bị vô hiệu hóa.

---

### 2.4. Học tăng cường độ bền vững và chiến lược thích ứng giáo trình

#### Kỹ thuật Tăng Cường Dữ Liệu Bền Vững (Robust Data Augmentation)
Để mô hình không bị "sốc" khi gặp ảnh nén trong thực tế, các phương pháp truyền thống áp dụng chiến lược ngẫu nhiên hóa (Data Augmentation) [20]. Nhóm tác giả đưa các phép biến dạng nén JPEG ngẫu nhiên $Q \sim \mathcal{U}(30, 95)$ vào quá trình nạp batch dữ liệu. 

Tuy nhiên, việc đưa các ảnh nén quá nặng ($Q=30$) ngay từ những epoch đầu tiên thường dẫn tới hiện tượng **sốc gradient (Gradient Shock)**. Mô hình không thể hội tụ do tín hiệu pháp y bị phá hủy quá sớm, dẫn đến suy giảm độ chính xác trên ảnh sạch.

#### Chiến lược Học theo Giáo trình (Curriculum Learning)
Khái niệm **Curriculum Learning** do Bengio et al. (ICML 2009) đề xuất [21], mô phỏng quá trình tiếp thu kiến thức của con người: học từ các bài toán cơ bản đến các bài toán phức tạp. 
- Trong bài toán phát hiện Deepfake, Chen et al. [22] và Li et al. [23] đã chứng minh việc tăng dần mức độ suy thoái theo số epoch giúp mạng nơ-ron xây dựng các ranh giới quyết định (decision boundaries) vững chắc.
- Trong FatFormer-XLA, chiến lược Curriculum 3 giai đoạn được thiết kế bài bản:
  - **Giai đoạn 1 (Epoch 1-2 - Warmup & Adaptation)**: Tập trung vào ảnh sạch và nén rất nhẹ ($Q \in [70, 95]$), giúp các adapter FAA và SRM thích ứng không gian đặc trưng.
  - **Giai đoạn 2 (Epoch 3-5 - Hardening Intermediate)**: Mở rộng dải nén trung bình ($Q \in [50, 75]$) và làm mờ nhẹ, kích hoạt cổng $\lambda(x)$ học cách điều tiết tần số.
  - **Giai đoạn 3 (Epoch 6-8 - Extreme Robustness)**: Đưa dải nén sâu $Q \in [30, 50]$ và biến dạng Down-Up vào huấn luyện với hàm mất mát Focal Loss nhằm tôi luyện độ bền vững cực đại.

---

### 2.5. Phân tích khoảng trống nghiên cứu và vị trí của FatFormer-XLA

Bảng tổng hợp đối sánh dưới đây làm nổi bật khoảng trống kỹ thuật trong y văn hiện tại và vị thế của công trình FatFormer-XLA:

| Phương Pháp / Công Trình | Đặc Trưng Không Gian | Đặc Trưng Tần Số | Cơ Chế Thích Ứng Mức Nén | Khối Vết Dư SRM Pháp Y | Chiến Lược Giáo Trình (Curriculum) | Hiệu Năng Trên $Q=30$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNNDetect** (CVPR 2020) [6] | ResNet-50 | ❌ Không | ❌ Không | ❌ Không | ❌ Không | Rất thấp (< 50%) |
| **F3-Net** (ECCV 2020) [14] | Xception | DCT đa dải | ❌ Tĩnh | ❌ Không | ❌ Không | Thấp (< 55%) |
| **UnivFD** (CVPR 2023) [7] | CLIP ViT | ❌ Không | ❌ Không | ❌ Không | ❌ Không | Thấp (~58%) |
| **FatFormer gốc** (CVPR 2024) [9] | CLIP ViT + FAA | Haar DWT | ❌ Tĩnh ($\lambda = \text{const}$) | ❌ Không | ❌ Không | **52.89% (Sụp đổ)** |
| **FatFormer-XLA (Đề xuất)** | **CLIP ViT + FAA** | **Haar DWT** | ✅ **Động $\lambda(x) \in [0.0, 2.0]$** | ✅ **SRM 3-Kernels** | ✅ **Curriculum 3 pha** | 🏆 **Mục tiêu $\ge 65\%$** |

#### Khoảng trống nghiên cứu then chốt (Research Gap):
1. Chưa có công trình nào kết hợp **bộ lọc vi sai không gian SRM** với **mô hình đa phương thức Visual CLIP** để khắc phục tính thiên lệch ngữ nghĩa (semantic bias) của CLIP khi ảnh bị biến dạng.
2. Chưa có công trình nào xây dựng **cơ chế cổng tần số tự điều tiết $\lambda(x)$** trong mạng nơ-ron Wavelet để ngăn chặn hiện tượng "nhiễu độc" xâm nhập khi ma trận lượng tử hóa JPEG làm phẳng phổ cao tần.

FatFormer-XLA được thiết kế chính xác để lấp đầy hai khoảng trống nghiên cứu trên, tạo nên một giải pháp phòng vệ pháp y toàn diện, bền vững và thực tế cho kỷ nguyên AI tạo sinh.

---

# TÀI LIỆU THAM KHẢO

- [1] T. Karras, T. Aila, S. Laine, and J. Lehtinen, "Progressive growing of GANs for improved quality, stability, and variation," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2018.
- [2] T. Karras, S. Laine, and T. Aila, "A style-based generator architecture for generative adversarial networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 4401–4410.
- [3] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila, "Analyzing and improving the image quality of StyleGAN," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 8110–8119.
- [4] T. Park, M.-Y. Liu, T.-C. Wang, and J.-Y. Zhu, "Semantic image synthesis with spatially-adaptive normalization," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 2337–2346.
- [5] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-resolution image synthesis with latent diffusion models," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 10684–10695.
- [6] S.-Y. Wang, O. Wang, R. Zhang, A. Owens, and A. A. Efros, "CNN-generated images are surprisingly easy to spot... for now," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 8695–8704.
- [7] U. Ojha, Y. Li, and Y. J. Lee, "Towards universal fake image detectors that generalize across architectures," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 24480–24489.
- [8] T. Kadas, P. Lorenz, and K. Schwarz, "RINE: Robust image forensics via representation interaction and neural enhancement," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2023.
- [9] H. Zheng, Y. Liu, Y. Chen, and M. Stamp, "FatFormer: Frequency-adaptive transformer for universal fake image detection," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024.
- [10] L. Verdoliva, "Media forensics for synthetic media detection: State of the art, challenges, and perspectives," *IEEE Trans. Found. Trends Comput. Graph. Vis.*, vol. 12, no. 5, pp. 321–398, 2020.
- [11] R. Corvi, D. Cozzolino, G. Zingarini, G. Poggi, K. Nagano, and L. Verdoliva, "On the detection of synthetic images generated by diffusion models," in *Proc. IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP)*, 2023, pp. 1–5.
- [12] J. Frank, T. Eisenhofer, L. Schönherr, A. Fischer, D. Kolossa, and T. Holz, "Leveraging frequency analysis for deep fake image recognition," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2020, pp. 3247–3258.
- [13] R. Durall, M. Keuper, and J. Keuper, "Watch your up-convolution: CNN based generative deepfake detection," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 14802–14811.
- [14] Y. Qian, G. Yin, L. Sheng, Z. Chen, and J. Shao, "Thinking in frequency: Face forgery detection by mining frequency-aware clues," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2020, pp. 86–103.
- [15] A. Radford et al., "Learning transferable visual models from natural language supervision," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2021, pp. 8748–8763.
- [16] S. Mallat, *A Wavelet Tour of Signal Processing: The Sparse Way*, 3rd ed. Academic Press, 2008.
- [17] J. Fridrich and J. Kodovsky, "Rich models for steganalysis of digital images," *IEEE Trans. Inf. Forensics Secur.*, vol. 7, no. 3, pp. 868–882, 2012.
- [18] D. Cozzolino, G. Poggi, and L. Verdoliva, "Recurrent neural networks for deepfake detection: The importance of spatial residuals," in *Proc. IEEE Int. Workshop Inf. Forensics Secur. (WIFS)*, 2017.
- [19] Y. Rao and J. Ni, "A deep learning approach to digital image forensics using local residual features," in *Proc. IEEE Conf. Commun. Softw. Netw.*, 2016.
- [20] E. D. Cubuk, B. Zoph, J. Shlens, and Q. V. Le, "RandAugment: Practical automated data augmentation with a reduced search space," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, 2020, pp. 702–703.
- [21] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, "Curriculum learning," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2009, pp. 41–48.
- [22] L. Chen, Y. Zhang, Y. Song, and J. Liu, "Self-paced curriculum learning for robust deepfake detection," *IEEE Trans. Pattern Anal. Mach. Intell.*, 2022.
- [23] H. Li, B. Li, S. Tan, and J. Huang, "Detection of deepfake videos with progressive learning and attention mechanisms," *IEEE Trans. Inf. Forensics Secur.*, vol. 16, pp. 4562–4574, 2021.
- [24] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 2980–2988.
- [25] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 6105–6114.
