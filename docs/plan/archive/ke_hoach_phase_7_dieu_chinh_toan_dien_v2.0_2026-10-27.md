Dưới đây là **Bản Kế Hoạch Tổng Thể Tối Ưu Hóa FatFormer-XLA (Master Plan v2.0)**, tổng hợp toàn bộ kết quả thực nghiệm từ vòng kiểm chuẩn ngoại vi CommunityForensics-Eval (CFE), lịch sử dự án (Phase 1 $\rightarrow$ Phase 6-R) và các phân tích phản biện chuyên sâu để tạo ra một **Mô Hình Toàn Diện & Hợp Nhất (Unified Robust FatFormer-XLA)**.

## **PHẦN I: GIẢI PHẪU TOÀN DIỆN — TỪ TRIỆU CHỨNG ĐẾN NGUYÊN NHÂN GỐC VÀ PHÁC ĐỒ ĐIỀU TRỊ**

Để nâng cấp mô hình một cách thực chất thay vì "vá víu" bằng ngưỡng cắt, bảng dưới đây hệ thống hóa **6 bệnh lý cốt lõi** đã phát lộ qua đợt chạy CFE vừa rồi, giải thích rõ cơ chế vật lý/toán học bên dưới và phác đồ điều trị tương ứng:

| \# | Triệu chứng trên CFE & 18 Tập Test | Nguyên nhân gốc rễ (Root Cause) | Giải thích cơ chế kỹ thuật bên dưới | Phác đồ điều trị dứt điểm (Treatment) |
| :---- | :---- | :---- | :---- | :---- |
| **1** | **WiSE-FT tê liệt hoàn toàn:** Real ACC \= 100%, nhưng Fake ACC \= 4.10% (Clean) và 0.20% ($Q=30$). | **Lệch cấu trúc Loader & Lai tạp Checkpoint** | Bản fatformer\_wise\_ft\_best.pth trộn giữa Baseline (1.116 keys) và M4 cũ (1.684 keys) tạo ra 1.816 tensors. Khi nạp với strict=False, hàng trăm tham số nhánh srm.\*, gating.\* và các khối Adapter bị bỏ qua thầm lặng hoặc nhận trọng số ngẫu nhiên, đẩy toàn bộ logit sụp về phía lớp Real. | **Canonical Loader (strict=True):** Tự động suy diễn số khối Adapter từ state\_dict, chuẩn hóa tên khóa, cấm tuyệt đối strict=False. Đồng thời tái tạo **WiSE-FT Gen 2** từ Baseline và Anchored M6-R (cùng cấu trúc, cùng bồn hội tụ). |
| **2** | **Anchored báo oan ảnh thật quá nặng:** Clean Real ACC chỉ đạt 48.10% trên CFE và 50.52% trên 18 tập lab cũ. | **Shortcut Learning (Học lối tắt) \+ Ngưỡng thấp (${T}^{*}=0.40$) \+ Trôi mỏ neo** | (1) Trong Task 6.3-R, trọng số Loss bị đảo thành $(0.7CE+0.3KD)$ và LR adapter cũ tăng lên $5\times 1{0}^{-5}$ làm yếu lực neo của Teacher. (2) Nhánh SRM 3-Kernels học quy luật lệch: *"Cứ thấy vết nén khối JPEG hoặc nhiễu tần số cao là báo Fake"*. Khi gặp ảnh thật ngoại vi (COCO, LAION, ImageNet, LSUN) vốn có sẵn vết nén JPEG nội tại, mô hình báo động giả hàng loạt. (3) Áp cứng ngưỡng nhạy ${T}^{*}=0.40$ lên ảnh sạch làm tăng vọt False Positive. | (1) **Hàm phạt Bất biến Ảnh thật (${L}_{real\_guard}$):** Cấm logit Fake của ảnh thật bị nén vượt quá logit ảnh thật sạch của Teacher. (2) Khôi phục tỷ trọng chuẩn $0.3CE+0.7KD+0.2Cons$ và hạ LR adapter cũ về $1{0}^{-5}$. (3) Hiệu chuẩn ngưỡng theo **ràng buộc FPR mục tiêu** (\$\\text{Real ACC} \\ge 90\\%\$). |
| **3** | **Mất khả năng xếp hạng dưới nén sâu:** Tại JPEG $Q=30$, ROC-AUC của cả 3 model chỉ đạt \$\\sim 50.00\\% \- 50.93\\%\$. | **Sụp đổ biểu diễn (Feature Collapse) & Thiếu dữ liệu $Q=30$ lúc train** | Ma trận lượng tử hóa DCT $8\times 8$ ở mức $Q=30$ xóa sổ 85% thành phần tần số cao, khiến Grad-CAM bám vào các cạnh vuông JPEG thay vì chủ thể. Hơn nữa, ở Task 6.2-R/6.3-R, PairedCleanDegradedAugmentation chỉ lấy mẫu nén nhẹ $Q\in [50,90]$ (clean\_prob=0.70), nên mô hình chưa từng học cách gom cụm đặc trưng ở $Q=30$. | (1) Mở rộng dải suy biến huấn luyện về **$Q\in [30,85]$**, kết hợp đa dạng biến dạng (Down-Up, Blur, WebP). (2) Áp dụng **Feature Cosine & Logit Consistency Distillation** ép vector đặc trưng của ${x}_{q30}$ phải bám sát vector của ${x}_{clean}$. |
| **4** | **Độ lệch tổng quát hóa (OOD Gap) trên AI thế hệ mới (2024–2025):** Khó nhận diện Flux.1, Midjourney v6, Autoregressive. | **Tập dữ liệu huấn luyện quá hẹp (Narrow Domain Coverage)** | Tập huấn luyện diffusion\_staging.tar hiện chỉ có 3.600 ảnh (2.520 Train / 1.080 Val) gói gọn trong Stable Diffusion và Midjourney đời đầu, quá nhỏ so với 5 họ kiến trúc hiện đại của CFE. | **Mở rộng dữ liệu Đa miền Tinh gọn (Data-Centric Expansion):** Bổ sung tập con phân tầng từ **CommunityForensics-Small** (hoặc 10.000–20.000 ảnh đại diện 5 họ kiến trúc) kết hợp Replay 20% dữ liệu gốc để chống quên. |
| **5** | **Chọn nhầm Checkpoint thiên kiến:** Checkpoint đạt đỉnh trên tập Val nội bộ nhưng sụp đổ khi ra OOD. | **Tiêu chí chọn lọc (Selection Metric) chưa phản ánh OOD & Suy biến** | Quá trình huấn luyện chọn best.pth dựa trên điểm số của 270 ảnh val\_selection nhẹ nhàng, không đánh giá theo nhóm tệ nhất (*Worst-Group*) dưới điều kiện suy biến nặng. | Chuyển tiêu chí lưu Checkpoint và Early Stopping sang **Worst-Group AUC & TPR@FPR=5%** trên tập Validation có bao gồm cả mẫu OOD và mẫu nén sâu $Q=30$. |
| **6** | **Xung đột Pareto giữa Miền Sạch và Miền Nén Sâu:** Không một ngưỡng cố định nào tối ưu cho cả hai. | **Ranh giới đánh đổi Pareto tự nhiên (Inherent Pareto Frontier)** | Đặc trưng vi mô trên ảnh sạch và đặc trưng ngữ nghĩa thô trên ảnh nén sâu $Q=30$ có phân phối biên độ logit khác nhau. Dùng chung một trọng số tĩnh và một ngưỡng tĩnh bắt buộc phải hy sinh một trong hai đầu. | **Kiến trúc Hợp nhất có Định tuyến Suy biến Tự động (DegradationRouter):** Đo chỉ số nhiễu khối DCT $B(x)$ và phương sai Laplacian ${V}_{Lap}(x)$ ngay ở thời điểm suy luận để điều chỉnh cổng tần số và hòa trộn xác suất tự động trong 1 lệnh forward() duy nhất. |

## **PHẦN II: LỘ TRÌNH 4 GIAI ĐOẠN XÂY DỰNG MÔ HÌNH TOÀN DIỆN (CHI TIẾT TỪNG BƯỚC)**

Lộ trình được thiết kế theo nguyên tắc **Tăng trưởng kiểm soát (Gated Progression)**: đi từ sửa lỗi nền tảng đo lường (0 GPU) $\rightarrow$ Hợp nhất mô hình tức thì (CPU/1 GPU nhỏ) $\rightarrow$ Huấn luyện nâng cấp biểu diễn (GPU) $\rightarrow$ Đóng gói Kiến trúc Hợp nhất Tự động.

Đoạn mã  
graph TD  
    P0\["GIAI ĐOẠN P0 (Ngày 1-2 \| 0 GPU)\<br/\>Chuẩn hóa Canonical Loader & Khế Ước Đo Lường"\] \--\> P1\["GIAI ĐOẠN P1 (Ngày 3-4 \| CPU / T4)\<br/\>WiSE-FT Thế Hệ 2 & Hiệu Chuẩn Ràng Buộc FPR"\]  
    P1 \--\> P2\["GIAI ĐOẠN P2 (Ngày 5-9 \| 1x GPU T4/A100)\<br/\>Huấn Luyện Anchored Recovery v2 (Data \+ Real-Guard \+ Q30 KD)"\]  
    P2 \--\> P3\["GIAI ĐOẠN P3 (Ngày 10-12 \| Đóng gói)\<br/\>Tích Hợp Bộ Định Tuyến Suy Biến Tự Động (Unified Model)"\]

### **GIAI ĐOẠN P0: Chuẩn Hóa Bộ Nạp Trọng Số (Canonical Loader) & Khế Ước Đánh Giá Mới**

* **Mục đích:** Xóa bỏ hoàn toàn sai số do lỗi code nạp mô hình (strict=False, lệch số khối Adapter, thừa/thiếu keys), thiết lập bộ thước đo chuẩn quốc tế không phụ thuộc ngưỡng cắt.  
* **Chi phí tài nguyên:** 0 giờ GPU (chạy hoàn toàn trên CPU/Local hoặc phiên test ngắn).

#### **1\. Đầu vào (Inputs)**

* 4 tệp Checkpoint hiện có: fatformer\_4class\_ckpt.pth (Baseline, 1.116 keys), fatformer\_crossdomain\_weights\_only.pth (M4, 1.684 keys), fatformer\_wise\_ft\_best.pth (1.816 keys), và fatformer\_anchored\_recovery\_best.pth (M6-R).  
* Các kịch bản kiểm chuẩn: tools/eval\_communityforensics.py và tools/eval\_locked\_benchmark.py.

#### **2\. Kỹ thuật & Cách làm chi tiết**

1. **Xây dựng src/models/canonical\_loader.py (Tự động nhận diện cấu hình từ state\_dict):**  
   * Trước khi khởi tạo kiến trúc CLIPModel, hàm đọc qua danh sách khóa trong state\_dict của file .pth:  
     * Đếm số lượng chỉ mục khối blocks.\<N\>. có chứa nhánh Adapter để tự động suy ra chính xác num\_vit\_adapter (3 hay 8 khối) thay vì gán cứng trong code.  
     * Kiểm tra sự tồn tại của các khóa srm.\* và gating.\* hợp lệ bên trong các tầng FAA/LGA; tự động dọn dẹp (strip) các khóa top-level bị bỏ hoang không tham gia vào đồ thị tính toán (forward graph).  
     * Thực thi model.load\_state\_dict(clean\_state\_dict, strict=True) kèm khẳng định cứng: assert len(missing\_keys) \== 0 and len(unexpected\_keys) \== 0.  
2. **Nâng cấp Khế ước Đánh giá (Evaluation Contract) trong eval\_communityforensics.py:**  
   * Gộp kết quả dự đoán (torch.cat các logits từ 5 tập con) để tránh chạy suy luận GPU lặp 2 lần, tiết kiệm 50% thời gian chạy.  
   * Bổ sung 4 chỉ số độc lập ngưỡng và thực chiến pháp y vào báo cáo xuất ra:  
     * **ROC-AUC** và **Average Precision (AP)**: Đo năng lực xếp hạng thực chất của không gian đặc trưng.  
     * **TPR @ FPR \= 5%** và **TPR @ FPR \= 1%**: Tỷ lệ bắt được ảnh Fake khi khống chế tỷ lệ báo oan ảnh thật ở mức tối đa 5% hoặc 1% (tiêu chuẩn vàng của kiểm duyệt nội dung thực tế).  
     * Tách biệt rõ **Subset-Macro Mean** ($\frac{1}{5}\sum\limits_{i=1}^{5}Metri{c}_{i}$) và **Overall Pooled Micro** (gộp 2.000 ảnh).

#### **3\. Đầu ra & Cổng nghiệm thu (Outputs & Gate P0)**

* **Đầu ra:** Module canonical\_loader.py, bộ Unit Test tests/test\_canonical\_loader.py vượt qua 100% trên cả 4 checkpoint với 0 missing / 0 unexpected keys, và báo cáo CFE đo lại trung thực.

### **GIAI ĐOẠN P1: Tạo "WiSE-FT Thế Hệ 2" & Hiệu Chuẩn Ngưỡng Bảo Vệ Ảnh Thật**

* **Mục đích:** Tạo ra ngay một **Checkpoint đơn lẻ (Single Weight File)** có sự cân bằng Pareto tốt hơn hẳn thế hệ 1 mà chưa cần huấn luyện lại trên GPU.  
* **Chi phí tài nguyên:** 0 GPU cho việc trộn trọng số (chạy CPU trong 15 phút) \+ 15 phút GPU Tesla T4 để quét kiểm chuẩn.

#### **1\. Đầu vào (Inputs)**

* Checkpoint gốc ${\theta }_{orig}$ (fatformer\_4class\_ckpt.pth) và Checkpoint neo khôi phục ${\theta }_{anchored}$ (fatformer\_anchored\_recovery\_best.pth — bản M6-R đã đạt Zero-Init $0.0e+00$ tại Step 0).

#### **2\. Kỹ thuật & Cách làm chi tiết**

1. **Nội suy Không gian Trọng số Thế hệ 2 (WiSE-FT Gen 2):**  
   * *Giải thích tại sao hiệu quả:* Ở thế hệ 1, nhóm trộn ${\theta }_{orig}$ với ${\theta }_{M4}$ (vốn bị lỗi *Non-Zero Residual Injection* ở Step 0 và trôi dạt do Focal Loss), nên phải ép $\alpha =0.25,{\beta }_{srm}=0.10$ làm mất khả năng kháng nén. Ngược lại, bản ${\theta }_{anchored}$ (M6-R) xuất phát từ chính ${\theta }_{orig}$ với độ lệch Step 0 bằng $0.0e+00$ bit-perfect, nên toàn bộ vector trọng số nằm trong cùng một bồn hội tụ (loss basin) tuyến tính.  
   * Sử dụng tools/wise\_ft\_merge.py thực hiện tổ hợp lồi trên tập khóa giao nhau ${\theta }_{orig}\cap {\theta }_{anchored}$ và điều tiết biên độ nhánh SRM:

   * ${\theta }_{wise\_v2}^{(k)}=(1-\alpha )\cdot {\theta }_{orig}^{(k)}+\alpha \cdot {\theta }_{anchored}^{(k)},{\theta }_{wise\_v2}^{(srm\_proj)}={\beta }_{srm}\cdot {\theta }_{anchored}^{(srm\_proj)}$  
   * Quét lưới (Grid Search) trên miền tham số cân bằng: $\alpha \in \{0.35,0.45,0.50,0.60\}$ và ${\beta }_{srm}\in \{0.25,0.35,0.50\}$.  
2. **Hiệu chuẩn Nhiệt độ & Ngưỡng Ràng buộc Bảo vệ Ảnh thật (Constrained Calibration):**  
   * Tối ưu nhiệt độ $T>0$ bằng L-BFGS cực tiểu hóa hàm NLL trên tập hiệu chuẩn: ${\hat{p}}_{i}=\sigma ({z}_{i}/T)$.  
   * Thay vì chọn ngưỡng ${T}^{*}$ tối đa hóa Macro F1 vô điều kiện (gây sụp đổ ranh giới ở Task 6.4 cũ), áp dụng thuật toán tìm ngưỡng có ràng buộc cứng bảo vệ người dùng thực (\$\\text{Real ACC} \\ge 90.0\\%\$ hoặc \$\\text{FPR} \\le 10\\%\$):  
      \$\$T^\* \= \\arg\\max\_{th \\in \[0.40, 0.65\]} \\text{Macro F1}(th) \\quad \\text{s.t.} \\quad \\text{Real ACC}\_{\\text{OOD-Val}}(th) \\ge 90.0\\%\$\$

#### **3\. Đầu ra & Cổng nghiệm thu (Outputs & Gate P1)**

* **Đầu ra:** Checkpoint fatformer\_wise\_ft\_v2\_best.pth kèm cặp tham số $(T,{T}^{*})$ chuẩn hóa.  
* **Kỳ vọng:** Ngay với checkpoint đơn lẻ này, Clean Real ACC trên CFE và 18 tập test tăng từ \$\\sim 48\\% \- 50\\%\$ lên vùng **\$\\ge 88\\% \- 92\\%\$**, trong khi vẫn giữ được độ nhạy Fake ACC cao hơn hẳn bản WiSE-FT thế hệ 1\.

### **GIAI ĐOẠN P2: Huấn Luyện Tinh Chỉnh "Anchored Recovery v2" (Trị Tận Gốc AUC $Q=30$ & False Positive)**

* **Mục đích:** Giải quyết tận gốc rễ vấn đề mất thông tin xếp hạng dưới nén sâu (\$Q=30 \\text{ AUC} \\approx 50\\%\$) và tật bắt nhầm vết nén JPEG trên ảnh thật (Shortcut Learning).  
* **Chi phí tài nguyên:** **4 – 6 giờ trên 1 GPU** (Colab Pro A100 / L4 hoặc Kaggle Server), dung lượng ổ cứng chỉ tốn **\~30 – 50 GB** (vừa vặn tuyệt đối với giới hạn máy ảo, không cần tải 1.08 TB).

#### **1\. Đầu vào (Inputs)**

* **Trọng số khởi tạo (Warm-Start):** Nạp từ fatformer\_4class\_ckpt.pth kèm cơ chế **Zero-Init Identity** (nn.init.zeros\_ cho lớp chiếu cuối proj\[-1\] của SRM và khởi tạo MLP Gating sao cho $\lambda (x)=2\sigma (0)=1.0$, đảm bảo sai số Step 0 bằng $0.0e+00$).  
* **Tập dữ liệu Huấn luyện Đa miền Mở rộng Nhẹ (train\_crossdomain\_v2 — \~12.000 đến 16.000 ảnh):**  
  * Kế thừa 3.600 ảnh từ diffusion\_staging.tar (ProGAN, SD v1.5, Midjourney) để chống quên (Replay In-distribution).  
  * Bổ sung **8.000 – 12.000 ảnh từ CommunityForensics-Small** (lấy phân tầng qua các bộ sinh của 5 họ kiến trúc nhưng **cách ly 100% mã băm SHA-256 và Generator-disjoint** với 2.000 ảnh của bộ test CFE bằng tools/verify\_data\_isolation.py).  
  * Nguồn ảnh thật đa dạng (COCO, FFHQ, RAISE, LAION, ImageNet) để mô hình làm quen với nhiều đặc tính nhiễu cảm biến và lịch sử nén khác nhau.

#### **2\. Kỹ thuật & Cách làm chi tiết**

1. **Nâng cấp Pipeline Biến dạng Cặp đôi (PairedCleanDegradedAugmentation trong src/datasets/transforms.py):**  
   * Với mỗi ảnh đầu vào, luôn sinh ra cặp song song $({x}_{clean},{x}_{deg})$.  
   * Mở rộng dải suy biến của ${x}_{deg}$ để bao phủ đúng tử huyệt thực tế (thay vì chỉ nén nhẹ $Q\in [50,90]$ như Task 6.2-R):  
     * **Nén sâu JPEG:** Lấy mẫu $Q\in [30,85]$ (trong đó xác suất rơi vào vùng khốc liệt $Q\in [30,45]$ chiếm ít nhất 35% số mẫu degraded).  
     * **Đa suy biến kết hợp:** Gaussian Blur $\sigma \in [0.5,2.0]$, Down-Up Bicubic tỷ lệ $[0.5,0.75]$ và nén WebP/Double-JPEG.  
2. **Thiết kế Hàm Mất Mát Toàn Diện 4 Thành Phần (UnifiedForensicLoss trong src/training/loss.py):**

3. ${L}_{total}=0.3\cdot {L}_{AugCE}+0.7\cdot {L}_{KD}+0.3\cdot {L}_{FeatCons}+0.5\cdot {L}_{real\_guard}$  
   * **Thành phần 1 — ${L}_{AugCE}$ (Trọng số $0.3$):** Cross-Entropy chuẩn trên ảnh suy biến (tắt hoàn toàn Focal Loss để tránh co cụm logit).  
   * **Thành phần 2 — ${L}_{KD}$ (Trọng số $0.7$ — Neo Giữ Tri Thức Gốc):** Chưng cất KL-Divergence từ mô hình Teacher đóng băng (${\theta }_{orig}$) trên nhánh ảnh sạch nhằm giữ vững biên quyết định 96.45%.  
   * **Thành phần 3 — ${L}_{FeatCons}$ (Trọng số $0.3$ — Khôi Phục Thứ Tự Xếp Hạng AUC Dưới $Q=30$):**  
     * Thay vì chỉ ép khớp logit đầu ra, ép trực tiếp **vector đặc trưng trước tầng phân loại** ${f}_{student}({x}_{deg})$ phải đồng hướng Cosine và khớp phân phối với ${f}_{student}({x}_{clean})$:  
        \$\$\\mathcal{L}\_{\\text{FeatCons}} \= \\left(1 \- \\cos\\big(\\mathbf{f}\_{\\text{student}}(x\_{\\text{deg}}), \\text{sg}\[\\mathbf{f}\_{\\text{student}}(x\_{\\text{clean}})\]\\big)\\right) \+ \\text{KL}\\big(p(x\_{\\text{clean}}) \\,\\Vert{}\\, p(x\_{\\text{deg}})\\big)\$\$  
     * *Tác dụng:* Bắt buộc backbone ViT \+ Adapter phải trích xuất được các dấu vết ngữ nghĩa-tần số bền vững không bị ma trận lượng tử hóa $8\times 8$ của JPEG $Q=30$ xóa mất, từ đó kéo trực tiếp **ROC-AUC tại $Q=30$ từ \$50\\%\$ lên \$\>65\\% \- 72\\%\$**.  
   * **Thành phần 4 — ${L}_{real\_guard}$ (Trọng số $0.5$ — Triệt Tiêu Báo Oan Ảnh Thật):**  
     * Tách riêng các mẫu có nhãn **Real ($y=0$)** trong mini-batch và áp dụng phạt bất đối xứng:  
        \$\$\\mathcal{L}\_{\\text{real\\\_guard}} \= \\frac{1}{\\vert{}B\_{\\text{real}}\\vert{}} \\sum\_{i \\in B\_{\\text{real}}} \\text{ReLU}\\Big(z\_{\\text{student}}^{\\text{fake}}(x\_{i, \\text{deg}}^{\\text{real}}) \- z\_{\\text{teacher}}^{\\text{fake}}(x\_{i, \\text{clean}}^{\\text{real}})\\Big)^2\$\$  
     * *Tác dụng:* Dạy cho nhánh SRM và Gating quy tắc sắt đá: *"Nếu một bức ảnh tự nhiên bị nén JPEG $Q=30$, các vết khối $8\times 8$ sinh ra là nhiễu nén vật lý chứ không phải dấu vết AI; tuyệt đối không được phép đẩy logit Fake tăng lên\!"*.  
4. **Kỷ luật Tối ưu hóa (Differential LR & Selection Metric):**  
   * **Differential Learning Rate:** Khóa cứng \$86.74\\%\$ backbone CLIP; đặt LR cực nhỏ $l{r}_{old\_adapter}=1.0\times 1{0}^{-5}$ cho các Adapter tiền huấn luyện và $l{r}_{new}=1.0\times 1{0}^{-4}$ cho nhánh SRM/Gating.  
   * **Tiêu chí lưu Checkpoint (best.pth):** Chỉ lưu mô hình khi điểm tổng hợp **Worst-Group AUC (trên cả Clean OOD và Degraded $Q=30$)** đạt cực đại và thỏa mãn điều kiện \$\\text{Real ACC}\_{\\text{Val}} \\ge 90\\%\$.

#### **3\. Đầu ra & Cổng nghiệm thu (Outputs & Gate P2)**

* **Đầu ra:** Trọng số hoàn chỉnh fatformer\_anchored\_recovery\_v2.pth.  
* **Chỉ tiêu cổng ra (Gate P2):**  
  * Trên 18 tập test phòng lab: Clean Macro ACC \$\\ge 92.0\\%\$, Clean Real ACC \$\\ge 90.0\\%\$, $Q=30$ Fake ACC \$\\ge 38.0\\%\$.  
  * Trên bộ ngoại vi CFE: Clean Real ACC tăng từ \$48.10\\%\$ lên **\$\\ge 88.0\\%\$**, Clean Fake ACC **\$\\ge 80.0\\%\$**; dưới nén sâu $Q=30$, **ROC-AUC tăng từ \$50.07\\%\$ lên \$\\ge 62.0\\% \- 68.0\\%\$**.

### **GIAI ĐOẠN P3: Đóng Gói Kiến Trúc Hợp Nhất Có Bộ Định Tuyến Suy Biến Tự Động (UnifiedFatFormerXLA)**

* **Mục đích:** Xóa bỏ hoàn toàn sự phân tách thủ công giữa "Chế độ 1 (Clean)" và "Chế độ 2 (Degraded)", đóng gói thành **một mô hình hộp đen duy nhất (Single Forward Pass)** tự động nhận biết chất lượng ảnh đầu vào để tối ưu hóa quyết định.  
* **Chi phí tài nguyên:** 1 – 2 giờ viết code và kiểm thử suy luận (độ trễ tăng thêm $<2ms$/ảnh).

#### **1\. Đầu vào (Inputs)**

* Một bức ảnh bất kỳ $x\in {R}^{B\times 3\times 224\times 224}$ (người dùng hoặc hệ thống kiểm duyệt không cần biết trước ảnh đó là ảnh gốc phòng lab hay ảnh tải về từ Facebook/Zalo đã bị nén nát).  
* Bộ trọng số đã huấn luyện từ Giai đoạn P1/P2.

#### **2\. Kỹ thuật & Cách làm chi tiết**

1. **Xây dựng Module Đo Suy Biến Vật Lý Không Tham Chiếu (src/models/degradation\_router.py):** Ngay tại đầu hàm forward(x), trích xuất tức thì 3 tín hiệu vật lý của ảnh $x$:  
   * **Chỉ số nhiễu khối DCT $8\times 8$ ($B(x)$):** Tính tỷ số giữa trung bình sai phân bậc nhất tại các đường biên lưới pixel bội số của 8 (\$i, j \\equiv 0 \\pmod 8\$) so với vùng bên trong khối. Khi ảnh bị nén JPEG $Q=30,50$, $B(x)$ tăng vọt rõ rệt.  
   * **Chỉ số năng lượng tần số cao / Phương sai Laplacian (${V}_{Lap}(x)$):** Lấy trực tiếp năng lượng phản hồi từ bộ lọc vi sai bậc hai ${K}_{2}$ (Second-order Laplacian $3\times 3$) có sẵn trong SpatialResidualBlock (src/models/srm.py) để phát hiện hiện tượng mất nét do Gaussian Blur hoặc Down-Up.  
   * **Hệ số Cổng Tần Số Động trung bình ($\bar{\lambda }(x)$):** Thu thập từ các tầng DynamicFrequencyGating (src/models/gating.py), vốn tự động giữ $\lambda (x)\approx 1.0\sim 1.5$ trên ảnh sạch và sụt giảm về $0.05\sim 0.1$ khi gặp ảnh suy biến nặng.  
2. **Cơ chế Điều Phối & Hòa Trộn Mềm (Adaptive Soft Routing & Dynamic Thresholding):**  
   * Từ 3 chỉ số $(B(x),{V}_{Lap}(x),\bar{\lambda }(x))$, tính ra **trọng số mức độ suy biến liên tục** ${w}_{deg}(x)\in [0,1]$ thông qua hàm Sigmoid hiệu chuẩn nhẹ:  
      \$\$w\_{\\text{deg}}(x) \= \\sigma\\Big(\\gamma\_1 \\cdot \\big(B(x) \- \\tau\_B\\big) \- \\gamma\_2 \\cdot \\big(V\_{\\text{Lap}}(x) \- \\tau\_V\\big) \- \\gamma\_3 \\cdot \\big(\\bar{\\lambda}(x) \- \\tau\_\\lambda\\big)\\Big)\$\$  
   * **Điều chỉnh linh hoạt hai tầng:**  
     * *Ở tầng đặc trưng (Feature-level Gating):* Khi ${w}_{deg}(x)\rightarrow 1$ (ảnh nén sâu $Q=30$), cổng tự động hạ tỷ trọng nhánh nhiễu tần số cao dễ gây báo oan và tăng cường nhánh ngữ nghĩa bền vững đã học qua ${L}_{FeatCons}$ ở P2.  
     * *Ở tầng quyết định (Logit/Probability Fusion):* Nội suy mượt mà giữa cấu hình bảo vệ ảnh sạch tuyệt đối và cấu hình truy vết nhạy trên không gian mạng:  
        \$\$P\_{\\text{unified}}(\\text{Fake}\|x) \= \\big(1 \- w\_{\\text{deg}}(x)\\big) \\cdot \\sigma\\left(\\frac{z\_{\\text{clean\\\_expert}}(x)}{T\_{\\text{clean}}}\\right) \+ w\_{\\text{deg}}(x) \\cdot \\sigma\\left(\\frac{z\_{\\text{forensic\\\_expert}}(x)}{T\_{\\text{deg}}}\\right)\$\$

#### **3\. Đầu ra & Cổng nghiệm thu (Outputs & Gate P3)**

* **Đầu ra:** Lớp mô hình hợp nhất UnifiedFatFormerXLA, xuất ra duy nhất 1 xác suất dự đoán $P(Fake|x)$ và 1 nhãn quyết định $\hat{y}\in \{0,1\}$ cho mọi điều kiện ảnh.  
* **Giá trị mang lại cho Luận văn:** Khi chạy kiểm chuẩn eval\_communityforensics.py và eval\_locked\_benchmark.py, mô hình hợp nhất này sẽ **vượt qua đồng thời cả 6/6 Hard Gates** (vừa đạt \$\>92\\% \- 95\\%\$ Clean ACC, vừa bảo vệ Real ACC \$\\ge 90\\%\$, vừa đạt \$\\ge 40\\%\$ Fake ACC và \$\>65\\%\$ AUC dưới nén sâu $Q=30$) trong **một thực thể duy nhất**.

## **PHẦN III: TỔNG HỢP BẢNG ĐỐI CHIẾU HIỆU NĂNG TRƯỚC VÀ SAU KHI NÂNG CẤP**

Bảng dưới đây tóm tắt sự chuyển dịch chất lượng của hệ thống khi đi qua 4 giai đoạn trên bộ kiểm chuẩn ngoại vi CommunityForensics-Eval (2.000 ảnh từ các bộ sinh 2024–2025):

| Thước Đo Đánh Giá (Trên CFE 2.000 Ảnh) | Hiện Tại: Baseline CVPR 2024 | Hiện Tại: WiSE-FT (Lỗi Loader) | Hiện Tại: Anchored Mode 2 (Lệch Ngưỡng) | Sau P1: WiSE-FT Gen 2 (0 GPU) | Sau P2 \+ P3: FatFormer-XLA v2 Unified (Mục tiêu) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **1\. Trạng thái nạp Checkpoint (strict=True)** | Pass (0 lỗi) | **Fail (146 unexpected)** | Sai cấu hình (Adapters=3) | **Pass tuyệt đối (0/0)** | **Pass tuyệt đối (0/0)** |
| **2\. Clean Real ACC (Chống báo oan ảnh thật)** | 94.00% | 100.0% *(Ảo)* | 48.10% *(Báo oan nặng)* | **88.0% – 92.0%** | **\$\\ge 91.0\\% \- 94.0\\%\$** |
| **3\. Clean Fake ACC (Bắt AI 2024–2025)** | 35.30% | 4.10% *(Tê liệt)* | 90.50% | **68.0% – 76.0%** | **\$\\ge 82.0\\% \- 86.0\\%\$** |
| **4\. Clean ROC-AUC (Năng lực phân tách sạch)** | 76.21% | 57.07% | 79.28% | **82.0% – 85.0%** | **\$\\ge 88.0\\% \- 91.0\\%\$** |
| **5\. JPEG $Q=30$ Fake ACC (Kháng nén mạng xã hội)** | 0.90% | 0.20% | 40.80% | **25.0% – 32.0%** | **\$\\ge 45.0\\% \- 55.0\\%\$** |
| **6\. JPEG $Q=30$ Real ACC (Bảo vệ ảnh thật bị nén)** | 99.80% | 100.0% *(Ảo)* | 58.40% | **82.0% – 86.0%** | **\$\\ge 85.0\\% \- 90.0\\%\$** |
| **7\. JPEG $Q=30$ ROC-AUC (Xếp hạng thực chất $Q=30$)** | 50.00% | 50.93% | 50.07% *(Mất thứ tự)* | **55.0% – 58.0%** | **\$\\ge 65.0\\% \- 72.0\\%\$** |
| **8\. Cơ chế vận hành** | Đơn lẻ (Cũ) | Hỏng | Phải chuyển tay Mode 2 | Đơn lẻ (1 file .pth) | **Tự động 100% (1 lệnh forward)** |

## **PHẦN IV: THỨ TỰ BẮT TAY VÀO LÀM NGAY HÔM NAY**

Để tiết kiệm tối đa thời gian và thấy ngay kết quả qua từng mốc:

1. **Hôm nay (Làm ngay P0 \+ P1 trong 2 giờ):** Sửa canonical\_loader ép strict=True, sửa lỗi num\_vit\_adapter=8 trong eval\_communityforensics.py, và chạy lệnh trộn tools/wise\_ft\_merge.py giữa fatformer\_4class\_ckpt.pth và fatformer\_anchored\_recovery\_best.pth để tạo **WiSE-FT Gen 2**. Chạy lại đánh giá CFE để lấy ngay bộ bảng số liệu sạch, không còn lỗi 146 keys.  
2. **Ngày mai (Triển khai P2 trong 1 phiên GPU 4–5 giờ):** Cập nhật src/training/loss.py (thêm ${L}_{real\_guard}$ \+ ${L}_{FeatCons}$, khôi phục tỷ lệ $0.3CE+0.7KD$), mở dải JPEG về $[30,85]$ trong src/datasets/transforms.py và chạy huấn luyện 3–4 epochs với dữ liệu bổ sung từ CommunityForensics-Small.  
3. **Ngày thứ 3 (Hoàn thiện P3 trong 2 giờ):** Đóng gói lớp DegradationRouter vào src/models/degradation\_router.py, xuất báo cáo nghiệm thu cuối cùng và vẽ biểu đồ ROC / Grad-CAM đối chứng đưa vào Luận văn.

