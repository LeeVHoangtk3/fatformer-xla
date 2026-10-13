# BÁO CÁO NGHIỆM THU KỸ THUẬT: TASK 6.1
## PHÂN TÍCH ĐỊNH LƯỢNG HIỆN TƯỢNG LỆCH PHÂN PHỐI & PHÂN TẦNG LỖI 18 BỘ SINH (CROSS-GENERATOR DISTRIBUTION SHIFT DIAGNOSIS)

> **Mã nhiệm vụ**: `Task 6.1` *(Milestone 6)*  
> **Người thực hiện**: Thành viên C (Training & Evaluation Lead)  
> **Người phối hợp**: Thành viên B (Architecture & Loss Lead), Thành viên A (Data & Infra Lead)  
> **Thời gian thực hiện**: 13/10/2026  
> **Môi trường thực thi**: Môi trường phân tích số liệu cục bộ  
> **Dữ liệu đầu vào**: [`DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv`](../../../DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv) kế thừa từ kết quả nghiệm thu Task 5.1 ([`docs/bao_cao/C/bao_cao_task_5.1_full_benchmark_18_datasets.md`](bao_cao_task_5.1_full_benchmark_18_datasets.md))  
> **Công cụ phân tích**: [`tools/analyze_distribution_shift.py`](../../../tools/analyze_distribution_shift.py)  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH NGHIỆM THU`  

---

### I. MỤC TIÊU & BỐI CẢNH KỸ THUẬT

Dựa trên kết quả đo đạc thực tế đã được nghiệm thu tại Task 5.1 ([`bao_cao_task_5.1_full_benchmark_18_datasets.md`](bao_cao_task_5.1_full_benchmark_18_datasets.md)), mô hình `fatformer_plan_a_best_weights.pth` (2.15 GB, 1.684 tensor) đã hoàn thành xuất sắc mục tiêu bảo vệ miền phân phối ProGAN (Clean ACC 97,95%, Q30 61,60%). Tuy nhiên, khi đối diện với 17 bộ sinh nằm ngoài miền dữ liệu huấn luyện, mô hình bộc lộ sự suy giảm hiệu năng rõ rệt (Macro Clean ACC 67,34%, Pooled ACC 70,37%).

Mục tiêu của Task 6.1 là:
1. **Xác minh minh bạch trạng thái dữ liệu nhật ký (Logging State Verification)**: Xác định phương thức phân tích theo Kịch bản A hay Kịch bản B, đảm bảo không vi phạm nguyên tắc bảo toàn dữ liệu kiểm định.
2. **Bóc tách định lượng hiện tượng lệch phân phối (Distribution Shift)**: Đo lường chi tiết mức độ thiên lệch giữa tỷ lệ nhận diện ảnh thật (Real Recall / $r\_acc$) và tỷ lệ nhận diện ảnh giả (Fake Recall / $f\_acc$) trên từng bộ sinh.
3. **Phân tầng 4 cụm lỗi kỹ thuật (Error Profiles)**: Phân định ranh giới giữa hiện tượng "mù đặc trưng thực sự" (Feature Blind Spot) và hiện tượng "lệch điểm cắt phân lớp do thiên lệch dự đoán" (Threshold Misalignment / Calibration Bias), cũng như "suy sụp dưới nén sâu" (Compression Sensitivity).
4. **Cung cấp cơ sở khoa học cho Milestone 6**: Chuyển giao các kết luận định lượng cho Lead A (Task 6.2 - thu thập dữ liệu bù đắp) và Lead B (Task 6.3-B - thiết kế hàm mất mát cân bằng và bảo toàn biểu diễn).

---

### II. XÁC MINH TRẠNG THÁI ĐẦU VÀO: KÍCH HOẠT KỊCH BẢN B

Theo quy chuẩn kiểm tra đầu vào (DoD Tiền đề của Task 6.1):
- **Hiện trạng kho lưu trữ**: Thư mục `DATASET/logs/m4_full_benchmark/` lưu trữ bảng tổng hợp ma trận 140 dòng [`final_full_benchmark_matrix.csv`](../../../DATASET/logs/m4_full_benchmark/final_full_benchmark_matrix.csv) bao gồm đầy đủ các giá trị đo đạc phân tầng cho 18 tập kiểm thử trên 7 biến thể suy giảm (clean, jpeg_q50, jpeg_q30, blur_s1, blur_s2, down_up, v.v.).
- **Tệp xác suất thô từng ảnh (per-sample logits)**: Không được lưu lại trong phiên chạy Kaggle nhằm tối ưu hóa bộ nhớ đĩa cứng và thời gian ghi đĩa khi xử lý hơn 120.000 lượt ảnh.
- **Quyết định phương pháp**: Tuân thủ nghiêm ngặt nguyên tắc **"Không chạy lại Task 5.1 chỉ để quét ngưỡng (tune threshold) nhằm tránh biến tập kiểm thử thành tập điều chỉnh tham số"**, nhóm kỹ thuật chính thức **kích hoạt Kịch bản B**.
- Toàn bộ việc chẩn đoán trong Task 6.1 dựa hoàn toàn trên các chỉ số định lượng khách quan có sẵn (`acc`, `ap`, `auc`, `r_acc`, `f_acc`, `delta_drop`). Việc quét ngưỡng tối ưu cục bộ và hiệu chuẩn xác suất (Temperature Scaling) sẽ được chuyển giao thực hiện trên tập kiểm định độc lập `val_crossdomain` tại Task 6.4.

---

### III. BẢNG DỮ LIỆU ĐỊNH LƯỢNG CHI TIẾT 18 BỘ SINH (TASK 6.1)

Dữ liệu được trích xuất và tính toán tự động thông qua công cụ [`tools/analyze_distribution_shift.py`](../../../tools/analyze_distribution_shift.py):

| Nhóm Kiến Trúc | Bộ Sinh (Subset) | Clean ACC (%) | AP (%) | ROC-AUC (%) | Real ACC (%) | Fake ACC (%) | Độ Lệch Real-Fake (Bias pp) | ACC Q=30 (%) | Fake ACC Q=30 (%) | Độ Sụt Giảm ACC ($\Delta_{\text{drop}}$ pp) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GANs** | progan | 97,95 | 99,85 | 99,83 | 99,65 | 96,25 | +3,40 | 61,60 | 55,20 | -36,35 |
| **GANs** | stylegan | 80,35 | 89,50 | 84,90 | 97,28 | 63,41 | +33,87 | 48,60 | 16,80 | -31,75 |
| **GANs** | stylegan2 | 73,97 | 87,14 | 82,73 | 98,51 | 49,44 | +49,07 | 49,80 | 18,40 | -24,17 |
| **GANs** | biggan | 56,07 | 58,69 | 52,33 | 91,00 | 21,15 | +69,85 | 56,80 | 38,00 | +0,73 |
| **GANs** | cyclegan | 53,14 | 58,99 | 61,92 | 91,75 | 14,53 | +77,22 | 53,80 | 44,40 | +0,66 |
| **GANs** | stargan | 98,00 | 99,86 | 99,83 | 97,10 | 98,90 | -1,80 | 51,40 | 24,40 | -46,60 |
| **GANs** | gaugan | 45,02 | 35,70 | 24,59 | 88,02 | 2,02 | +86,00 | 54,20 | 38,80 | +9,18 |
| **GANs** | deepfake | 64,74 | 85,19 | 82,51 | 98,78 | 30,58 | +68,20 | 49,80 | 0,00 | -14,94 |
| **Diffusion** | guided | 58,50 | 65,38 | 62,73 | 90,20 | 26,80 | +63,40 | 59,20 | 18,40 | +0,70 |
| **Diffusion** | ldm_200 | 67,50 | 77,11 | 74,27 | 83,90 | 51,10 | +32,80 | 48,80 | 34,80 | -18,70 |
| **Diffusion** | ldm_200_cfg | 73,35 | 83,56 | 80,17 | 83,90 | 62,80 | +21,10 | 52,20 | 41,60 | -21,15 |
| **Diffusion** | ldm_100 | 66,75 | 76,72 | 73,68 | 83,90 | 49,60 | +34,30 | 49,20 | 35,60 | -17,55 |
| **Diffusion** | glide_50_27 | 65,65 | 74,25 | 69,16 | 83,90 | 47,40 | +36,50 | 38,20 | 13,60 | -27,45 |
| **Diffusion** | glide_100_10 | 66,00 | 75,21 | 70,83 | 83,90 | 48,10 | +35,80 | 37,40 | 12,00 | -28,60 |
| **Diffusion** | glide_100_27 | 64,85 | 73,43 | 68,09 | 83,90 | 45,80 | +38,10 | 36,80 | 10,80 | -28,05 |
| **Diffusion** | dalle | 48,80 | 46,48 | 40,29 | 83,90 | 13,70 | +70,20 | 48,20 | 33,60 | -0,60 |
| **Diffusion** | pndm | 49,60 | 44,88 | 41,17 | 96,50 | 2,70 | +93,80 | 62,80 | 29,60 | +13,20 |
| **Diffusion** | vqdiffusion | 81,90 | 92,54 | 90,23 | 96,50 | 67,30 | +29,20 | 62,80 | 29,60 | -19,10 |

---

### IV. PHÂN TẦNG 4 CỤM LỖI KỸ THUẬT (ERROR TAXONOMY)

Dựa trên mối tương quan giữa năng lực xếp hạng (AP, ROC-AUC), độ chính xác tại ngưỡng cố định 0.5 (ACC, Real ACC, Fake ACC) và độ suy giảm dưới nén JPEG sâu, 18 bộ sinh được phân tách thành 4 cụm lỗi kỹ thuật rõ rệt:

#### 1. Cụm Vùng Mù Đặc Trưng (Feature Blind Spot)
*Tiêu chí định lượng*: $\text{AP} < 60\%$ và $\text{ROC-AUC} < 60\%$.
- **Các bộ sinh tiêu biểu**:
  - `gaugan` (GANs): AP 35,70%, AUC 24,59%, ACC 45,02%, Fake ACC 2,02%.
  - `pndm` (Diffusion): AP 44,88%, AUC 41,17%, ACC 49,60%, Fake ACC 2,70%.
  - `dalle` (Diffusion): AP 46,48%, AUC 40,29%, ACC 48,80%, Fake ACC 13,70%.
  - `biggan` (GANs): AP 58,69%, AUC 52,33%, ACC 56,07%, Fake ACC 21,15%.
  - `cyclegan` (GANs): AP 58,99%, AUC 61,92%, ACC 53,14%, Fake ACC 14,53%.
- **Bản chất kỹ thuật**:
  - ROC-AUC dưới 50% (đặc biệt ở GauGAN đạt 24,59%, PNDM đạt 41,17%, DALL-E đạt 40,29%) chứng minh rằng mô hình xếp hạng ngược hoàn toàn: các mẫu ảnh giả của các bộ sinh này bị mô hình gán điểm số dự đoán còn "thật hơn cả ảnh thật".
  - Mô hình hoàn toàn thiếu vắng các mẫu vân đặc trưng (fingerprints/artifacts) của các kiến trúc này trong không gian biểu diễn (feature representation).
  - **Kết luận giải pháp**: Việc điều chỉnh điểm cắt phân loại (threshold tuning) hay co giãn nhiệt độ (temperature scaling) hoàn toàn **vô hiệu** đối với cụm này. Biện pháp cần áp dụng là **phải bổ sung dữ liệu huấn luyện đa dạng** từ các họ kiến trúc này vào tập huấn luyện của Task 6.2.

#### 2. Cụm Lệch Điểm Cắt Phân Lớp (Threshold Misalignment / Calibration Bias)
*Tiêu chí định lượng*: $\text{AP} \ge 70\%$ hoặc $\text{ROC-AUC} \ge 70\%$, nhưng Fake ACC $< 65\%$ tại ngưỡng mặc định 0.5 (Độ lệch Bias Real - Fake rất lớn từ $+21\%$ đến $+68\%$).
- **Các bộ sinh tiêu biểu**:
  - `deepfake` (GANs): AP 85,19%, AUC 82,51% (rất cao!), nhưng Fake ACC chỉ đạt 30,58% (Real ACC đạt 98,78%, Bias = +68,20%).
  - `stylegan` (GANs): AP 89,50%, AUC 84,90%, Fake ACC đạt 63,41% (Real ACC 97,28%, Bias = +33,87%).
  - `stylegan2` (GANs): AP 87,14%, AUC 82,73%, Fake ACC đạt 49,44% (Real ACC 98,51%, Bias = +49,07%).
  - `ldm_200_cfg` (Diffusion): AP 83,56%, AUC 80,17%, Fake ACC 62,80% (Bias = +21,10%).
  - `ldm_200` (Diffusion): AP 77,11%, AUC 74,27%, Fake ACC 51,10% (Bias = +32,80%).
  - `ldm_100` (Diffusion): AP 76,72%, AUC 73,68%, Fake ACC 49,60% (Bias = +34,30%).
  - Họ `glide` (`glide_100_10`, `glide_50_27`, `glide_100_27`): AP từ 73,4% đến 75,2%, AUC ~68-71%, nhưng Fake ACC chỉ đạt từ 45,8% đến 48,1% (Bias ~ +36% đến +38%).
- **Bản chất kỹ thuật**:
  - Chỉ số AP và ROC-AUC vượt trội ($>75-89\%$) chứng minh bộ trích xuất đặc trưng của mô hình **đã nhận diện được tín hiệu khác biệt** giữa ảnh thật và ảnh giả của các bộ sinh này. Mô hình có năng lực phân tách tốt.
  - Tuy nhiên, do quá trình huấn luyện tập trung vào ProGAN và tỷ lệ phân bố nhãn bị lệch, phân phối xác suất dự đoán (output logits) bị dịch chuyển mạnh về phía nhãn Thật (Real). Kết quả là tại điểm cắt cứng 0.5, mô hình thiên vị phán đoán mẫu là Thật (Real ACC đạt gần như tuyệt đối 97-98%, trong khi Fake ACC bị kéo sụt xuống 30-50%).
  - **Kết luận giải pháp**: Cụm này có thể cải thiện hiệu năng rõ rệt thông qua **tối ưu hóa hàm mất mát (Focal Loss / Balanced Loss)** để cân bằng gradient ở Task 6.3-B, kết hợp cùng **hiệu chuẩn xác suất (Temperature Scaling / Optimal Thresholding)** trên tập kiểm định độc lập ở Task 6.4.

#### 3. Cụm Nhạy Cảm Nén Sâu (Compression Sensitivity)
*Tiêu chí định lượng*: Tỷ lệ nhận diện ảnh giả (Fake ACC) bị sụp đổ nghiêm trọng khi chuyển từ ảnh sạch sang biến thể nén JPEG sâu $Q=30$.
- **Các bộ sinh tiêu biểu**:
  - `deepfake` (GANs): Fake ACC từ 30,58% rơi thẳng về **0,00%** (mất sạch hoàn toàn khả năng phát hiện mẫu giả).
  - `stargan` (GANs): Fake ACC từ 98,90% sụt giảm sâu xuống **24,40%** (mức sụt giảm lên tới 74,50 pp).
  - `stylegan` (GANs): Fake ACC từ 63,41% sụt giảm xuống **16,80%** (mức sụt giảm 46,61 pp).
  - `stylegan2` (GANs): Fake ACC từ 49,44% sụt giảm xuống **18,40%** (mức sụt giảm 31,04 pp).
  - Nhánh `glide` (`glide_100_27`, `glide_100_10`, `glide_50_27`): Fake ACC sạch quanh mức 46-48% sụt giảm sâu xuống chỉ còn **10,80% - 13,60%** (mức sụt giảm ~35 pp).
- **Bản chất kỹ thuật**:
  - Các đặc trưng phân biệt mà mô hình khai thác trên các bộ sinh này chủ yếu tập trung ở dải tần số cao (high-frequency artifacts).
  - Khi thuật toán nén JPEG áp dụng ma trận lượng tử hóa mạnh ở mức chất lượng $Q=30$, các khối biến đổi cosin rời rạc (DCT) $8 \times 8$ đã triệt tiêu hoàn toàn các thành phần tần số cao này. Mô hình mất đi điểm tựa nhận diện và mặc định dự đoán về nhãn Thật.
  - **Kết luận giải pháp**: Cần áp dụng chiến lược **tăng cường dữ liệu cặp đôi (Paired Augmentation)** kết hợp **hàm mất mát nhất quán / chưng cất biểu diễn (Consistency Loss / Feature Distillation)** tại Task 6.3-B để buộc mô hình học các biểu diễn ngữ nghĩa bền vững (semantic-level representations) thay vì chỉ phụ thuộc vào nhiễu tần số cao.

#### 4. Cụm Vùng Lõi Bền Vững (Robust Core)
*Tiêu chí định lượng*: Clean ACC $\ge 80\%$, AP $\ge 90\%$, Fake ACC $\ge 65\%$.
- **Các bộ sinh tiêu biểu**:
  - `progan` (GANs): Clean ACC 97,95%, AP 99,85%, ROC-AUC 99,83%, Real ACC 99,65%, Fake ACC 96,25%. Kháng nén $Q=30$ đạt ACC 61,60%, Fake ACC 55,20%.
  - `stargan` (GANs): Clean ACC 98,00%, AP 99,86%, ROC-AUC 99,83%, Real ACC 97,10%, Fake ACC 98,90%.
  - `vqdiffusion` (Diffusion): Clean ACC 81,90%, AP 92,54%, ROC-AUC 90,23%, Real ACC 96,50%, Fake ACC 67,30%. Kháng nén $Q=30$ đạt ACC 62,80%.
- **Bản chất kỹ thuật**:
  - Mô hình thể hiện năng lực tổng quát hóa xuất sắc và duy trì phong độ vượt trội trên các bộ sinh này.
  - **Lưu ý bảo toàn**: Trong toàn bộ quá trình fine-tune chống domain shift ở Task 6.3, các bộ sinh thuộc Cụm Lõi Bền Vững (đặc biệt là ProGAN) phải được sử dụng làm mốc neo kiểm soát (retention anchor) để tránh hiện tượng quên thảm họa (catastrophic forgetting).

---

### V. ĐỊNH HƯỚNG TRIỂN KHAI CHO CÁC NHIỆM VỤ TIẾP THEO

Từ các phát hiện định lượng trên, nhóm nghiên cứu đề xuất phân công kỹ thuật cụ thể cho các thành viên:

#### 1. Chuyển Giao Dữ Liệu Cho Lead A (Task 6.2)
- **Tập trung thu thập bù đắp theo cụm lỗi**:
  - Ưu tiên hàng đầu: Thu thập dữ liệu từ các họ kiến trúc thuộc **Cụm Vùng Mù Đặc Trưng** (GauGAN, PNDM, DALL-E, BigGAN).
  - Ưu tiên thứ hai: Bổ sung dữ liệu đa dạng từ họ Diffusion (LDM, Glide) để hỗ trợ cân bằng phân phối.
- **Nguyên tắc cô lập dữ liệu (Strict Data Isolation)**:
  - Tuyệt đối không trích xuất dữ liệu từ thư mục `DATASET/test/m4_full_benchmark/` (tập dữ liệu Task 5.1).
  - Nguồn dữ liệu mới phải được lấy độc lập từ GenImage / bộ sinh bên ngoài, sau đó phân chia nghiêm ngặt thành 2 tập: `train_crossdomain` (dùng huấn luyện) và `val_crossdomain` (dùng kiểm định song song).

#### 2. Chuyển Giao Kiến Trúc & Hàm Mất Mát Cho Lead B (Task 6.3-B)
- **Giải quyết thiên lệch điểm cắt (Threshold Misalignment)**:
  - Tích hợp hàm mất mát có trọng số điều chỉnh hoặc **Focal Loss** nhằm giảm bớt độ tự tin quá mức đối với nhãn Thật (Real) và tập trung gradient vào các mẫu Giả (Fake) khó phân loại.
- **Giải quyết suy sụp dưới nén (Compression Sensitivity)**:
  - Thiết kế cấu trúc huấn luyện **Paired Augmentation**: Đưa đồng thời cặp ảnh gốc $(x_{\text{clean}})$ và ảnh nén $(x_{\text{jpeg}})$ qua mô hình.
  - Áp dụng **Consistency Loss** (hoặc Feature Distillation Loss) giữa đặc trưng biểu diễn của ảnh sạch và ảnh nén để ép mô hình trích xuất đặc trưng kháng nén bền vững.

#### 3. Chuyển Giao Quy Trình Huấn Luyện & Kiểm Định Cho Lead C (Task 6.3-C & Task 6.4)
- **Đánh giá trên tập `val_crossdomain` độc lập**:
  - Khóa chặt việc theo dõi checkpoint trong Task 6.3-C trên cả hai chiều: Hiệu năng trên miền mới (`val_crossdomain`) và Khả năng bảo toàn miền cũ (`val_progan_clean`).
- **Hiệu chuẩn xác suất tại Task 6.4**:
  - Thực hiện Temperature Scaling và dò tìm điểm cắt tối ưu trên tập `val_crossdomain` độc lập để giải phóng năng lực xếp hạng của các bộ sinh thuộc Cụm Lệch Điểm Cắt trước khi bước vào bài kiểm tra khóa cuối cùng.

---

### VI. BIÊN BẢN NGHIỆM THU TIÊU CHUẨN HOÀN THÀNH (DoD)

| Tiêu chuẩn hoàn thành (Definition of Done) | Trạng thái | Minh chứng kỹ thuật |
| :--- | :---: | :--- |
| **Xác minh trạng thái log thô** | **ĐẠT** | Xác nhận kích hoạt Kịch bản B, tuân thủ nguyên tắc không can thiệp hay chạy lại tập Task 5.1. |
| **Lập bảng số liệu chẩn đoán phân tầng** | **ĐẠT** | Bảng phân tích chi tiết 18 bộ sinh với đầy đủ các chỉ số Clean ACC, AP, ROC-AUC, Real ACC, Fake ACC, Bias, Q30 ACC và $\Delta_{\text{drop}}$. |
| **Phân tầng 4 cụm lỗi kỹ thuật** | **ĐẠT** | Phân định ranh giới rõ ràng: Vùng Mù Đặc Trưng (5 bộ sinh), Lệch Điểm Cắt (9 bộ sinh), Nhạy Cảm Nén (5 bộ sinh), Lõi Bền Vững (3 bộ sinh). |
| **Báo cáo nghiệm thu hoàn chỉnh** | **ĐẠT** | Hoàn tất văn kiện nghiệm thu khoa học, làm tiền đề dữ liệu cho Task 6.2 và hàm mất mát cho Task 6.3-B. |

**Chữ ký nghiệm thu**:
- Thành viên C (Training & Evaluation Lead): *Đã hoàn tất phân tích và nghiệm thu.*
- Thành viên B (Architecture Lead): *Đã tiếp nhận yêu cầu thiết kế hàm mất mát (Focal & Consistency Loss).*
- Thành viên A (Data Lead): *Đã tiếp nhận yêu cầu phân vùng và thu thập dữ liệu đa-generator.*
