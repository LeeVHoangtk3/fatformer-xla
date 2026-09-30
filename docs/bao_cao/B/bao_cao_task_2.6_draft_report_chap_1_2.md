# BÁO CÁO NGHIỆM THU NHIỆM VỤ 2.6 (TASK 2.6)
## SOẠN THẢO BÁO CÁO ĐỒ ÁN - CHƯƠNG 1 (GIỚI THIỆU) & CHƯƠNG 2 (CÁC CÔNG TRÌNH LIÊN QUAN)

> **Người thực hiện**: Thành viên B *(Architecture & Loss Lead - Chủ trì)*  
> **Người nghiệm thu / Review kỹ thuật**: Thành viên C *(Training & Evaluation Lead - Trưởng Dự Án)*  
> **Thời gian hoàn thành**: 2026-09-30  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC (PASS 100% TIÊU CHUẨN DoD)`  
> **Văn bản hoàn thiện**: [`docs/report/chuong_1_va_2_ban_thao.md`](../../report/chuong_1_va_2_ban_thao.md)  
> **Quy mô văn bản**: 15+ trang tài liệu học thuật chuyên sâu | 25 tài liệu tham khảo chuẩn mực IEEE  

---

## I. MỤC TIÊU VÀ Ý NGHĨA KHOA HỌC

1. **Hiện thực hóa mô hình viết báo cáo cuốn chiếu (*Progressive Paper Writing*)**:
   - Hoàn thành dứt điểm 40% khối lượng văn bản học thuật của toàn bộ đồ án tốt nghiệp ngay tại thời điểm kết thúc Tuần 2, loại bỏ triệt để rủi ro dồn khối lượng công việc vào tuần cuối cùng.
2. **Khai thác tức thời số liệu thực nghiệm đo đạc của Milestone 1**:
   - Trực tiếp đưa số liệu kiểm thử mốc trần từ Task 2.2 (**Clean Overall ACC 96.45%**) và mốc sàn sụp đổ từ Task 2.3 (**$Q=30$ Overall ACC 52.89%, Fake ACC chỉ còn 5.78%**) cùng bản đồ nhiệt Grad-CAM phân tán từ Task 2.4 vào Chương 1 để làm luận chứng thực nghiệm sắc bén chứng minh "tử huyệt" của bài báo CVPR 2024.
3. **Xác lập khoảng trống nghiên cứu và vị thế của FatFormer-XLA trong y văn**:
   - Phân tích có hệ thống tiến trình 3 thế hệ phát hiện ảnh AI (CNN không gian, phổ tần số Fourier/DCT, mô hình đa phương thức VLM/CLIP).
   - Chỉ rõ hai khoảng trống nghiên cứu: Chưa có mô hình nào tích hợp khối lọc vết dư SRM 3-Kernels với CLIP ViT, và chưa có cơ chế cổng thích ứng tần số động $\lambda(x)$ để tự vệ trước nhiễu khối lượng tử hóa JPEG.

---

## II. KẾ THỪA THÔNG SỐ TỪ CÁC TASK TIỀN ĐỀ (RULE 4)

1. **Kế thừa từ Task 2.2** ([`docs/bao_cao/C/bao_cao_task_2.2_baseline_clean_benchmark.md`](../C/bao_cao_task_2.2_baseline_clean_benchmark.md) - Lead C):
   - Mốc trần Clean tham chiếu của FatFormer gốc: Overall ACC **96.45%**, GANs **98.39%**, Diffusion **94.91%**, Guided Diff **76.05%**.
2. **Kế thừa từ Task 2.3** ([`docs/bao_cao/C/bao_cao_task_2.3_baseline_degraded_benchmark.md`](../C/bao_cao_task_2.3_baseline_degraded_benchmark.md) - Lead C):
   - Mốc sàn sụt giảm đo đạc trên 54,000 ảnh:
     - Nén sâu $Q=30$: Overall ACC rơi xuống **52.89%** ($\Delta = -43.56\%$).
     - Fake ACC sụp đổ hoàn toàn xuống **5.78%** (để lọt $94.22\%$ ảnh Deepfake).
     - Nén trung bình $Q=50$: Overall ACC **56.24%**, Fake ACC **12.49%**.
     - Nén nhẹ $Q=70$: Overall ACC **61.28%**, Fake ACC **22.60%**.
     - Làm mờ Gaussian $\sigma=1.0$: Nhóm Diffusion sụt giảm xuống **60.20%**.
3. **Kế thừa từ Task 2.4** ([`docs/bao_cao/B/bao_cao_task_2.4_gradcam_baseline_extraction.md`](bao_cao_task_2.4_gradcam_baseline_extraction.md) - Lead B):
   - Minh chứng định tính XAI: Bản đồ chú ý Grad-CAM trên ảnh $Q=30$ bị phân tán hoàn toàn khỏi các đường biên giả mạo.
4. **Kế thừa từ Task 2.5** ([`docs/milestones/milestone_1_freeze.md`](../../milestones/milestone_1_freeze.md) - Cả nhóm):
   - Ba chỉ tiêu định lượng bắt buộc của đồ án: Kéo $Q=30$ tăng $+10\% \sim +15\%$, bảo toàn ảnh sạch $\ge 96.00\%$, kéo Fake ACC trên $Q=30$ từ $5.78\% \rightarrow >50\%$.

---

## III. NỘI DUNG CHI TIẾT BẢN THẢO HOÀN THIỆN

Văn bản đầy đủ được lưu trữ tại [`docs/report/chuong_1_va_2_ban_thao.md`](../../report/chuong_1_va_2_ban_thao.md) với cấu trúc chuẩn mực:

### 1. Chương 1: Giới Thiệu và Động Lực Nghiên Cứu
- **Mục 1.1**: Bối cảnh bùng nổ của Generative AI (GANs, Diffusion, Midjourney, SDXL, FLUX) và hiểm họa Deepfake đối với an ninh, kinh tế, chính trị và quyền riêng tư.
- **Mục 1.2**: Thách thức thực tiễn về độ bền vững pháp y (Forensic Robustness) trước các biến dạng cưỡng bức trên mạng xã hội (Nén JPEG $8 \times 8$, Gaussian Blur, Down-Up).
- **Mục 1.3**: Phân tích chuyên sâu lỗ hổng kỹ thuật của mô hình FatFormer (CVPR 2024), đưa số liệu thực nghiệm đo đạc từ Task 2.2 và 2.3 làm bằng chứng cho hiện tượng *One-Sided Classification Collapse* (Real ACC 100%, Fake ACC 5.78%).
- **Mục 1.4**: Mục tiêu nghiên cứu và 4 đóng góp khoa học cốt lõi của đồ án FatFormer-XLA:
  1. Khối không gian SRM 3-Kernels cố định (`requires_grad=False`).
  2. Cổng thích ứng tần số động `DynamicFrequencyGating` $\lambda(x) \in [0.0, 2.0]$.
  3. Hàm mất mát `DualStreamFocalLoss` ($\gamma=2.0, \alpha=0.25$).
  4. Lập lịch suy thoái `CurriculumDegradationScheduler` 3 giai đoạn kết hợp dữ liệu Staging 90/10.
- **Mục 1.5**: Bố cục 5 chương của đồ án tốt nghiệp.

### 2. Chương 2: Các Công Trình Nghiên Cứu Liên Quan (Related Work)
- **Mục 2.1**: Tiến trình 3 thế hệ phát hiện ảnh AI: Từ CNN không gian thuần túy (CNNDetect) $\rightarrow$ Dấu vết phổ tần số (Frank et al., Durall et al., F3-Net) $\rightarrow$ Mô hình đa phương thức VLM/CLIP (UnivFD, RINE, FatFormer).
- **Mục 2.2**: Cơ sở lý thuyết của biến đổi Wavelet Haar DWT trong pháp y số và phân tích bản chất suy biến của các dải băng tần cao dưới lượng tử hóa JPEG.
- **Mục 2.3**: Kỹ thuật trích xuất vết dư không gian Spatial Rich Models (SRM), cơ chế triệt tiêu ngữ nghĩa cảnh vật và tính bền vững của tương quan vi sai điểm ảnh lân cận.
- **Mục 2.4**: Lý thuyết học tăng cường độ bền vững và chiến lược học theo giáo trình (Curriculum Learning) giúp loại bỏ hiện tượng sốc gradient ở các epoch đầu.
- **Mục 2.5**: Bảng ma trận đối sánh toàn diện và phân tích khoảng trống nghiên cứu (Research Gap).
- **Danh mục tài liệu tham khảo**: Trích dẫn đầy đủ 25 bài báo đỉnh cao (CVPR, ICCV, ECCV, NeurIPS, ICLR, ICML, IEEE TIFS, IEEE TPAMI).

---

## IV. ĐỐI CHIẾU TIÊU CHUẨN NGHIỆM THU (DoD VERIFICATION GATE)

| Tiêu Chuẩn Nghiệm Thu (DoD) | Yêu Cầu Thiết Kế | Thực Nghiệm & Văn Bản Hoàn Thành | Đánh Giá Nghiệm Thu |
| :--- | :--- | :--- | :---: |
| **Độ dài và quy mô** | Tối thiểu 8–10 trang chuẩn format luận văn | Đạt 15+ trang phân tích chuyên sâu đầy đủ bảng biểu | ✅ **PASS XUẤT SẮC** |
| **Trích dẫn khoa học** | Chuẩn mực IEEE/BibTeX, các hội nghị đỉnh cao | Đầy đủ 25 trích dẫn SOTA cập nhật nhất | ✅ **PASS 100%** |
| **Tích hợp số liệu thực tế** | Bảng số liệu Task 2.2, 2.3 và ảnh CAM Task 2.4 | Tích hợp chi tiết bảng mốc trần, mốc sàn và cơ chế sụp đổ | ✅ **PASS 100%** |
| **Review kỹ thuật của Lead C** | Đảm bảo tính nhất quán với lộ trình và code | Đã review, đối soát từng công thức và ký duyệt | ✅ **ĐÃ KÝ DUYỆT** |

---

## V. KÝ DUYỆT NGHIỆM THU HAI BÊN

| Vai Trò | Người Thực Hiện / Nghiệm Thu | Xác Nhận Phê Duyệt | Ngày Ký |
| :--- | :--- | :---: | :---: |
| **Chủ trì soạn thảo (Lead B)** | Thành viên B *(Architecture & Loss Lead)* | *(Đã hoàn thiện & Ký duyệt)* | 2026-09-30 |
| **Trưởng dự án Review (Lead C)** | Thành viên C *(Training & Evaluation Lead)* | *(Đã thẩm định & Phê duyệt)* | 2026-09-30 |
