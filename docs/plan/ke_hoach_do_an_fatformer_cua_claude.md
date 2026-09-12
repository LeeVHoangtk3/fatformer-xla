# KẾ HOẠCH THỰC HIỆN ĐỒ ÁN: FINETUNE FATFORMER TĂNG ĐỘ BỀN NÉN ẢNH
### Bài tập lớn Xử lý ảnh — Nhóm 3 người | Nền tảng: Google Colab Pro

**Mục tiêu tổng thể**: Finetune mô hình FatFormer (CVPR 2024, checkpoint 4-class có sẵn) để tăng độ bền phát hiện ảnh AI-sinh khi ảnh bị nén JPEG / resize kiểu mạng xã hội, đồng thời (tùy chọn) bổ sung nhánh SRM high-pass. Đánh giá bằng bộ số liệu Clean vs Degraded trên 8 GANs + 6-10 Diffusion test set.

**Phân công vai trò cố định** (xuyên suốt cả 2 giai đoạn):
| Thành viên | Vai trò | Trách nhiệm chính |
| :--- | :--- | :--- |
| **A — Data & Infra Lead** | Dữ liệu, môi trường, augmentation | Tải/tổ chức dataset, viết pipeline augmentation, quản lý Colab/Drive |
| **B — Model & Training Lead** | Kiến trúc, huấn luyện | Sửa code model (SRM, tham số train), chạy finetune, quản lý checkpoint |
| **C — Evaluation & Report Lead** | Đánh giá, trực quan hóa, báo cáo | Viết script benchmark, Grad-CAM, tổng hợp số liệu, soạn báo cáo/slide |

> Lưu ý: 3 người vẫn nên đọc và hiểu toàn bộ code (không làm việc kiểu "hộp đen") vì khi bảo vệ đồ án, giảng viên có thể hỏi bất kỳ ai về bất kỳ phần nào.

---

## LỘ TRÌNH 1 — GIAI ĐOẠN ĐẦU: THIẾT LẬP & TÁI LẬP BASELINE
**Thời lượng đề xuất: Tuần 1–2 (khoảng 10-14 ngày)**
**Mục tiêu ra**: Toàn nhóm chạy được code gốc, có số liệu baseline (checkpoint gốc, chưa finetune) trên cả tập Clean và tập Degraded — đây là "đường mốc" để so sánh sau này.

### Tuần 1: Setup môi trường + Hiểu kiến trúc

| # | Công việc | Người phụ trách | Đầu ra (Deliverable) |
| :-- | :--- | :--- | :--- |
| 1.1 | Clone repo FatFormer chính thức, tạo Colab notebook dùng chung, mount Google Drive nhóm | A | Notebook `00_setup.ipynb` chạy được, cấu trúc thư mục chuẩn |
| 1.2 | Cài `pytorch_wavelets`, các dependency (`ftfy`, `regex`, `scipy`...), kiểm tra tương thích CUDA/PyTorch trên Colab | A | File `requirements_colab.txt` đã test OK |
| 1.3 | Tải `ViT-L-14.pt` (OpenAI) và `fatformer_4class_ckpt.pth`, lưu vào Drive chung, chạy `check_weights.py` xác nhận load state_dict thành công (giữ đúng tên `patch_basaed_enhancer`) | A | Checkpoint đã verify, log kiểm định |
| 1.4 | Đọc và tóm tắt kiến trúc FAA (`faa.py`) — vẽ sơ đồ luồng tensor DWT/IDWT, Inter/Intra-band Attention | B | Slide/note giải thích FAA (1-2 trang) |
| 1.5 | Đọc và tóm tắt kiến trúc LGA (`lga.py`) — patch-based enhancer, text-guided interactor, soft prompt | B | Slide/note giải thích LGA (1-2 trang) |
| 1.6 | Đọc `evaluate.py`, `benchmark_metrics.py` — hiểu cách tính ACC/AP/AUC/EER, cấu trúc thư mục test set yêu cầu | C | Note quy trình evaluate + checklist thư mục dữ liệu |
| 1.7 | Tải các tập test: 8 GANs (ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, Deepfake) + 6-10 Diffusion (PNDM, Guided Diffusion, DALL-E, VQ-Diffusion, LDM, Glide...) | A + C | Dataset đã tải, đối chiếu đúng cấu trúc `0_real/1_fake` |

### Tuần 2: Chạy baseline + Xây bộ test Degraded

| # | Công việc | Người phụ trách | Đầu ra |
| :-- | :--- | :--- | :--- |
| 2.1 | Chạy `evaluate.py` với checkpoint gốc trên toàn bộ tập Clean (8 GANs + Diffusion) | B | Bảng số liệu baseline Clean (ACC/AP mỗi tập + trung bình) |
| 2.2 | Viết script sinh phiên bản Degraded của tập test: JPEG Q=30/50/70, resize-down-up mô phỏng MXH | A | Script `make_degraded_testset.py` + bộ ảnh degraded đã sinh |
| 2.3 | Chạy `evaluate.py` với checkpoint gốc trên tập Degraded (Q=30/50/70) | B | Bảng số liệu baseline Degraded — cho thấy mức sụt giảm hiện tại |
| 2.4 | Chạy `inference.py` + `visualize_cam.py` trên vài ảnh mẫu (clean & degraded) để có Grad-CAM baseline | C | 6-10 ảnh Grad-CAM baseline (dùng để so sánh sau finetune) |
| 2.5 | Tổng hợp toàn bộ số liệu baseline vào 1 file chung, vẽ biểu đồ sơ bộ ACC theo mức Q | C | File `baseline_results.xlsx` + biểu đồ |
| 2.6 | **Họp nhóm giữa kỳ**: rà soát baseline, thống nhất % sụt giảm khi nén (dùng làm "vấn đề cần giải quyết" trong báo cáo), chốt phạm vi finetune (có làm SRM hay không tùy thời gian còn lại) | Cả 3 | Biên bản họp + quyết định phạm vi |

**Tiêu chí hoàn thành Giai đoạn 1**: Có bảng số liệu baseline đầy đủ (Clean vs Degraded ở 3 mức Q) chạy trên checkpoint gốc, chưa chỉnh sửa gì — đây là số liệu bắt buộc phải có trước khi bắt đầu finetune, nếu không sẽ không có gì để so sánh.

---

## LỘ TRÌNH 2 — GIAI ĐOẠN FINETUNE: HUẤN LUYỆN, CẢI TIẾN & ĐÁNH GIÁ
**Thời lượng đề xuất: Tuần 3–5 (khoảng 15-18 ngày)**
**Mục tiêu ra**: Checkpoint đã finetune tăng độ bền nén ảnh, có đầy đủ số liệu so sánh trước/sau, báo cáo và slide hoàn chỉnh.

### Tuần 3: Augmentation mạnh hơn + Chuẩn bị training

| # | Công việc | Người phụ trách | Đầu ra |
| :-- | :--- | :--- | :--- |
| 3.1 | Sửa `datasets/transforms.py`: tăng xác suất JPEG lên p=0.7-0.8, mở rộng dải Q∈[30,95], thêm resize-down-up ngẫu nhiên, thêm double-JPEG compression | A | `transforms.py` đã cập nhật, unit test augmentation (xem ảnh trước/sau) |
| 3.2 | Chuẩn bị `train.py`: cấu hình đóng băng CLIP backbone, chỉ mở băng FAA+LGA (~55M tham số), load checkpoint gốc làm điểm khởi đầu (không train from scratch) | B | `train.py` chạy thử 1 epoch không lỗi |
| 3.3 | (Tùy chọn — nếu còn thời gian) Thêm nhánh SRM: viết 3 kernel SRM cố định trong `faa.py`, chèn trước Conv nhánh không gian | B | Module SRM đã tích hợp, forward pass không lỗi shape |
| 3.4 | Thiết lập hyperparameter: AdamW, learning rate 1e-4 (thấp hơn gốc vì finetune tiếp), Cosine Annealing, batch 32, AMP FP16, 5-10 epoch | B | File config `finetune_config.yaml` |
| 3.5 | Viết checklist đánh giá cuối kỳ (metric nào cần, bảng nào cần cho báo cáo) để B biết cần log gì trong lúc train | C | Checklist đánh giá |

### Tuần 4: Chạy Finetune + Theo dõi

| # | Công việc | Người phụ trách | Đầu ra |
| :-- | :--- | :--- | :--- |
| 4.1 | Chạy finetune phiên bản A: chỉ tăng cường augmentation (không SRM) — 5-10 epoch trên ProGAN 4-class | B | Checkpoint `ft_augment_only.pth` + log loss/ACC theo epoch |
| 4.2 | Chạy finetune phiên bản B (nếu làm SRM): augmentation + SRM branch — 5-10 epoch | B | Checkpoint `ft_augment_srm.pth` + log |
| 4.3 | Theo dõi log training, vẽ đường cong loss/accuracy theo epoch, phát hiện overfitting sớm nếu có | A | Biểu đồ training curve |
| 4.4 | Chuẩn bị sẵn script benchmark 2 chiều (Clean/Degraded) để chạy ngay khi có checkpoint | C | Script `benchmark_compare.py` đã test với checkpoint baseline |
| 4.5 | Dự phòng: nếu Colab bị ngắt phiên giữa chừng, đảm bảo checkpoint được lưu định kỳ vào Drive (mỗi 1-2 epoch) | A + B | Cơ chế auto-save hoạt động ổn định |

### Tuần 5: Đánh giá, Trực quan hóa & Viết báo cáo

| # | Công việc | Người phụ trách | Đầu ra |
| :-- | :--- | :--- | :--- |
| 5.1 | Chạy `evaluate.py` với checkpoint đã finetune trên toàn bộ Clean + Degraded (Q=30/50/70) — 8 GANs + Diffusion | C | Bảng số liệu đầy đủ sau finetune |
| 5.2 | So sánh Baseline vs Finetune: lập bảng tổng hợp (ACC/AP mỗi tập, mỗi mức Q), tính % cải thiện | C | Bảng so sánh trung tâm của báo cáo |
| 5.3 | Chạy lại Grad-CAM trên đúng bộ ảnh mẫu đã dùng ở Tuần 2 (clean & degraded) với checkpoint mới, so sánh cạnh nhau với baseline | C | Ảnh so sánh Grad-CAM trước/sau (minh họa trực quan) |
| 5.4 | (Nếu có SRM) Chạy ablation: so sánh 3 cấu hình — Baseline / Augment-only / Augment+SRM | B + C | Bảng ablation |
| 5.5 | Viết báo cáo: Đặt vấn đề (Chương 1-2) → Phương pháp (kiến trúc + cải tiến) → Thực nghiệm (bảng số liệu) → Kết luận & hạn chế | Cả 3 (C tổng hợp) | Báo cáo hoàn chỉnh |
| 5.6 | Làm slide thuyết trình, chuẩn bị demo `inference.py` trực tiếp trên ảnh mới (kể cả ảnh nén nặng) để trình diễn trực quan lúc bảo vệ | Cả 3 | Slide + demo chạy được live |
| 5.7 | Rà soát chéo toàn bộ code + báo cáo, mỗi người phải giải thích được phần việc của người khác (chuẩn bị hỏi đáp) | Cả 3 | Buổi review chéo nội bộ |

**Tiêu chí hoàn thành Giai đoạn 2**: Có checkpoint đã finetune, bảng số liệu so sánh Baseline vs Finetune rõ ràng (đặc biệt trên tập Degraded — đây là điểm chứng minh giá trị đóng góp), báo cáo + slide sẵn sàng bảo vệ.

---

## MỐC THỜI GIAN TỔNG QUAN (GANTT RÚT GỌN)

| Tuần | Giai đoạn | Trọng tâm |
| :---: | :--- | :--- |
| 1 | Lộ trình 1 | Setup môi trường, hiểu kiến trúc, tải dữ liệu |
| 2 | Lộ trình 1 | Chạy baseline Clean + Degraded, họp chốt phạm vi |
| 3 | Lộ trình 2 | Viết augmentation mới, chuẩn bị training/SRM |
| 4 | Lộ trình 2 | Chạy finetune, theo dõi training |
| 5 | Lộ trình 2 | Đánh giá, Grad-CAM, viết báo cáo & slide |
| (6) | Dự phòng | Buffer nếu Colab hết quota / cần chạy lại |

---

## RỦI RO & PHƯƠNG ÁN DỰ PHÒNG

| Rủi ro | Khả năng xảy ra | Phương án xử lý |
| :--- | :--- | :--- |
| Colab Pro hết giờ chạy / bị ngắt session giữa lúc train | Cao | Lưu checkpoint mỗi 1-2 epoch vào Drive; thiết kế `train.py` có thể resume từ checkpoint gần nhất |
| Dataset quá lớn, tải chậm/hết dung lượng Drive | Trung bình | Chỉ tải tập con đại diện (vd 2-3 GANs + 2-3 Diffusion) nếu Drive giới hạn 15GB free, ghi rõ giới hạn này trong báo cáo |
| Load checkpoint bị lỗi `Unexpected key/Missing key` | Trung bình | Kiểm tra kỹ mục 2 Chương 9 (giữ nguyên tên `patch_basaed_enhancer`), test bằng `check_weights.py` trước khi finetune |
| Finetune làm giảm hiệu năng trên tập Clean (trade-off) | Trung bình-Cao | Đây là kết quả khoa học hợp lệ, không phải lỗi — báo cáo cần phân tích trade-off Clean vs Degraded, không chỉ báo cáo con số đẹp |
| SRM branch làm phức tạp thêm mà không đủ thời gian debug | Trung bình | Coi đây là phần "tùy chọn nâng cao" — nếu Tuần 3 không kịp, bỏ qua, tập trung 100% vào augmentation (vẫn đủ để có kết quả tốt) |

---

## CHECKLIST BÀN GIAO CUỐI CÙNG

- [ ] Bảng baseline (checkpoint gốc) trên Clean + Degraded (Q=30/50/70)
- [ ] Bảng kết quả sau finetune trên cùng bộ test
- [ ] Bảng so sánh % cải thiện, đặc biệt trên tập Degraded
- [ ] (Tùy chọn) Bảng ablation Baseline / Augment-only / Augment+SRM
- [ ] Ảnh Grad-CAM so sánh trước/sau trên cùng mẫu ảnh
- [ ] Code đã dọn dẹp, có README hướng dẫn chạy lại
- [ ] Báo cáo hoàn chỉnh + Slide thuyết trình
- [ ] Mỗi thành viên có thể giải thích toàn bộ pipeline khi được hỏi
