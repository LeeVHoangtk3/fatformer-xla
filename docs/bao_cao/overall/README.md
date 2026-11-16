# FatFormer-XLA: Master Architectural & Project Blueprint

## Hai báo cáo cuối cùng

[Báo cáo tổng thể](final/Bao_Cao_Tong_The_Mo_Hinh_FatFormer_XLA.docx) và [bản thảo paper hai cột](final/Paper_FatFormer_XLA_CVPR_Format.docx) có chung tám phần chính: giới thiệu và mục tiêu → nền tảng FatFormer và phạm vi kế thừa → quá trình phát triển → phương pháp → thiết lập thực nghiệm → kết quả và phân tích → thảo luận → kết luận và hướng phát triển. Phần bài gốc trình bày FAA/LGA, dữ liệu ProGAN, xử lý ảnh, huấn luyện và ACC/AP do tác giả công bố; các số này được phân biệt với mô hình nền tái hiện và số đo của FatFormer XLA.

Hai bản có phần **Tóm tắt** khoảng 200 từ đặt ở cuối tài liệu, nêu kết quả chính, phạm vi nghiên cứu và quyết định dùng mô hình cho demo. Bảng thuật ngữ giải thích các khái niệm cần thiết; tên **XLA được chọn** được dùng thống nhất cho mô hình cuối. Phần dữ liệu bổ sung phân bố Real/Fake, 17 bộ sinh, nguồn ảnh thật và quan hệ giữa train với tập chọn mô hình. Data v2 thực giữ 2.991 ảnh replay, khác kho bổ sung ban đầu 3.600 ảnh.

Nội dung ngoài tiêu đề thống nhất cỡ chữ 13, gồm bảng, chú thích, công thức, chân trang và tài liệu tham khảo. Hai bản đã bỏ toàn bộ phần đầu trang (header), giữ nguyên số trang ở chân trang. Hai báo cáo không hiển thị ngày biên soạn hoặc ngày của các đợt chạy. Giới thiệu tập trung vào bài toán và mục tiêu; tóm tắt có số liệu theo yêu cầu biên tập mới. Ý chính và câu chốt kết luận được in đậm.

Bản tổng thể có bốn biểu đồ định lượng của dự án; paper có ba biểu đồ, gồm ROC tính từ xác suất từng ảnh CFE. Phần FatFormer gốc bổ sung hai hình trích từ PDF CVPR chính thức: kiến trúc FAA/LGA ở mục 2.1 và phân bố logits Real/Fake so sánh với UniFD ở mục 2.3. Hai hình có phần hướng dẫn đọc, chú thích tác giả, số hình và trang nguồn; trong paper, hình trải ngang trang. [Nguồn hình bài báo gốc](final/hinh_bai_bao_goc/README.md) lưu thông tin đối chiếu. Tổng số hình hiện là bảy ở báo cáo tổng thể và sáu ở paper. Số liệu của mô hình cuối được giữ nguyên; tài liệu có 18 bảng tổng thể và 13 bảng paper. Các số AUC CFE tính lại khớp kết quả đã lưu. [Bằng chứng tóm tắt](final/bang_chung_tom_tat/README.md) kèm phân bố dữ liệu, nguồn ảnh, bảng điều kiện, tọa độ ROC, danh sách dự đoán sai và mã kiểm tra nguồn. Đây là dữ liệu trích để đối chiếu; chưa thay cho một lần tái lập từ ảnh và trọng số gốc.

Kết luận so sánh được giới hạn rõ: XLA cải thiện so với baseline nội bộ đã lưu, **chưa chứng minh vượt FatFormer gốc**. Nguyên nhân baseline CFE thấp chưa được xác định đầy đủ. Ablation hiện có chỉ đổi trọng số Feature và Guard; ablation tách SRM/gating/KD được mô tả là công việc cần chạy, không phải kết quả đã có. Chưa đưa ảnh lỗi hoặc Grad-CAM khi thiếu ảnh gốc của mẫu tương ứng. Hai bản đã được kiểm tra nội dung và trực quan qua Microsoft Word, gồm 26 trang tổng thể và 17 trang paper.

Văn phong đã được biên tập cho người đọc có kiến thức cơ bản về học máy: mở đầu đoạn bằng ý chính hoặc kết quả, chia câu dài, ưu tiên từ tiếng Việt và giải thích thuật ngữ khi xuất hiện lần đầu. Các điều kiện cần để đọc đúng kết quả vẫn nằm cạnh phép đo; các giới hạn về đối chứng, dữ liệu và ứng dụng được gom vào mục **Giới hạn**. Phần văn xuôi bỏ mã nguồn nội bộ dạng `[S…]`; chú thích bảng/hình và danh mục nguồn giữ thông tin truy vết. Lượt biên tập này bảo toàn toàn bộ bảng số, giá trị số trong phần diễn giải, công thức, hình và liên kết nguồn; không thay mô hình hoặc kết luận khoa học.

Quy trình hiện tập trung vào **bảy giai đoạn tạo thành phẩm cuối**: chuẩn bị dữ liệu và mô hình nền; thiết lập dữ liệu và giao thức đánh giá; xây dựng XLA; chuẩn bị Data v2; huấn luyện có neo; chọn trọng số và hiệu chuẩn; kiểm chuẩn và đóng gói cho demo. Hai báo cáo đã bỏ diễn biến của các nhánh nội suy, ứng viên P0/P1 và hiệu chuẩn bổ sung không được chọn. Bảng 18 tập chỉ giữ cột mô hình cuối; bảng đối chiếu CFE giữ baseline nội bộ và XLA. Đối chứng thành phần, kết quả đánh giá và giới hạn của mô hình cuối vẫn được trình bày đúng bằng chứng. Hai báo cáo dùng tên dễ đọc và liên kết nguồn để truy vết; lịch sử thí nghiệm gốc trong dự án được giữ nguyên.

Hai báo cáo đã bổ sung [master full-data audit](../../../DATASET/logs/phase7/master_audit/README.md): 8 archive, 37 job, 1.618.061 lượt dự đoán; cập nhật latency L4 mới và phạm vi kiểm chứng. Package không đổi; **G7.2 FAIL 6/8, G7.3 PENDING**. Các khối ngày 10/11 và tài liệu lịch sử bên dưới giữ ngữ cảnh của thời điểm lập, không thay số đo master audit ngày 15/11.

<!-- PHASE7_CURRENT_BEGIN -->
## Kết quả Phase 7 hiện hành — cập nhật 10/11/2026

Run `corrected_20261109_102954_3d28ad`, package `248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`. Số liệu hiện hành: [hậu kiểm chi tiết](../../../DATASET/logs/phase7/task_7.15/current/analysis/README.md), [view JSON v3 cho tài liệu](../../../DATASET/logs/phase7/task_7.16/final_phase7_report.json) và [bảng nguồn số cũ → mới](bao_cao_ra_soat_so_lieu_phase7.md). View v3 được tổng hợp từ raw corrected schema v1, **không phải một phép đo mới**.

| Cổng | Kết quả hiện hành | Bằng chứng / phần còn thiếu |
| :--- | :--- | :--- |
| G7.0 | **PENDING** theo DoD toàn repo | Canonical core: 3 checkpoint nạp 0/0, Gen 1 bị từ chối; còn loader legacy và CI chẩn đoán n3/n8 chưa đủ. |
| G7.1 | **FAIL** | Gen 2: Clean Fake@Real90 48,88% <55%; Q30 AUC 43,67% <55%. |
| G7.2 | **FAIL — 6/8** | Clean Fake@Real90 64,12% <65%; worst-subset Fake ACC 30,86% <45%. |
| G7.3 | **PENDING** | 20 ảnh và overhead L4 0,188055 ms/ảnh đạt; thiếu reference tốt nhất theo điều kiện. |

Hoàn tất đo/biên soạn không có nghĩa đạt tất cả mục tiêu. Không dùng số Gen 1 như hiệu năng hợp lệ; không khẳng định router thắng, không đổi FAIL/pending thành PASS. Chấp nhận GPU CUDA đủ VRAM; latency chỉ áp dụng L4 đã ghi. Xác nhận chéo/chữ ký A/B/C còn chờ, không tạo chữ ký tự động.


## Tài liệu dùng để viết báo cáo và bàn giao

- [bao_cao_ra_soat_so_lieu_phase7.md](bao_cao_ra_soat_so_lieu_phase7.md)
- [chuong_5_ket_luan_va_han_che.md](chuong_5_ket_luan_va_han_che.md)
- [slide_phase7_5phut.md](slide_phase7_5phut.md)
- [milestone_phase7_acceptance.md](../../milestones/milestone_phase7_acceptance.md)

- [Catalog nguồn và JSON pointer](../../../DATASET/logs/phase7/task_7.16/evidence_catalog.json)
- [Hồ sơ package thực đã hậu kiểm](../../../DATASET/logs/phase7/task_7.15/current/README.md)
- [Đặc tả và checklist 7.16](../../tasks/member_c_training_eval/phase_7/task_7.16_cap_nhat_tai_lieu_du_an.md)

Kế hoạch quy định mục tiêu; JSON measured quy định kết quả. Không dùng kế hoạch hoặc ảnh lịch sử làm số đo hiện hành. Các tài liệu trước đây bên dưới chỉ để truy vết.

### Bổ sung ngày 10/11/2026 — vòng cải thiện 1 và chẩn đoán vòng 2

Pilot `pilot_20261109_182359` đã train hai nhánh mới ×5 epoch/seed42, chọn P0 bằng `val_select`: worst-group P0 42%, P1 35,33%. Matched evaluation `p0_eval_20261110_065901` sau đó cho P0 **G7.2 FAIL, đạt 4/8 tiêu chí**, incumbent refit **FAIL, đạt 6/8**. Vòng 1 chưa tìm được model thay thế; package `24869052…` và kết quả corrected hiện hành được giữ nguyên. Worst-group validation khác worst-subset CFE; pilot này không thay ablation bốn nhánh 7.12.

[Nhật ký và báo cáo](../C/bao_cao_phase7_improvement_round1_va_chan_doan_vong2.md) phân biệt weights nguồn, package deployment và refit. [Chẩn đoán](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/README.md) đã tái lập 4.320 lượt dự đoán. [Calibration transfer mới](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/README.md), run `calib_transfer_20261110_155948`/source `253942a`, fit đúng `val_calib`, tăng mạnh Fake recall nhưng Real Q30 chỉ **86,95% P0/87,29% incumbent**, không đạt guard90 đã khai báo. **Không chọn điểm clean-only mới; chưa train hoặc thay deployment.** Raw ranking không đổi, không lấy số validation thay G7.2 CFE. [Kế hoạch tiếp](../../plan/ke_hoach_cai_thien_phase7_sau_vong1.md) tách mục tiêu operating point robust khỏi cải thiện ranking bằng weights/data. Deployment và G7.3/chữ ký giữ trạng thái trên.

<!-- PHASE7_CURRENT_END -->

<details>
<summary>Kế hoạch nền và hồ sơ các giai đoạn trước — chỉ để tham khảo lịch sử, không dùng như kết quả Phase 7 hiện hành</summary>

Các bảng, hình, kết luận và lệnh dưới đây được giữ để truy vết. Số Gen 1 là `invalid_layout`; chỉ mục nguồn và nội dung hiệu lực nằm ở khối cập nhật phía trên. Không dùng kết luận “PASS toàn bộ” trong hồ sơ cũ thay cho phán quyết mới.


Tài liệu thiết kế kiến trúc và cấu trúc tổng thể toàn diện của đồ án **FatFormer-XLA** được đặt tại:

👉 **[Bản Thiết Kế Kiến Trúc & Tổng Quan Hệ Thống (tong_quan_du_an.md)](tong_quan_du_an.md)**  
👉 **[Báo Cáo Nghiệm Thu Kiểm Định Ngoại Vi CommunityForensics-Eval (CVPR 2025)](bao_cao_kiem_dinh_communityforensics_eval.md)**  

---

## 🚨 Đính Chính Quan Trọng & Kế Hoạch Hiện Hành (27/10/2026)

> **Mọi tài liệu trong thư mục này phải được đọc kèm hai văn kiện dưới đây.**

| Văn kiện | Vai trò | Ghi chú |
| :--- | :--- | :--- |
| **[Kế hoạch Phase 7 — Tái lập đo lường & Hợp nhất mô hình](../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md)** | **Source of truth** cho toàn bộ công việc tiếp theo | `v3.0`; 16 task, 4 cổng G7.0–G7.3; bản `v2.0` lưu tại [`archive/`](../../plan/archive/) |
| **[Phần IX — Đính chính kỹ thuật của báo cáo CFE](bao_cao_kiem_dinh_communityforensics_eval.md#ix-đính-chính-kỹ-thuật-ngày-27102026)** | Vô hiệu hóa số liệu WiSE-FT, giữ hiệu lực Baseline/Anchored | Bắt buộc đọc trước khi trích dẫn bất kỳ con số nào |
| **[Điểm dừng 6 — Phase 7](../../plan/ke_hoach_cac_cot_moc_va_diem_dung.md)** | Cổng nghiệm thu thay thế Gate 4 | G7.0–G7.3 + điều khoản kết quả âm |
| **[Khế ước đo lường v3 (Phụ lục A)](../../plan/ke_hoach_test_communityforensics_eval.md)** | Quy tắc báo cáo metric | Cấm báo cáo "overall accuracy" đơn lẻ |

**Ba điều phải nhớ:**

1. **WiSE-FT Gen 1 bị cách ly** (`status: invalid_layout`): checkpoint chứa 10 khối adapter `{2,5,7,8,11,14,15,17,20,23}`, trong khi kiến trúc chỉ sinh được vị trí `(i+1)·24/n − 1` với `n ∈ {3,4,6,8,12,24}` ⇒ không nạp đầy đủ được. Mọi con số của nó (`52,05%`, `4,10%`, `0,20%`, `96,58%`) **không còn hiệu lực**.
2. **Baseline và Anchored Recovery vẫn hợp lệ** — đây là hai mốc được phép trích dẫn.
3. **Thiết kế đích là một checkpoint duy nhất** (`UnifiedFatFormerXLA`, G7.3), không phải hệ thống hai chế độ. Kiến trúc kép chỉ là phương án dự phòng nếu G7.3 thất bại.

---

## Tóm Tắt Nhanh Nội Dung Tài Liệu:
1. **Bối Cảnh & Đột Phá Kỹ Thuật**: Nâng cao độ bền pháp y trước suy thoái mạng xã hội (JPEG, Gaussian Blur, Resize down-up) bằng cách tích hợp SRM 3-Kernels + Dynamic Frequency Gating $\lambda(x)$ + Dual-Stream Augmentation (Phương Án 2).
2. **Bản Đồ Cấu Trúc Toàn Bộ Dự Án**: Phân loại chi tiết toàn bộ các thư mục (`src/`, `FatFormer-main/`, `tools/`, `notebooks/`, `docs/`, `configs/`).
3. **Sơ Đồ Luồng Dữ Liệu (Mermaid)**: Trực quan hóa đường đi của ảnh từ đầu vào qua FAA Wavelet, SRM, Language-Guided Alignment đến hàm mất mát Eq. (10).
4. **Kiến Trúc Hạ Tầng (Colab A100 & Drive 5TB)**: Cơ chế kết nối Shortcut 0 byte và quy tắc giải nén SSD NVMe chống sập I/O.
5. **Ma Trận Phân Vai 3 Thành Viên**: Phân định ranh giới trách nhiệm giữa Member A (Data/Drive), Member B (Model/Loss), Member C (Train/Eval).
6. **Sổ Tay Kỹ Thuật & Cheat-Sheet Lệnh**: Cú pháp chạy Smoke Test, Fast-Eval và Huấn luyện chuẩn hóa.

</details>
