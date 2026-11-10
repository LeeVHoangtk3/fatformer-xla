# Hồ sơ báo cáo riêng sau Task 7.15

Cập nhật ngày **11/11/2026**, chốt bằng chứng đến **10/11/2026**. Thư mục này tổng hợp công việc đã làm sau mốc đóng gói Task 7.15: tài liệu Task 7.16, pilot cải thiện vòng 1, đánh giá đối chứng, chẩn đoán validation và calibration transfer vòng 2. Kết quả Task 7.15 chỉ được nhắc lại làm mốc đối chứng.

**Kết luận hiện hành: giữ package `24869052…` làm bản chuẩn cho báo cáo. Chưa có ứng viên thay thế được chấp nhận. G7.2 vẫn FAIL, đạt 6/8 tiêu chí; G7.3 vẫn PENDING.** “Bản chuẩn” là lựa chọn đang được giữ trong các phương án đã đánh giá, không phải tuyên bố tốt nhất ở mọi miền hoặc đã đạt toàn bộ Phase 7.

## 1. Đọc và sử dụng

| File | Nội dung | Khi cần dùng |
| :--- | :--- | :--- |
| [01_bao_cao_tong_hop.md](01_bao_cao_tong_hop.md) | Báo cáo liền mạch: mục tiêu, các giai đoạn, kết quả, kết luận | Đưa vào một chương hoặc phụ lục báo cáo |
| [02_bang_so_lieu.md](02_bang_so_lieu.md) | Bảng model, gate, pilot, CFE, validation và calibration | Lập bảng kết quả hoặc slide; mỗi bảng có nguồn |
| [03_quy_trinh_va_kiem_chung.md](03_quy_trinh_va_kiem_chung.md) | Lịch sử run, thiết kế đối chứng, phạm vi hậu kiểm, notebook, test | Viết phương pháp và giải thích độ tin cậy |
| [04_ket_luan_va_huong_tiep_theo.md](04_ket_luan_va_huong_tiep_theo.md) | Đánh giá công tâm, quyết định giữ bản chuẩn, việc còn thiếu | Viết thảo luận, hạn chế và hướng phát triển |
| [05_danh_muc_nguon.md](05_danh_muc_nguon.md) | Đường dẫn log/notebook/Drive, danh tính model và nguồn trích dẫn | Truy vết hoặc bàn giao |

Nếu cần một file để đọc từ đầu đến cuối, dùng **file 01**. Nếu cần số liệu để sao chép, dùng **file 02**; không ghép các hàng khác tập dữ liệu hoặc khác ngưỡng thành một bảng so sánh chung.

## 2. Phạm vi và quy ước

- Hồ sơ tổng hợp lại bằng chứng đã hậu kiểm; lần biên soạn này **không tạo run GPU, checkpoint, calibration hay benchmark mới**.
- Các giá trị phần trăm được ghi rõ trong bảng. **pp** là điểm phần trăm; `0,05` dạng tỷ lệ tương ứng `5%`.
- `Real recall` là tỷ lệ ảnh thật được nhận đúng; `FPR = 100% − Real recall`. `Fake recall` là tỷ lệ ảnh giả được nhận đúng tại ngưỡng đang xét.
- `Fake@Real90` là chỉ số tại mốc Real recall 90% của đường cong đánh giá, khác Fake recall tại ngưỡng triển khai cố định. Không dùng hai số này thay cho nhau.
- `val_select` đã được dùng chọn và phân tích model. CFE cũng đã được xem. Không gọi các kết quả đó là một lần kiểm thử mù mới.
- Notebook và raw logs giữ tại nơi đã lưu; thư mục này chỉ chứa Markdown. Không sao chép weights hoặc tạo thêm một folder log cạnh tranh với `task_7.15/current`.

## 3. Mốc chính để viết báo cáo

1. Task 7.16 đã có hồ sơ tài liệu; cross-review/chữ ký người thật còn chờ.
2. Pilot chọn P0 thay vì P1 bằng validation; chưa chọn P0 thay incumbent.
3. Matched evaluation cho P0 đạt 4/8 tiêu chí, incumbent refit 6/8; vòng 1 là kết quả âm đối với mục tiêu thay model.
4. Diagnostics làm rõ khác biệt giữa ranking và ngưỡng, đồng thời sửa các diễn giải quá mức về generator và JPEG.
5. Calibration transfer làm tăng Fake recall nhưng Real recall Q30 xuống dưới guard 90%; không chấp nhận điểm clean-only mới.
6. Giữ nguyên model/calibration Task 7.15; hoàn thiện bằng chứng G7.3, chỉ cải thiện model trong thí nghiệm có đối chứng riêng.

Nguồn chính: [Task 7.15 hiện hành](../../../DATASET/logs/phase7/task_7.15/current/README.md), [Task 7.16](../../../DATASET/logs/phase7/task_7.16/README.md), [sổ nguồn chi tiết](05_danh_muc_nguon.md).

## 4. Kiểm tra khi biên soạn

Ngày 11/11/2026: đã kiểm 109 liên kết local trong 6 file, đối chiếu 60 ô số với CSV gốc ở bảng gate/calibration và xác nhận hồ sơ XML của lần audit calibration có 30 test đạt. Không chạy lại test hoặc GPU trong lần biên soạn này. Đối chiếu hash cây thư mục trước/sau xác nhận toàn bộ `notebooks`, log `task_7.15/current`, `task_7.16`, `improvement_round1` và `improvement_round2` giữ nguyên. Phần số liệu còn lại được đối chiếu với báo cáo hậu kiểm được dẫn tại từng mục.
