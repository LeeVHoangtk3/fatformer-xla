# Triển khai FatFormer-XLA với ảnh đầu vào 336×336

**Đã bổ sung hỗ trợ kỹ thuật cho 336×336 trên nhánh `test`; kết quả huấn luyện và master audit 336 còn chờ thực thi trên Colab GPU.**

## Căn cứ và phạm vi

Kế thừa cấu hình train Anchored v2 và danh mục master audit đã nghiệm thu. Giữ ViT-L/14, adapter tại các block 7/15/23, SRM, gating, loss sáu thành phần, seed 42 và năm epoch. Lượt mới không thay dữ liệu hoặc checkpoint lịch sử.

Mã hóa vị trí được nội suy từ lưới 16×16 sang 24×24 khi forward. Nhánh SRM nhận lưới patch thực tế. Cách này giữ nguyên cấu trúc tensor trong checkpoint, hỗ trợ nạp nghiêm ngặt và duy trì hành vi gốc ở 224.

Resolution được lưu trong checkpoint train, calibration và package; inference và master audit đọc giá trị này để chọn transform. Calibration khác resolution bị từ chối. Từng tệp kết quả audit ghi hash package thực và kích thước đầu vào.

## Quy trình thực nghiệm đã chuẩn bị

Notebook [train_336_master_audit_colab.ipynb](../../../notebooks/train_336_master_audit_colab.ipynb) lấy source từ Git nhánh `test`, chốt commit, kiểm dữ liệu và đo 100 mini-batch trước train. Batch thực 4, tích lũy 8 lần, batch hiệu dụng danh định 32.

Baseline được hiệu chuẩn ở 336 trước train để đóng băng điểm vận hành chọn epoch. Sau train, checkpoint được chọn bằng validation và được hiệu chuẩn lại trên 400 ảnh CFE calibration; 1.600 ảnh evaluation không tham gia fit. Package được kiểm chứng tensor và dự đoán trên 20 ảnh thật trước master audit.

Master audit giữ **8 archive, 37 job** và đầy đủ số ảnh. Notebook xuất bảng đối chiếu 77 dòng giữa bản 224 và 336, kèm khác biệt về batch và hiệu chuẩn. Nếu checkpoint chỉ là best-effort, báo cáo ghi rõ điều kiện chọn mô hình không đạt.

## Kiểm chứng local

Nguồn bằng chứng: [logs/resolution336](../../../logs/resolution336). **Toàn bộ 429 kiểm thử đạt**, bao quát forward/backward 336, bảo toàn kết quả 224, SRM/DWT/gating, metadata resolution, calibration, đóng gói, phạm vi audit và phép tính chênh lệch trong bảng so sánh. Kiểm tra trên checkpoint thật dùng hai ảnh Real/Fake: logits 224 trước/sau khớp chính xác, forward 336 hữu hạn. Đây là smoke kỹ thuật, không phải kiểm thử chất lượng của model train mới.

Các tệp đóng băng có hash byte được giữ LF bằng `.gitattributes`, đúng byte trong Git; không sửa số liệu hoặc giá trị hash kỳ vọng để làm test đạt. Reviewer độc lập không phát hiện lỗi Critical/Important trong thay đổi.

## Giới hạn và trạng thái kết luận

Máy local chỉ có PyTorch CPU, CUDA không khả dụng. Phiên công cụ không có trình duyệt kết nối và không mở được trình duyệt tích hợp, nên chưa thể tự điều khiển Colab. Vì vậy chưa có checkpoint train 336 hoặc số đo master audit 336 để kết luận mô hình mới tốt hơn.

Train/validation giữ chuẩn hóa ImageNet như cấu hình train cũ, trong khi luồng triển khai dùng CLIP. Sai khác này chưa được giải quyết riêng. Thay batch thực cũng ảnh hưởng thống kê BatchNorm của SRM, nên không xem đây là ablation chỉ thay duy nhất độ phân giải. CFE vẫn là hậu kiểm đã được xem; kiểm trùng master audit mới ở mức SHA-256.

**Kết luận: mã và quy trình 336 đã được chuẩn bị để kiểm chứng; việc chọn thành phẩm cuối phải dựa trên lượt train và master audit thực tế, hiện chưa hoàn tất.**
