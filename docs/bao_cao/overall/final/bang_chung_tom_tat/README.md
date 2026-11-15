# Bằng chứng tóm tắt của báo cáo FatFormer XLA

Các tệp này được trích từ kết quả kiểm chuẩn đã lưu, không phải lần huấn luyện hoặc suy luận mới. Có thể đọc số liệu mà không cần mở liên kết nội bộ trong Word.

- `phan_bo_data_v2.csv`: số Real/Fake theo ba split, đếm trực tiếp manifest.
- `nguon_data_v2.csv`: toàn bộ tên bộ sinh, nguồn ảnh thật, nguồn nhập và số lượng.
- `ket_qua_theo_dieu_kien.csv`: bảng kết quả đã lưu của CFE và 18 tập.
- `duong_roc_cfe.csv`: tọa độ ROC tính lại từ điểm dự đoán từng ảnh, nhãn 1 là Fake.
- `mau_du_doan_sai.csv`: các dự đoán sai CFE clean và Q30 tại ngưỡng triển khai; không kèm ảnh gốc.
- `doi_chieu_nguon.json`: nguồn trích và SHA-256 để kiểm tra đúng tệp bằng chứng.
- Các PNG là biểu đồ đã chèn vào báo cáo; không phải ảnh lỗi của mô hình.

Danh mục dữ liệu chứng minh cách gán nhãn và phân chia đã ghi, chưa thay cho việc xem ảnh để xác nhận ngữ nghĩa. ROC chỉ mô tả mô hình được chọn, không tự tạo đối chứng cho FatFormer gốc. Các đường dẫn kỹ thuật trong JSON/CSV dùng để truy vết khi có đầy đủ repo. Gói này không bao gồm logits gốc, trọng số hay ảnh; chưa thay cho một lần tái lập độc lập. Không giả định dữ liệu nội bộ đã được công bố công khai.
