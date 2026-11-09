# Tiếp tục Task 7.15 sau lỗi cell 4

**Cập nhật sau khi hậu kiểm run đã chạy:** lỗi truyền split đã hết, nhưng phát hiện bộ fit T dừng sớm. Code đã được sửa và xuất trong [bản sửa và bundle Colab mới](../../../../DATASET/logs/phase7/task_7.15/current/README.md). Dùng hướng dẫn/bundle đó để kiểm chứng toàn bộ package và Task 7.14. Chạy lại hai cell bên dưới với code GitHub cũ sẽ không tự nhận sửa hội tụ chưa push; hai cell này chỉ khắc phục lỗi đối số phân vùng.

Không chỉnh sửa notebook đã chạy. Hai tệp `.py` cạnh tài liệu này là nội dung để copy vào hai code cell mới trong phiên Colab hiện tại, sau khi cell 1–3 đã chạy thành công.

1. Copy toàn bộ `cell_4_extract_fixed.py` vào cell mới và chạy thay cell 4 cũ.
2. Copy toàn bộ `cell_5_calibrate_fixed.py` vào cell mới kế tiếp và chạy thay cell 5 cũ.
3. Khi cả hai báo `SUCCESS`, tiếp tục cell 6 của notebook để đóng gói; các bước đo và kiểm thử vẫn phải chạy riêng. Không coi việc sửa lỗi extraction là nghiệm thu xong Task 7.15.

## Nguyên nhân

`--evaluation_split` của `tools.extract_calibration_logits` nhận **đường dẫn NPZ/JSON của tập evaluation độc lập**, không nhận tên `val_calib`. Chuỗi `val_calib` không có phần mở rộng hợp lệ nên gây `Partition must be NPZ or JSON`.

- `--split_file`: file frozen calibration 400 ảnh, 200 real + 200 fake.
- `--evaluation_split`: file frozen evaluation 1.600 ảnh, 800 real + 800 fake, không trùng ID calibration. File này chỉ kiểm tra tính độc lập; không lấy logits của nó để fit.
- File baseline split chỉ cung cấp ID/nhãn phân vùng đã chốt. Logits đầu ra vẫn phải được đo mới bằng checkpoint v2; không sao chép logits baseline.

Cell 4 mới tìm file trong biến từ cell 3 và các vị trí `DATASET/logs/phase7/{wise_ft_sweep/,calibration/,}`. Không tạo phân vùng ngẫu nhiên khi thiếu file. Nếu thiếu, cần đưa đúng hai file frozen split vào một trong những vị trí trên rồi chạy lại cell mới.

Cell 5 cũ cũng thiếu `--calibration_split` và `--evaluation_split`; hai đối số này bắt buộc với logits gắn SHA256 checkpoint. Cell 5 thay thế đã truyền đủ chúng và giữ các biến đầu ra mà cell 6 và cell 8 cần dùng.

Có thể dùng bất kỳ GPU CUDA đủ VRAM. Extraction dùng FP32, batch 32; nếu OOM, đổi `EXTRACT_BATCH_SIZE = 8` hoặc `4` trong cell mới rồi chạy lại cell đó. Không cần giải nén dữ liệu hoặc chạy lại cell 1–3 nếu phiên Colab vẫn còn đủ biến và dữ liệu.

Các cell chạy tiến trình con trực tiếp nên log hiện trong output Colab. Chỉ xuất bản NPZ đích sau khi extraction thành công và kiểm tra shape, giá trị hữu hạn, provenance và phân vùng. File `.pending.npz` là đầu ra tạm, không dùng để đóng gói.
