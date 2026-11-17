# FatFormer-XLA 336: kiểm chứng kỹ thuật

Nhánh triển khai: `test`. Hướng dẫn: [train_336.md](../../docs/demo/train_336.md).

- [local_real_smoke.json](local_real_smoke.json): forward bằng checkpoint thật trên hai ảnh Real/Fake ở CPU FP32. Logits 224 của mã mới khớp chính xác phương thức cũ; sai lệch tối đa bằng 0. Forward 336 có đầu ra hữu hạn.
- [tests.xml](tests.xml): kết quả pytest toàn repository của lượt kiểm chứng local.
- `readiness.json`: trạng thái triển khai và phần thực thi còn chờ.

Các số trong smoke là phép kiểm tra tương thích trên weights đã huấn luyện ở 224. **Không dùng chúng làm kết quả chất lượng của một mô hình đã train ở 336.** Test tự động dùng fixture nhỏ để kiểm tra logic; các fixture không phải bằng chứng benchmark khoa học.

Train 336 và master audit 336 chỉ được xác nhận sau khi notebook thực sự chạy xong trên Colab GPU. Báo cáo và package 224 hiện có vẫn là bằng chứng lịch sử riêng.
