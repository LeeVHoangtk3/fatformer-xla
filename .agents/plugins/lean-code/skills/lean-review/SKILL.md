---
name: lean-review
description: Review diff, commit, pull request hoặc file để tìm code có thể xóa, đơn giản hóa hoặc thay bằng thứ có sẵn. Dùng khi người dùng nói "review", "dọn code", "rút gọn", "có thừa gì không", "simplify", "trước khi commit", "xem diff", hoặc muốn kiểm tra mức độ phức tạp của thay đổi. Chỉ tìm thứ để bớt, không đề xuất thêm tính năng.
---

# Lean Review

Đọc phạm vi được chỉ định (diff, file hoặc thư mục) và tìm những gì có thể
**xóa, gộp, hoặc thay bằng cái đã có**. Không đề xuất thêm tính năng mới.

## Quy trình

1. Xác định phạm vi: nếu người dùng không chỉ rõ, dùng thay đổi chưa commit (`git diff`)
   hoặc commit gần nhất.
2. Đọc toàn bộ phạm vi trước, rồi mới đánh giá.
3. Với mỗi phát hiện, kiểm tra trong codebase xem có thật sự thừa không
   (ví dụ tìm nơi gọi hàm trước khi kết luận hàm không dùng).
4. Sắp xếp kết quả theo mức tiết kiệm từ lớn đến nhỏ.

## Danh sách kiểm tra

**Dependency**
- Thư viện mới có thể thay bằng thư viện chuẩn hoặc dependency đã cài?
- Thư viện nặng chỉ dùng một hàm nhỏ?

**Cấu trúc**
- Hàm, class hoặc file chỉ được gọi từ một chỗ và không cần tách riêng?
- Interface, base class, factory chỉ có một cài đặt?
- Lớp bọc (wrapper) chỉ chuyển tiếp lời gọi?
- Cấu hình, tham số hoặc cờ (flag) không bao giờ được dùng?

**Trùng lặp**
- Có util tương tự trong dự án mà code mới không dùng?
- Hai đoạn code làm cùng một việc?

**Code chết & Bộ nhớ thừa**
- Biến, import, hàm không dùng?
- Nhánh `if/else` không thể xảy ra?
- Code bị comment out?
- TODO cũ, comment lặp lại điều code đã nói?
- Trong PyTorch/ML: tensor trung gian không được `detach()` hoặc giữ graph tính toán không cần thiết gây tốn VRAM?

**Phạm vi**
- Có thay đổi không liên quan đến mục tiêu (đổi tên, format, refactor lan man)?

**Xử lý lỗi**
- `catch` nuốt lỗi không làm gì?
- Kiểm tra điều mà kiểu dữ liệu hoặc framework đã đảm bảo?

**Test**
- Test kiểm tra chi tiết cài đặt thay vì hành vi?
- Test trùng nhau, hoặc test cho getter/setter?

## Không đề xuất xóa

Kiểm tra đầu vào ở ranh giới hệ thống, xử lý lỗi cho tình huống thực tế, bảo mật,
accessibility, test cho logic quan trọng, tính toàn vẹn dữ liệu.
Nếu thấy chúng bị thiếu, nêu riêng ở mục "Cần bổ sung".

## Định dạng kết quả

```
## Có thể bớt (lớn nhất trước)
1. `đường/dẫn/file.py:42-80` – <vấn đề>. Đề xuất: <xóa / thay bằng X>. Tiết kiệm ~N dòng.
2. ...

## Cần bổ sung (nếu có)
- <thứ bị thiếu thuộc nhóm "không bao giờ cắt">

## Tổng kết
Có thể bớt khoảng N dòng và M dependency.
```

Nếu không tìm thấy gì đáng bớt, nói thẳng "Không thấy gì cần bớt" thay vì bịa ra
đề xuất cho đủ danh sách.

## Quy tắc

- Chỉ đọc và báo cáo. Không sửa code trừ khi người dùng đồng ý rõ ràng.
- Nêu bằng chứng cho mỗi phát hiện (nơi gọi, nơi trùng lặp), không chỉ nêu cảm giác.
- Không chê phong cách cá nhân nếu không ảnh hưởng đến độ dài hay độ phức tạp.
