# Lean Code: quy tắc luôn áp dụng

Mục tiêu: ít code nhất mà vẫn đúng. Code không viết ra thì không có bug,
không cần bảo trì và không cần review.

## Thang quyết định
Trước khi viết bất kỳ code mới nào, đi lần lượt và dừng ở bậc đầu tiên thỏa mãn:

1. Có thật sự cần không? Nếu yêu cầu không đòi, bỏ và nói rõ là đã bỏ.
2. Dự án đã có chưa? Tìm trong codebase rồi dùng lại hoặc mở rộng.
3. Thư viện chuẩn của ngôn ngữ làm được không?
4. Nền tảng/framework có tính năng native làm được không?
5. Dependency đã cài làm được không?
6. Một dòng có đủ không?
7. Chỉ khi cả sáu bậc trên không đủ: viết đoạn nhỏ nhất chạy đúng.

## Cấm
- Thêm dependency mới mà chưa nêu lý do và chưa hỏi người dùng.
- Tạo interface, factory, base class, config, plugin system cho thứ chỉ dùng một chỗ.
- Viết tính năng "phòng khi sau này cần".
- Refactor, đổi tên, format lại code không liên quan đến yêu cầu.
- Comment lặp lại điều code đã nói rõ.
- Bọc try/catch rồi nuốt lỗi.

## Luôn giữ (không bao giờ cắt)
- Kiểm tra dữ liệu đầu vào ở ranh giới hệ thống (API, form, file, CLI).
- Xử lý lỗi cho tình huống thực tế có thể xảy ra.
- Bảo mật: injection, xác thực, phân quyền, lộ secret.
- Accessibility của giao diện (label, focus, contrast, bàn phím).
- Test cho logic quan trọng và đường đi chính.

## Đường tắt phải được đánh dấu
Khi chọn giải pháp đơn giản mà có giới hạn, ghi một comment ngắn dạng
`lean: <giới hạn> -> <hướng nâng cấp>`.
Ví dụ: `// lean: lưu trong Map, mất khi restart -> chuyển sang DB khi cần bền vững`

## Báo cáo sau khi xong
Tối đa 3 dòng trong phản hồi chat: đã làm gì, đã dùng lại cái gì có sẵn, đã cố ý bỏ gì.
(Ghi chú: Vẫn duy trì tạo tệp báo cáo nghiệm thu task theo chuẩn dự án khi có yêu cầu).

## Khi người dùng yêu cầu rõ ràng
Nếu người dùng nói thẳng "tạo class X", "thêm thư viện Y", "tách thành service",
thì làm đúng như vậy. Quy tắc này nhằm ngăn việc tự thêm thứ thừa, không phải
để cãi lại yêu cầu rõ ràng. Có thể nêu một câu về phương án gọn hơn, rồi vẫn làm.
