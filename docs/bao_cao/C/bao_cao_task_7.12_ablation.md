# Task 7.12 — Sửa triển khai ablation, chưa có kết quả GPU

Cập nhật: **06/11/2026**.

**Trạng thái: đã sửa mã và kiểm thử CPU; chưa chạy thí nghiệm GPU, chưa nghiệm thu kết quả mô hình hoặc G7.2.** Không có bảng điểm A1–A4 đo thật trong báo cáo này.

Các sửa đổi:

- Thêm explicit `loss.mode: v2` cho cả bốn cấu hình. A1 giữ CE `0.3 clean + 0.3 degraded` khi tắt Guard/Feature; công thức legacy vẫn được giữ cho lời gọi cũ không chỉ định mode.
- Runner mới chạy cả bốn cấu hình trong experiment riêng. Resume dùng full latest có checksum/context, kiểm horizon thật và metadata best; không bỏ qua run chỉ vì có best + CSV và không ghi cứng số epoch.
- Preflight đủ gate, benchmark trên GPU hiện tại và source commit được lưu. Non-finite gradient làm dừng v2 run thay vì dùng kết quả thiếu kiểm chứng để kết luận ổn định.
- Eval hỗ trợ danh sách evaluation IDs gốc, kiểm calibration NPZ đúng SHA256 và tách IDs. Chỉ số CFE chính thức trong ablation không dùng lại 400 mẫu calibration của bộ 2.000 ảnh.
- Aggregator đọc đúng schema `metrics` và đơn vị phần trăm, kiểm provenance, canonical load, checksum checkpoint và dự đoán NPZ. Thiếu bằng chứng không đổi thành số 0.
- Bootstrap thêm Fake@Real90 theo đúng định nghĩa quantile; so sánh CFE dùng resample phân tầng, ghép cặp cùng sample IDs giữa các model. Worst-group delta trên tập selection vẫn được ghi là mô tả, chưa có CI độc lập.
- Tách tương phản tương tác `A4-A2-A3+A1` khỏi `A4-max(A2,A3)`. Không ép kết luận một loss thắng và không gọi A1 là “Data only”.
- Loại chế độ dummy ghi đè artifact thật. Summary dùng schema ablation riêng và không tự đánh dấu đủ điều kiện G7.2.

Kiểm chứng CPU gồm các ca regression cho lỗi A1 đổi CE, parser trả 0 khi đọc JSON thật, thiếu CI, sai split, chạy dở bị coi là đủ epoch, metadata checkpoint không khớp và tổng hợp bốn run bằng fixture nhỏ. Các fixture là dữ liệu tổng hợp trong thư mục tạm, không phải kết quả mô hình thực. Xem [bộ regression](../../../tests/test_ablation_v2.py).

Đặc tả vận hành và DoD còn mở được ghi tại [Task 7.12](../../tasks/member_c_training_eval/phase_7/task_7.12_ablation_mini_v2.md). Trước khi chạy Colab phải publish commit có bản sửa và chuẩn bị archive/split gốc; chưa có phép đo để bảo đảm thời gian GPU hoặc kết quả mô hình. Không sử dụng số dummy hoặc log Task 7.11 thay cho thí nghiệm mới.

Kiểm chứng sau sửa: **110/110 ca CPU đạt** trong nhóm test ablation, loss, selection, training runtime, preflight và acceptance (12,82 giây trong lần chạy local). Đã kiểm cú pháp notebook/script, liên kết tài liệu và `git diff --check`. Đây không phải thời gian train GPU hoặc bằng chứng checkpoint mô hình thật đã đạt.
