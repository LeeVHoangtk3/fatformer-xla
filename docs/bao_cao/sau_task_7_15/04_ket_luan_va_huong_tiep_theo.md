# Đánh giá, quyết định và hướng tiếp theo

Chốt ngày **10/11/2026**. Phần này dùng cho chương thảo luận/kết luận; [file 01](01_bao_cao_tong_hop.md) là bản báo cáo liền mạch và [file 02](02_bang_so_lieu.md) cung cấp số liệu.

## 1. Kết quả đạt được và phần chưa đạt

Các công việc sau Task 7.15 đã tạo một chuỗi bằng chứng từ pilot đến đánh giá đối chứng và chẩn đoán. P0 được chọn trước khi chấm CFE; những kết quả không đạt được giữ lại thay vì loại bỏ. Hậu kiểm tái tính dự đoán, đối chiếu source/hash và sửa kết luận không được dữ liệu hỗ trợ. Calibration transfer cũng cho thấy rõ mức đánh đổi giữa phát hiện ảnh giả và bảo vệ ảnh thật.

Tuy nhiên, **chưa tìm được model thay thế được chấp nhận**. P0 chỉ đạt 4/8 tiêu chí G7.2, ít hơn incumbent 6/8. Hai điểm calibration clean-only đều làm Real recall Q30 xuống dưới guard90. Đã hoàn tất chạy hoặc hậu kiểm không đồng nghĩa đạt tiêu chí chất lượng.

Model chuẩn vẫn còn hai điểm yếu G7.2: Clean Fake@Real90 thiếu 0,875 pp so với mốc65%, và worst-subset Fake accuracy thiếu khoảng14,14 pp so với mốc45%. Q30 Fake@Real90 của incumbent **đã đạt** mốc40%; không viết rằng đây là tiêu chí G7.2 incumbent còn trượt. Q30 vẫn là hướng robustness đáng nghiên cứu, nhưng không thay thế việc xử lý và báo cáo đúng hai tiêu chí còn thiếu.

Nguồn: [gates incumbent](../../../DATASET/logs/phase7/task_7.15/current/analysis/gate_summary.csv), [matched evaluation](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/README.md), [quyết định transfer](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/analysis/decision.json).

## 2. Quyết định giữ bản chuẩn

Giữ package SHA-256 **`248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`** cùng calibration T **2,7832491397857666**, ngưỡng **0,7918508648872375**. Đây là bản được giữ theo các tiêu chí và bằng chứng hiện có, không phải tuyên bố thắng ở mọi chỉ số hoặc mọi tập dữ liệu.

Hồ sơ chuẩn: [`DATASET/logs/phase7/task_7.15/current/`](../../../DATASET/logs/phase7/task_7.15/current/README.md). Weights theo reference GPU nằm trên Drive, không có bản `.pth` mới tương ứng ở local:

```text
MyDrive/Fatformer/checkpoint/phase7/corrected_20261109_102954_3d28ad/unified_fatformer_v2.pth
```

Giữ nguyên notebook, raw logs và calibration chuẩn. P0, incumbent refit và các fit clean-v2 là kết quả nghiên cứu có định danh riêng; không đổi tên hoặc ghi đè chúng thành package chuẩn. Việc biên soạn hồ sơ này không thay model, không tạo bản backup weights hoặc xác minh lại bytes weights trên Drive.

## 3. Những nhận xét cần viết đúng

- **Pilot:** P0 tốt hơn P1 theo luật selection của pilot một seed; chưa chứng minh mọi preprocessing CLIP đều kém.
- **CFE:** điểm P0 thấp hơn ở nhiều mục tiêu, nhưng không phải mọi delta recall đều có CI loại trừ 0. Quyết định không thay model dựa trên tiêu chí đã chốt, không cần phóng đại ý nghĩa thống kê.
- **Calibration:** Fake recall tăng mạnh đi kèm FPR Q30 trên12%. NLL/ECE tốt hơn không bảo đảm Real recall ở ngưỡng vận hành.
- **Ranking:** thay T/ngưỡng trên logits đóng băng không cải thiện raw ranking. Không lấy Fake recall validation84,69% để thay Clean Fake@Real90 CFE64,125%.
- **JPEG và generator:** CIPS có tín hiệu suy giảm trên12 ảnh; DiT recall đã thấp ở clean. Chưa chứng minh cơ chế biểu diễn nội tại “sụp đổ”.
- **Bit-exact:** bốn logits arrays tái dùng trong transfer có kiểm chứng bằng arrays thật. Bốn recall tổng hợp lịch sử khớp không đủ để tuyên bố bit-exact logits lịch sử.

Nguồn chi tiết: [đính chính diagnostics](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/README.md), [thảo luận transfer](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/README.md).

## 4. Trạng thái toàn Phase 7 cần giữ trong báo cáo

| Cổng/hồ sơ | Trạng thái hiện hành | Phần còn thiếu hoặc không đạt |
| :--- | :--- | :--- |
| G7.0 | PENDING theo DoD toàn repo | Loader legacy và CI chẩn đoán n3/n8 còn mở |
| G7.1 | FAIL | Gen2 Clean Fake@Real90 48,88% <55%; Q30 AUC43,67% <55% |
| G7.2 | FAIL, 6/8 tiêu chí | Clean Fake@Real90 và worst-subset Fake accuracy |
| G7.3 | PENDING | Verified best-by-condition reference trên cùng partition/protocol |
| Task 7.16 | Hồ sơ đã biên soạn; chờ xác nhận | Cross-review/chữ ký A/B/C người thật |

Nguồn trạng thái: [Task 7.16](../../../DATASET/logs/phase7/task_7.16/README.md), [human_signoff.json](../../../DATASET/logs/phase7/task_7.16/human_signoff.json), [hậu kiểm package](../../../DATASET/logs/phase7/task_7.15/current/analysis/README.md). Các run bổ sung không đóng các mục còn thiếu trên. Không dùng số WiSE-FT Gen1 đã bị vô hiệu hóa làm kết quả hợp lệ.

## 5. Công việc nên làm tiếp, theo thứ tự

### 5.1. Hoàn thiện báo cáo và G7.3

1. Dùng incumbent làm kết quả chính; trình bày P0 và calibration transfer như thí nghiệm bổ sung, gồm cả kết quả âm và hạn chế.
2. Rà soát reference tốt nhất từng điều kiện có cùng partition, tiền xử lý và bằng chứng nguồn để kiểm điều kiện delta AUC/Fake@Real90 ≤1,5 pp của G7.3. Không mặc định một incumbent là reference tốt nhất mọi điều kiện; chưa đủ bằng chứng thì giữ PENDING.
3. Bổ sung xác nhận chéo A/B/C thật. Việc ký không thay kết quả FAIL thành PASS.

Không cần chạy lại y nguyên notebook đã hậu kiểm chỉ để mong chỉ số tăng.

### 5.2. Nếu tiếp tục cải thiện model

Thiết kế một thí nghiệm riêng trước khi chạy: kiểm tra nhãn/nguồn và độ phủ của train/validation, khóa control, seed, ngân sách, luật selection và metric theo nhóm. Chỉ thay một yếu tố chính, ví dụ augmentation JPEG trong huấn luyện; không đồng thời đổi loss, preprocessing, dữ liệu và calibration rồi gán nguyên nhân cho một yếu tố. Có thể dùng CUDA GPU khác L4 nếu đủ bộ nhớ, nhưng hai nhánh đối chứng phải dùng protocol phù hợp và ghi rõ GPU/batch/precision.

Mục tiêu là cải thiện khả năng phân biệt trên dữ liệu train/validation có độ phủ tốt hơn mà vẫn bảo vệ ảnh thật. Không chọn dữ liệu, checkpoint, ngưỡng hay augmentation nhằm tối ưu các ảnh CFE đã xem. Muốn có kết luận tổng quát mới, cần holdout chưa được dùng điều chỉnh hoặc trình bày rõ tính chất hậu kiểm; nên có nhiều seed khi ngân sách cho phép.

Ứng viên chỉ được xem xét thay bản chuẩn sau khi có chuỗi bằng chứng cùng hash, calibration độc lập, so sánh công bằng trên các tiêu chí đã định và kiểm chứng đóng gói. Hoàn tất training không tự động thay incumbent. Đây là hướng đề xuất; **chưa có thí nghiệm train vòng 2 được khởi chạy trong hồ sơ này**.

### 5.3. Nếu mục tiêu chỉ là điểm vận hành chung clean/Q30

Có thể thử calibration robust riêng: fit trên các cặp clean/Q30 của `val_calib`, lấy một ngưỡng chung đáp ứng ngân sách FPR đã định ở cả hai điều kiện calibration. Không fit trên val_select, không dùng hai ngưỡng biết trước condition, và không tính hai view là hai ảnh gốc độc lập. Hướng này không tạo ranking gain và không bảo đảm đạt trên miền khác; không ưu tiên nó như lời giải cho mọi tiêu chí G7.2.

Nguồn định hướng: [kế hoạch sau kết quả](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/analysis/post_result_plan.json). Mọi hướng ở mục5 là việc còn cần thiết kế/xác minh, không phải kết quả đã chạy.

## 6. Đoạn kết luận có thể đưa vào báo cáo

> Sau Task 7.15, các thí nghiệm bổ sung chưa xác lập được phương án thay thế model hiện hành. Pilot lựa chọn P0 theo validation, nhưng đánh giá cùng protocol cho P0 đạt 4/8 tiêu chí G7.2, thấp hơn 6/8 của incumbent. Hiệu chuẩn trên clean-v2 giúp tăng nhận diện ảnh giả, song làm Real recall trên JPEG Q30 giảm dưới ngưỡng bảo vệ đã định. Nghiên cứu vì vậy giữ nguyên package incumbent và công bố các kết quả chưa đạt. Hướng tiếp theo là hoàn thiện đối chiếu G7.3 và khảo sát cải thiện dữ liệu/huấn luyện có đối chứng, thay vì tiếp tục điều chỉnh ngưỡng trên tập đánh giá đã xem.
