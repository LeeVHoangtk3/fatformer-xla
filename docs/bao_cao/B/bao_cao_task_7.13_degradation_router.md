# Task 7.13 — hậu kiểm và sửa DegradationRouter

Ngày cập nhật: 10/11/2026. Trạng thái: **code đã kiểm thử; chọn calibrated fixed fallback; hoãn thực nghiệm router; chưa nghiệm thu toàn bộ Task 7.13**. Task 7.15 sau đó đã đóng gói/đo fixed fallback trên GPU, không đồng nghĩa router đã được nghiệm thu.

Package báo cáo vẫn là `24869052…`, router tắt, G7.2 FAIL/đạt 6/8 và G7.3 pending. Pilot P0/P1 và matched evaluation bổ sung chưa tạo model thay thế. Run chẩn đoán validation `val_diag_20261110_142828` đã hoàn tất/hậu kiểm nhưng không thu thập bằng chứng router. Xem [báo cáo các giai đoạn tiếp theo](../C/bao_cao_phase7_improvement_round1_va_chan_doan_vong2.md). Phần implementation/hủy số synthetic bên dưới giữ phạm vi hậu kiểm từ 08/11, không được diễn giải thành thí nghiệm router mới.

## 1. Hủy bằng chứng không hợp lệ

Các tuyên bố AUC 100%, router thắng 5/5 subset và CI95 của bản cũ không được sử dụng. Script cũ chỉ lấy số lượng record val_calib để sinh đặc trưng Gaussian và đặt ngưỡng 0.50/0.55/0.60; ablation nội suy `fixed + 0.65*(oracle-fixed)` và gán CI ±2.8 pp. Artifacts hiện tại chỉ ghi trạng thái chờ đo; bản cũ còn trong Git commit 13749f6.

Chưa xác định router thắng hoặc thua. Không dùng các số cũ để thông qua G7.3.

## 2. Implementation hiện tại

- [router_data.py](../../../tools/router_data.py): đọc ảnh thật, kiểm tra SHA-256, resize/crop 224 và tạo clean/Q30 post-resize. Một forward mỗi view lấy logits và hook DynamicFrequencyGating. Trung bình qua adapter, không qua batch; dọn hook trong finally. Ảnh thiếu/lỗi hoặc không có gating làm collector dừng, không thay bằng ảnh giả hay lambda=1.
- [degradation_router.py](../../../src/models/degradation_router.py): blockiness, log1p Laplacian variance và lambda thật; tra ngưỡng theo ảnh; fallback riêng cho mẫu có tín hiệu lỗi. Thiếu lambda không được xem là tín hiệu hằng. state_dict giữ cả intercept, ngưỡng và chế độ fallback.
- [fit_router.py](../../../tools/fit_router.py): chỉ nhận cache measured val_calib; scaler và logistic; 5-fold CV nhóm theo hash ảnh gốc để clean/Q30 cùng ảnh không lọt qua hai fold. AUC lưu theo thang 0–1. Có histogram w và AUC từng tín hiệu; AUC dưới 0.5 có thể biểu thị tương quan ngược.
- Fit ngưỡng để FPR thực nghiệm trên Real calibration <=5%, comparator >= và float32 như suy luận. Đây không phải bảo đảm FPR trên evaluation/production; cần đo riêng sau khi khóa ngưỡng. Tạo tối đa ba bin theo quantile w; có thể gộp còn hai hoặc một bin nếu dưới 50 Real views. Tổng Real views dưới 50 thì dừng. Không làm tròn ngưỡng; ties bảo thủ; có thể dùng ngưỡng lớn hơn 1 một ULP để từ chối score bão hòa.
- [unified_fatformer.py](../../../src/models/unified_fatformer.py): một backbone, một forward. Clean/forensic **chia sẻ logits**, khác temperature fit riêng trên clean/Q30 cùng checkpoint. Không có hai nhánh feature độc lập, chưa khẳng định thiết kế có lợi. Fixed branch chọn bằng fake recall tại FPR5 trên pooled val_calib; hòa chọn clean. Oracle có ngưỡng riêng fit theo condition.
- [generate_router_ablation.py](../../../tools/generate_router_ablation.py): đọc evaluation cùng checkpoint/preprocess; từ chối hash ảnh giao calibration; so sánh fixed/router/condition-oracle trên đủ 5 subset. Bootstrap >=1.000 lần, phân tầng Real/Fake theo ảnh gốc, giữ hai view cùng lần lấy mẫu và ghép cặp ba phương án. Xuất CI từng phương án và CI chênh lệch.
- Helper compute_unified_router_predictions_v3 đã bỏ heuristic cũ; cần tín hiệu/params và đúng checkpoint. Không có params thì caller phải truyền ngưỡng đã hiệu chuẩn. Pipeline một backbone dùng wrapper mới.

## 3. Khế ước đánh giá và kích hoạt

Chỉ số chính: Fake@Real90 trên pooled clean/Q30 với hai view cân bằng cho mỗi ảnh, theo từng subset; có điểm clean/Q30 riêng. Đây là ROC diagnostic dùng held-out scores, không phải Fake ACC tại ngưỡng triển khai. Real/Fake ACC triển khai báo riêng tại ngưỡng val_calib.

Condition-oracle dùng clean calibration khi biết ảnh clean và forensic calibration khi biết ảnh Q30. Không lấy max chỉ số tổng hợp. Oracle không triển khai được và không bảo đảm luôn là cận trên toán học.

Thắng khi delta >=1 pp ở >=3/5 subset **và** CI95 router nằm trên CI95 fixed ở >=2 subset. Không đạt thì fixed fallback. Chưa đo hợp lệ thì “chưa xác định”, không gọi là thua.

Wrapper chỉ bật router khi ablation measured v2 thắng và fingerprint params/checkpoint khớp. Không có bằng chứng thắng thì dùng fixed branch và ngưỡng đúng checkpoint. Params v1 hoặc marker chờ đo không được dùng như fitted params. Không router vẫn chạy được với calibration/v1 measured đúng SHA-256 checkpoint. Thiếu calibration hợp lệ thì dừng, không tự đặt 0.50.

`validate_router_report` yêu cầu đủ đúng năm subset, chỉ số/CI hữu hạn và đúng miền, số ảnh Real/Fake và cặp view nhất quán, đủ clean/Q30, cùng protocol bootstrap phân tầng/ghép cặp >=1.000 lần với CI95. Hàm tính lại delta, số subset tăng >=1 pp và số CI tách rời, rồi đối chiếu toàn bộ verdict. Báo cáo thiếu, sai kiểu hoặc mâu thuẫn không bật router; wrapper vẫn dùng calibration fixed hợp lệ.

Báo cáo phải tham chiếu hash checkpoint, params file, cache calibration/evaluation và frozen evaluation split; canonical digest params phải khớp nội dung đang nạp. Đây là **kiểm tra hash và tính nhất quán**, không phải chữ ký số/xác thực người phát hành. Runtime không mở lại cache hoặc chạy lại bootstrap; số đo vẫn cần kiểm chứng bằng pipeline và artifacts gốc. Không gọi cơ chế này là bảo đảm an toàn tuyệt đối hoặc bằng chứng tự xác thực dữ liệu thật.

### 3.1. Đính chính lập luận về thứ hạng và kết quả âm

Với bài toán hai lớp, gọi z là hiệu logit fake trừ real. Mỗi nhánh có p=σ(z/T), T>0 cố định nên giữ thứ hạng theo z, ngoại trừ ties do độ chính xác hữu hạn. Khi w cố định trên mọi ảnh, tổ hợp lồi hai hàm này cũng đơn điệu tăng. Khi hai temperature bằng nhau, w không ảnh hưởng đến xác suất hợp nhất.

Nhưng router hiện dùng **w(x) khác nhau theo ảnh**. Xác suất hợp nhất không còn là một hàm cố định chỉ của z, nên không bảo đảm giữ thứ hạng, kể cả giữa các ảnh cùng condition. Phản ví dụ với T_clean=1, T_forensic=10: ảnh A có z=2, w=0.99 cho p_router≈0.553144; ảnh B có z=1, w=0.01 cho p_router≈0.728998. Hai nhánh riêng đều xếp A trên B, nhưng router xếp B trên A. Kiểm thử tham số hóa trong tests/test_router_pipeline.py kiểm tra cả trường hợp w cố định, w theo ảnh và hai temperature bằng nhau.

Đây chỉ là phản ví dụ toán học bác bỏ khẳng định “chia sẻ logits nên không thể đổi ranking/không thể thắng”; **không chứng minh router của dự án cải thiện**, không phải số đo held-out và không được đưa các số ví dụ này vào artifacts measured.

CI rộng khoảng 13 pp của cấu hình khác ở Task 7.12 không đặt giới hạn cho mức cải thiện hoặc độ rộng CI của router chưa đo. Quy tắc hai CI95 không chồng lấn là tiêu chí nghiệm thu hiện hành, không tương đương với việc chỉ xem CI ghép cặp của chênh lệch có chứa 0; không tự thay tiêu chí sau khi xem kết quả. Chưa thể suy ra router chắc chắn cần tăng đúng 10–14 pp hoặc chắc chắn thất bại từ CI đó.

Nếu chọn ưu tiên tiến độ và dùng fixed fallback, ghi **“chọn phương án triển khai fixed; lợi ích router chưa được đánh giá, hạng mục thực nghiệm được hoãn hoặc thu hẹp phạm vi”**. Đây là quyết định phạm vi/nguồn lực, không phải kết quả âm đã được chứng minh và không hoàn thành DoD thực nghiệm ban đầu. Kết quả âm cần phép đo hợp lệ không đạt luật thắng. Kết quả đo không đạt cũng không tự chứng minh router không có lợi trong mọi bối cảnh.

Các giá trị T=1.6402 và ngưỡng 0.9127 trong calibration/baseline.json chỉ gắn với checkpoint baseline có SHA-256 tương ứng. Không chuyển nguyên sang anchored v2 hoặc checkpoint khác. Chi phí “2–3 giờ GPU” trong nhận định bên ngoài cũng chưa được benchmark cho pipeline này; không dùng làm số đo lập kế hoạch đã xác minh.

## 4. Chạy lại trên môi trường có dữ liệu

Chạy từ repo root. Phải khôi phục đúng checkpoint và ảnh manifest trước; không tự thay checkpoint rồi kế thừa calibration cũ.

```powershell
python -m tools.router_data --manifest DATASET/manifests/data_manifest_v2_val_calib.json --checkpoint DATASET/checkpoints/phase7/fatformer_anchored_v2_best.pth --clip-path DATASET/pretrained/ViT-L-14.pt --device cuda --output DATASET/logs/phase7/router_val_calib.npz
python -m tools.fit_router --cache DATASET/logs/phase7/router_val_calib.npz
python -m tools.router_data --cfe-dir DATASET/test_communityforensics --evaluation-split DATASET/logs/phase7/wise_ft_sweep/logits_baseline_clean_eval_split.npz --checkpoint DATASET/checkpoints/phase7/fatformer_anchored_v2_best.pth --clip-path DATASET/pretrained/ViT-L-14.pt --device cuda --output DATASET/logs/phase7/router_evaluation.npz
python -m tools.generate_router_ablation --cache DATASET/logs/phase7/router_evaluation.npz
```

Evaluation dùng frozen partition 1.600 ảnh của Task 7.4/7.12, thay cho toàn bộ 2.000 ảnh theo kế hoạch cũ. Collector giữ chính xác sample IDs của partition; ablation kiểm tra thêm hash ảnh gốc không giao val_calib. Không tái chia sau khi xem kết quả.

Cache ghi hash checkpoint, nguồn manifest/partition và preprocess/Git. Output fit ghi hash cache và danh sách hash ảnh calibration. Metadata calibration/router đã có trong state_dict; đóng gói checkpoint hoàn chỉnh thuộc Task 7.15.

## 5. Kiểm thử và giới hạn

Unit/regression tests kiểm tra batch lẫn nhiều ảnh, gating thật trên model nhỏ, dọn hook khi lỗi, FPR/ties, gộp bin, từ chối synthetic/sai split, overlap, bootstrap, fallback theo ảnh, state roundtrip, wrapper và helper. Fixture trong thư mục tạm chỉ phục vụ kiểm thử, không phải số đo nghiệm thu.

Lần xác minh implementation trước đính chính lập luận: **68 tests PASS** trên tests/test_router.py, tests/test_router_pipeline.py, tests/test_phase7_acceptance.py, tests/test_ablation_v2.py và tests/test_canonical_loader.py. Đây là tổng kiểm thử nhiều thành phần, không phải 68 phép đo hiệu quả router và không bảo đảm chương trình không còn lỗi. CLI --help và Python compilation chạy được; git diff --check không có lỗi whitespace. Pytest dùng thư mục basetemp mới trong scratch vì thư mục pytest mặc định của máy bị hạn chế quyền.

Mốc trước sửa kiểm tra báo cáo: **71 tests PASS**, gồm router/pipeline 25, ablation 25, acceptance 8, canonical loader 13. Ba ca ranking trong số đó chỉ là phản ví dụ toán học. Kiểm thử không bảo đảm toàn bộ mã nguồn không lỗi hoặc thay thế số đo trên checkpoint thật.

Sau sửa kiểm tra kích hoạt: **107 tests PASS** trên cùng năm tệp, bổ sung 36 ca hồi quy cho report thiếu/mâu thuẫn, metric/CI/count sai, protocol/fingerprint sai và fallback tương ứng. Python compilation và git diff --check đạt. Report v2 cũ thiếu các trường bắt buộc mới phải được sinh lại bằng tools/generate_router_ablation.py; không kích hoạt chỉ bằng cờ tổng kết.

Preflight local xác nhận **0/1.080 ảnh val_calib có mặt** tại đường dẫn manifest và thiếu checkpoint đích fatformer_anchored_v2_best.pth. Collector dừng ở file thiếu, không tạo cache. Chưa chạy collect/fit/evaluation thật, chưa benchmark GPU/overhead <5ms hoặc đóng gói Task 7.15.

Theo [báo cáo 7.12 hiện tại](../C/bao_cao_task_7.12_ablation.md), A3 tốt nhất theo worst-group validation trong seed đã chạy, A4 không vượt trội; không kế thừa tuyên bố A4 tốt nhất của bản cũ.

## 6. Phạm vi hoàn thành và phần còn mở

- **Implementation:** đã có collector, fit, ablation, wrapper và regression tests; kiểm tra cấu trúc/nhất quán báo cáo kích hoạt đã được bổ sung. Có thể tiếp tục tích hợp API vào Task 7.15.
- **Lựa chọn triển khai:** ưu tiên calibrated fixed fallback, hoãn thực nghiệm router. Đây là quyết định phạm vi/tiến độ, chưa phải kết quả âm hoặc bằng chứng router không có lợi.
- **Calibration/ablation thật:** chưa có params router fitted hoặc kết quả evaluation hợp lệ; marker JSON vẫn unverified. Fallback cho v2 chỉ chạy khi được cung cấp calibration đo thật đúng checkpoint; không kế thừa trực tiếp các số baseline.
- **Đóng gói:** chưa xác minh artifact .pth hoàn chỉnh, end-to-end trên checkpoint thật hoặc latency GPU. Không ghi deployed hoặc hoàn thành toàn bộ DoD ban đầu.

