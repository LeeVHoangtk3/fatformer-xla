# TASK 7.13 — DegradationRouter và fusion đã hiệu chuẩn

- Phụ trách: thành viên B; phối hợp C.
- Trạng thái: `[ ] ĐÃ SỬA CODE — CHỜ ĐO THẬT`.
- [Báo cáo, giới hạn và lệnh chạy lại](../../../bao_cao/B/bao_cao_task_7.13_degradation_router.md).
- [Kế hoạch Phase 7](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md), §IV.2.

## 1. Mục tiêu và đính chính

Từng ảnh tự quyết định trọng số suy biến, không cần ảnh tham chiếu hoặc metadata điều kiện:

`P(fake|x) = (1-w)*p_clean + w*p_forensic`, với `w = sigmoid(beta0 + beta*standardized_signals)`.

Ba tín hiệu: blockiness, log1p Laplacian variance trên RGB [0,1], lambda thật từ adapter DynamicFrequencyGating. Trung bình lambda qua adapter, không qua batch.

Kết quả tại commit 13749f6 bị hủy vì fit Gaussian giả lập, đặt tay ngưỡng và nội suy ablation/CI. Không dùng AUC 100% hoặc thắng 5/5 để nghiệm thu. Chưa xác định router thắng hay thua.

Theo lựa chọn được cho phép ở Task 7.15: một backbone, hai xác suất chia sẻ logits, temperature riêng fit trên clean/Q30 cùng checkpoint. Không có hai chuyên gia feature độc lập hoặc mặc định tốt hơn fixed.

Temperature cố định giữ ranking của từng nhánh; tổ hợp với w cố định cũng giữ ranking. Tuy nhiên w(x) thay đổi theo ảnh có thể đảo ranking dù chia sẻ logits. Không dùng tính đơn điệu của từng nhánh hoặc độ rộng CI ở Task 7.12 để khẳng định router “100% không thể thắng”. Phản ví dụ và giới hạn suy luận có trong báo cáo §3.1; chưa có bằng chứng router cải thiện hoặc thất bại trên dữ liệu thật.

## 2. Dữ liệu và provenance

- Chỉ fit trên DATASET/manifests/data_manifest_v2_val_calib.json, split val_calib.
- tools/router_data.py đọc/xác minh hash ảnh, resize/crop 224, tạo clean và JPEG Q30 post-resize, chạy canonical model eval và lấy gating qua hook. Không thay ảnh lỗi/gating thiếu bằng hằng số.
- Cache router-data/v2: features[N,3], logits[N,2], targets[N], is_degraded[N], sample_ids[N] là hash ảnh gốc, subsets[N], metadata checkpoint/source/preprocess/Git.
- Mỗi ảnh gốc có một clean và một Q30 view, cùng nhãn thật/giả. Condition từ biến đổi thực thi, không từ model hoặc vị trí record.
- Evaluation dùng frozen CFE partition 1.600 ảnh của Task 7.4/7.12; không đưa lại 400 ảnh calibration vào evaluation. Cập nhật so với kế hoạch ban đầu 2.000 ảnh.
- Ablation từ chối cache khác checkpoint/preprocess hoặc hash ảnh gốc giao calibration. Không fit/chọn lại trên evaluation.

## 3. Fit và ngưỡng

tools/fit_router.py chỉ nhận cache measured val_calib:

1. 5-fold StratifiedGroupKFold theo hash ảnh gốc; clean/Q30 cùng ảnh trong một fold; scaler fit riêng trong fold. AUC thang 0–1.
2. Fit scaler và LogisticRegression(C=1, max_iter=1000, class_weight=balanced) cuối cùng trên toàn val_calib.
3. Fit temperature riêng trên clean và Q30 logits của cùng checkpoint.
4. Chọn fixed branch theo fake recall tại ngưỡng FPR5 trên pooled calibration views; hòa chọn clean. Oracle có ngưỡng riêng theo condition.
5. Chia tối đa ba bin theo quantile w, gộp bin dưới 50 Real views với bin bên cạnh và ghi log. Tổng Real views dưới 50 thì dừng.
6. Fit ngưỡng từng bin FPR <=5% bằng float32 và comparator >= như inference. Không làm tròn; ties bảo thủ; có thể dùng ngưỡng lớn hơn 1 một ULP để từ chối score bão hòa.
7. Lưu histogram w, AUC từng tín hiệu, scaler/coef/intercept, thresholds, số ảnh/views, CV, calibration và hashes trong degradation_router/v2. Fitted params chưa tự bật deployment.

## 4. Fusion, fallback và wrapper

- src/models/degradation_router.py: tín hiệu, tra bảng từng w, fusion và fallback từng ảnh; không dùng thống kê batch. Missing lambda báo lỗi khi extract/predict; forward dùng calibrated fixed fallback.
- src/models/unified_fatformer.py: UnifiedFatFormerXLA(...).eval()(normalized_x) trả prob_fake, label, w_deg, threshold; một backbone/một forward. Hook dọn trong finally; state_dict giữ tham số và metadata.
- Chỉ bật khi ablation measured v2 thắng, params fingerprint/checkpoint khớp. Chưa đo/không thắng/thiếu proof thì fixed fallback.
- Không có fitted params vẫn chạy được với calibration/v1 measured đúng SHA-256 backbone. Thiếu calibration hợp lệ thì dừng, không tự gán 0.50.
- Helper compute_unified_router_predictions_v3 không còn heuristic; routing cần tín hiệu thật, đúng checkpoint và proof. Không có params thì caller phải truyền ngưỡng đã hiệu chuẩn.
- Đóng gói .pth hoàn chỉnh và latency thuộc Task 7.15, chưa nghiệm thu.

## 5. Ablation và luật kết luận

tools/generate_router_ablation.py chạy từ dự đoán từng ảnh:

- Fixed: branch/ngưỡng toàn cục chọn trên val_calib.
- Router: w theo ảnh và bảng ngưỡng fit trên val_calib.
- Condition-oracle: biết condition để chọn calibration; không triển khai được, không lấy max hai chỉ số tổng hợp và không bảo đảm là cận trên toán học.

Chỉ số chính: Fake@Real90 trên pooled clean/Q30 từng subset, hai view cân bằng mỗi ảnh; báo clean/Q30 riêng. Đây là ROC diagnostic trên held-out scores; Real/Fake ACC triển khai báo riêng tại ngưỡng val_calib.

Bootstrap >=1.000 lần, seed mặc định 42; phân tầng Real/Fake theo ảnh gốc, giữ clean/Q30 cùng lần lấy mẫu, ba phương án cùng resample. CI95 cho fixed/router/oracle và chênh lệch.

Thắng khi delta >=1 pp ở >=3/5 subset **và** CI95 router nằm trên CI95 fixed ở >=2 subset. Không đạt thì KHÔNG THẮNG/fixed fallback. Chưa đo hợp lệ thì CHƯA XÁC ĐỊNH. Thắng router chưa tự chứng minh toàn bộ G7.3 đạt.

Chọn fixed fallback để ưu tiên tiến độ không phải nghiệm thu kết quả âm của router. Nếu hoãn hoặc thu hẹp hạng mục thực nghiệm, phải ghi rõ thay đổi phạm vi; các mục DoD đo thật bên dưới vẫn chưa hoàn thành theo yêu cầu ban đầu. Calibration fallback phải đúng checkpoint, không kế thừa trực tiếp T/ngưỡng của baseline cho v2.

## 6. Definition of Done

- [x] Tín hiệu, logistic inference, fusion theo ảnh và fallback.
- [x] Collector đo thật, CLI fit không tạo đặc trưng giả lập; ngưỡng fit theo FPR.
- [x] Ablation từ dự đoán, kiểm tra overlap, bootstrap và luật kết luận.
- [x] Wrapper một backbone; từ chối artifact cũ; chỉ bật khi proof khớp.
- [x] Unit/regression tests cho hook, batch lẫn nhiều ảnh, FPR/ties, provenance, overlap, bootstrap, fallback, state và wrapper.
- [ ] Khôi phục ảnh/checkpoint, collect/fit để có router_params measured.
- [ ] Chạy held-out ablation thật, kết luận thắng/không thắng với CI.
- [ ] Nhúng metadata checkpoint, kiểm chứng latency và G7.3 ở Task 7.15/7.14.

Artifacts DATASET/logs/phase7/router_params.json và router_ablation.{json,md} hiện chỉ là marker chờ đo, không phải model deployable. Xem báo cáo để biết lệnh chạy lại và giới hạn local.
