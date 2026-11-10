# Hậu kiểm calibration transfer — 10/11/2026

**Thí nghiệm chạy đúng, tăng mạnh Fake recall nhưng calibration clean-only không đạt guard Q30. Không chọn điểm mới để thay deployment.** Model báo cáo vẫn là package `24869052…`, G7.2 FAIL/đạt 6/8, G7.3 pending. Không có weights mới hoặc kết quả CFE mới trong lần này.

## 1. Nguồn và mức độ kiểm chứng

- Nhận từ `wait/calibration_transfer_without_weights/`: **56 file, 2.199.957 byte**; đã sao nguyên byte vào [`raw/`](raw/) và ghi [`import_manifest.json`](import_manifest.json).
- Run `calib_transfer_20261110_155948`, source `253942a5262432fbed7ce7ca8b12b4e455bfc2ce`; **170 source hashes** khớp Git, hash mọi output trong status khớp. L4, Torch 2.14.1+cu130/CUDA13.0, FP16, batch16, TF32 tắt.
- Chế độ **`reuse_frozen_logits`**: bốn raw validation logits matrices và IDs tái dùng đúng bit từ diagnostics đã hậu kiểm. Đây là xác nhận với arrays thật có sẵn, khác tuyên bố bit-exact từ bốn recall tổng hợp ở log train cũ. Không forward backbone validation mới trong run này.
- Hai bộ calibration mới: 2 model ×1.080 clean `val_calib` = **2.160 record**, mỗi model 590 real/490 fake. Validation dùng cùng 1.080 ID ở clean/Q30; calibration và validation không trùng SHA. Tám scored arrays = **8.640 quyết định**; không phải 8.640 ảnh kiểm thử độc lập.
- Đã tái tính báo cáo từng model, **224 dòng group metrics**, **1.471 dòng lỗi quyết định**, CSV summary/delta và toàn bộ paired CI. Có **30 test CPU** đạt cho auditor/workflow (8 audit mới +22 transfer); khác bộ 30 test preparation trước đó, không cộng thành 60 test độc lập.

[`analysis/verification.json`](analysis/verification.json) ghi chi tiết, sai số CPU/GPU và scope; [`analysis/recomputed/`](analysis/recomputed/) là báo cáo tái tính từ **scores/labels GPU đã lưu**. Bản calibration notebook hiện trong repo vẫn có execution counts null; chưa có notebook thực thi để đối chiếu cell/output. Không sửa template hoặc các notebook người dùng đã chạy. Việc này không cản kiểm tra gói arrays/source, nhưng không được báo “đã kiểm notebook thực thi thành công”.

Local không có `.pth` hoặc full archive ảnh, nên không forward lại model hoặc kiểm nhãn bằng ảnh gốc. Canonical load GPU báo đúng source SHA, 0 missing/0 unexpected, 1.200 keys. NPZ và các bằng chứng binary giữ local/Drive; text metadata/báo cáo đồng bộ riêng vào Git.

## 2. Fit calibration hợp lệ và đúng ngân sách

T được fit chỉ trên clean `val_calib` v2, không dùng `val_select` để fit. Một ngưỡng dùng chung clean/Q30:

- **P0:** T **1,9718821049**, ngưỡng **0,3712310195**.
- **Incumbent:** T **1,7667391300**, ngưỡng **0,5055273175**.

Cả hai có **29/590 false positives =4,915254%** trên calibration, đúng budget floor(5%×590)=29 với comparator `>=`. Bộ fit nhiệt độ hội tụ; nghiệm T CPU/kiểm tra NLL độc lập khớp. Ngưỡng CPU của P0 khác GPU khoảng **8,94×10⁻⁸**, trong tolerance float32; replay CPU tại ngưỡng GPU vẫn cho đúng 29 FP. Không ép sửa ngưỡng GPU bằng số CPU. File dùng research schema và `deployable=false`.

FPR≤5% trên tập fit không bảo đảm cho validation hoặc ảnh Q30. Việc calibration đạt 29 FP nhưng validation vượt 5% không tự chứng minh lỗi code.

## 3. P0 — recall tăng, báo oan cũng tăng

So `cfe_control` → `v2_clean_fit`, cùng weights và raw logits:

- **Clean:** Fake **62,04% →84,69%**; Real **99,32% →94,41%**; FPR **0,68% →5,59%**. FP **4→33**, TP **304→415**, accuracy **82,41%→90%**.
- **Q30:** Fake **49,18% →80%**; Real **98,14% →86,95%**; FPR **1,86% →13,05%**. FP **11→77**, TP **241→392**, accuracy **75,93%→83,80%**.

Guard đã khai báo: Real recall ≥90% **ở cả hai điều kiện**. P0 clean đạt, Q30 thiếu khoảng **3,05 pp** nên điểm mới **không đạt guard**. Đây là tradeoff lớn, không phải tăng Fake miễn phí.

Paired CI95 V2−control: clean Fake **+22,65 pp [18,98;26,33]**, Real **−4,92 pp [−6,61;−3,39]**; Q30 Fake **+30,82 pp [26,53;34,90]**, Real **−11,19 pp [−13,90;−8,64]**. Những CI này cho thấy hai mặt tradeoff trên validation đã đo, không chứng minh tổng quát sang CFE hoặc mọi generator.

## 4. Incumbent — cùng kiểu tradeoff

- **Clean:** Fake **63,06% →84,49%**; Real **98,98% →94,75%**; FPR **1,02% →5,25%**. FP **6→31**, TP **309→414**, accuracy **82,69%→90,09%**.
- **Q30:** Fake **49,18% →78,78%**; Real **97,29% →87,29%**; FPR **2,71% →12,71%**. FP **16→75**, TP **241→386**, accuracy **75,46%→83,43%**.

Q30 thiếu khoảng **2,71 pp Real recall**, nên điểm mới cũng **không đạt guard**. Hai control cũ đều đạt guard trên validation này; không có căn cứ tự đổi sang point mới vì accuracy/Fake recall lớn hơn.

CI95 V2−control: clean Fake **+21,43 pp [17,76;25,31]**, Real **−4,24 pp [−5,76;−2,71]**; Q30 Fake **+29,59 pp [25,51;33,67]**, Real **−10 pp [−12,54;−7,62]**. Nguồn đầy đủ: [`raw/incumbent/comparison/summary.csv`](raw/incumbent/comparison/summary.csv), [`paired_deltas.csv`](raw/incumbent/comparison/paired_deltas.csv).

## 5. Điểm tốt có thật và điều không được suy ra

NLL và confidence ECE điểm đều giảm trên validation:

- P0 clean NLL **0,28764→0,26401**, ECE **6,92%→2,30%**; Q30 NLL **0,38609→0,37197**, ECE **6,45%→4,90%**.
- Incumbent clean NLL **0,27139→0,24835**, ECE **5,54%→2,32%**; Q30 NLL **0,38221→0,36975**, ECE **5,49%→3,39%**. Q30 incumbent delta NLL có CI **[−0,02522;+0,00101]**, chứa 0; không khẳng định mọi cải thiện có ý nghĩa thống kê.

ECE ở đây là confidence ECE 10 bins theo argmax0,5, không phải bảo đảm báo oan tại ngưỡng vận hành. CI chỉ conditional on hai fit hiện tại/một seed, không bao quát fit uncertainty hoặc phụ thuộc generator. Recipe đổi đồng thời domain, cỡ mẫu và prior lớp calibration, chưa tách nguyên nhân riêng.

**Không có cải thiện biểu diễn:** raw-margin AUC của P0 **96,3691% clean/92,1361% Q30**, incumbent **96,4348%/91,8049%**, cùng raw logits cho hai point. Chênh score AUC rất nhỏ chỉ do lượng tử/ties float32. Calibration không tạo ranking gain. Fake recall84,69% trên v2 validation tại ngưỡng mới không thay Fake@Real90 của CFE hoặc biến G7.2 thành PASS.

Kết quả hỗ trợ nhận xét điểm CFE cũ bảo thủ trên miền v2 và calibration clean-only chưa bảo vệ đủ Q30. Nó không xác lập nguyên nhân P0 thua CFE, không loại trừ hạn chế representation/domain coverage, không chứng minh model mới tốt hơn. `val_select` đã dùng chọn P0 và phân tích lỗi, không phải holdout mù.

## 6. Quyết định và bước tiếp theo

[`analysis/decision.json`](analysis/decision.json): **không chấp nhận điểm clean-only mới để thay deployment**, giữ package `24869052…`, G7.2 FAIL6/8, G7.3 pending. Không hạ guard sau khi thấy kết quả; chưa có training, package hoặc CFE run mới.

Nếu cần một operating point chung clean/Q30, có thể làm một **thí nghiệm riêng** fit T trên clean/Q30 `val_calib` ghép cặp, rồi lấy **một ngưỡng chung bảo thủ** đáp ứng budget Real của cả hai điều kiện trên calibration. Không fit ngưỡng theo `val_select`, không dùng hai ngưỡng biết trước condition, không bảo đảm sẽ đạt trên validation. Đây chỉ là sửa tradeoff vận hành, không nâng ranking.

Nếu mục tiêu chính là nâng G7.2/Fake@Real90 trên CFE, cần thí nghiệm weights/data/robustness có control và chọn bằng train/validation; calibration đơn thuần không khép được khoảng thiếu ranking. Không tiếp tục chỉnh ngưỡng theo CFE đã xem. [Kế hoạch sau kết quả](analysis/post_result_plan.json) ghi hai mục tiêu riêng; chưa khởi chạy thí nghiệm tiếp theo.
