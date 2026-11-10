# Vòng cải thiện 2 — chẩn đoán trước khi train

Cập nhật 10/11/2026. **Đã hậu kiểm chẩn đoán validation GPU**, run [`val_diag_20261110_142828`](runs/val_diag_20261110_142828/README.md): 4.320 lượt dự đoán, 112 dòng chỉ số nhóm, 933 dòng lỗi tái lập. Chưa fit calibration hoặc train mới. Model báo cáo vẫn là `24869052…`; G7.2 FAIL/đạt 6/8, G7.3 pending verified best-by-condition reference.

Bản tóm tắt gửi kèm có số gộp đúng nhưng vài suy diễn sai: recall khớp không phải bit-exact logits, SD v1.5 P0 Q30 là 33,33%, DiT đã thấp ở clean; ngưỡng 0,9127 không phải ngưỡng P0 dùng khi chấm CFE. Xem [đính chính và bước tiếp theo](runs/val_diag_20261110_142828/README.md), [index hiện hành](index.json). Các mục kiểm kê/chuẩn bị dưới đây giữ lịch sử trước run; kết quả mới được lưu riêng, không ghi đè preparation hoặc notebook.

Phép đối chứng tiếp theo đã được chuẩn bị tại [`../calibration_transfer/`](../calibration_transfer/README.md): code/controls/notebook mới, chưa chạy GPU calibration, chưa train hoặc thay model. Không cần chạy lại notebook diagnostics chỉ để bắt đầu bước này.

## 1. Những gì đã xác nhận

Manifest v2 có 14.400 record: train 12.240 (6.683 real/5.557 fake), `val_calib` 1.080 và `val_select` 1.080; mỗi tập validation gồm 590 real/490 fake. Paths và SHA-256 trong manifest đều duy nhất, không có trùng byte-exact ID giữa split; nhãn metadata nhất quán với family real/fake. Đây không phải kiểm duyệt ngữ nghĩa nhãn qua ảnh. Hậu kiểm isolation trước đó gắn đúng hash manifest và báo pass/0 vi phạm; chưa lặp lại kiểm tra pixel/content tại local.

Độ phủ generator fake theo family: autoregressive **1** (`tamingTransformers`), commercial **1** (`GPT-4o`), pixel diffusion **1** (`DeepFloyd`), modern GAN **6**, latent diffusion **8**. Tổng 17 generator fake. Autoregressive có 1.200 ảnh, chia 1.020 train/90 calibration/90 selection; không thiếu cả họ này, nhưng độ đa dạng generator hẹp. Chưa có bằng chứng mapping một-một với subset test khó, nên chưa kết luận đây là nguyên nhân gây lỗi CFE.

**Toàn bộ generator của `val_select` đã có ở train.** Vì vậy validation hiện tại kiểm tra ảnh chưa thấy của các generator đã thấy, không kiểm tra khái quát sang generator mới. Mỗi generator latent nhỏ chỉ có khoảng 2–3 ảnh mỗi split validation; recall riêng những nhóm này có độ bất định lớn.

Q30 lịch sử chỉ dùng 300/1.080 ảnh: 150 real và 150 fake. Phần fake gồm autoregressive 33, modern GAN 30, commercial 26, pixel diffusion 19, latent diffusion 42. Chỉ số Q30 gộp có thể che lỗi theo nhóm; cần giữ lịch sử 300 ảnh và đo thêm đủ validation, không thay lịch sử bằng số mới.

Chi tiết máy đọc được: [`preparation/audit.json`](preparation/audit.json), [`family_coverage.csv`](preparation/family_coverage.csv), [`generator_coverage.csv`](preparation/generator_coverage.csv), [`historical_q30_coverage.csv`](preparation/historical_q30_coverage.csv).

## 2. Log hiện có cho biết gì và chưa cho biết gì?

Tại epoch 5 và cùng operating point lịch sử T=1,6402, threshold=0,9127:

- P0 clean: đúng 586/590 real (99,32%), đúng 296/490 fake (60,41%). Q30 lịch sử: đúng 147/150 real (98%), đúng 63/150 fake (42%).
- P1 clean: đúng 589/590 real (99,83%), đúng 264/490 fake (53,88%). Q30 lịch sử: đúng 147/150 real (98%), đúng 53/150 fake (35,33%).

Ngưỡng lịch sử tương đương yêu cầu raw logit margin fake−real khoảng **3,84964**. Cả hai giữ real rất cao nhưng bỏ sót nhiều fake tại operating point này. Đây là dấu hiệu cần kiểm tra mức bảo thủ của điểm/ngưỡng, **chưa chứng minh chỉ cần hạ ngưỡng là giải quyết được vấn đề**. AUC và phân bố điểm từng ảnh còn thiếu; hạ ngưỡng có thể tăng báo oan.

Cả hai nhánh vẫn cải thiện worst-group từ epoch 4 sang 5: P0 37,33%→42%; P1 34%→35,33%. Điều này không đủ để bảo đảm tăng epoch sẽ cải thiện benchmark. Chưa có bằng chứng để chọn tăng epoch, giảm guard/KD hoặc bổ sung dữ liệu làm cách sửa chính thức.

Ở bước chuẩn bị, đã quét các đường dẫn ảnh v2 ở local: **0/14.400 ảnh hiện diện**, gói log pilot không lưu logits validation. Run mới đã cung cấp logits/IDs để hậu kiểm AUC và lỗi theo nhóm; vẫn thiếu ảnh gốc ở local để xem ngữ nghĩa nhãn. Các file `preparation/` phản ánh trạng thái trước run, không dùng thay index/kết quả mới.

## 3. Phép đo đã chuẩn bị

Run mới đã thực thi [notebook](../../../../../notebooks/task/phase7_validation_diagnostics/phase7_validation_diagnostics_colab.ipynb), ba cell nguồn `0dabd72`, L4/FP16/batch16, cell đo khoảng 11,914 phút. [Hướng dẫn](../../../../../notebooks/task/phase7_validation_diagnostics/README.md), [plan nguồn](../../../../../configs/phase7_validation_diagnostics/plan.json) và [collector](../../../../../tools/phase7_validation_diagnostics.py) giữ quy trình tái lập, không yêu cầu chạy lại y nguyên. [Kế hoạch tổng hợp](../../../../../docs/plan/ke_hoach_cai_thien_phase7_sau_vong1.md) và [báo cáo](../../../../../docs/bao_cao/C/bao_cao_phase7_improvement_round1_va_chan_doan_vong2.md) đã ghi kết quả mới; raw và notebook giữ nguyên.

Đo P0 và incumbent trên cùng 1.080 ảnh `val_select`, clean và Q30 sau crop, cùng FP16/batch/GPU. Xuất arrays có danh tính/hashes, recall từng family/generator, AUC/AP, phân bố điểm và danh sách ảnh lỗi. Giữ riêng Q30 lịch sử 300 ảnh và operating point cũ để đối chiếu. Không đổi split, hiệu chuẩn lại hoặc đọc ảnh benchmark test. Probe bộ nhớ có thể dùng tensor tổng hợp; tensor này chỉ phục vụ kiểm tra dung lượng, không đưa vào số liệu chất lượng.

41 test đạt: 18 test mới cho frozen inputs, split isolation, JPEG parity, batch boundary, phân biệt ranking/ngưỡng, chống nhận nhầm/sửa arrays, phân tích tích hợp và luồng thu thập thành công/thất bại; 23 test workflow Git/evaluation liên quan. Hai test luồng thu thập dùng model/tensor CPU giả lập giao diện CUDA, không chạy CUDA thật. Xem [`preparation/validation/local_tests.xml`](preparation/validation/local_tests.xml). Đây là kiểm thử CPU/code, không phải bằng chứng forward model thật trên GPU. Snapshot [`preparation/preservation.json`](preparation/preservation.json) bảo vệ 337 file notebook/kết quả cũ; không sửa notebook đã chạy.

## 4. Quyết định sau khi có dữ liệu

[`preparation/next_steps.json`](preparation/next_steps.json) là kế hoạch trước run. Kế hoạch hậu kiểm hiện hành nằm ở [`post_diagnosis_plan.json`](runs/val_diag_20261110_142828/analysis/post_diagnosis_plan.json): ưu tiên so sánh calibration độc lập với weights cố định; robustness Q30 là giả thuyết riêng. Chưa train hoặc fit calibration mới. Các nhánh cân nhắc ban đầu được giữ để truy vết:

1. Nếu AUC/ranking còn tốt nhưng recall tại ngưỡng lịch sử thấp: kiểm tra chuyển miền calibration/operating point. Một thí nghiệm calibration phải fit trên `val_calib` độc lập; không lấy ngưỡng từ `val_select` để xuất deployment. So sánh FPR và recall, không chỉ tối đa recall.
2. Nếu lỗi/xếp hạng yếu tập trung ở family/generator: kiểm tra nhãn bằng ảnh thật và độ phủ train trước. Nếu bổ sung dữ liệu, phải audit isolation/denylist lại, không lấy ảnh test vào train hoặc mặc định lấy generator test làm mục tiêu.
3. Nếu clean ổn nhưng Q30 giảm ranking rõ: thử một thay đổi về robustness/degradation, giữ các yếu tố còn lại như control. Không đồng thời đổi loss, dữ liệu, preprocessing và epoch.
4. Nếu chưa tách được nguyên nhân: bổ sung bằng chứng nhỏ có mục tiêu trước khi train dài. Một lần forward chẩn đoán không cho phép kết luận nhân quả.

Thí nghiệm tiếp cần control phù hợp, ngân sách/seed và tiêu chí chọn theo validation chốt trước. Test CFE đã được xem, nên chỉ dùng để báo cáo hậu kiểm sau khi khóa model; kết luận tổng quát mạnh hơn cần holdout chưa dùng điều chỉnh. G7.3 vẫn cần xây dựng tham chiếu tốt nhất từng điều kiện có provenance; đây là công việc độc lập với chẩn đoán train/validation.
