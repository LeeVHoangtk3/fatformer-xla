# BÁO CÁO KẾT QUẢ THỰC NGHIỆM: TASK 7.12
## Ablation Guard / Feature trên dữ liệu v2 — seed 42

**Bổ sung 10/11/2026:** pilot preprocessing P0/P1 sau đó là thí nghiệm hai nhánh khác, không thay ablation A1–A4 trong báo cáo này. P0 được chọn theo validation nhưng chỉ đạt 4/8 tiêu chí G7.2 trong matched evaluation; incumbent hiện hành giữ 6/8 và G7.3 pending. Chưa có bằng chứng mới để quy kết Guard/Feature là nguyên nhân chính. Xem [báo cáo vòng cải thiện và chẩn đoán tiếp](bao_cao_phase7_improvement_round1_va_chan_doan_vong2.md).

> **Cập nhật**: 07/11/2026.
>
> **Run**: `ablation_v2_fixed_run1`, commit `7f1c5868ef7582c6d5f1f5a4abef1daef9476b75`.
>
> **Trạng thái**: `[~] Có log và số đo GPU của bốn cấu hình; chưa đóng toàn bộ nghiệm thu hậu kiểm.`
>
> **G7.2**: Chưa đạt đầy đủ theo số đo hiện có; còn chỉ số không đạt và thiếu kết quả 18 tập in-dist. Không dùng điều kiện worst-group >=50% vì bảng G7.2 hiện hành không quy định điều kiện này.
>
> **Phạm vi báo cáo**: Đối chiếu hiện vật đã lưu; không chạy lại GPU, nạp lại checkpoint hay tính lại bootstrap từ logits trong lần rà soát này.

## 1. Bằng chứng và phạm vi xác minh

Nguồn chính: [`DATASET/logs/phase7/ablation_v2_fixed_run1/`](../../../DATASET/logs/phase7/ablation_v2_fixed_run1/), gồm CSV/JSON tổng hợp, biểu đồ, fingerprints, và bốn thư mục run có `commands.log`, preflight, benchmark, config, identity, verification, history, validation và JSON evaluation.

Các đối chiếu đã thực hiện trên bản local:

- History của mỗi cấu hình có epoch 1–5, mỗi epoch 765 mini-batch. Các log chọn epoch 4 theo luật worst-group kèm `clean_r >=0.90`; A1 có cùng worst-group ở epoch 5 nhưng không vượt kỷ lục epoch 4.
- SHA256 của `train_history.csv` và `val_groups.csv` khớp giá trị trong `run_verification.json`. Đây là checksum, không phải chữ ký số.
- Config hash và source hash trong preflight khớp cấu hình và context đã lưu. Bốn context giống nhau sau khi bỏ hai trọng số Guard/Feature.
- Tám JSON CFE ghi `evidence_level=measured`, `git_dirty=false`, cùng partition evaluation 1.600 mẫu. Load report được lưu ghi canonical missing/unexpected=0/0, layout n=3, dropped=[]; SHA256 checkpoint trong load report khớp giá trị best trong verification.
- Các bảng dưới đây khớp CSV/JSON nguồn trong phạm vi làm tròn được trình bày.

**Giới hạn hậu kiểm**: Bộ hiện vật local hiện không kèm `best.pth`, `checkpoint_latest.pth` hoặc logits `.npz`. Vì vậy, chưa trực tiếp kiểm lại checksum/nội dung các tệp nhị phân, canonical load hoặc CI bootstrap. Các kết quả này được trích từ log/metadata của run, không gọi là đã tái lập độc lập.

### 1.1. Phần cứng, môi trường và thời gian

Cả bốn context và benchmark đều ghi **NVIDIA A100-SXM4-40GB**; không có bằng chứng sử dụng bản 80GB. Phần mềm được ghi: PyTorch `2.14.1+cu130`, CUDA `13.0`, Python `3.13.16`, torchvision `0.29.1+cu130`.

| Cấu hình | Benchmark (ms/mini-batch) | Peak allocated (GiB) | Peak reserved (GiB) | Tổng time_sec train+validation (phút) |
| :--- | ---: | ---: | ---: | ---: |
| A1 | 344.05 | 18.24 | 18.88 | 23.72 |
| A2 | 345.83 | 18.24 | 18.89 | 23.96 |
| A3 | 346.39 | 18.24 | 18.88 | 24.22 |
| A4 | 351.46 | 18.24 | 18.89 | 23.77 |

Benchmark là phép đo 100 mini-batch riêng, không bao gồm validation/backup; không gọi số của A1 là trung bình của cả bốn cấu hình. Cộng các `time_sec` chưa làm tròn trong history được **95.66 phút train+validation**. Theo mã nguồn, timer này dừng trước lưu/backup checkpoint mỗi epoch; không gồm toàn bộ staging, preflight, benchmark, CFE evaluation và gián đoạn giữa phiên. Chưa xác nhận tổng wall-clock toàn thí nghiệm; không dùng con số khoảng 40 phút của bản báo cáo trước.

### 1.2. Fingerprints đầu vào đã lưu

Nguồn: [`staged_inputs.json`](../../../DATASET/logs/phase7/ablation_v2_fixed_run1/staged_inputs.json).

- Train archive: `16e5952892035ea0058cc6e996be191c43f6b3af608a6c92fb0a75ebc0a78dff`.
- CFE archive: `159b47e501686295be8f0edf879fb1a3070a0cf5b85c8157b0abd6bd5744f059`.
- Calibration JSON: `4ba4cf620b5b85c2534591be9391a70853845d4c0defedabee8c3aac59be8854`.
- Calibration split: `128176bd4f914ce188be05ef964e05a81802cbf41b140bc99200a4d59646b70c` (400 mẫu theo artifact gốc).
- Evaluation split: `ed5af91047320147517c7a2a1283e242aaecb0cf9a91351a469902158b8bc4e9` (1.600 mẫu theo metadata).

Preflight của từng cấu hình ghi đủ 8 PASS, gồm ảnh staged và kiểm toán cách ly. Đây là kết quả kiểm tra được lưu của run; không suy rộng thành bảo đảm mọi dạng rò rỉ đã được loại trừ.

## 2. Thiết kế so sánh

Theo cấu hình và context: cùng dữ liệu v2 (12.240 train, 1.080 val_select), seed 42, baseline khởi tạo và teacher, batch 16, accumulation 2, AMP, 5 epoch, cùng hai LR `1e-4`/`1e-5`. Canonical report ghi ba vị trí adapter `{7,15,23}`; model bật SRM/gating. Các YAML dùng `loss.mode: v2` (CLI `loss_mode=v2`) cho cả A1.

Công thức với các thành phần **chưa nhân trọng số**:

`L = 0.3 CE_clean + 0.3 CE_deg + 0.7 KD_raw + 0.2 Cons_raw + w_feat Feat_raw + w_guard Guard_raw`.

Trong CSV, `loss_kd` và `loss_cons` đã nhân trọng số; `loss_feat`/`loss_guard` chưa nhân. A1 vẫn có CE, KD và Consistency; gọi A1 là **đối chứng không Guard/Feature**, tránh nhầm với checkpoint baseline gốc hoặc “Data only”.

| Cấu hình | Tên run | w_feat | w_guard |
| :--- | :--- | ---: | ---: |
| A1 | `a1_data` | 0.0 | 0.0 |
| A2 | `a2_data_guard` | 0.0 | 0.5 |
| A3 | `a3_data_featcons` | 0.3 | 0.0 |
| A4 | `a4_full` | 0.3 | 0.5 |

Thiết kế cho phép so sánh các cấu hình loss trong điều kiện đã chạy. Cả bốn cùng dùng dữ liệu v2 nên không tách được đóng góp riêng của dữ liệu mới. Chỉ một seed huấn luyện chưa đủ để suy rộng kết luận qua nhiều lần train.

## 3. Kết quả và định nghĩa phép đo

### 3.1. Validation dùng chọn checkpoint

Nguồn: `val_groups.csv` và `train_history.csv` của từng run. Mỗi epoch có 1.080 mẫu clean và cache Q30 300 mẫu (`in_memory_jpeg30_post_resize`). Recall tại threshold cố định 0.9127 và temperature 1.6402.

| Cấu hình | Epoch chọn | Worst-group (%) | Clean Real (%) | Clean Fake (%) | Q30 Real (%) | Q30 Fake (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 4 | 34.67 | 98.47 | 63.67 | 97.33 | 34.67 |
| A2 | 4 | 26.00 | 98.14 | 59.80 | 98.00 | 26.00 |
| A3 | 4 | 38.00 | 98.47 | 59.18 | 97.33 | 38.00 |
| A4 | 4 | 27.33 | 98.14 | 58.78 | 97.33 | 27.33 |

A3 cao nhất theo worst-group trong run này; nhóm thấp nhất tại các checkpoint đã chọn đều là Q30 Fake. Đây là số trên chính tập dùng selection, không thay thế đánh giá CFE độc lập. Epoch 5 không cải thiện worst-group so với epoch 4 ở bất kỳ cấu hình nào; điều đó chưa tự xác định nguyên nhân là overfitting.

### 3.2. CFE held-out: ranking, Fake@Real90 và calibration

Nguồn: [`ablation_mini_v2.json`](../../../DATASET/logs/phase7/ablation_v2_fixed_run1/ablation_mini_v2.json) và tám JSON CFE. Metadata ghi cùng 1.600 evaluation IDs, tách khỏi 400 calibration IDs, nén Q30 theo `post_resize`.

- AUC là chỉ số xếp hạng, không tính tại threshold 0.9127.
- Fake@Real90 dùng ngưỡng từ quantile 90% xác suất Fake trên các mẫu Real của tập đang đo; ngưỡng này được tính lại trong mỗi bootstrap resample. Đây là chỉ số ROC/operating point phục vụ đánh giá, không phải Fake ACC tại threshold triển khai 0.9127. Không dùng evaluation để fit ngưỡng triển khai.
- ECE dưới đây dùng xác suất với temperature baseline cố định 1.6402; chưa fit temperature riêng cho các model mới. ECE đo calibration của xác suất, không đồng nghĩa với FPR.
- **CI từng model trong bảng** lấy từ bootstrap ảnh của evaluator (1.000 resample không phân tầng theo cài đặt hiện tại). **CI các chênh lệch CFE ở mục 4** lấy từ paired stratified bootstrap riêng, dùng cùng chỉ số ảnh giữa các model. Không gộp hai phương pháp này thành một.

| Cấu hình | AUC clean (%) [CI 95%] | AUC Q30 (%) [CI 95%] | Fake@Real90 Q30 (%) [CI 95%] | ECE Q30, T cố định (%) |
| :--- | :--- | :--- | :--- | ---: |
| A1 | 89.18 [87.66, 90.61] | 79.14 [76.93, 81.47] | 41.88 [36.07, 49.11] | 14.05 |
| A2 | 89.32 [87.73, 90.82] | 79.01 [76.72, 81.30] | 42.50 [36.43, 47.44] | 10.79 |
| A3 | 88.88 [87.27, 90.32] | 78.83 [76.58, 81.00] | 42.12 [35.25, 49.20] | 12.87 |
| A4 | 88.83 [87.22, 90.32] | 79.14 [76.89, 81.37] | 42.00 [36.80, 49.02] | 11.82 |

Biểu đồ đã lưu: [`m7_ablation.png`](../../../DATASET/logs/phase7/ablation_v2_fixed_run1/m7_ablation.png).

Để tránh nhầm operating point, A4 tại threshold 0.9127 có **Clean Real 98.25%, Clean Fake 43.62%, Q30 Real 95.62%, Q30 Fake 27.25% trên CFE**. Các số này khác Fake@Real90 clean 67.50% và Q30 42.00%. Không trình bày Fake@Real90 như tỷ lệ phát hiện fake tại threshold hiện tại.

## 4. Diễn giải ablation và độ bất định

### 4.1. Feature: A3 so với A1

Worst-group validation tăng **3.33 điểm phần trăm** (34.67% lên 38.00%), nhưng Clean Fake giảm **4.49 điểm phần trăm**. Trên CFE, AUC Q30 giảm 0.31 điểm phần trăm và Fake@Real90 Q30 tăng 0.24 điểm phần trăm. CI ghép cặp của chênh lệch AUC Q30 là **[-1.70, +1.00] điểm phần trăm**, chứa 0.

Vì vậy, chỉ kết luận **A3 có worst-group validation cao nhất trong seed 42 này**; chưa chứng minh Feature giúp khái quát tốt hơn trên CFE. Thiết kế loss dùng cosine/KL và stop-gradient mô tả mục tiêu tối ưu, không chứng minh nguyên nhân của thay đổi recall. Stop-gradient ở một nhánh loss không bảo đảm biểu diễn sạch không đổi khi các loss khác vẫn cập nhật tham số chung. Chưa có phân tích biểu diễn để xác nhận cơ chế đó trong run này.

### 4.2. Guard: A2 so với A1

Worst-group validation giảm **8.67 điểm phần trăm**. Đồng thời, ECE Q30 giảm **3.26 điểm phần trăm** (14.05% xuống 10.79%), AUC clean tăng 0.14 và Fake@Real90 Q30 tăng 0.62 điểm phần trăm. Đây là các thay đổi điểm ước lượng; chưa có CI cho chênh lệch ECE trong hiện vật hiện tại.

CI ghép cặp chênh lệch AUC clean là **[-0.49, +0.75]**, AUC Q30 là **[-1.18, +0.90]**, Fake@Real90 Q30 là **[-5.63, +4.13] điểm phần trăm**. Các CI chứa 0, nên chưa đủ bằng chứng kết luận Guard cải thiện các chỉ số CFE này.

Guard phạt phần logit Fake của mẫu Real vượt logit teacher theo cài đặt. Giả thuyết mô hình trở nên bảo thủ hơn phù hợp với một số thay đổi recall, nhưng **chưa đo được việc toàn bộ phân phối logit dịch về phía âm** hoặc xác định đó là nguyên nhân giảm Q30 Fake. Không đồng nhất ECE thấp hơn với FPR thấp hơn, và không kết luận Guard có lợi hoặc có hại trên mọi mặt.

### 4.3. Hai tương phản khác nhau

- Tương phản 2×2 trên worst-group: `A4 - A2 - A3 + A1 = -2.00` điểm phần trăm.
- Full so với cấu hình đơn tốt nhất: `A4 - max(A2,A3) = -10.67` điểm phần trăm.
- Full so với đối chứng A1: worst-group giảm 7.33 điểm phần trăm; AUC Q30 bằng nhau 79.14%; Fake@Real90 Q30 chỉ hơn 0.12 điểm phần trăm.

Các delta worst-group trên tập selection chưa có CI độc lập trong bộ hiện vật; ghi như tương phản mô tả, không gọi là bằng chứng hiệp đồng/triệt tiêu hoặc Guard “chiếm ưu thế”. ECE Q30 A4 11.82% thấp hơn A1 nhưng vẫn cao hơn ngưỡng 8% của gate với calibration hiện tại.

### 4.4. Thống kê trên CFE

JSON ghi paired stratified bootstrap 1.000 lần, seed resampling 42. Các CI AUC Q30 của Guard, Feature và tương phản 2×2 lần lượt là **[-1.18,+0.90]**, **[-1.70,+1.00]**, **[-0.99,+2.02] điểm phần trăm**. Chưa có bằng chứng cải thiện rõ qua các tương phản này; **không suy ra bốn cấu hình tương đương** vì chưa thiết kế kiểm định tương đương với biên xác định trước.

Một kết quả khác cần báo cáo: AUC clean của full thấp hơn cấu hình đơn tốt nhất **0.49 điểm phần trăm**, CI tương phản `full_over_best_single` là **[-0.75,-0.29]**. Đây là mức giảm nhỏ trong phép đo/seed này; không suy rộng thành kết luận mọi cách kết hợp hai loss đều kém.

CI CFE chỉ phản ánh bất định do lấy mẫu ảnh, có điều kiện trên các model đã train với seed 42. Chưa có CI giữa các seed huấn luyện, chưa có CI độc lập cho worst-group selection và chưa tái tính các CI từ NPZ trong lần hậu kiểm local này. Không suy p-value cụ thể từ các khoảng được báo cáo.

## 5. Đối chiếu G7.2 theo kế hoạch hiện hành

Nguồn yêu cầu: [Kế hoạch Phase 7, §II.3](../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md). **Worst-group trên val_select không phải worst-subset Fake ACC trên CFE.** Bảng gate không đặt điều kiện worst-group >=50%; bỏ căn cứ đó của bản báo cáo trước.

Đối chiếu dưới đây dùng **A4**, không ghép chỉ số tốt nhất của nhiều cấu hình thành một model:

| Điều kiện Must | Yêu cầu | Số đo A4 đã lưu | Đánh giá trong phạm vi hiện có |
| :--- | :--- | :--- | :--- |
| CFE clean Fake@Real90 | >=65% | 67.50%, CI [61.86,72.06] | Điểm ước lượng đạt; CI đi qua ngưỡng |
| CFE clean AUC | >=78% | 88.83%, CI [87.22,90.32] | Đạt theo số đo đã lưu |
| CFE Q30 AUC | >=60% | 79.14%, CI [76.89,81.37] | Đạt theo số đo đã lưu |
| CFE Q30 Fake@Real90 | >=40% | 42.00%, CI [36.80,49.02] | Điểm ước lượng đạt; CI đi qua ngưỡng |
| 18 tập in-dist Macro ACC | >=90% | Chưa có kết quả checkpoint mới trong bộ hiện vật | Chưa đánh giá |
| 18 tập in-dist Real ACC | >=90% | Chưa có kết quả checkpoint mới trong bộ hiện vật | Chưa đánh giá |
| Worst-subset Fake ACC CFE | >=45% | A4 autoregressive: clean 24.69%, Q30 7.41% tại threshold 0.9127 | Không đạt ở operating point hiện tại |
| ECE | <=0.08, tức <=8% | Clean 9.27%, Q30 11.82%, dùng T baseline | Không đạt với calibration hiện tại |

Việc CI đi qua ngưỡng không tự thêm một tiêu chí gate mới; cần phân biệt điểm ước lượng đạt với mức chắc chắn của kết quả. Không có cấu hình nào đủ bằng chứng trong bộ hiện vật này để công bố đạt toàn bộ G7.2. A2 có ECE clean 7.86% nhưng ECE Q30 vẫn 10.79%; không dùng riêng một điều kiện thuận lợi để tuyên bố toàn gate đạt.

**Kết luận G7.2: chưa đạt đầy đủ theo số đo hiện có, chưa có đánh giá khóa toàn bộ gate.** Hiệu chuẩn riêng cho checkpoint mới, nếu thực hiện, phải dùng calibration độc lập và lưu thành artifact mới; không fit trên evaluation để ép đạt ECE hoặc recall. Kết quả âm vẫn có giá trị, nhưng không thay thế các hạng mục nghiệm thu còn thiếu.

## 6. Giới hạn kỹ thuật và phần cần bổ sung

Commit chạy dùng AMP GradScaler `init_scale=2048.0`. Trong [mã trainer](../../../src/training/train_crossdomain.py), `grad_norm` chỉ lấy trung bình các norm hữu hạn; nếu không có norm hữu hạn thì ghi 0.0. Khi gặp gradient không hữu hạn dưới AMP, code cho GradScaler xử lý bỏ qua cập nhật/điều chỉnh scale, và chỉ dừng theo điều kiện scale đã cài đặt. Do đó:

- Các `grad_norm` trong CSV hữu hạn **không chứng minh mọi bước đều có gradient hữu hạn**.
- Chưa có bộ đếm số bước optimizer thực sự cập nhật/bị bỏ qua hay lịch sử scale đủ chi tiết. Cùng 765 mini-batch/epoch chưa tự chứng minh cùng số cập nhật optimizer thực tế giữa các run.
- Đây là giới hạn ghi log, không phải bằng chứng đã có số bước bị bỏ qua hoặc run đã hỏng. Cần bổ sung đo đạc nếu muốn đánh giá độ ổn định và ngân sách cập nhật chính xác.

Để hậu kiểm đầy đủ cần lấy checkpoint và logits NPZ tương ứng trên Drive, đối chiếu hash, nạp canonical và tái tính các chỉ số/CI. Không sửa CSV, fingerprints hoặc chọn thêm kết quả thuận lợi để khớp kết luận. Đặc tả [Task 7.12](../../tasks/member_c_training_eval/phase_7/task_7.12_ablation_mini_v2.md) và báo cáo này phải phân biệt các hạng mục đã có log với các bước đã được kiểm chứng độc lập.

## 7. Kết luận và hiện vật

Log ghi đủ bốn cấu hình, mỗi cấu hình 5 epoch. **A3 tốt nhất theo worst-group validation trong run seed 42; chưa chứng minh Guard/Feature cải thiện tổng quát trên CFE, và A4 không vượt trội.** Thí nghiệm cung cấp kết quả ablation có giá trị, bao gồm các đánh đổi và kết quả âm. Chưa kết luận đã xác định cơ chế gây ra thay đổi recall, chưa đóng toàn bộ hậu kiểm và chưa công bố G7.2 đạt.

Hiện vật chính: [CSV](../../../DATASET/logs/phase7/ablation_v2_fixed_run1/ablation_mini_v2.csv), [JSON/CI](../../../DATASET/logs/phase7/ablation_v2_fixed_run1/ablation_mini_v2.json), [biểu đồ](../../../DATASET/logs/phase7/ablation_v2_fixed_run1/m7_ablation.png), [commit](../../../DATASET/logs/phase7/ablation_v2_fixed_run1/source_commit.txt), [fingerprints](../../../DATASET/logs/phase7/ablation_v2_fixed_run1/staged_inputs.json). Chi tiết từng run: [A1](../../../DATASET/logs/phase7/ablation_v2_fixed_run1/a1_data/), [A2](../../../DATASET/logs/phase7/ablation_v2_fixed_run1/a2_data_guard/), [A3](../../../DATASET/logs/phase7/ablation_v2_fixed_run1/a3_data_featcons/), [A4](../../../DATASET/logs/phase7/ablation_v2_fixed_run1/a4_full/).
