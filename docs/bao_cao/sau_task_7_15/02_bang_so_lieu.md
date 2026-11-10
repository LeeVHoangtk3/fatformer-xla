# Bảng số liệu sau Task 7.15

Chốt ngày **10/11/2026**. Các bảng là các phép so sánh riêng; không gộp validation, CFE và calibration thành một tập kiểm thử. Giá trị làm tròn để trình bày; JSON/CSV nguồn giữ độ chính xác gốc.

## 1. Danh tính và quyết định chọn model

| Phương án | Source SHA-256, rút gọn | Package SHA-256, rút gọn | Kết quả lựa chọn |
| :--- | :--- | :--- | :--- |
| Incumbent Task 7.15 | `5cd49252…` | `24869052…` | **Giữ làm bản chuẩn** |
| P0 pilot, sau đóng gói đánh giá | `9d97613c…` | `e26a1627…` | Không thay incumbent |
| Incumbent refit vòng 1 | `5cd49252…` | `de36061f…` | Đối chứng; không thay tham chiếu package chuẩn |
| Hai điểm calibration clean-v2 | Giữ hai source trên | Không tạo package triển khai mới | Không chấp nhận vì trượt Real guard Q30 |

Nguồn: [package hiện hành](../../../DATASET/logs/phase7/task_7.15/current/artifacts/package_reference.json), [matched evaluation](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/README.md), [quyết định calibration](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/analysis/decision.json). Hash đầy đủ và vị trí Drive nằm ở [file 05](05_danh_muc_nguon.md).

## 2. Tám tiêu chí G7.2 của bản chuẩn

Đơn vị **%**. ECE nguồn lưu dạng tỷ lệ đã được nhân 100 khi trình bày.

| Tiêu chí | Incumbent hiện hành | Yêu cầu | Kết quả |
| :--- | ---: | :--- | :--- |
| CFE clean Fake@Real90 | 64,125 | ≥65 | FAIL |
| CFE clean AUC | 88,6623 | ≥78 | PASS |
| CFE Q30 AUC | 79,8791 | ≥60 | PASS |
| CFE Q30 Fake@Real90 | 42,625 | ≥40 | PASS |
| In-dist macro balanced accuracy | 90,1018 | ≥90 | PASS |
| In-dist pooled Real accuracy | 90,4063 | ≥90 | PASS |
| Worst-subset Fake accuracy, clean | 30,8642 | ≥45 | FAIL |
| ECE Q30 sau calibration | 5,0823 | ≤8 | PASS |

**G7.2: 6/8, FAIL tổng thể.** Worst subset là `5_other_autoregressive`, Fake đúng **25/81**; không phải worst-group validation của pilot. Clean Fake@Real90 thiếu **0,875 pp**, worst-subset thiếu **14,1358 pp**.

Nguồn chính xác: [gate_summary.csv](../../../DATASET/logs/phase7/task_7.15/current/analysis/gate_summary.csv). Số CFE trong bảng 3 của đối chứng refit tái lập kết quả incumbent; điều đó không làm hai package có cùng hash.

## 3. Pilot train và matched evaluation vòng 1

### 3.1. Chọn ứng viên bằng validation

Pilot: seed42, 5 epoch mỗi nhánh; clean **1.080 ảnh**, Q30 lịch sử **300 ảnh**, T1,6402/ngưỡng0,9127; tất cả recall có đơn vị %.

| Chỉ số validation | P0: train legacy | P1: train CLIP |
| :--- | ---: | ---: |
| Clean Real recall | 99,32 | 99,83 |
| Clean Fake recall | 60,41 | 53,88 |
| Q30 Real recall, 150 real | 98,00 | 98,00 |
| Q30 Fake recall, 150 fake | 42,00 | 35,33 |
| Worst-group, min bốn recall | **42,00** | 35,33 |
| Epoch được chọn | 5 | 5 |

P0 được chọn để đánh giá tiếp. P0 có **1.914/1.915** optimizer updates áp dụng, P1 **1.913/1.915**; AMP bỏ qua lần có norm không hữu hạn. Không mô tả gradient luôn hữu hạn hoặc số update thực bằng nhau.

Nguồn: [pilot đã hậu kiểm](../../../DATASET/logs/phase7/improvement_round1/runs/pilot_20261109_182359/README.md).

### 3.2. Đánh giá ứng viên và đối chứng trên cùng protocol

400 calibration (200 Real/200 Fake), 1.600 CFE evaluation (800/800) ×7 điều kiện và 82.003 lượt ảnh/18 tập in-dist, mỗi model. Đơn vị %.

| Chỉ số | P0 | Incumbent refit | Mục tiêu G7.2 |
| :--- | ---: | ---: | :--- |
| CFE clean AUC | 87,96 | 88,66 | ≥78 |
| CFE clean Fake@Real90 | 61,000 | 64,125 | ≥65 |
| CFE Q30 AUC | 79,02 | 79,88 | ≥60 |
| CFE Q30 Fake@Real90 | 38,875 | 42,625 | ≥40 |
| In-dist macro balanced accuracy | 89,6794 | 90,1018 | ≥90 |
| In-dist pooled Real accuracy | 90,36 | 90,41 | ≥90 |
| Worst-subset Fake accuracy, clean | 25,93 (21/81) | 30,86 (25/81) | ≥45 |
| ECE Q30 | 6,17 | 5,08 | ≤8 |
| Số tiêu chí đạt | **4/8** | **6/8** | Đạt đủ 8 |

Nguồn: [README hậu kiểm](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/README.md), [gates.csv](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/gates.csv), [paired verification](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/paired_verification.json). Đối chứng refit tái lập dự đoán incumbent; một số số trong gates.csv làm tròn hai chữ số nên không dùng chúng để tính chênh lệch chính xác.

## 4. Validation diagnostics tại điểm lịch sử

Cùng **1.080 ảnh val_select (590 Real/490 Fake)** ở mỗi điều kiện; Q30 sau crop, T1,6402/ngưỡng0,9127. Khác Q30 300 ảnh ở pilot. Đơn vị %; AUC là score AUC của phép đo này.

| Model | Điều kiện | Accuracy | Real recall | Fake recall | AUC |
| :--- | :--- | ---: | ---: | ---: | ---: |
| P0 | Clean | 81,6667 | 99,3220 | 60,4082 | 96,3696 |
| Incumbent | Clean | 82,3148 | 98,9831 | 62,2449 | 96,4362 |
| P0 | Q30 | 75,0000 | 98,3051 | 46,9388 | 92,1347 |
| Incumbent | Q30 | 74,6296 | 97,2881 | 47,3469 | 91,8046 |

Nguồn: [diagnostics](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/README.md). AUC gần nhau không chứng minh hai model tương đương; CI paired delta AUC clean và Q30 đều chứa 0.

## 5. Calibration transfer trên cùng validation logits

Fit chỉ trên **clean val_calib: 590 Real/490 Fake**, độc lập SHA với val_select. “Control” là calibration CFE riêng của mỗi model; “mới” là calibration clean-v2. Các điểm ở bảng này khác điểm lịch sử bảng 4.

### 5.1. Tham số và ngân sách false positive trên tập fit

| Model/điểm | T | Ngưỡng | FP clean val_calib |
| :--- | ---: | ---: | :--- |
| P0 control | 3,0949893 | 0,76518965 | Không fit lại trên val_calib |
| P0 mới | 1,9718821 | 0,37123102 | 29/590 =4,9153% |
| Incumbent control | 2,7832491 | 0,79185086 | Không fit lại trên val_calib |
| Incumbent mới | 1,7667391 | 0,50552732 | 29/590 =4,9153% |

Control được fit trên 400 CFE calibration; không trộn FPR trên tập fit đó với FPR ở validation hoặc clean val_calib. Một ngưỡng của mỗi điểm được dùng chung clean/Q30.

### 5.2. Recall, báo oan và accuracy trên val_select

Đơn vị %. Guard đã khai báo: **Real recall ≥90% ở cả hai điều kiện**.

| Model | Điều kiện | Điểm | Real recall | Fake recall | FPR | Accuracy |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| P0 | Clean | Control | 99,32 | 62,04 | 0,68 | 82,41 |
| P0 | Clean | Mới | 94,41 | 84,69 | 5,59 | 90,00 |
| P0 | Q30 | Control | 98,14 | 49,18 | 1,86 | 75,93 |
| P0 | Q30 | Mới | **86,95** | 80,00 | **13,05** | 83,80 |
| Incumbent | Clean | Control | 98,98 | 63,06 | 1,02 | 82,69 |
| Incumbent | Clean | Mới | 94,75 | 84,49 | 5,25 | 90,09 |
| Incumbent | Q30 | Control | 97,29 | 49,18 | 2,71 | 75,46 |
| Incumbent | Q30 | Mới | **87,29** | 78,78 | **12,71** | 83,43 |

Cả hai control đạt guard trên validation này; **hai điểm mới không đạt**, không dùng để thay deployment. Fit đúng FPR5 trên calibration không bảo đảm FPR5 khi chuyển miền hoặc nén ảnh.

Nguồn: [P0 summary.csv](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/raw/candidate_p0/comparison/summary.csv), [incumbent summary.csv](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/raw/incumbent/comparison/summary.csv).

### 5.3. NLL, ECE và ranking

NLL không có đơn vị; ECE là **confidence ECE argmax0,5**, đơn vị %. Raw-margin AUC không đổi giữa hai điểm cùng model/điều kiện.

| Model | Điều kiện | NLL control → mới | ECE control → mới | Raw-margin AUC (%) |
| :--- | :--- | :--- | :--- | ---: |
| P0 | Clean | 0,28764 →0,26401 | 6,92 →2,30 | 96,3691 |
| P0 | Q30 | 0,38609 →0,37197 | 6,45 →4,90 | 92,1361 |
| Incumbent | Clean | 0,27139 →0,24835 | 5,54 →2,32 | 96,4348 |
| Incumbent | Q30 | 0,38221 →0,36976 | 5,49 →3,39 | 91,8049 |

Nguồn: hai summary.csv trên. Score AUC có chênh lệch float32/ties rất nhỏ; không gọi đó là cải thiện biểu diễn. CI95 delta NLL Q30 incumbent **[−0,02522; +0,00101]** chứa 0. Các CI đầy đủ nằm trong [báo cáo calibration transfer](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/README.md).

## 6. Độ trễ: ghi đúng package và phạm vi

L4/FP16, batch1; tổng thời gian package và median overhead có đơn vị **ms/ảnh**.

| Package | Tổng inference | Overhead |
| :--- | ---: | ---: |
| Incumbent hiện hành `24869052…` | 44,98 | 0,188055 |
| P0 evaluation `e26a1627…` | 44,53 | 0,151093 |
| Incumbent refit `de36061f…` | 45,82 | 0,205143 |

Nguồn: [latency bản chuẩn](../../../DATASET/logs/phase7/task_7.15/current/analysis/latency_summary.csv), [latency matched evaluation](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/latency.csv). Không lấy số refit thay latency package cũ; không suy sang GPU khác. Overhead <5 ms không có nghĩa tổng inference <5 ms. Phép đo không bao gồm đọc ảnh/tiền xử lý.
