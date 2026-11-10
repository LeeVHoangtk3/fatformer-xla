# Quy trình thực nghiệm và phạm vi kiểm chứng

Chốt ngày **10/11/2026**. Phần này dùng để viết phương pháp, lịch sử thực nghiệm và giải thích độ tin cậy. Số liệu chất lượng tập trung ở [file 02](02_bang_so_lieu.md).

## 1. Các giai đoạn đã hoàn tất

| Giai đoạn | Run hoặc hồ sơ | Code thực nghiệm | Sản phẩm và quyết định |
| :--- | :--- | :--- | :--- |
| Mốc đối chứng Task 7.15 | `corrected_20261109_102954_3d28ad` | Bundle nguồn được lưu trong hồ sơ run | Package `24869052…`, calibration GPU và benchmark; giữ làm bản chuẩn |
| Task 7.16 | `DATASET/logs/phase7/task_7.16/` | Công cụ đồng bộ tài liệu | Hồ sơ tài liệu; chữ ký/cross-review còn chờ |
| Pilot train vòng 1 | `pilot_20261109_182359` | `2665014` | P0 và P1 ×5 epoch; chọn P0 theo validation |
| Matched evaluation | `p0_eval_20261110_065901` | `977b282` | P0 4/8, đối chứng 6/8 G7.2; giữ incumbent |
| Validation diagnostics | `val_diag_20261110_142828` | `0dabd72` | Báo cáo nhóm/lỗi và đính chính diễn giải; không train |
| Calibration transfer | `calib_transfer_20261110_155948` | `253942a` | Hai calibration mới đều trượt Real guard Q30; không thay deployment |

Các run ID là định danh trong log, không tự suy giờ địa phương từ tên. Ngày chốt báo cáo dùng Asia/Saigon. Source code thực nghiệm khác checkpoint SHA và package SHA.

Nguồn theo từng run: [danh mục nguồn](05_danh_muc_nguon.md). Kế hoạch nền và chi tiết triển khai: [kế hoạch sau vòng 1](../../plan/ke_hoach_cai_thien_phase7_sau_vong1.md), [kế hoạch calibration transfer](../../plan/ke_hoach_calibration_transfer_phase7_round2.md).

## 2. Dữ liệu và vai trò của từng split

| Dữ liệu | Quy mô | Vai trò thực tế |
| :--- | :--- | :--- |
| CFE calibration | 400, 200 Real/200 Fake | Fit T/ngưỡng cho từng model khi đóng gói/đánh giá CFE |
| CFE evaluation | 1.600, 800 Real/800 Fake ×7 điều kiện | Chấm các chỉ số và gate; đã được xem, không dùng chọn lại tham số |
| In-dist | 82.003 lượt ảnh/18 tập mỗi model | Chấm macro, pooled và từng subset; không khẳng định mọi bản ghi là nội dung ảnh độc lập |
| v2 val_select clean | 1.080, 590 Real/490 Fake | Chọn pilot và chẩn đoán; đã được sử dụng nhiều lần |
| v2 val_select Q30 pilot | 300, 150 Real/150 Fake | Lát lịch sử để chọn pilot, biến đổi từ ảnh validation |
| v2 val_select Q30 diagnostics/transfer | 1.080, cùng IDs clean | Chẩn đoán đầy đủ ở JPEG Q30 sau resize/crop |
| v2 val_calib clean | 1.080, 590 Real/490 Fake | Fit calibration transfer; không trùng SHA với val_select |

Manifest v2 có 14.400 ảnh và 17 generator fake. Trong dữ liệu hiện có, generator của val_select đều đã xuất hiện ở train. Autoregressive, commercial và pixel diffusion mỗi họ chỉ có một generator; độ phủ hạn chế. Hash/metadata nhất quán chưa thay thế kiểm tra ngữ nghĩa nhãn bằng ảnh gốc.

Nguồn: [diagnostics và phân tích](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/README.md), [báo cáo trước đây](../C/bao_cao_phase7_improvement_round1_va_chan_doan_vong2.md), [manifest v2](../../../DATASET/manifests/data_manifest_v2.json).

## 3. Kiểm soát các phép so sánh

### 3.1. Pilot preprocessing

P0/P1 dùng cùng dữ liệu, teacher/khởi tạo, loss, seed42, 5 epoch và effective batch32; chỉ preprocessing train khác. Cả hai validation bằng CLIP. Số optimizer update thực khác một lần do AMP skip: P0 1.914, P1 1.913 trên 1.915 lượt thử mỗi nhánh. Bằng chứng trace cho phép nói các update có norm không hữu hạn đã bị bỏ qua trong run này; không suy ngược mọi NaN ở run cũ đều có cùng nguyên nhân.

### 3.2. Matched evaluation

P0 và incumbent được fit calibration riêng nhưng dùng chung split, preprocessing, L4/FP16/batch16 và TF32 tắt. Đối chứng là package refit từ weights incumbent. Arrays dự đoán refit trùng hoàn toàn arrays incumbent cũ trên CFE/in-dist, sai số xác suất tối đa 0; đây là tái lập dự đoán đã lưu, không phải băm lại hai file weights tại local.

### 3.3. Diagnostics và calibration transfer

Diagnostics dùng điểm lịch sử chung T1,6402/ngưỡng0,9127 để nhìn lỗi; không lấy điểm đó thay calibration của package. Calibration transfer so control CFE với fit clean-v2 trên cùng raw validation logits của từng model. Run thực tế dùng `reuse_frozen_logits`: bốn raw logits arrays và IDs bằng đúng bit với reference đã lưu. Không xem việc tái sử dụng arrays là một lần đo backbone validation mới.

Fit clean-v2 thay đồng thời miền calibration, số mẫu và tỷ lệ lớp so với CFE calibration. Thí nghiệm không tách được tác động nhân quả riêng của mỗi yếu tố. Temperature scaling dương và thay ngưỡng không cải thiện thứ tự raw margin; score AUC float32 có thể thay đổi rất nhỏ do ties.

## 4. Hậu kiểm đã thực hiện

| Run | File raw đã nhập | Phạm vi đối chiếu được ghi nhận |
| :--- | ---: | :--- |
| Pilot | 27 | 146 source hashes; 3.830 record optimizer trace; 28.832 phép đối chiếu dữ liệu |
| Matched evaluation | 90 | 155 source files; 186.406 lượt dự đoán; 2.446 phép kiểm tra dữ liệu; paired bootstrap tái lập |
| Diagnostics | 16 | 159 source hashes; 4.320 lượt dự đoán; 112 dòng group metrics; 933 dòng lỗi |
| Calibration transfer | 56 | 170 source hashes; 2.160 calibration records mới; 4.320 frozen validation records; 8.640 scored decisions; 224 dòng nhóm; 1.471 dòng lỗi |

Các số trên đo những đối tượng khác nhau. Không cộng thành số ảnh độc lập, số test độc lập hoặc số lần forward GPU. Diagnostics và transfer dùng lại cùng validation logits; matched evaluation có hai model trên cùng ảnh. File raw được sao giữ nguyên byte và có manifest; thư mục Markdown này không chứa bản raw thứ hai.

Nguồn: README/import manifest/verification của [pilot](../../../DATASET/logs/phase7/improvement_round1/runs/pilot_20261109_182359/README.md), [evaluation](../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/README.md), [diagnostics](../../../DATASET/logs/phase7/improvement_round2/diagnostics/runs/val_diag_20261110_142828/README.md), [transfer](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/analysis/verification.json).

## 5. Notebook và giới hạn kiểm chứng local

- Pilot: notebook thực thi có 4 code cell hoàn tất; log và lựa chọn checkpoint đã đối chiếu.
- Matched evaluation: 3 code cell hoàn tất, không có error output; trạng thái đối chiếu với 17 lệnh/step markers và run ID.
- Diagnostics: 3 code cell hoàn tất; notebook đã nộp được đối chiếu với run tương ứng.
- Calibration transfer: đã nhận gói kết quả và hậu kiểm arrays/source/metadata, nhưng notebook hiện có trong repo vẫn là template với execution counts null. **Chưa xác nhận notebook thực thi của run này.**

Không sửa source hoặc output notebook của người dùng. Các bản có output và NPZ có thể chỉ nằm ở local/Drive, không mặc định đã có trong Git. Đường dẫn chi tiết ở [file 05](05_danh_muc_nguon.md).

Gói nộp không chứa weights và full archive ảnh. Hậu kiểm local không băm lại `.pth`, không forward backbone thật hoặc xác nhận mọi nhãn bằng ảnh gốc. Báo cáo canonical load/kiểm chứng GPU được kiểm liên kết hash và nguồn; không được mô tả thành một lần chạy độc lập tại local.

## 6. Kiểm thử và khoảng tin cậy

Các bộ test đã được ghi trong hồ sơ lịch sử:

| Hồ sơ | Số case đạt được ghi nhận | Phạm vi |
| :--- | ---: | :--- |
| Task 7.16 | 156 trên 9 file | 103 pipeline +53 loss/selection/degradation; không phải toàn repo |
| Matched evaluation | 25 | Hậu kiểm và workflow liên quan |
| Diagnostics | 25 | 7 audit mới +18 validation |
| Calibration transfer preparation | 30 | 22 transfer +8 Git |
| Calibration transfer audit | 30 | 8 audit mới +22 transfer |

**Không cộng các hàng** thành tổng test độc lập: có case được dùng lại, đặc biệt 22 transfer cases. Đây là kết quả kiểm thử trong các lần trước; lần tạo hồ sơ Markdown này chỉ kiểm số liệu/đường dẫn/tính nguyên trạng, không chạy lại những bộ test trên. Nguồn XML tương ứng được dẫn trong [file 05](05_danh_muc_nguon.md).

Matched evaluation dùng paired bootstrap phân tầng nhãn 1.000 lần, seed42, cùng indices hai model; macro có phân tích 10.000 resample theo 18 tập. CI ảnh chưa bao quát đầy đủ phụ thuộc generator hoặc seed train, chưa điều chỉnh nhiều phép so sánh. Calibration transfer cũng dùng paired bootstrap 1.000 lần/seed42; CI chỉ có điều kiện trên hai fit đã có, chưa bao quát bất định của bước fit calibration. Không suy từ CI chứa 0 rằng hai model tương đương.
