# Hậu kiểm run corrected_20261109_102954_3d28ad

## Kết luận công tâm

**Quy trình chạy mới đã khắc phục các thiếu sót kỹ thuật chính: calibration đúng 400 ảnh, hội tụ nhiệt độ, thống nhất FP16/CUDA/batch 16, kiểm chứng 20 ảnh dữ liệu thực, lưu dự đoán từng ảnh và đo overhead cùng hash package.** Các metric và CI95 trong báo cáo được tính lại từ dữ liệu nộp và khớp. Không phát hiện sai khác số liệu trong các phép đối chiếu đã thực hiện.

**Năng lực mô hình chưa đạt toàn bộ mục tiêu: G7.2 đạt 6/8, tổng thể FAIL. G7.3 còn pending vì thiếu đối chiếu với reference tốt nhất theo từng điều kiện.** Không diễn giải notebook chạy hết thành nghiệm thu tất cả tiêu chí. Đây là xác nhận sau hậu kiểm trên evaluation đã xem trước, không phải phép thử mù mới.

## Nguồn và phạm vi xác minh

- Nguồn: `wait/phase7_corrected_rerun/corrected_20261109_102954_3d28ad_results/`, **55 tệp, 65.361.460 byte**. Bản byte-identical ở [raw](../raw/); hash từng tệp trong [import_manifest.json](../import_manifest.json).
- Notebook: `notebooks/task/phase7_corrected/phase7_corrected_rerun_colab.ipynb`, SHA256 `311a50b171a868581e89616458c32f84f599b07b8dbb622a27ad2d83a844173a`. Cả 11 cell có execution count 1–11, không có output lỗi; nội dung source các cell khớp notebook mẫu trong ZIP. Chỉ đọc notebook, không xóa output hoặc sửa source. [Bản có output được sao lưu nguyên byte](../evidence/executed_notebook.ipynb).
- ZIP sử dụng đúng bản chuẩn bị: SHA256 `3cd63cd9f5e07d6a70348e3d23809a8216c633b0d45224e815651378923ce657`. Kiểm tra đủ hash của **140 tệp code/input**. Hash manifest `9d7e622a…` khác hash ZIP là bình thường: đó là hai tệp khác nhau.
- GPU **NVIDIA L4, 22,03 GiB**, batch được probe chọn **16**, FP16 autocast cho backbone, logits/softmax float32; TF32 tắt. Không bắt buộc T4.
- Package ghi nhận SHA256 **`248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`**, thống nhất trong dự đoán, benchmark, kiểm chứng ảnh thật, latency và acceptance. Source checkpoint ghi nhận `5cd49252…` thống nhất.
- ZIP kết quả chủ ý không chứa weights. **Không có `.pth` mới ở local để băm lại hoặc chạy forward độc lập.** JSON/log GPU báo 1.200 tensor bằng source và standalone đạt; hậu kiểm local xác minh liên kết giữa các bản ghi, các ảnh verification và dự đoán. Không giả rằng package local cũ `16606e16…` là package mới.

## Calibration đúng nguồn và đã hội tụ

Cache có **400 logits hữu hạn, 200 real + 200 fake**, khớp frozen calibration, không trùng ID với **1.600 evaluation**. ID/nhãn/đường dẫn nhất quán, hash split và cache khớp JSON/metadata/preflight. Hai split được lưu trong [evidence](../evidence/).

- T **2,7832491397857666**, ngưỡng 5% **0,7918508648872375**.
- T được fit lại từ cache và khớp; kiểm tra tối ưu NLL độc lập xác nhận nghiệm hội tụ. Không dùng evaluation để fit.
- FPR calibration **10/200 = 5%**, Real ACC **95%**, Fake ACC **58,5%**.
- ECE calibration **14,38% → 5,39%**, NLL **0,6444 → 0,4227**.

T khác bản local FP32 `2,78636` là phù hợp với thay đổi precision/batch/backend; không phải lý do sửa số cho bằng bản local. FPR 5% ở đây là thực nghiệm trên 200 real calibration, không bảo đảm FPR trên mọi miền mới đều ≤5%.

## Chất lượng CFE: phân biệt ranking và ngưỡng vận hành

Mỗi điều kiện có **1.600 ảnh, 800 real + 800 fake**; cả 7 điều kiện dùng cùng ID/nhãn/thứ tự, không lẫn calibration. Dự đoán lưu đủ logits, `prob_fake`, nhãn, threshold, ID và hash package. Nhãn tái lập đúng từ **xác suất GPU đã lưu**, threshold cố định và router fallback clean (`w_deg=0`).

- **Clean:** AUC **88,66%** [87,11; 90,16], Fake@Real90 **64,12%** [59,25; 69,77], accuracy **74,375%**, ECE **4,39%**, NLL **0,4288**.
- **Q30:** AUC **79,88%** [77,65; 82,07], Fake@Real90 **42,62%** [37,87; 47,91], accuracy **65,8125%**, ECE **5,08%**, NLL **0,5515**.
- Q50: AUC **81,56%**, accuracy **66,50%**.
- Q75: AUC **83,32%**, accuracy **66,4375%**.
- WebP40: AUC **79,01%**, accuracy **66,3125%**.
- Blur1.5: AUC **84,21%**, accuracy **71,0625%**.
- Down/up0.5: AUC **86,99%**, accuracy **77,5625%**.

CI95 là bootstrap **1.000 lần, seed 42, đơn vị ảnh**, đã tính lại cho cả 7 điều kiện và pooled in-dist. Nó không mô tả toàn bộ bất định giữa generator/dataset. Clean Fake@Real90 có CI bao gồm 65%; vì vậy chỉ kết luận **điểm ước lượng không đạt cổng đã định**, không khẳng định đã chứng minh hiệu năng tổng thể thấp hơn 65% với ý nghĩa thống kê.

Fake@Real90 dùng quy ước mốc Real90 trên tập đánh giá để mô tả khả năng phân biệt; **khác Fake ACC tại ngưỡng triển khai cố định**. Với ngưỡng 0,79185086:

- Clean: **35/800 false positive = 4,375%**, Fake ACC **53,125%**.
- Q30: **69/800 = 8,625%**, Fake ACC **40,25%**.
- Q50: FPR **7,125%**; Q75 **6%**; WebP40 **11%**; Blur **11,875%**; down/up **12,125%**.

Vì vậy chưa có cơ sở tuyên bố mô hình giữ FPR ≤5% trên tất cả suy biến. Bảng đầy đủ ở [condition_summary.csv](condition_summary.csv), [per_subset_cfe.csv](per_subset_cfe.csv).

## In-dist: đạt điểm gộp, nhưng mức bảo vệ real không đồng đều

Đã đối chiếu đủ **82.003 bản ghi trên 18 tập**, số ảnh mỗi tập đúng hợp đồng, ID đầy đủ/đường dẫn không trùng trong định nghĩa này. Tính lại metric pooled và từng tập từ dự đoán thực:

- Macro ACC, tức trung bình **18 class-balanced accuracies**: **90,101847%**.
- Pooled accuracy: **93,464873%**; pooled AUC **98,41%**, CI95 [98,34; 98,48].
- Pooled Real ACC **90,406282%**; Fake ACC **96,524136%**.
- Có **3.934 false positive / 41.006 lượt real**, tương ứng FPR **9,593718%**.
- Mean-subset Real ACC khoảng **85,79%**; nhiều tập diffusion có Real ACC **75,8%**. Cổng hiện tại định nghĩa Real ACC theo pooled nên vẫn PASS; không được diễn giải thành mọi tập đều đạt 90%.

Macro chỉ vượt mốc 90% khoảng **0,10 pp**. Bootstrap chẩn đoán theo 18 subset cho CI khoảng **[86,76%; 93,32%]**, không phải quy tắc nghiệm thu mới và không dùng để tự đổi PASS thành FAIL. Nó cho thấy không nên gọi mức vượt mốc này là bền vững trên mọi miền.

Một số tập benchmark có dấu hiệu dùng chung real reference (tên tương đối giống nhau, có trường hợp logits bằng nhau). Chưa có đủ ảnh gốc để kiểm chứng trùng nội dung toàn bộ. Do đó **82.003 là số lượt/bản ghi đánh giá, không khẳng định là 82.003 nội dung ảnh độc lập**. CI theo ảnh có thể chưa phản ánh sự phụ thuộc đó. Chi tiết ở [distribution_cautions.json](distribution_cautions.json), [per_subset_indist.csv](per_subset_indist.csv).

## Phán quyết G7.2: 6/8, FAIL tổng thể

Sáu tiêu chí đạt: Clean AUC; Q30 AUC; Q30 Fake@Real90; in-dist Macro ACC; pooled Real ACC; ECE Q30.

Hai tiêu chí chưa đạt:

- Clean Fake@Real90 **64,12% <65%**, thiếu khoảng **0,88 pp** theo số công bố.
- Worst clean subset `5_other_autoregressive`: Fake ACC **25/81 =30,8642% <45%**, thiếu khoảng **14,14 pp**. Real ACC của nhóm này cũng chỉ **86,49%** tại ngưỡng cố định.

Điểm yếu lớn hơn nằm ở nhóm khó, không chỉ ở chênh lệch nhỏ của mốc Clean Fake@Real90. Không hạ ngưỡng dựa vào evaluation hoặc đổi định nghĩa gate để ép đạt. [gate_summary.csv](gate_summary.csv) chứa đủ 8 giá trị tái tính.

So run 7.14 cũ, tiêu chí ECE Q30 chuyển từ không đạt (13,49%) sang đạt (5,08%); AUC/Fake@Real90 nhìn chung gần như giữ nguyên. Đó là phù hợp với temperature scaling chủ yếu sửa độ tin cậy, không huấn luyện lại khả năng phân biệt. In-dist cũ dùng stretch 224×224, run mới dùng CLIP resize cạnh ngắn/bicubic/center crop; không gán mọi khác biệt in-dist cho calibration hoặc kết luận không có quên mô hình chỉ từ hai con số macro.

## Kiểm chứng đóng gói và latency

Đã đọc/verify đủ **20 PNG thực, 10 real +10 fake**, hai ảnh mỗi lớp/mỗi CFE subset. Tất cả thuộc calibration, không thuộc evaluation, và 20 file có hash nội dung khác nhau. JSON GPU báo đối chiếu đủ 1.200 tensor và xác suất/nhãn reference cùng calibration, sai số tối đa **0**. Đây là kiểm chứng tương đương đóng gói, không phải accuracy test 20 ảnh hay so sánh với model tốt nhất.

Latency đã tính lại từ **200 cặp đo/batch**, **20 warmup**, cùng package/input/precision, luân phiên thứ tự backbone/package:

- Batch 1: package **44,98 ms/ảnh**, overhead median **0,188055 ms/ảnh**, p95 overhead **2,481857 ms/ảnh**, **22,2 FPS**.
- Batch 8: **9,60 ms/ảnh**, overhead **0,059618 ms/ảnh**, khoảng **104,2 FPS**.
- Batch 16: **9,61 ms/ảnh**, overhead **−0,003034 ms/ảnh**, khoảng **104,1 FPS**.
- Batch 32: **10,38 ms/ảnh**, overhead **0,019385 ms/ảnh**, khoảng **96,4 FPS**.

**Overhead batch 1 <5 ms đạt trên L4/FP16.** 44,98 ms là thời gian package, không phải overhead. Median từng cặp chênh lệch không nhất thiết bằng hiệu hai median; do đó 0,188 ms không mâu thuẫn với hai median package/backbone đã làm tròn. Delta âm nhỏ ở batch 16 là nhiễu đo, không chứng minh wrapper làm backbone nhanh hơn.

Đo bằng tensor tổng hợp, không tính đọc file/tiền xử lý, không suy diễn sang GPU khác. Xem [latency_summary.csv](latency_summary.csv).

## G7.3 còn thiếu gì, có cần chạy lại không?

Phần một package/một backbone forward có bằng chứng từ code/instrumentation; kiểm chứng ảnh thật và overhead đã có. **Còn thiếu đối chiếu package với reference tốt nhất theo từng điều kiện trên cùng partition và protocol, bảo đảm delta AUC/Fake@Real90 ≤1,5 pp.** Tương đương source/package không thay thế phép so sánh đó. G7.3 giữ **pending**, đúng với `acceptance.json`.

`package/metadata.json` ghi trạng thái tại lúc đóng gói, còn `benchmark/final_report.json` được tạo trước bước latency nên phần overhead trong đó vẫn để trống. Phán quyết sau cùng của run nằm ở `acceptance.json`, đã ràng buộc hash cả ba báo cáo. Không sửa bằng chứng gốc chỉ để các trạng thái theo thời điểm khác nhau có cùng dòng chữ.

Không có lý do kỹ thuật từ hậu kiểm này để chạy lại y nguyên cả notebook chỉ để hy vọng metric tăng. Có thể dùng run này làm kết quả thực nghiệm và công bố trung thực phần chưa đạt. Nếu cần hoàn tất G7.3, bổ sung reference có bằng chứng phù hợp. Nếu cần cải thiện hai tiêu chí G7.2, cần thiết kế thí nghiệm dữ liệu/huấn luyện trên train và val_select; không fit lại trên evaluation đã xem.

## Lưu ý khi tái lập và triển khai

- Đầu ra gốc dùng `phase7_corrected_benchmark/v1` và `phase7_corrected_acceptance/v1`, khác tên/schema v3 trong mẫu 7.14 cũ. Không đổi nhãn schema của tệp gốc để giả đã đáp ứng đúng mẫu. Run cung cấp bảng theo subset; chưa có bảng per-generator được xác minh bằng nhãn generator chuẩn, nên không tự suy loại generator từ tên file để bổ sung số liệu.
- Kiểm tra CPU softmax từ logits khác xác suất GPU tối đa khoảng **3×10⁻⁷**. Có **1 mẫu Q50 và 5 mẫu in-dist** đổi nhãn khi tự tính lại trên CPU vì nằm sát ngưỡng. Dự đoán GPU lưu sẵn nhất quán với nhãn của chính nó; không coi đây là tệp lỗi và không thay bằng nhãn CPU. Danh sách ở [cpu_gpu_boundary_cases.csv](cpu_gpu_boundary_cases.csv).
- Chất lượng đã xác minh theo FP16/CUDA/batch 16. CLI ảnh thông thường trong code nộp mặc định chạy FP32, còn latency GPU dùng FP16; các kênh đó cần được kiểm chứng theo protocol triển khai nếu đem dùng thực tế. Không gắn accuracy FP16/batch16 sang mọi cách gọi hoặc backend khác.
- Bản model đang ở `MyDrive/Fatformer/checkpoint/phase7/corrected_20261109_102954_3d28ad/unified_fatformer_v2.pth`. Muốn kiểm chứng lại bytes/forward độc lập ở local cần đúng file này, không dùng package của run trước.

Hậu kiểm này không chỉnh notebook, source kết quả hoặc huấn luyện model. Mọi phép tính local là tái tính trên dữ liệu đã nộp; không tạo thêm số đo GPU.
