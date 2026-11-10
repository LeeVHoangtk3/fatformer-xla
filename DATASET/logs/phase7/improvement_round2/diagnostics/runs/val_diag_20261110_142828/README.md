# Hậu kiểm validation diagnostics — 10/11/2026

**Run hoàn tất và các số liệu gộp được xác nhận; phần diễn giải nguyên nhân cần sửa.** Có cơ sở thiết kế thí nghiệm calibration transfer và robustness Q30, chưa chứng minh nguyên nhân P0 thua incumbent trên CFE hoặc loại trừ hạn chế biểu diễn. Package báo cáo vẫn là `24869052…`, G7.2 FAIL/đạt 6/8 và G7.3 pending. Không có train mới hoặc thay deployment trong run này.

## 1. Bằng chứng và vị trí file

- Nguồn nhận: `wait/validation_diagnostics_without_weights/`, **16 file, 607.637 byte**.
- [`raw/`](raw/): sao nguyên byte toàn bộ kết quả; [`import_manifest.json`](import_manifest.json) ghi hash từng file.
- [`evidence/phase7_validation_diagnostics_colab.ipynb`](evidence/phase7_validation_diagnostics_colab.ipynb): sao nguyên byte notebook đã chạy. SHA-256 `4fd74f6e5f70e9f4b9ce032087ea67b9bcf34a9eb08e5093c6e878517b20f359`. Không sửa notebook gốc hoặc xóa outputs.
- [`analysis/verification.json`](analysis/verification.json): source commit `0dabd725e5b7a351514a9b7ee34f4bd6014b584f`, **159 source hashes** khớp Git, toàn bộ hashes trong status khớp. Plan xuất ra có định dạng JSON khác file config, nhưng nội dung và hash config nguồn được kiểm đúng; không nhầm hai hash này.
- Notebook ba code cell thực thi 1–2–3, đều `ok`, không error output, cùng run ID; cell đo mất khoảng **11,914 phút**. GPU **L4**, FP16, batch chung **16**, TF32 tắt. Hai canonical loads đều 0 missing/0 unexpected, 1.200 keys, n3/[7,15,23].
- Tái tính từ **4 arrays ×1.080 = 4.320 lượt dự đoán**, đúng IDs/hash ảnh thuộc `val_select`; **112 dòng group metrics và 933 dòng lỗi** khớp. Báo cáo gộp/historical/matched correctness tái lập. Bản tái tính: [`analysis/recomputed/`](analysis/recomputed/).

Arrays NPZ và bản sao notebook thực thi được giữ local/Drive; không mặc định là đã đưa vào Git. Báo cáo và text metadata có thể đồng bộ riêng. 25 test CPU hậu kiểm/workflow liên quan đạt (7 test audit mới +18 test validation); không cộng chồng với 41 test chuẩn bị lịch sử như một full-suite mới. [Kết quả test](analysis/validation/local_tests.xml).

Local không có weights/full archive ảnh, nên không forward lại model thật, không băm lại `.pth` hoặc xem ảnh để xác nhận nhãn ngữ nghĩa. Canonical/GPU bằng chứng là báo cáo được ràng buộc với source/arrays, không phải một lần chạy model độc lập ở local. Tensor probe chỉ kiểm bộ nhớ, không cung cấp chỉ số chất lượng.

## 2. Bảng gộp của bản tóm tắt đúng

Trên 590 real và 490 fake, cùng T=1,6402/ngưỡng=0,9127 lịch sử:

- P0 clean: accuracy **81,6667%**, Real **99,3220%**, Fake **60,4082%**, AUC **96,3696%**. Incumbent: **82,3148% / 98,9831% / 62,2449% / 96,4362%**.
- P0 Q30 post-crop: accuracy **75%**, Real **98,3051%**, Fake **46,9388%**, AUC **92,1347%**. Incumbent: **74,6296% / 97,2881% / 47,3469% / 91,8046%**.
- P0 clean FP/TP **4/296**, Q30 **10/230**; incumbent clean **6/305**, Q30 **16/232**. Recall/fixed-threshold accuracy khác AUC và Fake@Real90.

Điểm AUC gần nhau chưa chứng minh tương đương thống kê. Paired bootstrap 1.000 lần/seed42, phân tầng nhãn, P0−incumbent: clean AUC delta −0,0666 pp, CI95 **[−0,3976; +0,2675]**; Q30 +0,3302 pp, CI **[−0,2079; +0,8311]**. Cả hai chứa 0; chưa có bằng chứng đủ để gọi P0 tốt hơn hoặc tương đương theo một equivalence margin đã định. CI theo ảnh không bao quát đầy đủ phụ thuộc generator và một seed train.

## 3. Các kết luận cần sửa

### 3.1. Khớp bốn recall không phải bit-exact logits

Bốn recall lịch sử P0 đều chênh **0,0**: clean 586/590 real, 296/490 fake; lát Q30 lịch sử 147/150 real, 63/150 fake. Nhưng **clean dùng toàn bộ 1.080 ảnh**, chỉ Q30 dùng lát lịch sử 300 ảnh. Log cũ chỉ lưu số tổng hợp, không có logits/nhãn từng ảnh để so bit. Cách viết đúng là “tái lập chính xác bốn recall tổng hợp”, không phải bit-exact model/arrays hoặc toàn bộ phép đo trên 300 ảnh.

### 3.2. Điểm fake phân bố rộng; real không phải đều dưới 0,02

Với P0 clean, commercial/latent/pixel có AUC lần lượt **95,4906% / 94,0704% / 94,6582%** và median score **0,87045 / 0,88284 / 0,90531**, dưới ngưỡng lịch sử. Điều này hỗ trợ giả thuyết operating point bảo thủ. Tuy nhiên chỉ **15/90**, **12/131**, **10/89** score thuộc [0,80;0,90], không phải phần lớn fake tập trung trong dải đó; quantiles p10–p90 khá rộng.

Real clean có median **0,01885**, nhưng chỉ **306/590=51,86%** dưới 0,02; p90 **0,1700**. DiT riêng lẻ có clean AUC **92,0143%**, không thuộc khẳng định mọi generator nêu trên đều AUC 94–95,5%. Đây là score tại T lịch sử, chưa phải xác suất được chứng minh hiệu chuẩn cho miền validation này.

### 3.3. Số theo generator và ảnh thật

Tất cả số sau là **P0**, clean→Q30 trên cùng ảnh:

- **CIPS, n=12:** Fake recall **10/12=83,33% → 1/12=8,33%**, AUC **99,6045% → 78,6017%**. Có 9 ảnh đúng clean chuyển thành sai Q30, không có chiều ngược. Đây là suy giảm mạnh trên mẫu đã đo. CI exact recall 95% khoảng **[51,59%;97,91%] → [0,21%;38,48%]**; kiểm định paired khám phá p=0,00390625, chưa điều chỉnh nhiều so sánh. Không gọi kết quả 12 ảnh là chứng minh kiến trúc/biểu diễn nội tại “sụp đổ”.
- **Gansformer, n=23:** **23/23=100% → 17/23=73,91%**. Có suy giảm quan sát được; cần xem cùng AUC, số mẫu và các nhóm còn lại.
- **SD v1.5, n=90:** **42/90=46,67% → 30/90=33,33%**. Số 43,3% trong tóm tắt không đúng với P0 Q30; incumbent Q30 cũng là 36,67%, không phải 43,3%.
- **DiT, n=13:** **3/13=23,08% → 3/13=23,08%**. Recall đã thấp ở clean, không có sụt recall khi chuyển Q30. AUC giảm **92,0143% → 88,5072%**, nên vẫn có dấu hiệu ranking kém hơn dưới nén, nhưng không được lấy recall thấp sau nén làm bằng chứng nguyên nhân do nén.
- **COCO:** FPR **1/114=0,88% → 5/114=4,39%**. **FFHQ:** **1/114=0,88% → 2/114=1,75%**. **ImageNet replay:** **2/134=1,49% → 2/134=1,49%**, không tăng FPR dù phân bố điểm có dịch. Không kết luận mọi nguồn real đều tăng lên 2–4%.

Nguồn tái tính và CI/paired counts: [`analysis/interpretation.json`](analysis/interpretation.json).

### 3.4. Không quy kết nguyên nhân CFE từ ngưỡng chẩn đoán

Ngưỡng **0,9127/T=1,6402** là điểm chọn model lịch sử dùng chung để chẩn đoán. Khi chấm CFE vòng 1, P0 đã được fit riêng **T=3,0949893, threshold=0,76518965**; incumbent **T=2,7832491, threshold=0,79185086** trên 400 calibration CFE độc lập. Vì vậy P0 thua CFE không thể được giải thích trực tiếp là “do còn dùng 0,9127”.

Ngưỡng xác suất còn đổi ý nghĩa khi T đổi: ba raw logit margin tương đương lần lượt **3,84964**, **3,65625**, **3,71875**. Không so trực tiếp 0,9127 với 0,7652 như hai ngưỡng trên cùng thang điểm. AUC cao trên validation gồm generator đã thấy không loại trừ lỗi biểu diễn/shortcut hoặc domain shift trên CFE.

Kiểm tra phụ dùng **CPU float32 trên logits đã nộp**, giữ calibration CFE sẵn có, không fit thêm: P0 val Fake recall clean **62,0408%**, Q30 **49,1837%**, chỉ tăng khoảng **1,63/2,24 pp** so với điểm lịch sử; Real clean **99,3220%**, Q30 **98,1356%**. Đây không phải số GPU mới hoặc kết quả CFE mới. Incumbent clean có một nhãn sát biên khác giữa CPU float32/float64; giữ ghi chú sai số thay vì khẳng định mọi backend giống nhau.

## 4. Hướng tiếp theo có căn cứ

**Ưu tiên một phép đối chứng calibration độc lập trước, chưa train augmentation ngay.** Giữ weights/preprocessing/precision cố định, so calibration CFE sẵn có với calibration fit trên `val_calib` v2 độc lập (1.080 ảnh, không dùng `val_select` để fit T/ngưỡng). Đo cả clean/Q30 trên `val_select`, công bố FPR và recall theo nhóm, giữ ràng buộc Real đã định. Đây là thử giả thuyết chuyển miền calibration, không bảo đảm sẽ cải thiện CFE hoặc đổi model triển khai.

Sau đó mới xét một thí nghiệm robustness Q30 riêng nếu ranking/lỗi theo nhóm còn là giới hạn; CIPS là tín hiệu đáng kiểm tra nhưng mẫu nhỏ, DiT đã yếu ở clean. Không đồng thời đổi calibration, loss, dữ liệu và augmentation rồi quy công cho một yếu tố. Temperature/threshold không tự sửa thứ tự ranking; augmentation cũng chưa chắc giải quyết được lỗi miền ngoài.

`val_select` đã dùng để chọn P0 và hiện tiếp tục dùng chẩn đoán. Không gọi nó là holdout mù, không chọn mô hình/ngưỡng từ CFE đã xem. Muốn kết luận tổng quát mới cần tập kiểm tra chưa dùng điều chỉnh hoặc trình bày rõ kết quả là hậu kiểm. Xem [kế hoạch hậu chẩn đoán](analysis/post_diagnosis_plan.json). Chưa khởi chạy calibration mới hoặc train mới; model/gate hiện hành giữ nguyên.
