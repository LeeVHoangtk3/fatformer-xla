# Đánh giá hai báo cáo Word và kết quả model ngày 13 11 2026

**Bổ sung hình trực quan từ FatFormer gốc:** Cả hai Word thêm sơ đồ kiến trúc (Hình 3 của bài gốc, trang 10772) tại mục 2.1 và biểu đồ phân bố logits UniFD/FatFormer (Hình 2 của bài gốc, trang 10771) tại mục 2.3. Hình trích trực tiếp từ PDF CVPR chính thức ở 300 dpi, có chú thích tác giả, trang và nguồn [1], cùng đoạn giải thích cách đọc. Paper dùng hình trải ngang trang; các hình của XLA được đánh số lại từ Hình 3. Đối chiếu bảo toàn 18/13 bảng, sáu công thức ở mỗi bản, toàn bộ ảnh cũ và nội dung cũ ngoài số tham chiếu hình. Tóm tắt vẫn ở cuối, không header, cỡ chữ ngoài tiêu đề 13. Đã kết xuất Word và kiểm tra 26 trang tổng thể, 17 trang paper. Hình nguồn và thông tin trích được lưu trong `hinh_bai_bao_goc/`; không tạo kết quả thực nghiệm XLA mới.

**Tập trung quy trình tạo mô hình cuối:** Theo yêu cầu biên tập, mục 3 của cả hai Word được viết thành bảy giai đoạn từ chuẩn bị dữ liệu đến đóng gói cho demo. Đã bỏ phần diễn biến các nhánh không được chọn: nội suy trọng số, ứng viên P0/P1 và hiệu chuẩn bổ sung. Bỏ bảng so sánh P0, hàng WiSE-FT và các cột ứng viên lịch sử; rút gọn thuật ngữ, trạng thái và danh mục nguồn tương ứng. Giữ baseline nội bộ để đối chiếu và thí nghiệm Feature/Guard để diễn giải bằng chứng về thành phần. Đánh số lại mục và bảng; có 18 bảng tổng thể và 13 bảng paper. Đối chiếu xác nhận các bảng số của mô hình cuối, sáu công thức và năm/bốn hình được bảo toàn. Không thay chất lượng 6/8, trọng số hay hiệu chuẩn; các thí nghiệm gốc trong kho dự án không bị xóa. Tóm tắt vẫn ở cuối, không có header và cỡ chữ ngoài tiêu đề vẫn là 13. Bản kết xuất Word đã được kiểm tra trực quan đủ 25 trang tổng thể và 16 trang paper.

**Chuyển Tóm tắt xuống cuối:** Di chuyển nguyên tiêu đề và hai đoạn tóm tắt đến cuối cả hai tài liệu. Đối chiếu XML xác nhận toàn bộ nội dung được giữ nguyên, chỉ đổi vị trí ba đoạn; header vẫn được bỏ. Bản kết xuất Word giữ 26 trang tổng thể và 18 trang paper, với Tóm tắt ở trang cuối của từng tài liệu.

**Bỏ đầu trang theo yêu cầu:** Hai Word đã bỏ toàn bộ tham chiếu header mặc định, trang chẵn và trang đầu, đồng thời xóa các phần header không còn dùng trong gói DOCX. Nội dung chính chỉ thay đổi các tham chiếu header; nội dung chân trang và hình giữ nguyên byte. Kết xuất Word xác nhận số trang giữ nguyên 26/18. So sánh ảnh từng trang xác nhận phần nội dung không thay đổi; đầu trang trống. Số trang ở chân trang được giữ nguyên.

**Biên tập văn phong theo yêu cầu mới:** Hai báo cáo đã được sửa trực tiếp cho người đọc ngoài dự án có kiến thức cơ bản về học máy. Mỗi đoạn diễn giải mở bằng kết quả hoặc ý chính; các câu dài được tách, thuật ngữ được giải thích bằng tiếng Việt kèm tiếng Anh khi cần. Phần kết quả giữ điều kiện đo và phạm vi đối chứng. Các câu giới hạn được tập trung vào mục Giới hạn; bỏ mã `[S…]` khỏi văn xuôi, giữ danh mục nguồn và chú thích để truy vết. Không thêm phép đo hoặc kết luận khoa học mới.

**Đối chiếu sau biên tập văn phong:** Bảo toàn toàn bộ nội dung 19 bảng tổng thể và 14 bảng paper, sáu công thức OMML ở mỗi tài liệu, đủ năm/bốn hình, các giá trị số và số lần xuất hiện của giá trị thập phân trong phần diễn giải, cùng các liên kết nguồn. Các số được kiểm tra theo đoạn gốc trước khi ghi; mã nguồn và trích dẫn được tách khỏi số liệu thực nghiệm khi đối chiếu. Cỡ chữ nội dung vẫn 13. Tóm tắt hiện có 196 từ. Bản kết xuất Word cuối đã được kiểm tra đủ 26 trang tổng thể và 18 trang paper. Renderer chuẩn thiếu LibreOffice nên dùng Word ẩn với bản sao chỉ đọc. Các ghi chú phía dưới mô tả những lần sửa trước, gồm cả số trang và độ dài tóm tắt của các bản cũ.

**Hiệu chỉnh theo phản biện về độ dễ đọc:** Hai báo cáo có tóm tắt 219 từ với số liệu, bảng thuật ngữ và tên XLA được chọn thống nhất trong bảng so sánh. Phần Giới thiệu vẫn chỉ trình bày bài toán và mục tiêu. Data v2 được đếm trực tiếp từ manifest: 7.863 Real, 6.537 Fake; train 6.683/5.557, mỗi split chọn mô hình và hiệu chuẩn v2 là 590/490. Liệt kê 17 bộ sinh và năm nguồn ảnh thật, phân biệt 2.991 ảnh replay cuối với kho ban đầu 3.600 ảnh; nêu rõ các bộ sinh của tập chọn mô hình đều có trong train v2.

**Bằng chứng so sánh và hình:** Bổ sung bốn biểu đồ định lượng trong tổng thể, ba trong paper; ROC tính từ xác suất CFE đã lưu, tái lập AUC sạch 88,6623% và Q30 79,8791%. Giải thích sự khác nhau về tập ảnh, chỉ số và tiền xử lý giữa bài gốc với CFE; ghi rõ nguyên nhân đầy đủ của baseline thấp chưa được xác định. Không biến kiểm nạp đúng cấu trúc thành xác nhận đã tái lập bài gốc. Các cấu hình ablation đổi nhãn AB1–AB4 để phân biệt với phụ lục; chưa có kết quả tách SRM/gating/KD và chưa thêm ảnh lỗi hoặc Grad-CAM khi thiếu ảnh gốc. Đề xuất ablation được ghi dưới dạng công việc tương lai.

**Khả năng đọc độc lập và kiểm tra bản cuối:** Bảng tóm tắt bằng chứng được chèn vào hai Word; [gói số liệu đi kèm](bang_chung_tom_tat/README.md) có nguồn dữ liệu v2, bảng kết quả, tọa độ ROC, các mẫu dự đoán sai và SHA-256 nguồn. Liên kết nội bộ vẫn được phân biệt với nguồn công khai; tài liệu tham khảo [7] chỉ rõ kho chính thức FatFormer. Đối chiếu bảo toàn số liệu của 15 bảng tổng thể và 10 bảng paper cũ, sáu phương trình và sơ đồ kiến trúc ở mỗi bản. Có thêm bốn bảng ở mỗi tài liệu; cỡ chữ ngoài tiêu đề là 13. Bản kết xuất cuối đã được kiểm tra đầy đủ 26 trang tổng thể và 17 trang paper. Dùng Word ẩn mở bản sao chỉ đọc vì renderer LibreOffice thiếu executable. Đây là hiệu chỉnh tài liệu và tái tính thống kê từ dự đoán, không huấn luyện/suy luận model mới, không đổi hiệu chuẩn hay trạng thái chất lượng 6/8. Các mục phía dưới là lịch sử của các bản trước.

**Biên tập theo mạch báo cáo nghiên cứu:** Hai tài liệu thống nhất tám phần chính, phân biệt phương pháp với cài đặt và kết quả. Bảy giai đoạn phát triển được viết ngắn hơn; bổ sung mục tiêu cụ thể, giải thích thuật ngữ và chỉ số, phạm vi so sánh, câu chốt có nguồn và tham chiếu bảng trong phần diễn giải. Bảng chi tiết của báo cáo tổng thể chuyển xuống phụ lục; thông tin demo và danh mục nguồn đặt sau phần kết luận. Đối chiếu giữ đủ 15 bảng tổng thể và 10 bảng paper với nội dung không đổi, sáu dòng công thức và hình ở mỗi tài liệu, cùng toàn bộ đích liên kết nguồn. Cỡ chữ ngoài tiêu đề vẫn là 13; phần đầu không có kết quả dự án và không thêm ngày biên soạn. Kiểm tra dàn trang cuối gồm 21 trang tổng thể và 13 trang paper hai cột. Thay đổi là biên tập tài liệu, không phải huấn luyện hay phép đo mới.

**Chuẩn hóa cỡ chữ và phần mở đầu:** Hai bản Word dùng cỡ chữ 13 cho nội dung ngoài tiêu đề, bao gồm bảng, chú thích, công thức, đầu/chân trang và tài liệu tham khảo. Đã bỏ ngày biên soạn, ngày đợt kiểm chuẩn trong nội dung và đầu trang; giữ thông tin xuất bản trong trích dẫn khoa học và các đại lượng đo độ trễ. Phần giới thiệu và tóm tắt không nêu kết quả định lượng hoặc phán quyết dự án. Kết quả và câu chốt in đậm được giữ tại phần đánh giá, kết luận. Bảng số, nội dung công thức, hình kiến trúc và đích liên kết nguồn giữ nguyên. Đã kiểm tra trực quan bản mới gồm 19 trang tổng thể và 11 trang paper hai cột.

**Bổ sung nhấn mạnh và kết luận ngày 16/11/2026:** In đậm có chọn lọc mục đích, quyết định giữ mô hình, trạng thái chất lượng và giới hạn diễn giải; làm rõ tiêu đề bảng và trạng thái đạt/chưa đạt. Cả hai tài liệu có câu chốt riêng được in đậm: chọn FatFormer XLA đã hiệu chuẩn cho demo nghiên cứu, đạt 6/8 tiêu chí chất lượng và chưa đủ điều kiện nghiệm thu toàn bộ. Nội dung cũ được giữ nguyên ngoài tên mục kết luận của báo cáo tổng thể; bảng số, công thức, hình và liên kết nguồn không thay đổi. Bản dàn trang cuối vẫn gồm 14 trang tổng thể và 7 trang paper, đã kiểm tra trực quan.

**Biên tập cách dẫn nguồn ngày 16/11/2026:** Hai báo cáo cuối dùng tên báo cáo, tên mô hình và tên dữ liệu thay cho đường dẫn mã nguồn, tên tệp triển khai, mã lần chạy và mã kiểm tra dài. Danh mục đối chiếu gồm 30 tên tài liệu có liên kết mở nguồn; 29 đích liên kết cũ được giữ và bổ sung hướng dẫn demo. Ví dụ mã suy luận trong báo cáo tổng thể được chuyển thành mô tả quy trình demo. Đối chiếu giữ nguyên 261 ô số trong báo cáo tổng thể và 118 ô số trong paper, sáu dòng công thức và hình kiến trúc ở mỗi tài liệu, bảy giai đoạn nghiên cứu và kết luận chất lượng chưa đạt toàn bộ yêu cầu. Đã kiểm tra trực quan đầy đủ 14 trang tổng thể và 7 trang paper bằng Microsoft Word. Đây là biên tập tài liệu; không thay mô hình, hiệu chuẩn hoặc dữ liệu thực nghiệm. Các mục ghi ngày trước bên dưới là lịch sử.

**Bổ sung quy trình theo giai đoạn ngày 15/11/2026:** Hai báo cáo đã bổ sung lịch sử từ chuẩn bị hạ tầng, baseline, SRM/gating và curriculum ban đầu, Plan A, thích ứng đa miền/phục hồi đến chuẩn hóa và kiểm chuẩn cuối. Quy trình được viết theo bảy giai đoạn có mục đích, cách làm và kết quả; loại cách kể theo task khỏi phần chính, giữ mã trong nguồn để truy vết. Các bảng định lượng, sáu dòng công thức và ảnh kiến trúc được kiểm tra không thay đổi so với bản trước; thêm chín nguồn lịch sử, tổng 29 liên kết local hợp lệ. **52 kiểm tra biên tập đạt; rà trực quan bản render Word gồm 14 trang tổng thể và 7 trang paper.** Đây là kiểm tra tài liệu, không phải unit test model hoặc phép đo mới. Các lần hiệu chỉnh trước dưới đây được giữ làm lịch sử.

**Cập nhật ngày 15/11/2026:** [Báo cáo tổng thể](Bao_Cao_Tong_The_Mo_Hinh_FatFormer_XLA.docx) và [paper hai cột](Paper_FatFormer_XLA_CVPR_Format.docx) đã được sắp xếp lại theo yêu cầu: giới thiệu dự án và mục đích → nền tảng FatFormer gốc → phần triển khai FatFormer XLA. Phần bài gốc có FAA/LGA, dữ liệu ProGAN và 18 cấu hình kiểm thử, preprocessing/huấn luyện, ACC/AP do tác giả công bố và giới hạn so sánh với dự án. Bổ sung master audit 8 archive/37 job/1.618.061 lượt dự đoán, cập nhật latency và kiểm exact SHA; G7.2 vẫn FAIL 6/8, G7.3 PENDING. Đối chiếu **201 kiểm tra nội dung/số liệu đạt**, kiểm tra trực quan đủ **13 trang báo cáo và 6 trang paper** bằng ảnh trang Microsoft Word 16 qua `Page.EnhMetaFileBits` và Windows GDI. Renderer LibreOffice không có executable trong môi trường; bộ xuất PDF Word bị chờ nên không dùng PDF làm bằng chứng QA. Không train, đổi model/calibration hoặc xác nhận nghiệm thu thay con người.

**Lịch sử hiệu chỉnh ngày 13/11/2026:** Hai file Word khi đó đã được sửa, đối chiếu 150 kiểm tra nội dung/số liệu đạt và kiểm tra trực quan bản render Word gồm 10 trang báo cáo, 5 trang paper. Các nhận xét và SHA-256 bên dưới mô tả **bản gốc trước sửa**, được giữ để truy vết; không phải danh sách lỗi còn tồn tại trong bản cập nhật ngày 15/11. Việc sửa tài liệu không thay đổi model, calibration hoặc trạng thái gate.

Kết luận: **giữ model incumbent Task 7.15 làm bản chuẩn theo bằng chứng hiện có; cả hai file Word cần sửa lớn trước khi dùng làm báo cáo chính thức.** Nhiều metric chính của model đúng, nhưng bảng số liệu có ô sai, paper mô tả sai công thức gating và Guard Loss, kết luận ablation trái với log, và trạng thái phát hành/nghiệm thu được diễn giải quá mức.

Đây là đánh giá kỹ thuật hai tài liệu hiện hữu. Không sửa hai file Word, không thay model/calibration, không xác nhận nghiệm thu thay A/B/C.

## 1 Phạm vi kiểm tra và giới hạn

Đã đọc nội dung đoạn văn và bảng trong cả hai DOCX, đối chiếu với mã nguồn, cấu hình, lịch sử run, JSON/CSV và các mảng dự đoán lưu sẵn. Các vị trí bên dưới dùng tên mục/bảng trong Word, không dùng số trang chưa được render.

Hai bản được đánh giá:

- [Báo cáo tổng thể](Bao_Cao_Tong_The_Mo_Hinh_FatFormer_XLA.docx), SHA-256 `c25df47dcd38398dbb9a9b1973d557e562ed0d49937585fc73f1f2d941d8bedd`.
- [Paper](Paper_FatFormer_XLA_CVPR_Format.docx), SHA-256 `5165c21f0fd463556fc1cf7d74fe8208bbd598cd852c66bf6c2a1c2cddc70e2e`.

Đã tái tính độc lập bằng NumPy trên **93.203 lượt dự đoán incumbent**: 1.600 CFE ×7 điều kiện +82.003 lượt ảnh in-dist. Đối chiếu accuracy, Real/Fake accuracy, AUC, Fake@Real90, ECE; tái tính macro balanced accuracy 18 tập và latency từ mẫu đo. **64/64 kiểm tra nhất quán đạt** sau khi xét đúng quy ước xuống dòng của Git. Đây là kiểm tra dữ liệu, không phải 64 unit test hay một phép đo GPU mới.

Ba JSON được acceptance tham chiếu có working-tree CRLF nên hash bytes hiện tại khác nguồn LF. `git ls-files --eol` xác nhận index LF/working tree CRLF; sau chuẩn hóa CRLF→LF cả ba khớp hash nguồn. Không sửa raw để làm khớp hash, không gọi working-tree là bản nguyên byte nguồn.

Model hiện hành không được nộp weights tại local theo [package reference](../../../../DATASET/logs/phase7/task_7.15/current/artifacts/package_reference.json). Lần rà soát này không forward lại model, không băm lại actual `.pth`, không train hay bootstrap lại toàn bộ CI. Bản runtime Windows không có LibreOffice và không tìm thấy renderer native; chưa render/kiểm tra trực quan từng trang. Các nhận xét hình/cột bên dưới dựa trên cấu trúc OOXML, không khẳng định chất lượng dàn trang toàn tài liệu.

## 2 Kết quả model được giữ làm chuẩn

Run: `corrected_20261109_102954_3d28ad`.

Package: `unified_fatformer_v2.pth`, SHA-256 `248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`.

Weights nguồn Anchored v2: `5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c`, chọn epoch 4 của Task 7.11. Không đồng nhất file này với checkpoint A4 của thí nghiệm ablation Task 7.12 chỉ vì dùng cấu hình loss đầy đủ.

Calibration hiện hành: **T=2,7832491397857666**, **threshold=0,7918508648872375**. Router deployment tắt, dùng calibrated fixed fallback. Nguồn: [metadata](../../../../DATASET/logs/phase7/task_7.15/current/artifacts/metadata.json), [calibration](../../../../DATASET/logs/phase7/task_7.15/current/calibration/anchored_v2.json).

Các số nên trích dẫn, đơn vị %:

- **CFE clean:** AUC 88,6623; Fake@Real90 64,125; accuracy 74,375; Real 95,625; Fake 53,125; ECE 4,3936.
- **CFE Q30:** AUC 79,8791; Fake@Real90 42,625; accuracy 65,8125; Real 91,375; Fake 40,25; ECE 5,0823.
- **Các AUC còn lại:** Q50 81,5570; Q75 83,3243; WebP40 **79,0089**; blur15 84,2148; downup05 86,9880.
- **18 tập in-dist:** macro balanced accuracy 90,1018; pooled accuracy 93,4649; pooled AUC 98,4105; pooled Fake 96,5241; pooled Real 90,4063.
- **Worst-subset CFE clean:** `5_other_autoregressive`, Fake đúng 25/81=30,8642%.

Nguồn: [final report](../../../../DATASET/logs/phase7/task_7.15/current/raw/benchmark/final_report.json), [condition summary](../../../../DATASET/logs/phase7/task_7.15/current/analysis/condition_summary.csv), [gate summary](../../../../DATASET/logs/phase7/task_7.15/current/analysis/gate_summary.csv).

Độ trễ **package** trên L4/FP16: batch1 44,9796 ms/ảnh, 22,2323 FPS; batch8 104,1807 FPS; batch16 104,0881 FPS; batch32 **96,3539 FPS**. Overhead batch1 **0,188055 ms/ảnh** là median chênh lệch cặp đo giữa wrapper package và backbone FatFormer được bọc; không phải tổng thời gian inference hay chi phí cộng toàn bộ SRM/gating/FAA vào vanilla CLIP. Phép đo loại I/O và tiền xử lý. Nguồn: [latency JSON](../../../../DATASET/logs/phase7/task_7.15/current/raw/latency_phase7.json), [mã đo](../../../../tools/infer_unified.py).

Trạng thái nghiệm thu cần giữ:

- **G7.0 PENDING:** canonical core đã có bằng chứng nhưng DoD toàn repo/legacy chưa đóng.
- **G7.1 FAIL:** nhánh Gen 2/ensemble không đạt cổng thắng nhanh; không mô tả cổng này chỉ là baseline gốc phải vượt ngưỡng.
- **G7.2 FAIL 6/8:** thiếu clean Fake@Real90 0,875 pp và worst-subset Fake accuracy 14,1358 pp so với mục tiêu.
- **G7.3 PENDING:** packaging/equivalence/overhead đã có phần đạt; còn thiếu verified best-by-condition comparison.
- Cross-review/chữ ký A/B/C vẫn chờ. Điều khoản kết quả âm không tự biến thành nghiệm thu toàn bộ.

Nguồn: [acceptance gốc](../../../../DATASET/logs/phase7/task_7.15/current/raw/acceptance.json), [biên bản](../../../milestones/milestone_phase7_acceptance.md), [human signoff](../../../../DATASET/logs/phase7/task_7.16/human_signoff.json).

P0 và calibration clean-v2 chưa cung cấp căn cứ thay incumbent. Matched evaluation dùng incumbent **refit** package `de36061f…`, khác package deployment `24869052…`; arrays tái lập kết quả deployment nhưng danh tính file và latency phải tách riêng. Không kết luận incumbent tối ưu toàn cục từ các thử nghiệm này. Nguồn: [báo cáo vòng cải thiện](../../C/bao_cao_phase7_improvement_round1_va_chan_doan_vong2.md).

## 3 Các lỗi số liệu cần sửa trong cả hai tài liệu

### 3.1 Bảng CFE lẫn metric và model lịch sử

Vị trí: báo cáo tổng thể §5.1; paper Bảng 1.

- **Baseline clean Accuracy 76,21%:** không có căn cứ cho ô này trong ma trận Task 7.6; 76,21% xuất hiện như **AUC lịch sử**, không phải accuracy. Không lấy AUC điền sang accuracy.
- **WiSE-FT Gen 2 clean Accuracy 52,05%:** số này được catalog đánh dấu là accuracy diagnostic của **Gen 1 n=8 invalid_layout**. Phải bỏ khỏi cột Gen 2.
- **Baseline/Gen 2 Q30 Accuracy <50%:** không được suy accuracy từ AUC dưới 50%. Các file 2.000 ảnh khác protocol còn cho accuracy lần lượt 50,35% và 50%; không được lấy các số đó ghép vào bảng evaluation 1.600 ảnh.
- **ECE baseline/Gen 2 18,25%; 17,50%; khoảng 34%; khoảng 33,5%:** chưa tìm thấy nguồn đúng protocol xác nhận các ô này. Vì vậy các delta ECE −13,86 pp/−28,92 pp cũng chưa đủ căn cứ. Xóa hoặc ghi chưa có số cùng protocol; chỉ điền lại khi có nguồn truy vết.
- **WebP AUC incumbent 83,05%:** sai; số đúng **79,01%**. P0 **82,37%** cũng sai nếu gắn WebP40; số đúng **78,18%**. P0 khoảng 82,38% thuộc Q75, không phải WebP40.

Nguồn: [bảng nguồn số cũ](../bao_cao_ra_soat_so_lieu_phase7.md), [ma trận Task 7.6](../../../../DATASET/logs/phase7/wise_ft_sweep/cfe_v3_baseline_matrix.json), [current condition summary](../../../../DATASET/logs/phase7/task_7.15/current/analysis/condition_summary.csv), [matched comparison](../../../../DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/analysis/comparison.csv).

Các cặp AUC/Fake@Real90 baseline→incumbent Clean và Q30 về cơ bản đúng nguồn lịch sử; +37,55 pp AUC Q30 và khoảng +27,75 pp Fake@Real90 Q30 có thể trình bày là chênh lệch với mốc đó. **+27,74 pp** trong Word là trừ hai số đã làm tròn 42,62−14,88; nên thống nhất cách làm tròn và không dùng chênh lệch lịch sử như bằng chứng nhân quả cho từng module.

### 3.2 FPR 5% không đúng cho mọi điều kiện evaluation

Bảng CFE ghi chú `FPR<=5%` cạnh Accuracy là gây hiểu sai. Threshold fit FPR 5% trên 400 calibration không bảo đảm FPR evaluation. Incumbent CFE clean FPR **4,375%**, Q30 **8,625%**, WebP40 **11%**, blur15 **11,875%**, downup05 **12,125%**.

Fake@Real90 là metric sử dụng ngưỡng suy từ phân bố Real của tập đánh giá theo định nghĩa hiện hành; nó khác Fake accuracy tại threshold triển khai 0,79185086. Cần báo riêng Real/Fake accuracy và operating point, không coi hai recall này là một.

### 3.3 Kết quả nội miền đúng nhưng cần giới hạn diễn giải

Các số 90,10%, 93,46%, 98,41%, 96,52%, 90,41% khớp arrays. Các số Task 5.1 67,34%, 70,37%, 73,57%, 47,00%, 93,73% có trong [CSV lịch sử](../../../../DATASET/logs/m4_full_benchmark/task_5.1_clean_results.csv). Tuy nhiên đó là so sánh hai hệ thống/operating point khác nhau, chưa cô lập đóng góp SRM/gating/Guard.

Pooled Real 90,41% **không bảo đảm từng tập đạt 90%**: 9/18 tập dưới 90%; DALL-E/GLIDE/LDM có Real 75,8%, deepfake 84,30%, StarGAN 77,19%. CFE WebP/blur/downup cũng dưới Real90. Không dùng các số pooled để viết Guard bảo đảm Real>=90% mọi miền.

82.003 là số lượt ảnh/record test, chưa được chứng minh là 82.003 nội dung ảnh độc lập. [Distribution cautions](../../../../DATASET/logs/phase7/task_7.15/current/analysis/distribution_cautions.json) ghi các nhóm Real có thể dùng chung reference; chưa có bytes để kiểm chứng trùng nội dung.

## 4 Các lỗi phương pháp nghiêm trọng trong paper

### 4.1 Công thức gating và đường dung hợp không khớp code

Vị trí: paper §3.2, phương trình (2)–(3); báo cáo tổng thể §2.1 cũng mô tả gây hiểu nhầm.

Word viết `lambda=sigmoid(W*[f_clip;f_srm]+b)` trong [0,1] và `f_fused=(1-lambda)*f_clip+lambda*f_srm`. Code thực tế là:

```text
lambda(x) = 2 * sigmoid(MLP(mean_tokens(x)))
adapter(x,I) = spatial_adapter(x) + 0.1 * SRM_tokens(I)
               + freq_scale * lambda(x) * DWT_branch(x)
```

Gating nhận token feature; không nhận concatenation CLIP/SRM trong công thức Word, không thực hiện convex mixture hai nhánh đó. Lambda có khoảng lý thuyết [0,2], khởi tạo bằng 1; SRM được cộng vào nhánh spatial, gating điều tiết nhánh DWT. Không tự khẳng định lambda tăng/giảm đơn điệu theo độ nén nếu chưa có phân tích đo tương ứng. Nguồn: [gating.py](../../../../src/models/gating.py), [adapter forward](../../../../src/models/clip/model.py), [srm.py](../../../../src/models/srm.py).

SRM là lọc vết dư **không gian**; nhánh phân tích sóng con DWT cần được mô tả riêng. Câu “tín hiệu không thể bị triệt tiêu bởi JPEG” không có bằng chứng bất biến; đổi thành mục tiêu khai thác thêm tín hiệu vết dư.

### 4.2 Guard Loss phương trình (5) là một loss khác

Word mô tả triplet margin với tâm `c_real`, khoảng cách feature và margin >=0,5. Code không có `c_real` hoặc margin này. Loss thực tế:

```text
L_guard = mean_real( ReLU(z_student_degraded,fake
                         - stopgrad(z_teacher_clean,fake)) ** 2 )
```

Hệ số 0,5 trong tổng loss là **trọng số loss**, không phải triplet margin hay “Real Anchor >=50%”. Đây là penalty một phía trên fake logit của mẫu Real, không phải bảo đảm toán học Real recall>=90%. Nguồn: `RealGuardLoss` trong [loss.py](../../../../src/training/loss.py).

### 4.3 KD và consistency được diễn giải sai

Vị trí: paper §3.4; báo cáo tổng thể §2.3.

KD là distillation KL từ teacher baseline đóng băng sang student clean, có chọn mẫu teacher-correct theo code; không phải “Triplet Knowledge Distillation” clean→degraded. Paired consistency dùng **MSE logits** clean/degraded với clean detached, không phải một latent-feature loss. FeatureConsistency dùng cosine feature **cộng KL logits**, cũng stop-gradient phía clean.

Tổng loss có **sáu số hạng** nếu tính CE clean và CE degraded riêng. Bộ hệ số 0,3/0,3/0,7/0,2/0,3/0,5 đúng cấu hình đầy đủ, nhưng phải định nghĩa từng số hạng đúng. Các module KD/Cons đã nhân trọng số nội bộ; không nhân lặp khi mô tả implementation.

### 4.4 Degradation v2 bị trộn với curriculum cũ

Word nói Q cố định {30,50,75}, clean100%→degrade30%→70%, thực hiện RAM/GPU. Cấu hình v2 đã dùng có `clean_prob=0.60`, JPEG Q trong [30,85] với heavy-share, WebP Q [30,70], **double JPEG**, blur sigma [0,5;2,0], down-up ratio [0,5;0,75], composite. Pipeline sử dụng PIL/BytesIO trên CPU; không có bằng chứng GPU augmentation/curriculum ba pha trong run này. Các mức JPEG Q30/Q50/Q75 là điều kiện benchmark, không phải toàn bộ phân bố train.

Nguồn: [degradation config](../../../../configs/degradation_v2.yaml), `DegradationPipelineV2` trong [transforms.py](../../../../src/datasets/transforms.py), [run config đã lưu](../../../../DATASET/logs/phase7/anchored_v2_run1/run_config.yaml).

### 4.5 Ablation được kết luận ngược với kết quả

Vị trí: paper §4.4; báo cáo tổng thể dòng Task 7.12 và các tuyên bố “chứng minh phối hợp tối ưu”.

Thí nghiệm bốn nhánh Task 7.12 có worst-group validation: **A1 34,6667%; A2 26%; A3 38%; A4 27,3333%**. A4 thấp hơn A1 7,3333 pp và A3 10,6667 pp. **46,67% thuộc run Task 7.11**, không phải A4 ablation; số 31,2% cho A1 trong Word không khớp CSV.

Tên đúng là `a1_data`, `a2_data_guard`, `a3_data_featcons`, `a4_full`. A1 vẫn có SRM/gating, KD và paired consistency; không phải “CE thường” hoặc chưa có gating. Thí nghiệm này thay hai loss Feature/Guard; không cô lập đóng góp của gating.

Phải viết lại: một seed 42, A3 tốt nhất theo worst-group selection; chưa chứng minh Guard+Feature cải thiện tổng quát hoặc phối hợp vượt trội. CI Q30 của các tương phản loss chưa đủ kết luận cải thiện rõ. Nguồn: [CSV gốc](../../../../DATASET/logs/phase7/ablation_v2_fixed_run1/ablation_mini_v2.csv), [A1 config](../../../../DATASET/logs/phase7/ablation_v2_fixed_run1/a1_data/run_config.yaml), [báo cáo ablation](../../C/bao_cao_task_7.12_ablation.md).

## 5 Các lỗi triển khai và đo hiệu năng

### 5.1 Đoạn code trong báo cáo tổng thể §7.2 không chạy đúng package

`checkpoint['model']` là **state_dict**, không phải `nn.Module`; gọi `.eval()` sẽ lỗi. Calibration nằm trong `metadata['calibration']`, không ở top-level `temperature`/`optimal_threshold`. `UnifiedFatFormerXLA.forward` trả dictionary `prob_fake`, `label`, `w_deg`, `threshold`, không trả logits để softmax lần nữa. Code còn gọi `.cuda()` vô điều kiện dù có nhánh chọn CPU.

Cần thay toàn bộ ví dụ bằng loader thật `load_unified_model` trong [infer_unified.py](../../../../tools/infer_unified.py), transform tương ứng và đọc output dictionary. Phải ghi precision/batch được kiểm chứng: backbone FP16 autocast, logits/softmax FP32, quality benchmark batch16. CLI thường mặc định FP32 khác hợp đồng quality này; không tự gán metric GPU hiện có cho mọi cách chạy.

“Standalone” nghĩa là đủ weights/calibration trong một checkpoint, không cần checkpoint CLIP/calibration ngoài trong đường nạp đã kiểm chứng; vẫn cần code kiến trúc và dependencies. Nguồn cấu trúc package: [package builder](../../../../tools/package_unified_fatformer.py); output: [unified model](../../../../src/models/unified_fatformer.py).

### 5.2 Throughput và hệ quy chiếu overhead

Vị trí: tổng thể §5.4/tóm tắt; paper §4.6.

**397,8 FPS batch32** không khớp current; số đúng **96,35 FPS**. `44,792 ms` backbone trong Word cũng không khớp batch1 current: raw lưu 44,71 ms/ảnh đã làm tròn. Không tính overhead bằng hiệu median hoặc số đã làm tròn; dùng paired delta 0,188055 ms.

Mã benchmark đo `model.backbone(x)` so với `model(x)` trên **cùng** backbone đã có SRM/gating/FAA. Vì vậy không dùng 0,188 ms để chứng minh tổng chi phí bổ sung các module đó so với CLIP thuần. L4/PyTorch autocast là backend có bằng chứng; bỏ **TensorRT** vì không có số đo engine tương ứng. Tổng latency là khoảng 44,98 ms/ảnh batch1, không phải 0,188 ms. Đây chưa phải độ trễ end-to-end Gateway/luồng kiểm duyệt.

## 6 Diễn giải và tính tái lập cần sửa

### 6.1 Trạng thái RELEASED và PASS IN PACKAGING

Vị trí: bảng đầu/tóm tắt/kết luận báo cáo tổng thể; Bảng 5 và kết luận paper.

Đổi thành **“bản model được giữ làm chuẩn thực nghiệm, G7.2 FAIL 6/8, G7.3 PENDING”**. Có thể ghi riêng các hạng mục packaging đã đạt. Không gọi thiếu best-reference là chỉ “pending bên thứ ba”; đó là phép so sánh reference tốt nhất theo điều kiện chưa được xác minh. G7.0 pending cũng còn việc kỹ thuật toàn repo, không chỉ “lưu vết lịch sử”.

### 6.2 Không chứng minh tối ưu hoặc an toàn trên mọi miền

Loại các câu “khắc phục triệt để”, “tối ưu nhất”, “loại bỏ hoàn toàn Real Drift”, “bảo vệ tuyệt đối”, “sẵn sàng giám định tư pháp”. Các kết luận đó vượt phạm vi benchmark. Giữ quyết định không promote P0/clean-only calibration, nhưng nói rõ quyết định dựa trên tiêu chí đã đặt và các run đã thử.

P0 không kém mọi metric: Q30 TPR tại FPR1%/5% cao hơn incumbent refit, in-dist ECE/NLL tốt hơn. CI delta Fake@Real90 clean/Q30 chứa 0 trong matched evaluation; không gọi mọi chênh recall đều có ý nghĩa thống kê. CFE đã được xem trước nên kết quả corrected/matched là hậu kiểm, không phải blind test mới.

Calibration clean-v2 có Real Q30 P0 **86,9492%**, incumbent **87,2881%**: FPR tương ứng **13,0508% và 12,7119%**. Câu “báo oan vượt 13%” không đúng cho cả hai. Thay calibration không tăng raw ranking và không được dùng metric validation này thay CFE.

Chẩn đoán CIPS 10/12→1/12 là đúng số đếm; nhưng DiT 3/13 và SD v1.5 42/90→30/90 không hỗ trợ câu “Diffusion vẫn duy trì nhận diện ổn định” theo nghĩa hiệu năng cao. Ngưỡng 0,9127 thuộc điểm chẩn đoán lịch sử, không phải threshold deployment hiện hành.

### 6.3 Cách ly ID không đồng nghĩa không trùng nội dung

Tổng thể §3.1 viết 400 calibration và 1.600 evaluation CFE “0% trùng SHA-256”. [Preflight](../../../../DATASET/logs/phase7/task_7.15/current/raw/input_preflight.json) ghi identity unit là `subset/class/filename`, **content duplicates not audited**. Sửa thành disjoint IDs đúng phạm vi đã kiểm chứng. Không nâng thành bảo đảm mọi loại rò rỉ đã được giải quyết.

Data v2 14.400 record với chia 12.240/1.080/1.080 là đúng; audit SHA có phạm vi reference riêng. Phải phân biệt `val_calib` v2 1.080 với calibration CFE 400. 186.406 là tổng hai model ×93.203 lượt dự đoán matched, không phải số ảnh độc lập hay số inference riêng của model chuẩn. Nguồn replay 3.600 trong báo cáo Data v2 là `diffusion_staging`/SD-Midjourney, không nên ghi chung thành 3.600 GenImage khi chưa chứng minh thành phần.

### 6.4 Tiền xử lý P0/P1 và các đường dẫn sản phẩm

Tổng thể §4.1 ghi P0 Resize256/RandomCrop224; implementation profile legacy dùng resize vuông rồi CenterCrop, còn profile CLIP dùng short-side bicubic rồi CenterCrop. Phải lấy kích thước/profile từ config runtime và [preprocessing.py](../../../../src/datasets/preprocessing.py), không điền mô tả augmentation quen thuộc.

Các tên sản phẩm `degradation_pipeline.py`, `loss_v2_anchored.py` trong bảng Task không trỏ đến implementation hiện hành; các lớp nằm trong `src/datasets/transforms.py` và `src/training/loss.py`. Tách tên dự kiến/khái niệm khỏi đường dẫn file thực.

## 7 Tài liệu tham khảo và hình thức paper

1. **Reference [1] sai tác giả/trang.** Nguồn CVF: Huan Liu, Zichang Tan, Chuangchuang Tan, Yunchao Wei, Jingdong Wang, Yao Zhao; *Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection*, CVPR 2024, **10770–10780**. Word ghi thứ tự/tên tác giả và 10452–10461 không khớp. [CVF chính thức](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Forgery-aware_Adaptive_Transformer_for_Generalizable_Synthetic_Image_Detection_CVPR_2024_paper.html).
2. **Reference [10] không khớp công trình nguồn.** Công trình Community Forensics là **Jeongsoo Park và Andrew Owens**, *Community Forensics: Using Thousands of Generators to Train Fake Image Detectors*, CVPR 2025, **8245–8257**. Không dùng tên bài/tác giả Zhang/Zhao/Dong trong Word. CFE của dự án phải được giải thích là tập con/protocol được dùng trong repo, không biến thành tên bài benchmark mạng xã hội độc lập. [CVF chính thức](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html).
3. **Reference [2] sai tiêu đề.** Đuôi đúng là *That Generalize Across Generative Models*, không phải *Across Architectures*. [CVF chính thức](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html).
4. **DiT không thuộc autoregressive.** Là diffusion với transformer backbone; cần sửa paper phần giới thiệu. [Công trình DiT](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html).
5. **Paper viện dẫn Hình 1 nhưng file không có hình.** OOXML có 0 drawing, 0 media và không thấy sơ đồ thay thế; cả tài liệu chỉ có một section, không đặt số cột=2. Các phương trình là văn bản thường, chưa có OMML. Có cấu trúc bài báo không đồng nghĩa đã dàn đúng template CVPR; cần bổ sung hình kiến trúc đúng code và dàn template mục tiêu nếu muốn gọi là bản CVPR format.
6. URL repo `github.com/nam4/fatformer-xla` trong paper không khớp `git remote origin` hiện tại `https://github.com/LeeVHoangtk3/fatformer-xla.git`. Cần xác nhận đường dẫn công bố; không tự coi tên lab/affiliation trong Word là thông tin đã được xác nhận.
7. Câu các công trình trước “đều giả định ảnh nguyên vẹn lý tưởng” quá rộng; cần đọc lại đánh giá robustness trong [supplement FatFormer](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Liu_Forgery-aware_Adaptive_Transformer_CVPR_2024_supplemental.pdf) trước khi nêu giới hạn liên quan.

## 8 Thứ tự sửa đề xuất

Trước tiên sửa số liệu CFE, công thức gating/Guard/KD và toàn bộ kết luận ablation. Tiếp theo sửa trạng thái nghiệm thu, ví dụ nạp package, hệ quy chiếu overhead/throughput và backend. Sau đó bổ sung bảng đủ bảy điều kiện kèm Real/Fake accuracy, CI sẵn có, operating point và nguồn; sửa bibliography/hình/template và giảm các tuyên bố vượt phạm vi.

Đoạn kết luận có thể dùng sau khi cập nhật các phần liên quan:

> Theo bằng chứng hiện có, nhóm giữ package unified_fatformer_v2.pth SHA-256 24869052… làm bản chuẩn thực nghiệm. Trên CFE evaluation 1.600 ảnh, model đạt AUC 88,66% ở clean và 79,88% ở JPEG Q30; Fake@Real90 lần lượt 64,125% và 42,625%. Macro balanced accuracy trên 18 tập in-dist đạt 90,10%. G7.2 đạt 6/8 tiêu chí và vẫn FAIL; G7.3 còn chờ đối chiếu reference tốt nhất theo điều kiện. Các thử nghiệm P0 và calibration clean-v2 chưa cung cấp đủ căn cứ thay model hiện hành. Kết quả cho thấy cải thiện so với các mốc lịch sử đã nêu, đồng thời còn hạn chế về nhận diện autoregressive, kiểm soát false positive theo miền và bằng chứng đóng góp riêng của từng thành phần.
