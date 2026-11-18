# Kiến thức chung của cả ba thành viên

## Cách chia phần học

**Mỗi thành viên nắm sâu một chuyên môn và giải thích được toàn bộ luồng dự án ở mức tổng quan.** Cách chia này bám vai trò A/B/C hiện có, đồng thời giúp nhóm chuyển câu hỏi đúng người khi giảng viên hỏi sâu.

- **A — Dữ liệu và giao thức:** bài toán, dữ liệu bài gốc, Data v2, chia tập, tiền xử lý, chống rò rỉ và phạm vi tổng quát hóa. Học sâu câu 07–16.
- **B — Kiến trúc và phương pháp:** CLIP, ViT, FAA, LGA, DWT, SRM, gating, khởi tạo và hàm mất mát. Học sâu câu 17–28.
- **C — Huấn luyện, kết quả và demo:** cách tinh chỉnh, chọn epoch, hiệu chuẩn, chỉ số, kết quả chính, bản đóng gói và trình diễn. Học sâu câu 29–40.
- **Cả ba:** câu 01–06, câu 41–50 và các số liệu tối thiểu dưới đây. A và C cùng giải thích giao thức; B và C cùng giải thích huấn luyện và loss; A và B cùng giải thích tác động của tiền xử lý.

**Trình tự trình bày đề xuất là A → B → C.** A đặt bài toán và dữ liệu; B mô tả mô hình gốc rồi phần mở rộng; C trình bày cách huấn luyện, bằng chứng đánh giá và demo. Không dùng phân công ôn tập này để nhận thay công việc mà một thành viên chưa thực hiện.

## Những điều mọi người phải nhớ

**FatFormer gốc, baseline nội bộ và XLA là ba đối tượng cần phân biệt.** Số do tác giả bài gốc công bố không phải kết quả tái lập của nhóm. Baseline nội bộ là mốc đối chiếu đã lưu trong dự án. XLA được chọn là mô hình đã tinh chỉnh và hiệu chuẩn dùng cho đánh giá và demo.

- Bài FatFormer gốc: với dữ liệu ProGAN bốn lớp ngữ nghĩa, trung bình tám tập GAN đạt **98,4% ACC và 99,7% AP**; mười cấu hình diffusion đạt **95,0% ACC và 98,8% AP**.
- XLA trên **1.600 ảnh CFE**: **AUC 88,66% ở ảnh sạch, 79,88% ở JPEG Q30**; Fake@Real90 tương ứng **64,125% và 42,625%**.
- XLA trên **18 tập ảnh sạch, 82.003 lượt ảnh mỗi điều kiện**: **Macro BA 90,10%**, pooled ACC **93,46%**. Đây là hai cách tổng hợp khác nhau.
- Chất lượng mô hình cuối: **đạt 6/8 tiêu chí**. Thành phẩm được chọn cho demo nghiên cứu; phạm vi sử dụng còn gắn với các giới hạn đã đo.
- Trọng số triển khai: **epoch 4 của đợt huấn luyện chính**, nhiệt độ khoảng **2,7832491398**, ngưỡng ảnh giả khoảng **0,7918508649**, bộ định tuyến tắt.

## Câu hỏi chung 01 đến 06

**Câu 01. Dự án giải quyết bài toán gì và vì sao cần làm?**

Nhóm nghiên cứu phân biệt ảnh thật với ảnh do AI sinh ra khi ảnh có thể bị nén, làm mờ, thay đổi kích thước hoặc khác nguồn sinh. Mô hình nền kế thừa FatFormer; dự án khảo sát cách tinh chỉnh và bổ sung thành phần để cải thiện độ bền. Đầu ra hiện tại là quyết định Real/Fake cho ảnh được xử lý, không phải kết luận giám định nguồn gốc tuyệt đối.

**Câu 02. Hai ý tưởng chính của bài FatFormer gốc là gì?**

Bài gốc đề xuất bộ thích nghi nhận biết dấu vết giả mạo (FAA) và căn chỉnh theo ngôn ngữ (LGA) trên nền CLIP. FAA kết hợp dấu vết trong miền ảnh và miền tần số. LGA dùng biểu diễn văn bản cùng đặc trưng ảnh để hướng mô hình vào thông tin phục vụ phân loại thật/giả. Đọc mục 3.2–3.3 và Hình 3 của bài gốc.

**Câu 03. Phần nào của nhóm, phần nào kế thừa tác giả?**

CLIP, FAA không gian–DWT và LGA được kế thừa từ FatFormer. Dự án bổ sung SRM, cơ chế điều tiết nhánh tần số phụ thuộc ảnh, dữ liệu và cách tinh chỉnh trên cặp ảnh sạch–suy thoái; đồng thời tổ chức hiệu chuẩn, kiểm chuẩn và demo. Chưng cất tri thức là kỹ thuật được áp dụng trong dự án, không phải phương pháp do nhóm phát minh. Hiệu quả riêng của từng thay đổi cần bằng chứng ablation tương ứng.

**Câu 04. Mô hình nhận gì, trả gì và phát hiện được vùng giả không?**

Demo nhận ảnh, tiền xử lý về crop 224×224 rồi trả điểm Fake và nhãn theo ngưỡng cố định. Quyết định dựa trên crop mà mô hình nhìn thấy. Demo có thể đo ảnh hưởng của từng vùng bằng cách che ô, nhưng bản đồ này chỉ giải thích độ nhạy của điểm dự đoán. Mô hình hiện không có bằng chứng kiểm định cho nhiệm vụ phân đoạn vùng giả mạo.

**Câu 05. Kết quả nào nên nói đầu tiên khi giảng viên hỏi nhóm đạt được gì?**

Trên CFE, XLA đạt AUC 88,66% ở ảnh sạch và 79,88% ở Q30. Trên 18 tập ảnh sạch, Macro BA đạt 90,10%. Mô hình cuối đạt 6/8 tiêu chí chất lượng và được chọn cho demo nghiên cứu. Khi trình bày, phải gắn mỗi số với tập và chỉ số; không gọi tất cả các số này là accuracy.

**Câu 06. Có thể giới thiệu toàn bộ dự án trong khoảng một phút thế nào?**

“Nhóm em nghiên cứu phát hiện ảnh AI dựa trên FatFormer. Bài gốc dùng CLIP, bộ thích nghi không gian–tần số và căn chỉnh ảnh–văn bản. Dự án bổ sung SRM và gating, tinh chỉnh bằng cặp ảnh sạch–suy thoái rồi chọn trọng số trên tập riêng. Sau đó, nhóm hiệu chuẩn, kiểm chuẩn nhiều bộ dữ liệu và đóng gói cho demo. XLA đạt AUC 88,66% trên CFE sạch và 79,88% ở Q30; Macro BA trên 18 tập sạch đạt 90,10%. Mô hình đạt 6/8 tiêu chí, nên nhóm xác định sản phẩm hiện phục vụ demo nghiên cứu và tiếp tục cải thiện khả năng nhận ảnh giả sau nén, khác nguồn.”

Đây là lời dẫn gợi ý. Người trình bày cần dùng cách nói tự nhiên và chỉ nhận những công việc nhóm thực tế đã làm.
