# Câu hỏi phản biện và cách luyện tập

## Câu hỏi 41 đến 50

**Cả ba thành viên cần trả lời được nhóm câu hỏi này.** Những câu trả lời sau giữ đúng phạm vi bằng chứng của sản phẩm cuối; không yêu cầu kể lại các nhánh thử nghiệm đã loại. A dẫn phần dữ liệu và giao thức, B dẫn phần thành phần mô hình, C dẫn phần kết quả và sản phẩm.

**Câu 41. Tại sao baseline nội bộ AUC Q30 chỉ 42,33%, thấp hơn mức xếp hạng ngẫu nhiên?**

Đó là số đo đã lưu của baseline nội bộ trên giao thức CFE Q30, không phải số công bố của tác giả FatFormer. Nguyên nhân chưa được xác định đầy đủ; nhóm không nên tự quy cho một nguyên nhân như JPEG hay nạp sai trọng số khi chưa có kiểm chứng. Kiểm nạp cấu trúc là một bằng chứng cần thiết nhưng chưa đủ xác nhận đã tái lập bài gốc. Cần chạy trọng số và mã chính thức theo giao thức công bố, rồi đối chiếu trên cùng dữ liệu và tiền xử lý. A và C cùng phụ trách.

**Câu 42. XLA có tốt hơn FatFormer gốc không?**

Bằng chứng hiện tại cho thấy XLA cải thiện so với baseline nội bộ đã lưu trên CFE. Ví dụ, AUC Q30 từ 42,33% lên 79,88%, chênh 37,55 điểm phần trăm. Chưa đủ cơ sở khẳng định vượt FatFormer gốc trong một phép so sánh công bằng đã tái lập. Kết quả 98,4% ACC của bài gốc và 88,66% AUC CFE của XLA thuộc các chỉ số và giao thức khác nhau. C trả lời chính, A bổ sung điều kiện dữ liệu.

**Câu 43. Nhóm đã chứng minh SRM, gating và KD đóng góp riêng bao nhiêu chưa?**

Ablation hiện có của XLA chỉ thay trọng số Feature và Guard qua bốn cấu hình; các thành phần khác được dùng chung và thí nghiệm mới có một seed. Do đó chưa tách được đóng góp riêng của SRM, gating hoặc KD. Bằng chứng phù hợp tiếp theo là các đợt huấn luyện đối chứng cùng dữ liệu và quy tắc chọn checkpoint, lần lượt bỏ SRM, cố định gating hoặc bỏ KD, rồi lặp seed khi có điều kiện. Không dùng việc tắt một nhánh lúc suy luận để thay hoàn toàn cho ablation huấn luyện. B trả lời chính.

**Câu 44. Dự án đặt mục tiêu tăng độ bền, vậy có đạt mục tiêu không?**

Mục tiêu đạt một phần trong phạm vi các phép đo hiện có. XLA cải thiện so với baseline nội bộ trên CFE, nhưng chất lượng vẫn giảm khi nén mạnh. Trên 18 tập, pooled ACC giảm từ 93,46% ở sạch xuống 70,91% ở Q30; Fake ACC giảm từ 96,52% xuống 56,37%. Kết luận phù hợp là đã tạo được mô hình và quy trình đánh giá cho nghiên cứu độ bền, đồng thời còn điểm yếu rõ ở ảnh giả sau nén và khác nguồn. C trả lời chính.

**Câu 45. Demo báo Fake với điểm rất cao có chứng minh ảnh chắc chắn là giả không?**

Điểm là đầu ra của mô hình sau hiệu chuẩn trong phạm vi dữ liệu và quy trình đã dùng. Điểm cao không phải bằng chứng pháp lý hoặc bảo đảm đúng trên một nguồn ảnh mới. Cần nhìn điều kiện tiền xử lý, crop, mức nén và sai số đã quan sát, nhất là khi ảnh nằm ngoài miền dữ liệu. Demo hỗ trợ đánh giá của mô hình, không thay cho xác minh nguồn gốc ảnh bằng bằng chứng khác. C trả lời chính, cả nhóm cần thống nhất cách diễn đạt.

**Câu 46. ECE thấp hoặc Real ACC gộp trên 90% có bảo đảm ít báo nhầm ở mọi trường hợp không?**

ECE đo mức khớp giữa độ tin cậy và tỷ lệ đúng theo cách nhóm điểm, không trực tiếp bảo đảm FPR ở ngưỡng triển khai. Dự án tính ECE bằng nhãn argmax tại ngưỡng 0,5, trong khi demo dùng ngưỡng khoảng 0,79185. Điểm Real gộp cũng có thể che các miền kém hơn; thực tế 9/18 tập có Real ACC dưới 90%. Cần kiểm tra riêng từng miền và điểm vận hành. A giải thích cách tổng hợp, C giải thích ngưỡng.

**Câu 47. Vì sao chỉ đạt 6/8 tiêu chí mà vẫn có sản phẩm cuối?**

Việc chọn một bản cố định để nghiên cứu và demo khác với việc đạt toàn bộ mục tiêu nghiệm thu. Hai tiêu chí còn thiếu là Clean Fake@Real90=64,125% so với mục tiêu ≥65%, và worst-subset Fake=30,8642% so với mục tiêu ≥45%. Nhóm giữ trạng thái này và công bố phạm vi demo nghiên cứu; không chuyển thành đạt toàn bộ chỉ vì đã đóng gói hoặc chạy được giao diện. C trả lời chính; cả ba phải nhớ hai điểm còn thiếu.

**Câu 48. Giảng viên hỏi hướng phát triển có cơ sở thì nên ưu tiên gì?**

Ưu tiên bám lỗi đã đo: cải thiện nhận ảnh giả sau nén và nhóm tự hồi quy khác nguồn, đồng thời bảo vệ ảnh thật. Cần dữ liệu và đánh giá độc lập theo nguồn sinh, hiệu chuẩn trên tập riêng, ablation tách SRM/gating/KD và nhiều seed. Có thể đo thêm thời gian toàn luồng demo và kiểm chứng Colab. Đây là công việc đề xuất, không phải kết quả đã hoàn thành. Bài gốc có phân tích mở rộng và độ bền; không nên nói tác giả hoàn toàn chưa xét suy thoái chỉ để tạo lý do cải tiến.

**Câu 49. Giải thích hình trong bài gốc và bản đồ vùng ảnh trong demo thế nào để không nhầm?**

Hình phân bố logits của bài gốc so UniFD ở hàng trên với FatFormer ở hàng dưới trên ProGAN, StyleGAN, Deepfake và LDM. Nó minh họa mức tách Real/Fake trên các mẫu đã chọn, không phải đường ROC hay bản đồ vị trí giả. Hình attention của bài gốc là một phép trực quan khác. Demo XLA dùng che từng ô trong lưới 4×4 của crop để đo thay đổi điểm Fake, cần thêm 16 lượt suy luận. Ô đỏ cho biết che vùng đó làm điểm Fake giảm; không chứng minh vùng đó bị chỉnh sửa, và không gọi đó là Grad-CAM. B và C phối hợp.

**Câu 50. Nếu bị hỏi một chi tiết ngoài phần mình nắm hoặc chưa có bằng chứng, nên trả lời ra sao?**

Trả lời phần đã xác nhận, nêu nguồn cần đối chiếu rồi chuyển cho người phụ trách. Ví dụ: “Phần em xác nhận là ngưỡng triển khai được hiệu chuẩn trên tập riêng và giữ cố định. Chi tiết tiêu chí tối ưu tham số, em xin để bạn phụ trách huấn luyện trình bày và đối chiếu hồ sơ.” Nếu cả nhóm chưa kiểm chứng, nói rõ chưa đo và đề xuất phép kiểm cụ thể. Không suy đoán số liệu, nhận kết quả tác giả thành kết quả nhóm hoặc khẳng định một tính năng chỉ vì có tên trong mã nguồn.

## Ba vòng luyện tập đề xuất

**Vòng 1 giúp từng người hiểu luồng tổng thể.** Cả nhóm đọc phần chung, sau đó A giải thích dữ liệu, B vẽ kiến trúc và C diễn giải kết quả. Mỗi người thử trả lời sáu câu chung bằng lời của mình; hai người còn lại yêu cầu nêu đúng nguồn và điều kiện của từng số.

**Vòng 2 kiểm tra chiều sâu và khả năng hỗ trợ nhau.** A học sâu câu 07–16, B học sâu câu 17–28, C học sâu câu 29–40. Mỗi người chuẩn bị sản phẩm ôn tập đã giao, rồi dạy lại phần đó cho hai bạn. Người nghe phải nhắc lại được ý chính mà không nhìn tài liệu. Cả nhóm đối chiếu mọi giá trị với báo cáo cuối và hướng dẫn demo.

**Vòng 3 mô phỏng buổi bảo vệ.** Trình bày theo A → B → C rồi đảo thứ tự hỏi ngẫu nhiên câu 41–50. Một người làm giảng viên, một người trả lời, một người ghi lỗi; luân phiên các vai. Sau mỗi câu, kiểm tra câu trả lời có đúng mô hình, đúng dữ liệu, đúng chỉ số và đúng phạm vi bằng chứng hay không.

## Một lượt hỏi nối tiếp để tự kiểm tra

- A trả lời “ProGAN bốn lớp là bốn nhãn đầu ra không?”; B tiếp nối bằng cách giải thích vì sao đầu ra vẫn Real/Fake.
- B giải thích gating; C tiếp nối bằng cách phân biệt nhiệt độ KD với nhiệt độ hiệu chuẩn triển khai.
- C nêu AUC CFE; A tiếp nối bằng việc chỉ ra 400 ảnh hiệu chuẩn khác 1.600 ảnh đánh giá thế nào.
- A giải thích 82.003 lượt ảnh; C tiếp nối bằng việc phân biệt số lượt dự đoán và số nội dung độc lập.
- C trình diễn nhãn và bản đồ che vùng; B giải thích vì sao bản đồ độ nhạy không phải mặt nạ giả mạo.

## Điều kiện cả nhóm sẵn sàng

- Mỗi người giới thiệu được bài toán, mô hình gốc, phần đóng góp và kết luận của dự án mà không cần đọc thuộc câu chữ.
- Mỗi người trả lời sâu được nhóm câu hỏi chính của mình và tóm tắt được hai nhóm còn lại.
- Cả nhóm nhớ các số liệu tối thiểu, phân biệt ACC/AP/AUC, Macro BA/pooled ACC và Fake ACC/Fake@Real90.
- Có thể mở đúng hình hoặc bảng để chứng minh câu trả lời; không dùng số cũ của ứng viên khác cho mô hình cuối.
- Demo được chạy thử trong đúng môi trường sẽ dùng khi trình bày; có phương án giải thích khi thiếu GPU hoặc có sai khác gần ngưỡng.
- Khi nói hạn chế, nêu rõ bằng chứng hiện có và phép kiểm cần làm tiếp; không kể lan sang lịch sử các nhánh đã loại.

**Câu chốt cả nhóm cần thống nhất:** “Nhóm đã xây dựng và kiểm chuẩn một bản FatFormer XLA cố định cho demo nghiên cứu. Kết quả cho thấy cải thiện so với baseline nội bộ theo giao thức đã lưu, đồng thời còn giới hạn về độ bền sau nén, khác nguồn và bằng chứng cho từng thành phần.”
