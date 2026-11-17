# Thành viên A Dữ liệu và giao thức đánh giá

## Phạm vi cần học

**A chịu trách nhiệm giải thích dữ liệu nào được dùng ở bước nào và kết quả được phép khái quát đến đâu.** Cần đọc mục 4.1 của bài FatFormer gốc; mục 2.2, 3, 4.3, 5.1 và 6.4 của báo cáo tổng thể XLA.

- Phân biệt ảnh thật/giả với lớp ngữ nghĩa; nắm danh sách tám tập GAN và mười cấu hình diffusion của bài gốc.
- Nắm Data v2, nguồn ảnh thật, số lượng hai nhãn, cách chia train/chọn mô hình/hiệu chuẩn.
- Phân biệt CFE, nhóm 18 tập và kho suy thoái lưu sẵn; giải thích bản ghi, nội dung độc lập và lượt dự đoán.
- Giải thích thứ tự resize → crop → suy thoái → normalize, các phép tăng cường và phạm vi chống rò rỉ.

**Sản phẩm ôn tập của A:** một sơ đồ chia dữ liệu trên một trang, một ví dụ đi qua tiền xử lý và phần giới thiệu bài toán trong khoảng hai phút. A cần trả lời được câu hỏi chỉ số cơ bản của C; nhờ B hỗ trợ khi giảng viên hỏi sâu về DWT hoặc tác dụng của SRM.

## Câu hỏi 07 đến 16

**Câu 07. “ProGAN bốn lớp” nghĩa là bài toán có bốn nhãn đầu ra không?**

Không. Bốn lớp là các nhóm nội dung car, cat, chair và horse dùng chọn dữ liệu huấn luyện. Hai lớp ngữ nghĩa trong thiết lập khác là chair và horse. Nhãn phân loại cuối vẫn là Real và Fake. Đây là điểm cần nói rõ vì nhầm lớp nội dung với nhãn phát hiện sẽ dẫn đến giải thích sai kiến trúc đầu ra.

**Câu 08. Bài gốc huấn luyện và kiểm thử trên những dữ liệu nào?**

Bài gốc huấn luyện trên ProGAN, với thiết lập hai hoặc bốn lớp ngữ nghĩa. Tám tập GAN kiểm thử gồm ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN và Deepfake. Mười cấu hình diffusion gồm PNDM, Guided, DALL-E, VQ-Diffusion, ba cấu hình LDM và ba cấu hình GLIDE. ProGAN có mặt trong huấn luyện và kiểm thử; không nên gọi toàn bộ các tập kiểm thử là nguồn sinh chưa từng thấy.

**Câu 09. Data v2 của dự án có bao nhiêu ảnh, cân bằng nhãn không và ảnh thật đến từ đâu?**

Data v2 có 14.400 ảnh, gồm 7.863 Real (54,60%) và 6.537 Fake (45,40%); hai nhãn gần cân bằng nhưng không bằng nhau. Nguồn Real theo danh mục gồm COCO, FFHQ, LandscapesHQ, VISION và ImageNet replay. Phần Fake có 17 bộ sinh được đặt tên trong danh mục. Dữ liệu replay thực giữ 2.991 ảnh; không đồng nhất số này với kho bổ sung ban đầu 3.600 ảnh.

**Câu 10. Tại sao phải tách train, tập chọn mô hình, tập hiệu chuẩn và test?**

Train cập nhật trọng số; tập chọn mô hình dùng chọn epoch hoặc cấu hình; tập hiệu chuẩn dùng ước lượng tham số điểm và ngưỡng; test dùng đánh giá sau khi các lựa chọn đã cố định. Data v2 chia thành 12.240 ảnh train, 1.080 ảnh chọn mô hình và 1.080 ảnh hiệu chuẩn. Nếu liên tục dùng test để chọn cấu hình, kết quả sẽ phản ánh việc thích nghi với test và cần được mô tả đúng phạm vi.

**Câu 11. CFE là gì, có phải toàn bộ Community Forensics không?**

CFE là tập con và giao thức của dự án lấy từ nguồn Community Forensics. Tổng 2.000 bản ghi gồm 400 ảnh hiệu chuẩn và 1.600 ảnh đánh giá, với 800 Real và 800 Fake ở phần đánh giá. Không gọi đây là toàn bộ Community Forensics. Cũng không gộp 400 ảnh này với 1.080 ảnh hiệu chuẩn của Data v2. CFE đã được xem trong quá trình nghiên cứu nên kết quả được trình bày là hậu kiểm.

**Câu 12. Đã đánh giá toàn bộ dữ liệu chưa, và 82.003 ảnh có ý nghĩa gì?**

Đợt master audit hoàn tất 37/37 lượt đánh giá trên tám kho dữ liệu đã nộp, tạo 1.618.061 lượt dự đoán. Nhóm 18 tập kiểm thử có 82.003 lượt ảnh cho mỗi điều kiện xử lý. Đây là phạm vi kho dữ liệu của dự án, không phải tất cả dữ liệu công khai trên thế giới. Một ảnh có thể xuất hiện ở nhiều điều kiện và ảnh thật tham chiếu có thể dùng chung, nên không coi mọi lượt ảnh là nội dung độc lập.

**Câu 13. Nhóm đã chống rò rỉ dữ liệu bằng cách nào?**

Hồ sơ kiểm chuẩn đối chiếu mã băm nội dung chính xác giữa các phần Data v2, các phần CFE và các nhóm dữ liệu được cấu hình để kiểm tra. Các phép kiểm đó ghi nhận không có vi phạm trong phạm vi đã xét. Mã băm chính xác phát hiện tệp hoặc nội dung trùng theo phép kiểm, không đủ phát hiện mọi ảnh gần giống, ảnh đổi kích thước hoặc cùng nguồn. Kiểm trùng theo cảm nhận và độc lập theo nguồn sinh vẫn cần xem xét riêng.

**Câu 14. Có thể nói toàn bộ 18 tập đều là dữ liệu chưa từng thấy không?**

Không thể suy ra điều đó chỉ từ tên nhóm 18 tập. ProGAN đã được dùng ở giai đoạn huấn luyện trước; cả 17 bộ sinh Fake trong tập chọn mô hình Data v2 đều có mặt ở train v2. CFE là đánh giá khác nguồn nhưng đã được dùng trong quá trình nghiên cứu. Khi nói tổng quát hóa, cần chỉ rõ độc lập ở mức ảnh, nguồn ảnh hay bộ sinh; ba mức này không đồng nghĩa.

**Câu 15. Tiền xử lý bài gốc và XLA có giống nhau hoàn toàn không?**

Bài gốc đổi ảnh thành 256×256 rồi lấy vùng cắt 224×224; huấn luyện cắt ngẫu nhiên và lật ngang, kiểm thử cắt giữa. Giao thức XLA dùng đổi cạnh ngắn thành 224 bằng bicubic, cắt giữa 224×224, tạo suy thoái tương ứng rồi chuẩn hóa theo CLIP. Thứ tự JPEG trước hay sau crop tạo ra đầu vào khác nhau. Vì vậy, JPEG Q30 trong kiểm chuẩn không đại diện cho mọi chuỗi xử lý ảnh của mạng xã hội.

**Câu 16. Vì sao dùng cặp ảnh sạch–suy thoái và phải dùng chung biến đổi hình học?**

Cặp ảnh giúp mô hình học cùng nội dung dưới hai điều kiện chất lượng và áp dụng các ràng buộc nhất quán. Dùng chung bước biến đổi hình học giúp hai nhánh so sánh cùng vùng nội dung. Nhánh suy thoái có xác suất 60% giữ nguyên ảnh; các phép còn lại gồm JPEG Q trong [30,85], WebP Q trong [30,70], JPEG hai lần, Gaussian blur và down-up. Các phép được chọn theo cấu hình, không phải tất cả luôn áp dụng trên mỗi ảnh. 60% là xác suất trong luồng tăng cường, không phải tỷ lệ Real của dataset.
