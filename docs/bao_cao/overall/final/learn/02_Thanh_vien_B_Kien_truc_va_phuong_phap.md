# Thành viên B Kiến trúc và phương pháp

## Phạm vi cần học

**B chịu trách nhiệm giải thích một ảnh đi qua mô hình thế nào và từng thành phần học điều gì.** Cần đọc mục 3.1–3.3, Hình 3 và Bảng 3 của bài FatFormer gốc; mục 2.1, 2.4 và 4 của báo cáo tổng thể XLA. Khi học hình, phải phân biệt sơ đồ FatFormer gốc với sơ đồ phần mở rộng XLA.

- Vẽ được luồng ảnh → CLIP ViT → FAA không gian/tần số → LGA → logits Real/Fake.
- Chỉ ra SRM và gating được bổ sung ở đâu; phân biệt tham số cố định, tham số học được và tham số đóng băng.
- Giải thích từng số hạng loss, vai trò teacher và thao tác chặn gradient; phân biệt mục đích thiết kế với hiệu quả đã kiểm chứng.
- Nắm ablation của bài gốc và phạm vi ablation hiện có của dự án, tránh dùng bằng chứng của tác giả để xác nhận đóng góp riêng của nhóm.

**Sản phẩm ôn tập của B:** một sơ đồ kiến trúc tự vẽ, một trang giải thích các thành phần loss và phần diễn giải một ảnh đi qua mô hình trong khoảng ba phút. B cần nắm số liệu chính của C và nhờ A xác nhận khi câu hỏi liên quan nguồn dữ liệu hoặc độ độc lập của test.

![Kiến trúc FatFormer gốc](../../overall/final/hinh_bai_bao_goc/fat_former_architecture_cvpr2024.png)

Sơ đồ trích Hình 3 của Liu và cộng sự, CVPR 2024, trang 10772. Đây là kiến trúc bài gốc; SRM và gating của XLA được giải thích riêng trong báo cáo dự án.

## Câu hỏi 17 đến 28

**Câu 17. Vì sao dùng CLIP thay vì huấn luyện một mạng phân loại từ đầu?**

CLIP cung cấp không gian biểu diễn ảnh–văn bản đã học trước trên dữ liệu quy mô lớn. FatFormer khai thác nền này và học các thành phần thích nghi cho dấu vết giả mạo. So với chỉ gắn bộ phân loại lên đặc trưng CLIP đóng băng như UniFD, bài gốc hướng tới việc điều chỉnh biểu diễn cho nhiệm vụ phát hiện. Lợi ích là mục tiêu nghiên cứu được kiểm tra bằng thực nghiệm, không phải bảo đảm CLIP luôn nhận đúng ảnh từ mọi bộ sinh.

**Câu 18. Patch token và CLS token trong ViT là gì?**

Ảnh được chia thành các ô để tạo chuỗi biểu diễn vùng ảnh, gọi là patch token. CLS token tổng hợp thông tin phục vụ biểu diễn toàn ảnh. Với ảnh 224×224 và patch 14×14 của ViT-L/14, có 16×16=256 patch token, cộng một CLS token; đây là số suy ra từ kích thước đầu vào. Trong cấu hình XLA, chiều đặc trưng là 1.024. LGA sử dụng cả thông tin toàn ảnh và thông tin cục bộ.

**Câu 19. FAA hoạt động ra sao và được đặt ở đâu?**

FAA kết hợp một nhánh không gian và một nhánh tần số giữa các giai đoạn của bộ mã hóa ảnh ViT. Nhánh không gian dùng Conv–ReLU–Conv để học dấu vết cục bộ. Nhánh tần số dùng DWT, attention giữa và trong các băng, rồi IDWT để đưa đặc trưng trở lại miền ảnh. Trong XLA, ba bộ thích nghi nằm ở các khối 7, 15 và 23. Không mô tả chúng như hai bộ phân loại độc lập được lấy trung bình ở đầu ra.

**Câu 20. DWT khác SRM thế nào và LL/LH/HL/HH có ý nghĩa gì?**

DWT là biến đổi wavelet tách đầu vào thành bốn băng theo tổ hợp lọc thông thấp và thông cao ở hai chiều. LL giữ phần biến thiên chậm; các băng còn lại mang chi tiết theo các tổ hợp hướng. Bài gốc dùng attention trong từng băng và giữa các băng để học quan hệ. DWT còn giữ cấu trúc vị trí phù hợp với thiết kế này. Không nên nói FAA chỉ sử dụng tần số cao, vì sơ đồ và phép xử lý có cả LL.

**Câu 21. LGA có vai trò gì và kiểm thử có cần nhập mô tả văn bản cho mỗi ảnh không?**

LGA gồm patch-based enhancer và text-guided interactor. Thành phần đầu dùng patch token để làm giàu ngữ cảnh của prompt; thành phần sau căn chỉnh patch token với biểu diễn văn bản của lớp. Mô hình kết hợp độ tương đồng từ CLS và các patch đã căn chỉnh để tạo điểm phân loại. Khi kiểm thử, người dùng chỉ cần đưa ảnh; các biểu diễn lớp Real/Fake thuộc cơ chế nội bộ, không yêu cầu người dùng tự viết chú thích cho từng ảnh.

**Câu 22. SRM là gì, có phải một mạng được huấn luyện đầy đủ không?**

SRM trong XLA tạo vết dư bằng ba bộ lọc cố định: sai phân bậc nhất, Laplacian và square 5×5. Áp dụng theo từng kênh RGB tạo chín bản đồ vết dư. Các lớp chiếu học được đưa thông tin này về biểu diễn tương thích với nhánh thích nghi. Bộ lọc cố định và lớp chiếu học được có vai trò khác nhau; không nên nói toàn bộ SRM đều đóng băng hoặc toàn bộ đều được học.

**Câu 23. Dự án ghép SRM và DWT vào mô hình như thế nào?**

Vết dư SRM được bổ sung vào nhánh không gian với hệ số 0,1 trong biểu thức adapter. Nhánh DWT được nhân với hệ số tần số và giá trị gating phụ thuộc ảnh. Đầu ra adapter đi theo kết nối residual của bộ mã hóa. SRM là phép lọc trong miền không gian, còn DWT là biến đổi wavelet; việc cùng khai thác dấu vết chi tiết không làm chúng trở thành cùng một phép biến đổi.

**Câu 24. Gating của XLA khác hệ số λ của FatFormer gốc ở đâu?**

Bài gốc đã có hệ số λ học được để phối hợp nhánh ảnh và tần số. XLA bổ sung hệ số phụ thuộc đặc trưng của từng ảnh: lấy trung bình token, đưa qua MLP rồi sigmoid và nhân 2. Vì vậy, giá trị có miền giới hạn lý thuyết từ 0 đến 2. Tầng cuối khởi tạo bằng không nên giá trị ban đầu bằng 1. Không khẳng định gating luôn giảm đơn điệu khi JPEG mạnh hơn nếu chưa đo và kiểm chứng hành vi đó.

**Câu 25. Vì sao dùng khởi tạo bằng không, và làm vậy có khiến mô hình không học được không?**

Khởi tạo lớp cuối của gating bằng không làm hệ số khởi đầu bằng 1, còn lớp chiếu cuối SRM bằng không làm phần bổ sung ban đầu bằng không. Mục đích là hạn chế thay đổi đột ngột khi gắn thành phần mới vào mô hình đã học. Không phải mọi trọng số trong mạng đều bằng không; các lớp vẫn có thể nhận cập nhật theo đường gradient của chúng. Đây là cách tổ chức điểm khởi đầu, không tự chứng minh mô hình sẽ bền hơn.

**Câu 26. Hàm mất mát cuối có những thành phần nào, vì sao không chỉ dùng CE?**

Sáu số hạng gồm CE ảnh sạch, CE ảnh suy thoái, KD, nhất quán logits (Cons), nhất quán đặc trưng (Feature) và bảo vệ ảnh thật (Guard). Trọng số lần lượt là 0,3; 0,3; 0,7; 0,2; 0,3; 0,5. CE học phân loại; các ràng buộc bổ sung nhằm giữ tri thức và giảm thay đổi giữa hai phiên bản ảnh. Trong mã, KD và Cons đã nhân trọng số trong mô-đun, nên không nhân thêm lần nữa khi cộng tổng loss. Mức đóng góp thực tế cần đọc cùng ablation.

**Câu 27. KD, Cons và Feature khác nhau thế nào? Vì sao cần stop-gradient?**

KD truyền phân bố dự đoán của teacher đóng băng sang student; cấu hình dùng nhiệt độ 2 và chọn các mẫu teacher dự đoán đúng. Cons giữ logits của hai nhánh nhất quán bằng sai số bình phương. Feature kết hợp khoảng cách cosine đặc trưng và KL giữa dự đoán sạch–suy thoái. Chặn gradient ở nhánh mục tiêu giúp phép tối ưu cập nhật phía đang học thay vì đồng thời đuổi theo một mục tiêu đang bị kéo bởi cùng ràng buộc. KD không phải triplet loss.

**Câu 28. Guard bảo vệ ảnh thật ra sao, có bảo đảm Real ACC luôn trên 90% không?**

Guard áp dụng trên mẫu Real và phạt khi logit Fake của student ở ảnh suy thoái vượt logit Fake của teacher ở ảnh sạch. Phần vượt được đưa qua ReLU, bình phương và lấy trung bình; đầu ra teacher được chặn gradient. Đây là ràng buộc tối ưu, không phải bảo đảm thống kê cho mọi tập test. Yêu cầu Real ACC ≥90% phải kiểm tra bằng số đo riêng ở từng điều kiện và từng miền dữ liệu.
