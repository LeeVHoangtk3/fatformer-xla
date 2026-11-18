# Thành viên C Huấn luyện kết quả và demo

## Phạm vi cần học

**C chịu trách nhiệm nối phương pháp với bộ trọng số cuối, giải thích đúng số đo và trình diễn sản phẩm.** Cần đọc mục 4.1–4.3 của bài FatFormer gốc; mục 3, 5.2–5.4, 6–8 và phần demo trong báo cáo tổng thể XLA. Khi nói về trạng thái chạy demo, đọc thêm hướng dẫn và biên bản kiểm chứng demo hiện có.

- Nắm cấu hình huấn luyện chính, teacher/student, quy tắc chọn epoch và sự khác nhau giữa lựa chọn mô hình với hiệu chuẩn triển khai.
- Đọc được ACC, AP, AUC, Macro BA, Real/Fake ACC, Fake@Real90 và ECE; luôn gắn số với giao thức.
- Nắm kết quả CFE, nhóm 18 tập, độ phủ master audit và hai tiêu chí chất lượng còn thiếu.
- Giải thích được quy tắc điểm/ngưỡng, pipeline demo, Windows CPU so với Colab GPU và giới hạn của bản đồ che vùng.

**Sản phẩm ôn tập của C:** một trang số liệu có điều kiện đo, một lần chạy thử demo đúng package và một phần kết luận trong khoảng hai phút. A hỗ trợ câu hỏi về độ độc lập dữ liệu; B hỗ trợ câu hỏi gradient, loss và kiến trúc.

## Câu hỏi 29 đến 40

**Câu 29. Cấu hình huấn luyện cuối của dự án là gì?**

Đợt chính chạy 5 epoch, seed 42, batch nhỏ 16 và tích lũy gradient 2 lần. Tốc độ học của nhóm mới là 10⁻⁴, adapter cũ là 10⁻⁵; mức tối thiểu 10⁻⁶, weight decay 10⁻⁴. Huấn luyện dùng AMP. Cần phân biệt cấu hình này với bài gốc dùng Adam, 25 epoch, batch tổng 256, LR ban đầu 4×10⁻⁴ và nhân 0,9 sau mỗi 10 epoch.

**Câu 30. Teacher, student và huấn luyện có neo nghĩa là gì?**

Teacher là mô hình tham chiếu giữ cố định để cung cấp phân bố mục tiêu; student là XLA đang được tinh chỉnh. Phần nền CLIP được đóng băng, các thành phần thích nghi được cập nhật với tốc độ học theo nhóm. “Có neo” mô tả việc dùng tri thức teacher và các ràng buộc để hạn chế thay đổi không mong muốn. Không có nghĩa toàn bộ student bị đóng băng hoặc chất lượng luôn được giữ nguyên trên mọi miền.

**Câu 31. Vì sao chọn epoch 4 thay vì epoch cuối, và có phải AB4 không?**

Chọn checkpoint dựa trên quy tắc của tập chọn mô hình, không mặc định epoch cuối tốt nhất. Đợt chính chọn epoch 4, với kết quả nhóm yếu nhất trên tập chọn đạt 46,67%. AB4 là một cấu hình ablation được huấn luyện độc lập, không phải tên của checkpoint triển khai. “Worst-group” trên tập chọn và “worst-subset” trên CFE cũng là hai cách nhóm dữ liệu khác nhau, nên không thay thế số của nhau.

**Câu 32. Hiệu chuẩn làm gì, nhiệt độ và ngưỡng cuối là bao nhiêu?**

Hiệu chuẩn điều chỉnh cách chuyển logits thành điểm và chọn điểm cắt theo giao thức; bước này không cập nhật lại trọng số mô hình. Bản triển khai dùng T=2,7832491398 và ngưỡng=0,7918508649, ước lượng trên 400 ảnh CFE hiệu chuẩn riêng. Nhiệt độ này khác nhiệt độ KD bằng 2 và khác điểm vận hành lịch sử dùng khi chọn mô hình. Mục tiêu FPR 5% trên tập hiệu chuẩn không tự bảo đảm FPR test giữ nguyên.

**Câu 33. Điểm Fake 0,75 thì demo quyết định thế nào, vì sao không dùng ngưỡng 0,5?**

Demo so điểm Fake đã hiệu chuẩn với ngưỡng cố định của package. Với ngưỡng khoảng 0,79185, điểm 0,75 được xếp Real; điểm bằng hoặc vượt ngưỡng được xếp Fake. Đây chỉ là ví dụ tính theo quy tắc quyết định, không phải kết quả chạy một ảnh cụ thể. Ngưỡng 0,5 thuộc quy tắc argmax mặc định và một số cách tính chỉ số, không phải ngưỡng triển khai đã chọn.

**Câu 34. ACC, AUC, AP và Macro BA khác nhau thế nào?**

ACC là tỷ lệ phân loại đúng tại một ngưỡng. AUC tóm tắt khả năng xếp hạng qua đường ROC; AP tóm tắt quan hệ precision–recall, nên AP và AUC không phải cùng chỉ số. Balanced accuracy là trung bình recall Real và recall Fake. Macro BA lấy trung bình balanced accuracy theo tập; pooled ACC tính trên mọi lượt ảnh và bị ảnh hưởng bởi kích thước từng tập. Không được trừ trực tiếp ACC bài gốc với AUC dự án để nói mức cải thiện.

**Câu 35. Fake@Real90 có phải tỷ lệ phát hiện ảnh giả tại ngưỡng demo không?**

Không. Fake@Real90 đo recall Fake tại ngưỡng tính theo giao thức để nhận đúng 90% ảnh thật; trong dự án, ngưỡng này dùng phân vị điểm Real của tập đánh giá. Fake ACC dùng ngưỡng triển khai cố định. Vì vậy, trên CFE sạch có Fake@Real90=64,125% trong khi Fake ACC tại ngưỡng triển khai là 53,125%. Không lấy ngưỡng tính từ tập đánh giá này làm ngưỡng demo để quảng bá hiệu năng triển khai.

**Câu 36. Kết quả chính của bài gốc là gì, và đọc Bảng 3 ablation thế nào?**

Với ProGAN bốn lớp ngữ nghĩa, bài gốc công bố 98,4% ACC và 99,7% AP trung bình tám tập GAN; 95,0% ACC và 98,8% AP trung bình mười cấu hình diffusion. Bảng 3 của tác giả kiểm tra nhánh không gian/tần số, attention giữa/trong băng, cách giám sát và prompt, cùng FAA/LGA. Đây là bằng chứng cho thiết kế của bài gốc theo giao thức của tác giả, không thay cho ablation SRM/gating của XLA.

**Câu 37. Kết quả XLA trên CFE thế nào, mô hình yếu ở điều kiện nào?**

Trên 1.600 ảnh CFE, AUC sạch là 88,66%, JPEG Q30 là 79,88%; Fake@Real90 tương ứng 64,125% và 42,625%. Tại ngưỡng triển khai, ACC sạch là 74,38% và Q30 là 65,81%; Fake ACC tương ứng 53,125% và 40,25%. Mô hình có mức nhận đúng Real cao hơn Fake ở các điều kiện này. CFE và nhóm 18 tập dùng dữ liệu khác nhau, nên không lấy AUC Q30 của hai nhóm thay thế lẫn nhau.

**Câu 38. Kết quả trên 18 tập có thể kết luận gì?**

Trên ảnh sạch, Macro BA đạt 90,10%, pooled ACC 93,46%, pooled Real ACC 90,41% và pooled Fake ACC 96,52%. Khi JPEG Q30, pooled ACC còn 70,91% và Fake ACC còn 56,37%. Điểm tổng hợp cho thấy chất lượng trên nhóm dữ liệu đã đo, nhưng 9/18 tập có Real ACC dưới 90%. Cần đọc cả từng tập và từng lớp để thấy sự khác biệt giữa các miền.

**Câu 39. Master audit gồm 1.618.061 dự đoán thì có nghĩa đã kiểm thử trên từng ấy ảnh mới không?**

Không. Đây là số lượt dự đoán của 37 lượt đánh giá trên tám kho dữ liệu, gồm nhiều vai trò dữ liệu và nhiều điều kiện biến đổi. Hồ sơ có 944.327 bản ghi ảnh và 815.689 mã băm nội dung phân biệt. Chạy suy luận trên phần train giúp kiểm tra trong phạm vi audit, nhưng không biến train thành test độc lập. Khi báo cáo khả năng tổng quát hóa, phải dùng đúng các phần đánh giá được chỉ định.

**Câu 40. Windows CPU và Colab GPU có làm mô hình thông minh hơn khác nhau không, demo đã được kiểm chứng đến đâu?**

Nếu dùng cùng trọng số, tiền xử lý và hiệu chuẩn thì thay CPU bằng GPU chủ yếu thay đổi tốc độ và cách tính số, không bổ sung tri thức cho mô hình. Có thể có sai khác FP32/FP16, nhất là gần ngưỡng, nên cần kiểm chứng từng môi trường. Hồ sơ demo Windows CPU đã đối chiếu 40 lượt sạch/Q30 trên 20 ảnh minh họa, nhãn khớp reference của cùng backend; đó không phải 40 dự đoán đều đúng nhãn nguồn. Phiên notebook Colab GPU mới còn cần chạy thực tế để nghiệm thu. Số đo L4 trong master audit chỉ gồm suy luận trên tensor, không phải thời gian upload và xử lý giao diện.
