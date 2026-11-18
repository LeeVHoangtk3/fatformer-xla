# Tài liệu học và chuẩn bị bảo vệ FatFormer XLA

Nhóm chia chuyên môn theo vai trò hiện có: **A phụ trách dữ liệu và giao thức đánh giá; B phụ trách kiến trúc và hàm mất mát; C phụ trách huấn luyện, kết quả và demo**. Cả ba cùng học phần kiến thức chung và câu hỏi phản biện. Đây là đề xuất phân công ôn tập, không phải xác nhận đóng góp thực tế của từng người.

- **Bản Word tổng hợp:** [Phân công học và hỏi đáp bảo vệ](Phan_Cong_Hoc_Va_Hoi_Dap_Bao_Ve_FatFormer_XLA.docx).
- **Cả nhóm đọc trước:** [Kiến thức chung và câu 01–06](00_Kien_thuc_chung.md).
- **Thành viên A:** [Dữ liệu và giao thức, câu 07–16](01_Thanh_vien_A_Du_lieu_va_giao_thuc.md).
- **Thành viên B:** [Kiến trúc và phương pháp, câu 17–28](02_Thanh_vien_B_Kien_truc_va_phuong_phap.md).
- **Thành viên C:** [Huấn luyện, kết quả, demo, câu 29–40](03_Thanh_vien_C_Huan_luyen_ket_qua_demo.md).
- **Cả nhóm luyện vấn đáp:** [Câu phản biện 41–50 và cách luyện tập](04_Phan_bien_va_luyen_tap.md).

Các đáp án là gợi ý dựa trên tài liệu đã có, không phải danh sách câu hỏi chắc chắn giảng viên sẽ hỏi. Số liệu của bài FatFormer gốc và dự án XLA được ghi riêng. Tài liệu tập trung vào quy trình tạo mô hình cuối, kết quả và phạm vi sử dụng; không kể lại lịch sử các nhánh thử nghiệm đã loại.

## Nguồn cần mở khi học

- [Bài FatFormer gốc trên CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Forgery-aware_Adaptive_Transformer_for_Generalizable_Synthetic_Image_Detection_CVPR_2024_paper.pdf): mục 3 về phương pháp, mục 4 về dữ liệu, cài đặt, kết quả và ablation.
- [Báo cáo tổng thể XLA](../../overall/final/Bao_Cao_Tong_The_Mo_Hinh_FatFormer_XLA.docx): nguồn chính cho cấu hình và số liệu dự án.
- [Bản paper XLA](../../overall/final/Paper_FatFormer_XLA_CVPR_Format.docx): đọc để luyện trình bày ngắn gọn.
- [Hướng dẫn demo](../../../demo/README.md) và [kiểm chứng demo](../../../demo/validation.md): nguồn cho hành vi thực tế của demo Windows và trạng thái Colab.
- [Cấu hình huấn luyện cuối](../../../../configs/train_phase7_anchored_v2.yaml): dùng khi cần đối chiếu siêu tham số; ưu tiên hành vi thực thi và hồ sơ kiểm chứng hơn các lời mô tả trong chú thích cũ.

Nên trả lời bằng ý chính, dẫn một bằng chứng và nêu phạm vi áp dụng. Khi được hỏi số liệu, luôn nói đủ mô hình, tập dữ liệu, điều kiện ảnh và chỉ số.
