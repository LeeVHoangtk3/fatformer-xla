# BÁO CÁO KẾT QUẢ BENCHMARK BASELINE TRÊN TẬP DEGRADED (TASK 2.3)

```
==============================================================================================================
BẢNG TỔNG HỢP MỐC SÀN SUY GIẢM HIỆU NĂNG FATFORMER BASELINE TRÊN 6 BIẾN THỂ (TASK 2.3)
==============================================================================================================
Biến thể     Mô tả suy biến                 GANs (%)   Diff (%)   Overall (%)  Real ACC   Fake ACC   Δ Sụt giảm  
--------------------------------------------------------------------------------------------------------------
Clean        Mốc trần tham chiếu (Task 2.2) 98.39      94.91      96.45        99.37      93.54      0.00% (Mốc) 
--------------------------------------------------------------------------------------------------------------
jpeg_q30     Nén JPEG sâu Q=30 (Artifact khối 8x8) 45.85      47.52      46.78        64.02      29.53      -49.67%     
==============================================================================================================
```

## NHẬN XÉT KHOA HỌC:
1. **Độ dốc suy giảm JPEG**: Khi chất lượng giảm từ Q=70 -> Q=50 -> Q=30, độ chính xác sụt giảm phi tuyến tính.
2. **Hiện tượng sụp đổ Fake ACC**: Mô hình gốc phân loại nhầm ảnh giả mạo thành ảnh thật do các đặc trưng giả mạo vi mô bị làm phẳng bởi ma trận lượng tử hóa.
