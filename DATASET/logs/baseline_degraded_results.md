# BÁO CÁO KẾT QUẢ BENCHMARK BASELINE TRÊN TẬP DEGRADED (TASK 2.3)

```
==============================================================================================================
BẢNG TỔNG HỢP MỐC SÀN SUY GIẢM HIỆU NĂNG FATFORMER BASELINE TRÊN 6 BIẾN THỂ (TASK 2.3)
==============================================================================================================
Biến thể     Mô tả suy biến                 GANs (%)   Diff (%)   Overall (%)  Real ACC   Fake ACC   Δ Sụt giảm  
--------------------------------------------------------------------------------------------------------------
Clean        Mốc trần tham chiếu (Task 2.2) 98.39      94.91      96.45        99.37      93.54      0.00% (Mốc) 
--------------------------------------------------------------------------------------------------------------
jpeg_q30     Nén JPEG sâu Q=30 (Artifact khối 8x8) 54.65      51.48      52.89        100.00     5.78       -43.56%     
jpeg_q50     Nén JPEG trung bình Q=50       60.08      53.18      56.24        100.00     12.49      -40.21%     
jpeg_q70     Nén JPEG nhẹ Q=70              68.22      55.72      61.28        99.96      22.60      -35.17%     
blur_s1      Gaussian Blur (sigma=1.0)      86.20      60.20      71.76        99.47      44.04      -24.69%     
blur_s2      Gaussian Blur (sigma=2.0)      76.62      56.66      65.53        98.27      32.80      -30.92%     
down_up      Down-Up Bicubic (224 -> 112 -> 224) 82.85      58.86      69.52        99.29      39.76      -26.93%     
==============================================================================================================
```

## NHẬN XÉT KHOA HỌC:
1. **Độ dốc suy giảm JPEG**: Khi chất lượng giảm từ Q=70 -> Q=50 -> Q=30, độ chính xác sụt giảm phi tuyến tính.
2. **Hiện tượng sụp đổ Fake ACC**: Mô hình gốc phân loại nhầm ảnh giả mạo thành ảnh thật do các đặc trưng giả mạo vi mô bị làm phẳng bởi ma trận lượng tử hóa.
