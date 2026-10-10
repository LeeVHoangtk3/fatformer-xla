# BÁO CÁO KẾT QUẢ BENCHMARK BASELINE TRÊN TẬP DEGRADED (TASK 2.3)

```
==============================================================================================================
BẢNG TỔNG HỢP MỐC SÀN SUY GIẢM HIỆU NĂNG FATFORMER BASELINE TRÊN 6 BIẾN THỂ (TASK 2.3)
==============================================================================================================
Biến thể     Mô tả suy biến                 GANs (%)   Diff (%)   Overall (%)  Real ACC   Fake ACC   Δ Sụt giảm  
--------------------------------------------------------------------------------------------------------------
Clean        Mốc trần tham chiếu (Task 2.2) 93.99      87.16      90.19        99.37      93.54      0.00% (Mốc) 
--------------------------------------------------------------------------------------------------------------
jpeg_q30     Nén JPEG sâu Q=30 (Artifact khối 8x8) 74.50      68.26      71.03        97.22      44.84      -19.16%     
jpeg_q50     Nén JPEG trung bình Q=50       80.30      74.52      77.09        96.24      57.93      -13.10%     
jpeg_q70     Nén JPEG nhẹ Q=70              83.45      77.68      80.24        96.18      64.31      -9.95%      
blur_s1      Gaussian Blur (sigma=1.0)      87.15      73.36      79.49        97.00      61.98      -10.70%     
blur_s2      Gaussian Blur (sigma=2.0)      82.15      69.80      75.29        94.69      55.89      -14.90%     
down_up      Down-Up Bicubic (224 -> 112 -> 224) 83.17      67.62      74.53        97.62      51.44      -15.66%     
==============================================================================================================
```

## NHẬN XÉT KHOA HỌC:
1. **Độ dốc suy giảm JPEG**: Khi chất lượng giảm từ Q=70 -> Q=50 -> Q=30, độ chính xác sụt giảm phi tuyến tính.
2. **Hiện tượng sụp đổ Fake ACC**: Mô hình gốc phân loại nhầm ảnh giả mạo thành ảnh thật do các đặc trưng giả mạo vi mô bị làm phẳng bởi ma trận lượng tử hóa.
