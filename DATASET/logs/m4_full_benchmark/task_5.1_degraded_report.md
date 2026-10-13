# BÁO CÁO KẾT QUẢ BENCHMARK BASELINE TRÊN TẬP DEGRADED (TASK 2.3)

```
==============================================================================================================
BẢNG TỔNG HỢP MỐC SÀN SUY GIẢM HIỆU NĂNG FATFORMER BASELINE TRÊN 6 BIẾN THỂ (TASK 2.3)
==============================================================================================================
Biến thể     Mô tả suy biến                 GANs (%)   Diff (%)   Overall (%)  Real ACC   Fake ACC   Δ Sụt giảm  
--------------------------------------------------------------------------------------------------------------
Clean        Mốc trần tham chiếu (Task 2.2) 71.16      64.29      67.34        99.37      93.54      0.00% (Mốc) 
--------------------------------------------------------------------------------------------------------------
jpeg_q30     Nén JPEG sâu Q=30 (Artifact khối 8x8) 53.25      49.56      51.20        74.87      27.53      -16.14%     
jpeg_q50     Nén JPEG trung bình Q=50       56.60      53.20      54.71        66.96      42.47      -12.63%     
jpeg_q70     Nén JPEG nhẹ Q=70              59.35      54.46      56.63        70.98      42.29      -10.71%     
blur_s1      Gaussian Blur (sigma=1.0)      65.33      54.16      59.12        83.53      34.71      -8.22%      
blur_s2      Gaussian Blur (sigma=2.0)      56.35      50.16      52.91        85.91      19.91      -14.43%     
down_up      Down-Up Bicubic (224 -> 112 -> 224) 62.73      54.38      58.09        87.09      29.09      -9.25%      
==============================================================================================================
```

## NHẬN XÉT KHOA HỌC:
1. **Độ dốc suy giảm JPEG**: Khi chất lượng giảm từ Q=70 -> Q=50 -> Q=30, độ chính xác sụt giảm phi tuyến tính.
2. **Hiện tượng sụp đổ Fake ACC**: Mô hình gốc phân loại nhầm ảnh giả mạo thành ảnh thật do các đặc trưng giả mạo vi mô bị làm phẳng bởi ma trận lượng tử hóa.
