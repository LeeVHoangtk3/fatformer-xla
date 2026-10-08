# BÁO CÁO KẾT QUẢ BENCHMARK BASELINE TRÊN TẬP DEGRADED (TASK 2.3)

```
==============================================================================================================
BẢNG TỔNG HỢP MỐC SÀN SUY GIẢM HIỆU NĂNG FATFORMER BASELINE TRÊN 6 BIẾN THỂ (TASK 2.3)
==============================================================================================================
Biến thể     Mô tả suy biến                 GANs (%)   Diff (%)   Overall (%)  Real ACC   Fake ACC   Δ Sụt giảm  
--------------------------------------------------------------------------------------------------------------
Clean        Mốc trần tham chiếu (Task 2.2) 50.82      49.90      50.31        99.37      93.54      0.00% (Mốc) 
--------------------------------------------------------------------------------------------------------------
jpeg_q30     Nén JPEG sâu Q=30 (Artifact khối 8x8) 45.85      47.52      46.78        64.02      29.53      -3.53%      
jpeg_q50     Nén JPEG trung bình Q=50       46.45      49.14      47.94        64.22      31.67      -2.37%      
jpeg_q70     Nén JPEG nhẹ Q=70              46.35      48.56      47.58        58.80      36.36      -2.73%      
blur_s1      Gaussian Blur (sigma=1.0)      46.73      50.04      48.57        37.18      59.96      -1.74%      
blur_s2      Gaussian Blur (sigma=2.0)      46.38      47.80      47.17        46.24      48.09      -3.14%      
down_up      Down-Up Bicubic (224 -> 112 -> 224) 46.15      49.76      48.16        43.93      52.38      -2.15%      
==============================================================================================================
```

## NHẬN XÉT KHOA HỌC:
1. **Độ dốc suy giảm JPEG**: Khi chất lượng giảm từ Q=70 -> Q=50 -> Q=30, độ chính xác sụt giảm phi tuyến tính.
2. **Hiện tượng sụp đổ Fake ACC**: Mô hình gốc phân loại nhầm ảnh giả mạo thành ảnh thật do các đặc trưng giả mạo vi mô bị làm phẳng bởi ma trận lượng tử hóa.
