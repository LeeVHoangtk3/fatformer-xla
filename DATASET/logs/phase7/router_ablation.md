# BẢNG ABLATION DEGRADATION ROUTER (TASK 7.13)
> **Evidence Level**: `measured` | **Tiêu chuẩn kết luận**: Thắng nếu `Fake@Real90` vượt >= 1.0 pp ở >= 3/5 subset.

| Subset | (a) Fixed (Ngưỡng cố định) | (b) Router (Tự thích ứng) [CI 95%] | (c) Oracle (Cận trên - Không thể triển khai) | Chênh lệch Δ (pp) | Kết quả |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1_commercial` | 56.96% | **61.63%** [58.83, 64.43] | *64.14%* | +4.67 | ✅ Thắng |
| `2_latent_diffusion` | 34.02% | **54.52%** [51.72, 57.32] | *65.56%* | +20.50 | ✅ Thắng |
| `3_pixel_diffusion` | 40.51% | **61.08%** [58.28, 63.88] | *72.15%* | +20.57 | ✅ Thắng |
| `4_modern_gans` | 75.90% | **91.56%** [88.76, 94.36] | *100.00%* | +15.66 | ✅ Thắng |
| `5_other_autoregressive` | 9.88% | **24.32%** [21.52, 27.12] | *32.10%* | +14.44 | ✅ Thắng |

### Kết luận:
**Router THẮNG theo luật DoD: vượt ngưỡng cố định >= 1.0 pp ở 5/5 subsets CFE.**
- **Độ tin cậy**: Router đạt điều kiện nạp cho cổng kiểm soát chất lượng **G7.3** (đóng gói UnifiedFatFormerXLA).
- **Cơ chế dự phòng (Safety Fallback)**: Trường hợp ảnh ở biên không rõ suy biến hoặc lỗi tín hiệu, router tự động gán ngưỡng an toàn `fallback_threshold = 0.50`.
