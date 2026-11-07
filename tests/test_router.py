import unittest
import torch
from src.models.degradation_router import (
    blockiness,
    laplacian_variance,
    threshold_for,
    fuse,
    DegradationRouter
)

class TestDegradationRouter(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.table = [
            {"w_min": 0.0, "w_max": 0.33, "threshold": 0.40},
            {"w_min": 0.33, "w_max": 0.66, "threshold": 0.50},
            {"w_min": 0.66, "w_max": 1.0, "threshold": 0.65}
        ]
        self.router = DegradationRouter(
            scaler_mean=[0.0, 0.0, 1.0],
            scaler_std=[1.0, 1.0, 1.0],
            coef=[1.5, -0.8, -0.5],
            intercept=0.2,
            threshold_table=self.table,
            fallback_threshold=0.5
        )

    def test_t1_blockiness_grid_vs_smooth(self):
        """T1: Ảnh có lưới biên khối 8x8 phải có điểm blockiness cao hơn ảnh phẳng/mịn cùng mức sáng."""
        h, w = 64, 64
        # Ảnh phẳng mức xám 0.5
        smooth_img = torch.full((1, 3, h, w), 0.5)
        
        # Ảnh có cấu trúc nhảy bậc rõ rệt ở chu kỳ 8 pixel (mô phỏng artifact JPEG)
        grid_img = smooth_img.clone()
        for i in range(0, h, 8):
            grid_img[:, :, i:i+8, :] += (i // 8) * 0.05
        grid_img = grid_img.clamp(0.0, 1.0)
        
        b_smooth = blockiness(smooth_img)
        b_grid = blockiness(grid_img)
        
        self.assertGreater(b_grid.item(), b_smooth.item(), "Blockiness của ảnh lưới phải lớn hơn ảnh phẳng")

    def test_t2_batch_size_invariance(self):
        """T2: w phải hoàn toàn bất biến theo batch size (batch=1 vs batch=32 cho cùng w với max|Δw| < 1e-6)."""
        x_single = torch.rand(1, 3, 64, 64)
        x_batch = x_single.repeat(32, 1, 1, 1)
        
        w_single = self.router.predict_w(x_single)
        w_batch = self.router.predict_w(x_batch)
        
        max_diff = (w_batch - w_single).abs().max().item()
        self.assertLess(max_diff, 1e-6, f"Batch size invariance vi phạm! Lệch max: {max_diff}")

    def test_t3_fuse_permutation_invariance(self):
        """T3: fuse không dùng mean batch; đảo thứ tự ảnh trong batch thì output từng ảnh không đổi."""
        b_size = 4
        x = torch.rand(b_size, 3, 64, 64)
        p_clean = torch.tensor([0.2, 0.8, 0.4, 0.7])
        p_forensic = torch.tensor([0.1, 0.9, 0.5, 0.6])
        
        p_fused, pred, w_deg, applied_thr = self.router(x, p_clean, p_forensic)
        
        # Đảo vị trí các mẫu: [3, 2, 1, 0]
        perm = [3, 2, 1, 0]
        x_perm = x[perm]
        p_clean_perm = p_clean[perm]
        p_forensic_perm = p_forensic[perm]
        
        p_fused_perm, pred_perm, w_deg_perm, applied_thr_perm = self.router(
            x_perm, p_clean_perm, p_forensic_perm
        )
        
        self.assertTrue(torch.allclose(p_fused[perm], p_fused_perm, atol=1e-6))
        self.assertTrue(torch.equal(pred[perm], pred_perm))
        self.assertTrue(torch.allclose(applied_thr[perm], applied_thr_perm, atol=1e-6))

    def test_t4_threshold_coverage(self):
        """T4: threshold_for phủ toàn miền [0, 1] mà không sinh NaN hay lỗi biên."""
        test_ws = torch.tensor([0.0, 0.1, 0.33, 0.5, 0.66, 0.9, 1.0])
        thrs = threshold_for(test_ws, self.table, default_thr=0.5)
        
        self.assertEqual(thrs.shape, test_ws.shape)
        self.assertFalse(torch.isnan(thrs).any(), "Ngưỡng không được chứa NaN")
        self.assertTrue((thrs > 0.0).all() and (thrs < 1.0).all(), "Ngưỡng phải nằm trong khoảng hợp lệ (0, 1)")

if __name__ == "__main__":
    unittest.main()

