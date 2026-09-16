import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialResidualBlock(nn.Module):
    """
    Khối lọc vết dư không gian Spatial Rich Model (SRM) gồm 3 bộ lọc vi sai pháp y kinh điển:
    1. Kernel bậc 1 (First-order horizontal/vertical gradient): dò vết nứt vi sai điểm ảnh.
    2. Kernel bậc 2 (Second-order Laplacian): dò vi sai độ cong bề mặt.
    3. Kernel Square 5x5: dò vết nội suy song tuyến tính (Bilinear/Bicubic upsampling artifacts).
    
    Trọng số các bộ lọc vi sai là CỐ ĐỊNH (requires_grad = False), tiêu thụ 0 tham số huấn luyện.
    """
    def __init__(self, in_channels=3, out_dim=1024):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # 1. Kernel bậc 1 (dò sai phân bậc nhất liền kề ngang)
        k1 = torch.tensor([
            [0., 0., 0.],
            [-1., 1., 0.],
            [0., 0., 0.]
        ], dtype=torch.float32)

        # 2. Kernel bậc 2 (Laplacian 4 hướng chuẩn)
        k2 = torch.tensor([
            [0., -1., 0.],
            [-1., 4., -1.],
            [0., -1., 0.]
        ], dtype=torch.float32)

        # 3. Kernel Square 5x5 chuẩn Steganalysis (dò vết nội suy AI upsampler)
        k3 = torch.tensor([
            [-1.,  2.,  -2.,  2., -1.],
            [ 2., -6.,   8., -6.,  2.],
            [-2.,  8., -12.,  8., -2.],
            [ 2., -6.,   8., -6.,  2.],
            [-1.,  2.,  -2.,  2., -1.]
        ], dtype=torch.float32) / 12.0

        # Đăng ký buffers để tự động chuyển thiết bị (CPU/GPU) cùng mô hình
        self.register_buffer("k1", k1.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.register_buffer("k2", k2.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.register_buffer("k3", k3.view(1, 1, 5, 5).repeat(in_channels, 1, 1, 1))

        # Tầng Conv1d chiếu vết dư (9 channels -> out_dim) tương thích với định dạng token [N, B, D]
        # 3 kernels x 3 channels RGB = 9 kênh vết dư
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * 3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)), # Đưa về lưới 16x16 = 256 patch tokens
            nn.Conv2d(64, out_dim, kernel_size=1, bias=False)
        )

    def forward(self, x_img):
        """
        x_img: Tensor ảnh gốc dạng [B, 3, H, W]
        Trả về: đặc trưng vết dư không gian dạng [256, B, out_dim]
        """
        # Áp dụng 3 bộ lọc theo cơ chế depthwise convolution
        r1 = F.conv2d(x_img, self.k1, padding=1, groups=self.in_channels)
        r2 = F.conv2d(x_img, self.k2, padding=1, groups=self.in_channels)
        r3 = F.conv2d(x_img, self.k3, padding=2, groups=self.in_channels)

        # Ghép 3 vết dư: [B, 9, H, W]
        residuals = torch.cat([r1, r2, r3], dim=1)

        # Chiếu về không gian đặc trưng: [B, out_dim, 16, 16]
        feat = self.proj(residuals)

        # Reshape sang [256, B, out_dim] để khớp với chuỗi patch tokens của ViT-L/14
        B, C, H, W = feat.shape
        feat_tokens = feat.flatten(2).permute(2, 0, 1) # [H*W, B, C] = [256, B, out_dim]
        return feat_tokens
