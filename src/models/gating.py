import torch
import torch.nn as nn

class DynamicFrequencyGating(nn.Module):
    """
    Mạng Cổng Thích Ứng Tần Số Động (Dynamic Frequency Gating):
    Thay thế hệ số hằng số cố định freq_scale bằng mạng học nhẹ:
    lambda(x) = Sigmoid(MLP(GlobalAvgPool(feat))) * 2.0
    
    Cơ chế:
    - Ảnh sạch (Clean): lambda(x) giữ ở mức 1.0 ~ 1.5, khai thác trọn vẹn dải tần DWT.
    - Ảnh nén sâu (JPEG Q <= 50) hoặc Blur: lambda(x) tự động đóng về 0.05 ~ 0.1,
      ngăn chặn nhiễu khối ô vuông 8x8 tràn vào mô hình.
    """
    def __init__(self, d_model=1024, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Khởi tạo bias để ban đầu lambda(x) có giá trị xấp xỉ 1.0 (trung tính)
        nn.init.constant_(self.mlp[2].bias, 0.0)

    def forward(self, x):
        """
        x: Tensor đặc trưng dạng [N, B, D] (với N là số token, B là batch size, D là kênh)
        Trả về: lambda_scale có kích thước [1, B, 1] hoặc số vô hướng
        """
        # Pooling trên chuỗi token: [B, D]
        feat_pool = torch.mean(x, dim=0) # [B, D]
        # Dự đoán hệ số thích ứng trong khoảng [0, 2.0]
        lambda_val = self.mlp(feat_pool) * 2.0 # [B, 1]
        return lambda_val.unsqueeze(0) # [1, B, 1] để broadcast nhân với [N, B, D]
