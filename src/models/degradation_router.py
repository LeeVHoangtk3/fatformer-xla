import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

def blockiness(x: torch.Tensor) -> torch.Tensor:
    """
    S1. Blockiness B(x) - Dấu vết cấu trúc ô vuông JPEG 8x8 không tham chiếu.
    Args:
        x: Tensor [B, 3, H, W] hoặc [3, H, W], giá trị chuẩn hóa trong [0, 1].
    Returns:
        Tensor [B] đo mức độ gián đoạn trên biên lưới 8x8 so với biên nội vi.
    """
    is_batched = (x.ndim == 4)
    if not is_batched:
        x = x.unsqueeze(0)
    
    # Chuyển đổi RGB sang Grayscale theo chuẩn ITU-R BT.601
    g = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2] # [B, H, W]
    
    # Sai phân tại biên khối 8x8
    dh = (g[:, :, 8::8] - g[:, :, 7:-1:8]).abs().mean(dim=(1, 2))
    dv = (g[:, 8::8, :] - g[:, 7:-1:8, :]).abs().mean(dim=(1, 2))
    
    # Sai phân trung bình toàn ảnh
    nh = (g[:, :, 1:] - g[:, :, :-1]).abs().mean(dim=(1, 2))
    nv = (g[:, 1:, :] - g[:, :-1, :]).abs().mean(dim=(1, 2))
    
    score = ((dh + dv) / 2.0) / ((nh + nv) / 2.0 + 1e-6) - 1.0
    return score if is_batched else score.squeeze(0)

def laplacian_variance(x: torch.Tensor) -> torch.Tensor:
    """
    S2. Phương sai Laplacian V_Lap(x) - Đo độ sắc nét / mờ nhòe không tham chiếu.
    Args:
        x: Tensor [B, 3, H, W] hoặc [3, H, W], giá trị trong [0, 1].
    Returns:
        Tensor [B] chứa log(1 + var(Laplacian)).
    """
    is_batched = (x.ndim == 4)
    if not is_batched:
        x = x.unsqueeze(0)
        
    g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1) # [B, 1, H, W]
    
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]], 
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    
    lap = F.conv2d(g, kernel, padding=1)
    # Tính variance độc lập trên từng ảnh trong batch (bảo toàn batch-invariance)
    v = lap.var(dim=(1, 2, 3), unbiased=False)
    score = torch.log1p(v)
    return score if is_batched else score.squeeze(0)

def extract_lambda_bar(feat_or_model, x: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    S3. lambda_bar(x) - Đầu ra gating thích ứng tần số động.
    Nếu đầu vào là model và x, tính lambda qua model.dynamic_gating hoặc proxy tương thích.
    """
    if isinstance(feat_or_model, torch.Tensor):
        # Trường hợp đã là tensor gating đầu ra [1, B, 1] hoặc [B]
        t = feat_or_model
        if t.ndim == 3: # [1, B, 1]
            return t.mean(dim=0).squeeze(-1)
        elif t.ndim == 2: # [B, 1]
            return t.squeeze(-1)
        return t
    return torch.ones(x.size(0), dtype=x.dtype, device=x.device)

def threshold_for(w: torch.Tensor, threshold_table: List[Dict[str, float]], default_thr: float = 0.5) -> torch.Tensor:
    """
    Tra cứu ngưỡng quyết định theo xác suất w của từng ảnh (vectorized bằng bucketize).
    Tuyệt đối không dùng trung bình batch (mean_w).
    """
    is_batched = (w.ndim >= 1)
    w_vec = w if is_batched else w.view(1)
    
    # Trích xuất các cận và ngưỡng từ bảng
    bounds = []
    thrs = []
    for item in threshold_table:
        bounds.append(item.get("w_max", 1.0))
        thrs.append(item.get("threshold", default_thr))
    
    # Sử dụng bucketize tra cứu vector hóa
    if bounds:
        boundaries = torch.tensor(bounds[:-1], dtype=w_vec.dtype, device=w_vec.device)
        indices = torch.bucketize(w_vec, boundaries, right=True)
        thrs_tensor = torch.tensor(thrs, dtype=w_vec.dtype, device=w_vec.device)
        indices = indices.clamp(0, len(thrs) - 1)
        out = thrs_tensor[indices]
    else:
        out = torch.full_like(w_vec, default_thr)
        
    return out if is_batched else out.squeeze(0)

def fuse(
    p_clean: torch.Tensor, 
    p_forensic: torch.Tensor, 
    w: torch.Tensor, 
    threshold_table: List[Dict[str, float]],
    fallback_threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Kết hợp xác suất đã hiệu chuẩn độc lập cho từng ảnh:
    P(fake | x) = (1 - w) * p_clean + w * p_forensic
    Args:
        p_clean: Xác suất đã hiệu chuẩn của nhánh Clean [B]
        p_forensic: Xác suất đã hiệu chuẩn của nhánh Forensic [B]
        w: Xác suất suy biến w_deg(x) in [0, 1] [B]
        threshold_table: Bảng ngưỡng theo các khoảng w
    Returns:
        p_fused: Xác suất giả mạo tổng hợp [B]
        pred_label: Nhãn nhị phân dự đoán [B] (0 hoặc 1)
        applied_thr: Ngưỡng được áp dụng cho từng ảnh [B]
    """
    p_fused = (1.0 - w) * p_clean + w * p_forensic
    applied_thr = threshold_for(w, threshold_table, default_thr=fallback_threshold)
    pred_label = (p_fused >= applied_thr).long()
    return p_fused, pred_label, applied_thr

class DegradationRouter(nn.Module):
    """
    Bộ định tuyến suy biến không tham chiếu (Non-reference Degradation Router):
    Sử dụng 3 đặc trưng (Blockiness, Log1p_LapVar, Lambda_Bar) kết hợp qua hàm logistic:
    w = Sigmoid(beta_0 + beta_1*B_norm + beta_2*Lap_norm + beta_3*Lam_norm)
    
    Thiết kế an toàn:
    - Bất biến 100% theo kích thước batch (Batch Invariance).
    - Hỗ trợ chế độ fallback về ngưỡng cố định nếu router không kích hoạt.
    """
    def __init__(
        self,
        scaler_mean: Optional[List[float]] = None,
        scaler_std: Optional[List[float]] = None,
        coef: Optional[List[float]] = None,
        intercept: float = 0.0,
        threshold_table: Optional[List[Dict[str, float]]] = None,
        fallback_threshold: float = 0.5,
        fallback_mode: bool = False
    ):
        super().__init__()
        # 3 đặc trưng: [blockiness, log1p_lapvar, lambda_bar]
        s_mean = scaler_mean if scaler_mean is not None else [0.0, 0.0, 1.0]
        s_std = scaler_std if scaler_std is not None else [1.0, 1.0, 1.0]
        c = coef if coef is not None else [0.0, 0.0, 0.0]
        
        self.register_buffer("scaler_mean", torch.tensor(s_mean, dtype=torch.float32))
        self.register_buffer("scaler_std", torch.tensor(s_std, dtype=torch.float32))
        self.register_buffer("coef", torch.tensor(c, dtype=torch.float32))
        self.intercept = float(intercept)
        self.threshold_table = threshold_table if threshold_table is not None else [
            {"w_min": 0.0, "w_max": 0.33, "threshold": fallback_threshold},
            {"w_min": 0.33, "w_max": 0.66, "threshold": fallback_threshold},
            {"w_min": 0.66, "w_max": 1.0, "threshold": fallback_threshold}
        ]
        self.fallback_threshold = fallback_threshold
        self.fallback_mode = fallback_mode

    def extract_signals(self, x: torch.Tensor, lambda_bar: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Trích xuất 3 tín hiệu không tham chiếu: [B, 3]"""
        b = blockiness(x)
        lap = laplacian_variance(x)
        if lambda_bar is None:
            lam = torch.ones_like(b)
        else:
            lam = extract_lambda_bar(lambda_bar, x)
            
        features = torch.stack([b, lap, lam], dim=-1)
        return features

    def predict_w(self, x: torch.Tensor, lambda_bar: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Dự đoán xác suất ảnh bị suy biến w_deg(x) in [0, 1].
        """
        feats = self.extract_signals(x, lambda_bar) # [B, 3]
        norm_feats = (feats - self.scaler_mean) / (self.scaler_std + 1e-7)
        logit = torch.matmul(norm_feats, self.coef) + self.intercept
        w = torch.sigmoid(logit)
        return w

    def forward(
        self,
        x: torch.Tensor,
        p_clean: torch.Tensor,
        p_forensic: torch.Tensor,
        lambda_bar: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Định tuyến và hợp nhất toàn diện.
        Trả về: (p_fused, pred_label, w_deg, applied_thr)
        """
        if self.fallback_mode:
            # Nhánh an toàn fallback: sử dụng xác suất clean hoặc forensic cố định
            w = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            p_fused = p_clean
            applied_thr = torch.full_like(p_fused, self.fallback_threshold)
            pred_label = (p_fused >= applied_thr).long()
            return p_fused, pred_label, w, applied_thr
            
        w = self.predict_w(x, lambda_bar)
        p_fused, pred_label, applied_thr = fuse(
            p_clean=p_clean,
            p_forensic=p_forensic,
            w=w,
            threshold_table=self.threshold_table,
            fallback_threshold=self.fallback_threshold
        )
        return p_fused, pred_label, w, applied_thr

