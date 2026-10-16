"""
LoRA Adapter (Low-Rank Adaptation) cho ViT-L/14 Backbone — Task 6.3-B, Milestone 6.
Thành viên B — Architecture & Loss Lead.

Thiết kế:
    Chèn adapter tham số thấp vào ma trận chiếu W_q và W_v của N khối Transformer
    cuối cùng của backbone CLIP ViT-L/14. Cơ chế:

        W'(x) = W(x) + (α / r) · B(A(x))
        A ∈ R^{r × d},  B ∈ R^{d × r},  r = 8,  α = 16

    - W: trọng số gốc đóng băng (frozen) — KHÔNG cập nhật gradient
    - A, B: ma trận adapter nhỏ — học được, khởi tạo B = 0 (không ảnh hưởng forward pass ban đầu)
    - Tổng tham số thêm: 2 × 2 × r × d = 4 × 8 × 1024 ≈ 32K params / khối

Kích hoạt/vô hiệu hóa:
    Sử dụng cờ `enabled` trong cấu hình để bật/tắt minh bạch:
        lora_config = {"enabled": True, "rank": 8, "alpha": 16, "num_layers": 2}

Phạm vi kích hoạt:
    Chỉ dùng cho nhánh M5 trong Task 6.5. Mặc định `enabled=False` để không ảnh hưởng
    pipeline M4 và M6 Plan A/B đang chạy.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Lớp Linear tích hợp LoRA adapter.

    Thay thế nn.Linear gốc bằng cách bổ sung hai ma trận tham số thấp A, B:
        out = x @ W.T + (α / r) * x @ A.T @ B.T

    W (frozen) + B·A (trainable, khởi tạo B=0 → delta=0 ban đầu)

    Args:
        in_features:  Số chiều đầu vào (thường = 1024 cho ViT-L/14)
        out_features: Số chiều đầu ra
        rank:         Hạng LoRA r, mặc định 8
        alpha:        Hệ số tỷ lệ α, mặc định 16. Tỷ lệ thực tế = α / r = 2.0
        bias:         Có dùng bias không (kế thừa từ lớp gốc)
        enabled:      Nếu False → vô hiệu hóa adapter, hoạt động như nn.Linear thuần túy
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = True,
        enabled: bool = True,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank    = rank
        self.alpha   = alpha
        self.enabled = enabled
        self.scaling = alpha / rank  # tỷ lệ chuẩn hóa theo Hu et al. 2021

        # Trọng số gốc — đóng băng
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias_param = nn.Parameter(
            torch.zeros(out_features), requires_grad=False
        ) if bias else None

        if enabled:
            # Adapter A: khởi tạo bằng Kaiming Uniform (như nn.Linear gốc)
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            # Adapter B: khởi tạo bằng 0 → delta = 0 khi bắt đầu huấn luyện
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.lora_A = None
            self.lora_B = None

        # Khởi tạo weight gốc giống nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        enabled: bool = True,
    ) -> "LoRALinear":
        """
        Khởi tạo LoRALinear từ nn.Linear gốc đã có trọng số pretrained.

        Sao chép weight và bias từ lớp gốc, giữ nguyên frozen.
        """
        lora_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
            enabled=enabled,
        )
        # Copy trọng số gốc (không tính gradient)
        lora_layer.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and lora_layer.bias_param is not None:
            lora_layer.bias_param.data.copy_(linear.bias.data)
        return lora_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Nhánh chính: tính toán với W gốc (frozen)
        bias = self.bias_param if self.bias_param is not None else None
        out = nn.functional.linear(x, self.weight, bias)

        # Nhánh adapter: cộng thêm delta nếu enabled
        if self.enabled and self.lora_A is not None:
            # delta = x @ A^T @ B^T * scaling
            delta = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
            out = out + delta

        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, enabled={self.enabled}"
        )


class ViTLoRAAdapter(nn.Module):
    """
    Bộ điều khiển LoRA cho backbone CLIP ViT-L/14.

    Chèn LoRALinear vào ma trận W_q và W_v của `num_layers` khối Transformer
    cuối cùng của ViT. Tất cả adapter được quản lý tập trung qua module này.

    Cách sử dụng:
        # Khởi tạo adapter (KHÔNG sửa trực tiếp các layer trong vit_model)
        adapter = ViTLoRAAdapter(vit_model, config={"enabled": True, "rank": 8, "alpha": 16, "num_layers": 2})
        # Khi huấn luyện: chỉ có adapter.parameters() có gradient
        optimizer = torch.optim.AdamW(adapter.trainable_parameters(), lr=1e-4)

    Args:
        vit_model:  Module ViT có thuộc tính `transformer.resblocks` hoặc `visual.transformer.resblocks`
        config:     Dict cấu hình LoRA, các key:
                        enabled    (bool, mặc định False)
                        rank       (int,  mặc định 8)
                        alpha      (float, mặc định 16.0)
                        num_layers (int,  mặc định 2)
    """

    def __init__(self, vit_model: nn.Module, config: Optional[dict] = None):
        super().__init__()
        cfg = config or {}
        self.enabled    = cfg.get("enabled", False)
        self.rank       = cfg.get("rank", 8)
        self.alpha      = float(cfg.get("alpha", 16.0))
        self.num_layers = cfg.get("num_layers", 2)

        self.lora_layers = nn.ModuleList()

        if not self.enabled:
            return

        # Tìm danh sách ResBlock trong ViT (hỗ trợ cả cấu trúc OpenAI CLIP)
        resblocks = self._get_resblocks(vit_model)
        if resblocks is None or len(resblocks) == 0:
            raise RuntimeError(
                "Không tìm thấy resblocks trong vit_model. "
                "Hãy kiểm tra cấu trúc module với print(vit_model)."
            )

        target_blocks = list(resblocks)[-self.num_layers:]
        for block in target_blocks:
            attn = self._get_attn(block)
            if attn is None:
                continue
            # Thay thế in_proj hoặc q_proj / v_proj bằng LoRALinear
            for proj_name in ("q_proj", "v_proj"):
                proj = getattr(attn, proj_name, None)
                if isinstance(proj, nn.Linear):
                    lora_proj = LoRALinear.from_linear(
                        proj, rank=self.rank, alpha=self.alpha, enabled=True
                    )
                    setattr(attn, proj_name, lora_proj)
                    self.lora_layers.append(lora_proj)

    @staticmethod
    def _get_resblocks(vit_model: nn.Module) -> Optional[nn.ModuleList]:
        """Tìm danh sách ResBlock trong ViT — hỗ trợ cả 2 cấu trúc CLIP."""
        # Cấu trúc 1: model.transformer.resblocks (OpenAI CLIP visual)
        for path in [
            ["transformer", "resblocks"],
            ["visual", "transformer", "resblocks"],
        ]:
            obj = vit_model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                return obj  # type: ignore[return-value]
            except AttributeError:
                continue
        return None

    @staticmethod
    def _get_attn(block: nn.Module) -> Optional[nn.Module]:
        """Tìm module attention trong một ResBlock."""
        for attr in ("attn", "self_attn", "attention"):
            attn = getattr(block, attr, None)
            if attn is not None:
                return attn
        return None

    def trainable_parameters(self):
        """Trả về iterator chỉ gồm tham số adapter (A, B) — để truyền cho optimizer."""
        for layer in self.lora_layers:
            if layer.lora_A is not None:
                yield layer.lora_A
            if layer.lora_B is not None:
                yield layer.lora_B

    def set_enabled(self, enabled: bool):
        """Bật/tắt adapter runtime mà không cần khởi tạo lại."""
        self.enabled = enabled
        for layer in self.lora_layers:
            layer.enabled = enabled

    def summary(self) -> str:
        """In tóm tắt cấu hình LoRA và số tham số adapter."""
        n_params = sum(
            p.numel()
            for layer in self.lora_layers
            for p in [layer.lora_A, layer.lora_B]
            if p is not None
        )
        return (
            f"ViTLoRAAdapter | enabled={self.enabled} | "
            f"rank={self.rank} | alpha={self.alpha} | "
            f"num_layers={self.num_layers} | "
            f"adapter_layers={len(self.lora_layers)} | "
            f"trainable_params={n_params:,}"
        )
