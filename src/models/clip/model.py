from collections import OrderedDict
from typing import Tuple, Union
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError:
    class DWTForward(nn.Module):
        """Native PyTorch implementation cho Haar 2D DWT (J=1), khớp 100% state_dict buffer và API của pytorch_wavelets."""
        def __init__(self, J=1, wave='haar', mode='zero'):
            super().__init__()
            self.J = J
            self.wave = wave
            self.mode = mode
            h0 = torch.tensor([0.7071067690849304, 0.7071067690849304])
            h1 = torch.tensor([0.7071067690849304, -0.7071067690849304])
            self.register_buffer('h0_col', h0.reshape(1, 1, 2, 1))
            self.register_buffer('h1_col', h1.reshape(1, 1, 2, 1))
            self.register_buffer('h0_row', h0.reshape(1, 1, 1, 2))
            self.register_buffer('h1_row', h1.reshape(1, 1, 1, 2))

        def forward(self, x):
            B, C, H, W = x.shape
            w_ll = (self.h0_col @ self.h0_row).to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)
            w_lh = (self.h0_col @ self.h1_row).to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)
            w_hl = (self.h1_col @ self.h0_row).to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)
            w_hh = (self.h1_col @ self.h1_row).to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)

            ll = F.conv2d(x, w_ll, stride=2, groups=C)
            lh = F.conv2d(x, w_lh, stride=2, groups=C)
            hl = F.conv2d(x, w_hl, stride=2, groups=C)
            hh = F.conv2d(x, w_hh, stride=2, groups=C)

            yh = torch.stack([lh, hl, hh], dim=2)
            return ll, [yh]

    class DWTInverse(nn.Module):
        """Native PyTorch implementation cho Haar 2D IDWT, khớp 100% state_dict buffer và tái tạo hoàn hảo."""
        def __init__(self, wave='haar', mode='zero'):
            super().__init__()
            self.wave = wave
            self.mode = mode
            g0 = torch.tensor([0.7071067690849304, 0.7071067690849304])
            g1 = torch.tensor([0.7071067690849304, -0.7071067690849304])
            self.register_buffer('g0_col', g0.reshape(1, 1, 2, 1))
            self.register_buffer('g1_col', g1.reshape(1, 1, 2, 1))
            self.register_buffer('g0_row', g0.reshape(1, 1, 1, 2))
            self.register_buffer('g1_row', g1.reshape(1, 1, 1, 2))

        def forward(self, coeffs):
            yl, yh_list = coeffs
            yh = yh_list[0]
            B, C, H_half, W_half = yl.shape
            lh, hl, hh = yh[:, :, 0], yh[:, :, 1], yh[:, :, 2]

            w_ll = (self.g0_col @ self.g0_row).to(dtype=yl.dtype, device=yl.device).repeat(C, 1, 1, 1)
            w_lh = (self.g0_col @ self.g1_row).to(dtype=yl.dtype, device=yl.device).repeat(C, 1, 1, 1)
            w_hl = (self.g1_col @ self.g0_row).to(dtype=yl.dtype, device=yl.device).repeat(C, 1, 1, 1)
            w_hh = (self.g1_col @ self.g1_row).to(dtype=yl.dtype, device=yl.device).repeat(C, 1, 1, 1)

            rec = (
                F.conv_transpose2d(yl, w_ll, stride=2, groups=C) +
                F.conv_transpose2d(lh, w_lh, stride=2, groups=C) +
                F.conv_transpose2d(hl, w_hl, stride=2, groups=C) +
                F.conv_transpose2d(hh, w_hh, stride=2, groups=C)
            )
            return rec

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ForgeryAwareAdapterLayer(nn.Module):
    """
    Khối ForgeryAwareAdapterLayer chuẩn tác giả CVPR 2024,
    đã tích hợp sẵn hook cho SRM 3-Kernels và Dynamic Frequency Gating.
    """
    def __init__(self,
                 d_model=1024,
                 bottleneck=64,
                 dropout=0.1,
                 kernel_size=1,
                 adapter_scalar="0.1",
                 head=8,
                 args=None):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.d_model = d_model
        self.scale = float(adapter_scalar)

        # Spatial adapter
        self.first_conv_layer = nn.Conv1d(in_channels=self.n_embd, out_channels=self.down_size, kernel_size=kernel_size)
        self.non_linear_func = nn.ReLU()
        self.second_conv_layer = nn.Conv1d(in_channels=self.down_size, out_channels=self.n_embd, kernel_size=kernel_size)
        self.dropout = dropout

        # Frequency-aware adapter
        self.freq_scale = nn.Parameter(torch.zeros(1))
        self.dwt_transform = DWTForward(J=1, wave='haar')
        self.idwt_transform = DWTInverse(wave='haar')

        self.dwt_norm = nn.GroupNorm(128, d_model)
        self.intra_band = nn.MultiheadAttention(d_model, head)
        self.dropout_intra = nn.Dropout(dropout)
        self.norm_intra = nn.LayerNorm(d_model)

        self.inter_band = nn.MultiheadAttention(d_model, head)
        self.dropout_inter = nn.Dropout(dropout)
        self.norm_inter = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # Tùy chọn nâng cấp cho Phương án 2 (SRM & Gating)
        self.use_srm = getattr(args, "use_srm", False) if args else False
        self.use_gating = getattr(args, "use_gating", False) if args else False

        if self.use_srm:
            from ..srm import SpatialResidualBlock
            self.srm_block = SpatialResidualBlock(in_channels=3, out_dim=d_model)
        else:
            self.srm_block = None

        if self.use_gating:
            from ..gating import DynamicFrequencyGating
            self.dynamic_gating = DynamicFrequencyGating(d_model=d_model)
        else:
            self.dynamic_gating = None

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_freq(self, x):
        B, C = x.shape[:2]
        nq = x.shape[2]
        q = k = v = x.transpose(0, 1).flatten(1, 2)
        tgt2 = self.intra_band(q, k, v)[0].reshape(C, B, nq, self.d_model).transpose(0, 1)
        x = x + self.dropout_intra(tgt2)
        x = self.norm_intra(x)

        q = k = v = x.flatten(0, 1).transpose(0, 1)
        tgt2 = self.inter_band(q, k, v)[0].transpose(0, 1).reshape(B, C, nq, self.d_model)
        x = x + self.dropout_inter(tgt2)
        x = self.norm_inter(x)

        x = self.forward_ffn(x)
        return x

    def forward(self, x, x_raw_img=None):
        nq, bs, md = x.shape
        # 1. Frequency branch (Haar DWT)
        img_patchs = x[1:].transpose(0, 1).reshape(bs, int(math.sqrt(nq)), int(math.sqrt(nq)), md).permute(0, 3, 1, 2)
        dwt_img_l, dwt_img_h = self.dwt_transform(img_patchs)
        dwt_img = torch.cat([dwt_img_l[:, :, None], dwt_img_h[0]], dim=2)
        hh, ww = dwt_img.shape[-2:]
        dwt_img = dwt_img.flatten(-2)
        dwt_img = self.dwt_norm(dwt_img).permute(0, 2, 3, 1)
        dwt_feature = self.forward_freq(dwt_img)
        dwt_feature = dwt_feature.reshape(bs, 4, hh, ww, md).permute(0, 4, 1, 2, 3)
        dwt_feature = self.idwt_transform((dwt_feature[:, :, 0], [dwt_feature[:, :, 1:]])).flatten(-2).permute(2, 0, 1)
        dwt_feature = torch.cat([torch.zeros_like(x[:1]), dwt_feature], dim=0)

        # 2. Spatial branch
        down = self.first_conv_layer(x.permute(1, 2, 0))
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.second_conv_layer(down).permute(2, 0, 1)
        up = up * self.scale

        # Bổ sung vết dư SRM nếu bật chế độ nâng cấp
        if self.srm_block is not None and x_raw_img is not None:
            srm_res = self.srm_block(x_raw_img)
            # Ghép token CLS zê-rô vào đầu chuỗi
            srm_tokens = torch.cat([torch.zeros_like(x[:1]), srm_res], dim=0)
            up = up + srm_tokens * 0.1

        # 3. Dynamic Frequency Gating
        if self.dynamic_gating is not None:
            scale_factor = self.dynamic_gating(x)
        else:
            scale_factor = self.freq_scale

        output = up + dwt_feature * scale_factor
        return output


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_adapter=False, args=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        if add_adapter:
            self.forgery_aware_adapter = ForgeryAwareAdapterLayer(d_model=d_model, args=args)
        else:
            self.forgery_aware_adapter = None

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        if self.forgery_aware_adapter is not None:
            adapt_x = self.forgery_aware_adapter(x)
            x = x + self.mlp(self.ln_2(x)) + adapt_x
        else:
            x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, add_adapter=None, args=None):
        super().__init__()
        self.width = width
        self.layers = layers
        if add_adapter is None:
            add_adapter = [False] * layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask, add_adapter[idx], args=args) 
            for idx in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        out = {}
        for idx, layer in enumerate(self.resblocks.children()):
            x = layer(x)
            out['layer' + str(idx)] = x
        return out, x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, args=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        num_vit_adapter = getattr(args, "num_vit_adapter", 8) if args else 8
        add_adapter = [False] * layers
        if num_vit_adapter > 0:
            assert layers % num_vit_adapter == 0, f"layers {layers}, while num_vit_adapter {num_vit_adapter}!"
            n_spilt = layers // num_vit_adapter
            for i in range(num_vit_adapter):
                add_adapter[(i + 1) * n_spilt - 1] = True

        self.transformer = Transformer(width, layers, heads, add_adapter=add_adapter, args=args)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, return_full=False):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        out, x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj

        if return_full:
            return x
        return x[:, 0, :]


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 args=None
                 ):
        super().__init__()
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                args=args
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)[1]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def build_model(state_dict: dict, args=None):
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        args=args
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    return model.eval()
