import io
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, Set
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from .preprocessing import preprocessing_parts
try:
    import yaml
except ImportError:
    yaml = None  # lean: yaml optional — chỉ cần khi dùng from_config(path)


class RandomJPEGCompression:
    """Áp dụng nén JPEG ngẫu nhiên với hệ số chất lượng Q trong khoảng [min_q, max_q]."""
    def __init__(self, min_q=30, max_q=95):
        self.min_q = min_q
        self.max_q = max_q

    def __call__(self, img: Image.Image) -> Image.Image:
        q = self.min_q if self.min_q == self.max_q else random.randint(self.min_q, self.max_q)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=q)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomGaussianBlur:
    """Áp dụng Gaussian Blur ngẫu nhiên với bán kính radius trong khoảng [min_r, max_r]."""
    def __init__(self, min_r=0.5, max_r=3.0):
        self.min_r = min_r
        self.max_r = max_r

    def __call__(self, img: Image.Image) -> Image.Image:
        r = random.uniform(self.min_r, self.max_r)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


class RandomDownUpSampling:
    """Giảm độ phân giải rồi phóng to lại (mô phỏng resize trên mạng xã hội)."""
    def __init__(self, min_ratio=0.3, max_ratio=0.9):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        low_w, low_h = max(16, int(w * ratio)), max(16, int(h * ratio))
        img_down = img.resize((low_w, low_h), Image.BILINEAR)
        return img_down.resize((w, h), Image.BILINEAR)



class RandomWebPCompression:
    """Nén ảnh WebP ngẫu nhiên với chất lượng Q trong [min_q, max_q]. Dùng PIL."""
    def __init__(self, min_q=30, max_q=70, method=0):
        self.min_q = min_q
        self.max_q = max_q
        if type(method) is not int or not 0 <= method <= 6:
            raise ValueError("webp.method must be an integer in [0, 6]")
        self.method = method

    def __call__(self, img: Image.Image) -> Image.Image:
        q = random.randint(self.min_q, self.max_q)
        buf = io.BytesIO()
        try:
            img.save(buf, format="WEBP", quality=q, method=self.method)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        except (KeyError, OSError):
            # lean: fallback khi PIL build không hỗ trợ WebP
            return RandomJPEGCompression(self.min_q, self.max_q)(img)


class RandomDoubleJPEG:
    """Double JPEG: nén 2 lần liên tiếp (mô phỏng ảnh tải lên - tải xuống)."""
    def __init__(self, q1_range=(70, 90), q2_range=(30, 50)):
        self.q1_range = q1_range
        self.q2_range = q2_range

    def __call__(self, img: Image.Image) -> Image.Image:
        q1 = random.randint(*self.q1_range)
        q2 = random.randint(*self.q2_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q1)
        buf.seek(0)
        img2 = Image.open(buf).convert("RGB")
        buf2 = io.BytesIO()
        img2.save(buf2, format="JPEG", quality=q2)
        buf2.seek(0)
        return Image.open(buf2).convert("RGB")


class DualStreamRobustAugmentation:
    """
    Pipeline tăng cường dữ liệu kép (Dual-Stream):
    - 50% xác suất giữ ảnh nguyên bản (Clean Stream) để bảo toàn đặc trưng ban đầu và chống catastrophic forgetting.
    - 50% xác suất áp dụng một trong các suy thoái mạng xã hội (Degraded Stream: JPEG, Blur, Down-Up).
    """
    def __init__(self, clean_prob=0.5):
        self.clean_prob = clean_prob
        self.jpeg = RandomJPEGCompression(min_q=30, max_q=95)
        self.blur = RandomGaussianBlur(min_r=0.5, max_r=2.5)
        self.down_up = RandomDownUpSampling(min_ratio=0.4, max_ratio=0.8)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.clean_prob:
            return img
        
        # Chọn ngẫu nhiên 1 trong 3 kiểu suy thoái
        degradation_type = random.choice(["jpeg", "blur", "down_up"])
        if degradation_type == "jpeg":
            return self.jpeg(img)
        elif degradation_type == "blur":
            return self.blur(img)
        else:
            return self.down_up(img)


class CurriculumDegradationScheduler:
    """
    Bộ lập lịch suy thoái động theo giáo trình 3 giai đoạn (Curriculum Learning - Task 3.2):
    Tác giả: Thành viên A (Data & Infra Lead)
    Kế thừa thông số: configs/train_config.yaml
    
    Quy tắc 3 giai đoạn theo epoch:
      - Giai đoạn 1 (Epoch 1-2): Nén rất nhẹ JPEG Q ∈ [70, 90], Gaussian Blur σ ∈ [0.5, 1.0], không Down-Up.
      - Giai đoạn 2 (Epoch 3-5): Nén vừa Q ∈ [45, 70], Gaussian Blur σ ∈ [1.0, 1.5], Down-Up 224 -> 160 -> 224.
      - Giai đoạn 3 (Epoch 6-8): Nén sâu thử thách Q ∈ [30, 50], Blur σ ∈ [1.5, 2.0], Down-Up 224 -> 112 -> 224.
    
    Tỷ lệ phân bổ mẫu: 50% Clean (bảo toàn trần 96.45% ACC), 50% Degraded.
    """
    def __init__(self, clean_prob=0.5, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        p1_clean = self.config.get("phase1", {}).get("clean_ratio", clean_prob)
        self.clean_prob = p1_clean
        self.epoch = 1
        self.stage = 1
        self.lr_multiplier = 1.0
        self._update_stage_params()

    def set_epoch(self, epoch: int):
        self.epoch = max(1, epoch)
        self._update_stage_params()

    def _update_stage_params(self):
        c1 = self.config.get("phase1", {})
        c2 = self.config.get("phase2", {})
        c3 = self.config.get("phase3", {})

        p1_epochs = c1.get("epochs", [1, 2])
        p2_epochs = c2.get("epochs", [3, 4])
        p3_epochs = c3.get("epochs", [5, 5])

        if self.epoch <= p1_epochs[1]:
            # Giai đoạn 1: Nén rất nhẹ, ổn định gradient FAA
            self.stage = 1
            self.q_range = tuple(c1.get("jpeg_quality_range", [70, 90]))
            self.sigma_range = tuple(c1.get("blur_sigma_range", [0.5, 1.0]))
            self.down_size = (c1.get("downsample_target_size", 160), c1.get("downsample_target_size", 160)) if c1.get("downsample", False) else None
            self.clean_prob = c1.get("clean_ratio", self.clean_prob)
            self.lr_multiplier = 1.0
        elif self.epoch <= p2_epochs[1]:
            # Giai đoạn 2: Nén vừa, rèn luyện mạng cổng Gating λ(x)
            self.stage = 2
            self.q_range = tuple(c2.get("jpeg_quality_range", [45, 70]))
            self.sigma_range = tuple(c2.get("blur_sigma_range", [1.0, 1.5]))
            d_sz = c2.get("downsample_target_size", 160)
            self.down_size = (d_sz, d_sz) if c2.get("downsample", True) else None
            self.clean_prob = c2.get("clean_ratio", self.clean_prob)
            self.lr_multiplier = 1.0
        else:
            # Giai đoạn 3: Nén sâu thử thách, tôi luyện biểu diễn vi sai SRM
            self.stage = 3
            self.q_range = tuple(c3.get("jpeg_quality_range", [30, 50]))
            self.sigma_range = tuple(c3.get("blur_sigma_range", [1.5, 2.0]))
            d_sz = c3.get("downsample_target_size", 112)
            self.down_size = (d_sz, d_sz) if c3.get("downsample", True) else None
            self.clean_prob = c3.get("clean_ratio", self.clean_prob)
            self.lr_multiplier = float(c3.get("lr_multiplier", 0.5))

    def apply_jpeg(self, img: Image.Image) -> Image.Image:
        q = random.randint(*self.q_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def apply_blur(self, img: Image.Image) -> Image.Image:
        sigma = random.uniform(*self.sigma_range)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    def apply_down_up(self, img: Image.Image) -> Image.Image:
        if self.down_size is None:
            return img
        orig_size = img.size
        resample_mode = getattr(Image, "Resampling", Image).BICUBIC
        down = img.resize(self.down_size, resample_mode)
        up = down.resize(orig_size, resample_mode)
        return up

    def __call__(self, img: Image.Image) -> Image.Image:
        # 30% giữ sạch nguyên bản
        if random.random() < self.clean_prob:
            return img

        # 70% áp dụng suy biến vật lý ngẫu nhiên
        degrade_choice = random.choice(["jpeg", "blur", "down_up"])
        if degrade_choice == "jpeg":
            return self.apply_jpeg(img)
        elif degrade_choice == "blur":
            return self.apply_blur(img)
        else:
            return self.apply_down_up(img)


def get_eval_transforms(img_resolution=256, crop_resolution=224, degradation=None, degrade_order="pre_resize"):
    """
    Trả về bộ transforms chuẩn dùng cho pha đánh giá (Validation / Test).
    Hỗ trợ chèn thêm suy thoái tĩnh (degradation) để kiểm thử độ bền vững (robustness test).
    """
    transform_list = []
    if degrade_order not in ("pre_resize", "post_resize"):
        raise ValueError("degrade_order must be pre_resize or post_resize")
    
    if degradation is not None:
        deg_type = degradation.get("type", "")
        if deg_type == "jpeg":
            q = degradation.get("quality", 50)
            transform_list.append(RandomJPEGCompression(min_q=q, max_q=q))
        elif deg_type == "blur":
            r = degradation.get("radius", 1.5)
            transform_list.append(lambda img: img.filter(ImageFilter.GaussianBlur(radius=r)))
        elif deg_type == "down_up":
            ratio = degradation.get("ratio", 0.5)
            transform_list.append(lambda img: _apply_fixed_down_up(img, ratio))

    geometry = [
        transforms.Resize((img_resolution, img_resolution)),
        transforms.CenterCrop(crop_resolution),
    ]
    transform_list = geometry + transform_list if degrade_order == "post_resize" else transform_list + geometry
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(transform_list)


def get_train_transforms(img_resolution=256, crop_resolution=224, use_dual_stream=True, scheduler=None):
    """
    Trả về bộ transforms dùng cho huấn luyện / fine-tuning.
    Nếu có scheduler (CurriculumDegradationScheduler): Áp dụng giáo trình suy biến 3 giai đoạn.
    Nếu use_dual_stream=True và scheduler=None: Áp dụng suy biến tĩnh ngẫu nhiên (30% Clean + 70% Degraded).
    """
    transform_list = [
        transforms.Resize((img_resolution, img_resolution)),
        transforms.RandomCrop(crop_resolution),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    if scheduler is not None:
        transform_list.append(scheduler)
    elif use_dual_stream:
        transform_list.append(DualStreamRobustAugmentation(clean_prob=0.5))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(transform_list)


def _apply_fixed_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _apply_fixed_down_up(img: Image.Image, ratio: float) -> Image.Image:
    w, h = img.size
    low_w, low_h = max(16, int(w * ratio)), max(16, int(h * ratio))
    return img.resize((low_w, low_h), Image.BILINEAR).resize((w, h), Image.BILINEAR)


# ============================================================
# MILESTONE 6 — Task 6.3-B (Thành viên B)
# PairedCleanDegradedAugmentation
# ============================================================

class PairedCleanDegradedAugmentation:
    """
    Pipeline tăng cường ghép cặp ảnh sạch—suy biến — Task 6.3-B & Task 6.2-R (Phase 6-R).

    Với mỗi ảnh đầu vào x, pipeline xuất ra cặp tensor:
        (x_clean, x_degraded)  — cùng nhãn phân loại, cùng kích thước (3, 224, 224)

    - x_clean:    Chỉ áp dụng Resize → CenterCrop → ToTensor → Normalize. Bảo toàn cấu
                  trúc vi mô gốc phục vụ tính toán L_KD của CrossDomainDistillationLoss.
    - x_degraded: Áp dụng cơ chế Clean-First (clean_prob=0.70):
                    • 70% xác suất giữ nguyên bản sạch (không áp dụng suy biến vật lý).
                    • 30% xác suất áp dụng một trong các suy biến nhẹ:
                        - JPEG nén nhẹ Q ∈ [50, 90] (xác suất 50%)
                        - Gaussian Blur σ ∈ [0.5, 1.5] (xác suất 30%)
                        - Down-Up Sampling tỷ lệ 160/224 ≈ 0.714 (xác suất 20%)
                  Bảo toàn vết giả mạo vi mô của ảnh gốc, tránh trôi dạt biểu diễn.

    Trả về Tuple[torch.Tensor, torch.Tensor] để tương thích với DataLoader chuẩn.
    Sử dụng kết hợp với PairedConsistencyLoss và CrossDomainTotalLoss.
    """

    def __init__(
        self,
        img_resolution: int = 256,
        crop_resolution: int = 224,
        clean_prob: float = 0.70,
        jpeg_q_range: tuple = (50, 90),
        blur_sigma_range: tuple = (0.5, 1.5),
        down_ratio: float = 160.0 / 224.0,
        jpeg_prob: float = 0.50,
        blur_prob: float = 0.30,
        preprocessing: str = "legacy_imagenet",
    ):
        """
        Args:
            img_resolution:   Kích thước resize trước khi crop.
            crop_resolution:  Kích thước CenterCrop cuối cùng (mặc định 224 cho CLIP ViT-L/14).
            clean_prob:       Xác suất giữ nguyên bản sạch cho nhánh suy biến (mặc định 0.70).
            jpeg_q_range:     Khoảng chất lượng JPEG nhẹ (min_q, max_q), mặc định (50, 90).
            blur_sigma_range: Khoảng sigma Gaussian Blur nhẹ (min_σ, max_σ), mặc định (0.5, 1.5).
            down_ratio:       Tỷ lệ giảm độ phân giải cho Down-Up Sampling, mặc định 160/224.
            jpeg_prob:        Xác suất chọn JPEG làm suy biến.
            blur_prob:        Xác suất chọn Blur (phần còn lại là Down-Up).
        """
        self.img_resolution   = img_resolution
        self.crop_resolution  = crop_resolution
        self.clean_prob       = clean_prob
        self.jpeg_q_range     = jpeg_q_range
        self.blur_sigma_range = blur_sigma_range
        self.down_ratio       = down_ratio
        self.jpeg_prob        = jpeg_prob
        self.blur_prob        = blur_prob

        self.preprocessing = preprocessing
        self._resize_crop, self._to_tensor_norm, _, _ = preprocessing_parts(
            preprocessing, img_resolution, crop_resolution)
        self.clean_transform = transforms.Compose([self._resize_crop, self._to_tensor_norm])

    def _apply_degradation(self, img: Image.Image) -> Image.Image:
        """Áp dụng một suy biến ngẫu nhiên theo phân phối xác suất được cấu hình."""
        r = random.random()
        if r < self.jpeg_prob:
            # JPEG compression nhẹ
            q = random.randint(*self.jpeg_q_range)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        elif r < self.jpeg_prob + self.blur_prob:
            # Gaussian Blur nhẹ
            sigma = random.uniform(*self.blur_sigma_range)
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        else:
            # Down-Up Sampling nhẹ (224 -> 160 -> 224)
            w, h = img.size
            lw = max(16, int(w * self.down_ratio))
            lh = max(16, int(h * self.down_ratio))
            resample = getattr(Image, "Resampling", Image).BICUBIC
            return img.resize((lw, lh), resample).resize((w, h), resample)

    def __call__(self, img: Image.Image):
        """
        Args:
            img: PIL Image RGB

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                x_clean    — tensor (3, 224, 224) không suy biến
                x_degraded — tensor (3, 224, 224) với xác suất 70% sạch / 30% suy biến nhẹ
        """
        # Ảnh sạch
        x_clean = self.clean_transform(img)

        # Cơ chế Clean-First Stream (Task 6.2-R): 70% giữ ảnh sạch nguyên bản
        if self.clean_prob > 0.0 and random.random() < self.clean_prob:
            return x_clean, x_clean

        # 30% còn lại áp dụng suy biến nhẹ
        img_cropped  = self._resize_crop(img)
        img_degraded = self._apply_degradation(img_cropped)
        x_degraded   = self._to_tensor_norm(img_degraded)

        return x_clean, x_degraded


# ============================================================
# TASK 7.8 — DegradationPipelineV2
# Mở rộng PairedCleanDegradedAugmentation với phân phối
# mạng xã hội 2024-2025: JPEG Q30-85, WebP, Double-JPEG,
# Composite, Blur σ [0.5-2.0], Down-Up [0.50-0.75].
# ============================================================

class DegradationPipelineV2(PairedCleanDegradedAugmentation):
    """
    Pipeline suy biến v2 — Task 7.8.

    API tương thích ngược: __call__(img) -> Tuple[Tensor, Tensor]
    - x_clean:    Resize → CenterCrop → ToTensor → Normalize (không đổi)
    - x_degraded: phân phối đầy đủ theo config v2

    Attributes:
        last_tag: list[str] — tag suy biến vừa áp dụng (phục vụ test/phân tích).
    """

    _SINGLE_TYPES = ["jpeg", "webp", "double_jpeg", "blur", "down_up"]

    @classmethod
    def from_config(cls, path_or_dict, **kwargs):
        """Khởi tạo từ đường dẫn YAML hoặc dict config."""
        if isinstance(path_or_dict, dict):
            cfg = dict(path_or_dict)
        else:
            assert yaml is not None, "Cần cài pyyaml: pip install pyyaml"
            with open(path_or_dict, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        for k, v in kwargs.items():
            cfg[k] = v
        return cls(config=cfg, **kwargs)

    def __init__(self, config: Optional[Dict[str, Any]] = None, return_pair: bool = True, **kwargs):
        """
        Args:
            config: dict từ degradation_v2.yaml. Nếu None, dùng default v2.
            return_pair: nếu True trả về (x_clean, x_degraded), nếu False chỉ trả về x_degraded (đơn tensor).
            **kwargs: forward sang PairedCleanDegradedAugmentation (backward-compat).
        """
        cfg = config or {}
        crop_res = kwargs.pop("crop_resolution", cfg.get("crop_resolution", 224))
        img_res  = kwargs.pop("img_resolution",  cfg.get("img_resolution",  crop_res))
        clean_p  = kwargs.pop("clean_prob",      cfg.get("clean_prob",      0.60))
        self.return_pair = return_pair

        # Khởi tạo nhánh clean (dùng lại từ parent)
        super().__init__(
            img_resolution=img_res,
            crop_resolution=crop_res,
            clean_prob=clean_p,
            **kwargs,
        )

        # Tham số v2
        j = cfg.get("jpeg", {})
        self._jpeg_prob        = j.get("prob", 0.55)
        self._jpeg_heavy_share = j.get("heavy_share", 0.35)
        def quality_range(value, name):
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or any(type(q) is not int for q in value)
                    or not 1 <= value[0] <= value[1] <= 100):
                raise ValueError(f"jpeg.{name} must contain two ordered integers in [1, 100]")
            return tuple(value)

        q_min, q_max = quality_range(j.get("q_range", [30, 85]), "q_range")
        self._jpeg_q_range = (q_min, q_max)
        if not 0 <= self._jpeg_heavy_share <= 1:
            raise ValueError("jpeg.heavy_share must be in [0, 1]")
        # q_range is authoritative, including when a YAML still has old subranges.
        # A disjoint subrange collapses to the nearest allowed quality.
        def bounded_range(value, name):
            low, high = quality_range(value, name)
            return (max(q_min, min(q_max, low)), max(q_min, min(q_max, high)))

        split = max(q_min, min(q_max, 45))
        self._jpeg_heavy_range = bounded_range(j.get("heavy_range", [q_min, split]), "heavy_range")
        self._jpeg_light_range = bounded_range(j.get("light_range", [split, q_max]), "light_range")

        w = cfg.get("webp", {})
        self._webp_prob  = w.get("prob", 0.15)
        self._webp_q     = tuple(w.get("q_range", [30, 70]))

        dj = cfg.get("double_jpeg", {})
        self._djpeg_prob = dj.get("prob", 0.15)
        self._djpeg_q1   = tuple(dj.get("q1_range", [70, 90]))
        self._djpeg_q2   = tuple(dj.get("q2_range", [30, 50]))

        bl = cfg.get("blur", {})
        self._blur_prob  = bl.get("prob", 0.20)
        self._blur_sigma = tuple(bl.get("sigma_range", [0.5, 2.0]))

        du = cfg.get("down_up", {})
        self._du_prob    = du.get("prob", 0.20)
        self._du_ratio   = tuple(du.get("ratio_range", [0.50, 0.75]))

        comp = cfg.get("composite", {})
        self._comp_prob  = comp.get("prob", 0.25)
        self._comp_pairs = comp.get("pairs", [["jpeg", "down_up"], ["jpeg", "blur"],
                                               ["webp", "down_up"], ["double_jpeg", "blur"]])

        # Pre-build transforms tái sử dụng
        self._webp   = RandomWebPCompression(*self._webp_q, method=w.get("method", 0))
        self._djpeg  = RandomDoubleJPEG(self._djpeg_q1, self._djpeg_q2)

        self.last_tag: List[str] = []

    # ------------------------------------------------------------------
    # Primitive degraders
    # ------------------------------------------------------------------

    def _apply_jpeg(self, img: Image.Image) -> Tuple[Image.Image, str]:
        """JPEG với nhóm nặng/nhẹ tách rời — đảm bảo heavy_share chính xác."""
        if random.random() < self._jpeg_heavy_share:
            q = random.randint(*self._jpeg_heavy_range)
        else:
            q = random.randint(*self._jpeg_light_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB"), f"jpeg:{q}"

    def _apply_webp(self, img: Image.Image) -> Tuple[Image.Image, str]:
        result = self._webp(img)
        return result, f"webp"

    def _apply_double_jpeg(self, img: Image.Image) -> Tuple[Image.Image, str]:
        result = self._djpeg(img)
        return result, "double_jpeg"

    def _apply_blur(self, img: Image.Image) -> Tuple[Image.Image, str]:
        sigma = random.uniform(*self._blur_sigma)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma)), f"blur:{sigma:.2f}"

    def _apply_down_up(self, img: Image.Image) -> Tuple[Image.Image, str]:
        ratio = random.uniform(*self._du_ratio)
        w, h = img.size
        lw, lh = max(16, int(w * ratio)), max(16, int(h * ratio))
        resample = getattr(Image, "Resampling", Image).BICUBIC
        result = img.resize((lw, lh), resample).resize((w, h), resample)
        return result, f"down_up:{ratio:.2f}"

    def _dispatch(self, name: str, img: Image.Image) -> Tuple[Image.Image, str]:
        """Áp dụng suy biến theo tên."""
        if name == "jpeg":
            return self._apply_jpeg(img)
        elif name == "webp":
            return self._apply_webp(img)
        elif name == "double_jpeg":
            return self._apply_double_jpeg(img)
        elif name == "blur":
            return self._apply_blur(img)
        else:  # down_up
            return self._apply_down_up(img)

    def _pick_single(self) -> str:
        """Chọn 1 suy biến đơn theo trọng số đã chuẩn hoá."""
        weights = [self._jpeg_prob, self._webp_prob, self._djpeg_prob,
                   self._blur_prob, self._du_prob]
        total = sum(weights)
        r = random.uniform(0, total)
        acc = 0.0
        for name, w in zip(self._SINGLE_TYPES, weights):
            acc += w
            if r < acc:
                return name
        return self._SINGLE_TYPES[-1]

    # ------------------------------------------------------------------
    # Public API — tương thích ngược với PairedCleanDegradedAugmentation
    # ------------------------------------------------------------------

    def __call__(self, img: Image.Image, seed: Optional[int] = None):
        """
        Args:
            img:  PIL Image RGB
            seed: nếu không None, đặt random.seed trước khi xử lý (phục vụ test tái lập).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (x_clean, x_degraded), shape (3,224,224)
        """
        if seed is not None:
            random.seed(seed)

        # Share the deterministic PIL resize/crop between the two streams.
        # In single-tensor mode do not normalize an unused clean tensor.
        img_cropped = self._resize_crop(img)
        self.last_tag = []

        # Bước 1: clean_prob
        if random.random() < self.clean_prob:
            x_clean = self._to_tensor_norm(img_cropped)
            if not self.return_pair:
                return x_clean
            return x_clean, x_clean

        # Bước 2: composite?
        if random.random() < self._comp_prob:
            pair = random.choice(self._comp_pairs)
            result = img_cropped
            for name in pair:
                result, tag = self._dispatch(name, result)
                self.last_tag.append(tag)
        else:
            # Bước 3: suy biến đơn
            name = self._pick_single()
            result, tag = self._dispatch(name, img_cropped)
            self.last_tag.append(tag)

        x_degraded = self._to_tensor_norm(result)
        if not self.return_pair:
            return x_degraded
        x_clean = self._to_tensor_norm(img_cropped)
        return x_clean, x_degraded

