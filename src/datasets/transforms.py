import io
import random
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class RandomJPEGCompression:
    """Áp dụng nén JPEG ngẫu nhiên với hệ số chất lượng Q trong khoảng [min_q, max_q]."""
    def __init__(self, min_q=30, max_q=95):
        self.min_q = min_q
        self.max_q = max_q

    def __call__(self, img: Image.Image) -> Image.Image:
        q = random.randint(self.min_q, self.max_q)
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


class DualStreamRobustAugmentation:
    """
    Pipeline tăng cường dữ liệu kép (Dual-Stream):
    - 30% xác suất giữ ảnh nguyên bản (Clean Stream) để bảo toàn đặc trưng ban đầu.
    - 70% xác suất áp dụng một trong các suy thoái mạng xã hội (Degraded Stream: JPEG, Blur, Down-Up).
    """
    def __init__(self, clean_prob=0.3):
        self.clean_prob = clean_prob
        self.jpeg = RandomJPEGCompression(min_q=30, max_q=95)
        self.blur = RandomGaussianBlur(min_r=0.5, max_r=3.0)
        self.down_up = RandomDownUpSampling(min_ratio=0.3, max_ratio=0.9)

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
    
    Tỷ lệ phân bổ mẫu: 30% Clean (bảo toàn trần 96.45% ACC), 70% Degraded.
    """
    def __init__(self, clean_prob=0.3):
        self.clean_prob = clean_prob
        self.epoch = 1
        self.stage = 1
        self._update_stage_params()

    def set_epoch(self, epoch: int):
        self.epoch = max(1, epoch)
        self._update_stage_params()

    def _update_stage_params(self):
        if self.epoch <= 2:
            # Giai đoạn 1: Nén rất nhẹ, ổn định gradient FAA
            self.stage = 1
            self.q_range = (70, 90)
            self.sigma_range = (0.5, 1.0)
            self.down_size = None
        elif self.epoch <= 5:
            # Giai đoạn 2: Nén vừa, rèn luyện mạng cổng Gating λ(x)
            self.stage = 2
            self.q_range = (45, 70)
            self.sigma_range = (1.0, 1.5)
            self.down_size = (160, 160)
        else:
            # Giai đoạn 3: Nén sâu thử thách, tôi luyện biểu diễn vi sai SRM
            self.stage = 3
            self.q_range = (30, 50)
            self.sigma_range = (1.5, 2.0)
            self.down_size = (112, 112)

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


def get_eval_transforms(img_resolution=256, crop_resolution=224, degradation=None):
    """
    Trả về bộ transforms chuẩn dùng cho pha đánh giá (Validation / Test).
    Hỗ trợ chèn thêm suy thoái tĩnh (degradation) để kiểm thử độ bền vững (robustness test).
    """
    transform_list = []
    
    if degradation is not None:
        deg_type = degradation.get("type", "")
        if deg_type == "jpeg":
            q = degradation.get("quality", 50)
            transform_list.append(lambda img: _apply_fixed_jpeg(img, q))
        elif deg_type == "blur":
            r = degradation.get("radius", 1.5)
            transform_list.append(lambda img: img.filter(ImageFilter.GaussianBlur(radius=r)))
        elif deg_type == "down_up":
            ratio = degradation.get("ratio", 0.5)
            transform_list.append(lambda img: _apply_fixed_down_up(img, ratio))

    transform_list.extend([
        transforms.Resize((img_resolution, img_resolution)),
        transforms.CenterCrop(crop_resolution),
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
        transform_list.append(DualStreamRobustAugmentation(clean_prob=0.3))

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
