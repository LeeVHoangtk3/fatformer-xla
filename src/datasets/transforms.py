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


def get_eval_transforms(img_resolution=256, crop_resolution=224, degradation=None):
    """
    Trả về bộ transforms chuẩn dùng cho pha đánh giá (Validation / Test).
    Hỗ trợ chèn thêm suy thoái tĩnh (degradation) để kiểm thử độ bền vững (robustness test).
    
    degradation có thể là:
      - None (ảnh gốc / clean)
      - {"type": "jpeg", "quality": 30/50/70}
      - {"type": "blur", "radius": 1.5}
      - {"type": "down_up", "ratio": 0.5}
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


def get_train_transforms(img_resolution=256, crop_resolution=224, use_dual_stream=True):
    """
    Trả về bộ transforms dùng cho huấn luyện / fine-tuning.
    Nếu use_dual_stream=True: Tích hợp 30% Clean + 70% Degraded.
    """
    transform_list = [
        transforms.Resize((img_resolution, img_resolution)),
        transforms.RandomCrop(crop_resolution),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    if use_dual_stream:
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
