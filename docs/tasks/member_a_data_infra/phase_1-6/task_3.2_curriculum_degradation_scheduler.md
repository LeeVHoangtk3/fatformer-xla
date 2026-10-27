# TASK 3.2: XÂY DỰNG MODULE LẬP LỊCH SUY THOÁI CURRICULUM

- **Mã nhiệm vụ**: `Task 3.2`
- **Người phụ trách**: **Thành viên A** *(Data & Infra Lead)*
- **Môi trường thực hiện**: 🖥️ **Local** (phát triển module và viết unit test)
- **Thời hạn hoàn thành**: Tuần 3 (Ngày 18)
- **Độ ưu tiên**: 🔴 Khẩn cấp (Trụ cột Chiến lược 2 của đồ án)
- **Đầu vào (Input)**:
  - Thông số giáo trình suy biến 3 giai đoạn theo epoch:
    - *Pha 1 (Epoch 1–2)*: $Q \in [70, 90]$, $\sigma \in [0.5, 1.0]$, không Down-Up.
    - *Pha 2 (Epoch 3–5)*: $Q \in [45, 70]$, $\sigma \in [1.0, 1.5]$, Down-Up $224 \rightarrow 160 \rightarrow 224$.
    - *Pha 3 (Epoch 6–8)*: $Q \in [30, 50]$, $\sigma \in [1.5, 2.0]$, Down-Up $224 \rightarrow 112 \rightarrow 224$.
- **Đầu ra (Output)**:
  - Module `src/datasets/transforms.py` tích hợp class `CurriculumDegradationScheduler` hoàn chỉnh.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Triệt tiêu nguy cơ sốc gradient (*gradient shock*) khi đưa dải nén nặng vào huấn luyện adapter ngay từ đầu.
2. Thiết kế cơ chế phân bổ mẫu: **30% ảnh Clean** (bảo toàn đặc trưng nhận diện sạch) + **70% ảnh Degraded** (huấn luyện độ bền).
3. Cho phép cập nhật tham số suy biến động theo `epoch` hiện tại thông qua lời gọi `scheduler.set_epoch(epoch)`.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Cài đặt `CurriculumDegradationScheduler` trong `src/datasets/transforms.py`:
```python
import io
import random
from PIL import Image, ImageFilter
import torchvision.transforms as T

class CurriculumDegradationScheduler:
    """
    Bộ lập lịch suy thoái động theo giáo trình 3 giai đoạn (Curriculum Learning).
    Tỷ lệ cố định: 30% Clean, 70% Degraded.
    """
    def __init__(self, clean_prob=0.3):
        self.clean_prob = clean_prob
        self.epoch = 0
        self.stage = 1
        self._update_stage_params()

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._update_stage_params()

    def _update_stage_params(self):
        if self.epoch <= 2:
            # Giai đoạn 1: Nén rất nhẹ, ổn định gradient FAA
            self.stage = 1
            self.q_range = (70, 90)
            self.sigma_range = (0.5, 1.0)
            self.down_size = None
        elif self.epoch <= 5:
            # Giai đoạn 2: Nén vừa, rèn luyện cổng gating λ(x)
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
        return Image.open(buf).copy()

    def apply_blur(self, img: Image.Image) -> Image.Image:
        sigma = random.uniform(*self.sigma_range)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    def apply_down_up(self, img: Image.Image) -> Image.Image:
        if self.down_size is None:
            return img
        orig_size = img.size
        down = img.resize(self.down_size, Image.Resampling.BICUBIC)
        up = down.resize(orig_size, Image.Resampling.BICUBIC)
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
```

---

## 3. RỦI RO & PHÒNG NGỪA
* **Rủi ro**: Quên gọi `scheduler.set_epoch(epoch)` ở đầu vòng lặp epoch trong training loop khiến tham số bị kẹt ở Pha 1.
* **Biện pháp**: Viết hook trong `train.ipynb` tự động in thông báo: `[Curriculum] Epoch {epoch}: Giai đoạn {stage}, Q in {q_range}`.

---

## 4. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Unit test pass 100%: Khi đổi `epoch = 1, 4, 7`, các dải tham số tự động nhảy chuẩn xác theo đúng thông số 3 giai đoạn.
- [ ] Bàn giao module cho Thành viên B & C đưa vào Smoke Test (Task 3.6).
