# TASK 3.1: CHUẨN HÓA DỮ LIỆU HUẤN LUYỆN STAGING 90/10

- **Mã nhiệm vụ**: `Task 3.1`
- **Người phụ trách**: **Thành viên A** *(Data & Infra Lead)*
- **Môi trường thực hiện**: ☁️ **Colab** (tải dataset trực tiếp về Drive 5TB, giải nén và lọc)
- **Thời hạn hoàn thành**: Tuần 3 (Ngày 16)
- **Độ ưu tiên**: 🔴 Khẩn cấp (Chuẩn bị nguyên liệu cho chiến dịch huấn luyện Tuần 4)
- **Đầu vào (Input)**:
  - Tập ProGAN 4-class (`car, cat, chair, horse`) gồm ~32.400 ảnh.
  - Bộ dữ liệu GenImage mã nguồn mở (Midjourney v5 và Stable Diffusion v1.5).
- **Đầu ra (Output)**:
  - Tệp nén `datasets/diffusion_staging.tar` trên Google Drive 5TB (tổng ~36.000 ảnh: tỷ lệ 90% ProGAN + 10% GenImage catalyst).

---

## 1. MỤC TIÊU KỸ THUẬT
1. Giải quyết hiện tượng thiên lệch GAN (*GANs overfitting bias*): Mô hình huấn luyện thuần trên ProGAN thường thất bại khi gặp ảnh sinh bởi mô hình khuếch tán (Diffusion Models).
2. Chuẩn hóa tập **Chất xúc tác 10% (Catalyst Staging Set)** gồm đúng 3.600 ảnh:
   - **Stable Diffusion v1.5**: 1.200 ảnh Real + 1.200 ảnh Fake = 2.400 ảnh.
   - **Midjourney v5**: 600 ảnh Real + 600 ảnh Fake = 1.200 ảnh.
3. Hợp nhất với ~32.400 ảnh ProGAN 4-class gốc thành một tập dữ liệu Staging duy nhất theo tỷ lệ chuẩn **90/10** (đóng gói `diffusion_staging.tar`).

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Bước 1: Trích xuất 3.600 Ảnh GenImage Xúc Tác
Viết script lọc mẫu ngẫu nhiên cân bằng:
```python
import os
import shutil
import random

def extract_catalyst(src_dir: str, dst_dir: str, num_samples: int):
    """Trích xuất cân bằng num_samples ảnh cho mỗi class (0_real và 1_fake)."""
    for cls in ["0_real", "1_fake"]:
        s_cls = os.path.join(src_dir, cls)
        d_cls = os.path.join(dst_dir, cls)
        os.makedirs(d_cls, exist_ok=True)
        
        all_imgs = [f for f in os.listdir(s_cls) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        chosen = random.sample(all_imgs, min(num_samples, len(all_imgs)))
        
        for f in chosen:
            shutil.copy(os.path.join(s_cls, f), os.path.join(d_cls, f))
        print(f"[✓] Đã copy {len(chosen)} ảnh từ {s_cls} -> {d_cls}")

# Trích xuất 1.200 ảnh/nhãn cho SD 1.5 và 600 ảnh/nhãn cho Midjourney
extract_catalyst("/content/raw_genimage/sd_v15/train", "/content/staging/sd_v15", 1200)
extract_catalyst("/content/raw_genimage/midjourney/train", "/content/staging/midjourney", 600)
```

### Bước 2: Hợp Nhất ProGAN và Đổi Tên Tránh Trùng Khóa
```python
def merge_into_staging(progan_dir: str, staging_dir: str, output_dir: str):
    os.makedirs(os.path.join(output_dir, "0_real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "1_fake"), exist_ok=True)
    
    # 1. Copy ProGAN (90%)
    for cls in ["0_real", "1_fake"]:
        p_path = os.path.join(progan_dir, cls)
        for f in os.listdir(p_path):
            shutil.copy(os.path.join(p_path, f), os.path.join(output_dir, cls, f"progan_{f}"))
            
    # 2. Copy GenImage SD v1.5 & Midjourney (10%)
    for model_name in ["sd_v15", "midjourney"]:
        for cls in ["0_real", "1_fake"]:
            s_path = os.path.join(staging_dir, model_name, cls)
            for f in os.listdir(s_path):
                shutil.copy(os.path.join(s_path, f), os.path.join(output_dir, cls, f"{model_name}_{f}"))
```

### Bước 3: Đóng Gói Thành `diffusion_staging.tar`
```bash
cd /content/
tar -cf diffusion_staging.tar -C staging_final .
cp diffusion_staging.tar /content/drive/MyDrive/FatFormer_Hub/datasets/
```

---

## 3. RỦI RO & PHÒNG NGỪA
* **Rủi ro**: Tỷ lệ ảnh Diffusion quá lớn (>20%) sẽ làm mô hình bị mất khả năng phát hiện GAN gốc (*catastrophic forgetting*).
* **Biện pháp**: Kiểm soát chặt chẽ đúng số lượng 3.600 ảnh catalyst (chiếm ~10% tổng tập huấn luyện).

---

## 4. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Tệp `diffusion_staging.tar` có sẵn trên Drive 5TB (tổng số lượng ~36.000 ảnh).
- [ ] Tỷ lệ chính xác: $90\%$ ProGAN 4-class + $10\%$ GenImage (1.200 SD v1.5 + 600 Midjourney mỗi class Real/Fake).
- [ ] Bàn giao cho Thành viên B & C phục vụ Smoke Test (Task 3.6) và Train A100 (Task 4.1).
