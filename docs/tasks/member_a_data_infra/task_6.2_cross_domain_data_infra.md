# TASK 6.2: XÂY DỰNG KHO DỮ LIỆU ĐA BỘ SINH ĐỘC LẬP, AUDIT NÂNG CAO & ĐO MỐC SÀN VAL

- **Mã nhiệm vụ**: `Task 6.2` *(Milestone 6 - Tuần 6)*
- **Người thực hiện**: **Thành viên A (Data & Infra Lead)**
- **Người phối hợp**: **Thành viên C (Training Lead)**, **Thành viên B (Architecture Lead)**
- **Môi trường thực thi**: ☁️ **Google Drive 5TB** + 🖥️ **Local / Cloud**
- **Độ ưu tiên**: 🔴 Khẩn cấp (Tiền đề then chốt cho Task 6.3)
- **Trạng thái**: `[ ] Sẵn sàng triển khai`

---

## 1. Mục Tiêu & Cơ Sở Khoa Học

Dựa trên kết luận thực chứng tại [Báo cáo Task 5.1](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/C/bao_cao_task_5.1_full_benchmark_18_datasets.md), mô hình Plan A biểu hiện sự phân hóa rõ rệt giữa ProGAN (Clean 97.95%, Q30 61.6%) và các bộ sinh khác do thiên kiến tập huấn luyện.

Mục tiêu cốt lõi của Task 6.2:
1. Xây dựng kho dữ liệu huấn luyện và kiểm định đa bộ sinh (`train_crossdomain` và `val_crossdomain`) độc lập hoàn toàn với 18 tập test của bài báo gốc.
2. Kiểm toán phân ly dữ liệu đa tầng: Kết hợp mã băm SHA-256 và thuật toán `pHash` (Perceptual Hash) để loại bỏ 100% rủi ro trùng lặp hoặc biến đổi ảnh tương đồng (*near-duplicate*).
3. Phân tách miền ảnh Real theo nguồn thu thập để tránh mô hình học thuộc dấu vết camera/nhiễu nền (*domain shortcut learning*).
4. Thiết lập tệp kê khai nguồn gốc `data_manifest.json` ghi nhận minh bạch giấy phép, bộ sinh và thông số.
5. Thực hiện đo đạc mô hình baseline gốc trên tập `val_crossdomain` để xác lập bảng mốc sàn tham chiếu chuẩn phục vụ Cổng nghiệm thu Gate tại Task 6.4.

---

## 2. Ràng Buộc & Nguyên Tắc Phân Ly Dữ Liệu

1. **Khóa 18 tập test Task 5.1**:
   - Ba tệp nén `test_benchmark_gans.tar`, `test_benchmark_diffusion.tar`, `test_degraded.tar` đóng vai trò bài kiểm tra đối chứng hậu nghiệm.
   - Tuyệt đối không sao chép, không gộp và không lấy mẫu từ 3 tệp này vào `train_crossdomain` hay `val_crossdomain`.
2. **Nguồn dữ liệu độc lập**:
   - Dữ liệu ProGAN: Tập `progan_train` sẵn có (144.000 ảnh).
   - Dữ liệu Diffusion: Mở rộng từ kho `GenImage` (SD v1.4, SD v1.5, Midjourney, Wukong, VQ-DM, ADM).
   - Dữ liệu GANs mở rộng: Trích xuất từ các tập công khai (StyleGAN, BigGAN, CycleGAN) ngoài phân vùng của bài báo gốc.
3. **Phân chia 3 tập con chuẩn mực**:
   - `train_crossdomain`: Dùng để huấn luyện thích ứng tại Task 6.3-C.
   - `val_calibration` (50% của validation): Dùng cho Task 6.4 tối ưu nhiệt độ và quét ngưỡng.
   - `val_selection` (50% còn lại): Dùng cho Task 6.4 chấm điểm độc lập và chọn mô hình.

---

## 3. Nội Dung Thực Hiện Chi Tiết

### Bước 1: Chuẩn hóa cấu trúc thư mục & Phân tách nguồn ảnh Real
Để ngăn ngừa mô hình bắt chước đặc trưng riêng của từng tập ảnh thật thay vì dấu vết giả mạo AI, cấu trúc thư mục được phân tách tường minh theo miền nguồn:
```
data_crossdomain/
├── train/
│   ├── real_imagenet/
│   ├── real_coco/
│   ├── real_ffhq/
│   └── fake/
│       ├── progan/
│       ├── stylegan/
│       ├── stable_diffusion/
│       └── midjourney/
└── val/
    ├── real_imagenet/
    ├── real_coco/
    ├── real_ffhq/
    └── fake/
        ├── progan/
        ├── stylegan/
        ├── stable_diffusion/
        └── midjourney/
```

### Bước 2: Kiểm toán phân ly dữ liệu đa tầng (Dual Audit Protocol)
- Viết script `tools/verify_data_isolation.py`:
  - **Tầng 1 (SHA-256 Hash)**: So sánh đối chiếu toàn bộ tệp với 18 tập test Task 5.1 $\rightarrow$ Tỷ lệ trùng lặp bắt buộc: **0.0%**.
  - **Tầng 2 (Perceptual pHash)**: Tính khoảng cách Hamming ($d_H$). Mọi cặp ảnh có $d_H \le 3$ (dấu hiệu crop, resize, đổi màu từ tập test) đều bị loại bỏ tự động.
  - **Tầng 3 (Train-Val Isolation)**: Đối chiếu chéo giữa `train_crossdomain` và `val_crossdomain` bảo đảm không có mẫu nào xuất hiện ở cả hai tập.

### Bước 3: Thiết lập tệp kê khai dữ liệu `data_manifest.json`
Ghi nhận đầy đủ siêu dữ liệu của từng tệp:
- Tên tệp, mã băm SHA-256, mã pHash.
- Nguồn gốc (Dataset gốc, URL hoặc đường dẫn Drive).
- Giấy phép dữ liệu (License).
- Bộ sinh (Generator architecture, checkpoint release).
- Phân loại miền (Domain, Class, Prompt nếu có).

### Bước 4: Thiết kế bộ lấy mẫu `BalancedCrossDomainSampler`
- Viết module `src/datasets/samplers.py`:
  - Cân bằng trọng số nghịch đảo tần suất mẫu giữa các bộ sinh ($w_g = 1/N_g$).
  - Cân bằng tỷ lệ 1:1 giữa tổng ảnh thật và tổng ảnh giả trong mỗi mini-batch.
  - Phân bổ đều các miền nguồn ảnh thật nhằm triệt tiêu thiên vị nguồn camera.

### Bước 5: Đo đạc mốc sàn Baseline gốc trên `val_crossdomain`
- Chạy đánh giá mô hình baseline gốc `fatformer_4class_ckpt.pth` trên tập `val_crossdomain` (cả nhánh Clean và Degraded $Q=30$).
- Xuất bảng mốc tham chiếu chuẩn hóa:
  - $\text{Base}_{\text{clean}}$ (Clean Macro ACC).
  - $\text{Base}_{\text{fake\_clean}}$ (Clean Fake ACC).
  - $\text{Base}_{\text{q30}}$ (Q30 Overall ACC).
  - $\text{Base}_{\text{fake\_q30}}$ (Q30 Fake ACC).
  - $\text{Base}_{\text{real\_q30}}$ (Q30 Real ACC).
- Lưu kết quả vào `DATASET/logs/val_crossdomain_baseline_metrics.json` làm căn cứ khóa cứng cho Gate 4 điều kiện tại Task 6.4.

---

## 4. Tiêu Chuẩn Nghiệm Thu (Definition of Done)

- [ ] Kho dữ liệu `crossdomain_staging.tar` được đóng gói hoàn chỉnh trên Google Drive 5TB.
- [ ] Vượt qua kiểm toán 3 tầng: 0% trùng SHA-256 và 0% trùng pHash ($d_H \le 3$) so với 18 tập test; 0% trùng giữa train và val.
- [ ] Tệp `data_manifest.json` được tạo lập đầy đủ thông tin nguồn gốc và phân loại miền.
- [ ] Module `BalancedCrossDomainSampler` vượt qua unit test nạp batch cân bằng.
- [ ] Bảng mốc sàn tham chiếu `val_crossdomain_baseline_metrics.json` được đo đạc xong và bàn giao cho Lead C.
- [ ] Biên soạn báo cáo nghiệm thu tại [`docs/bao_cao/A/bao_cao_task_6.2_cross_domain_data_infra.md`](../../bao_cao/A/bao_cao_task_6.2_cross_domain_data_infra.md).
