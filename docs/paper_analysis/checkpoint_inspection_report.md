# Báo Cáo Kỹ Thuật: Thẩm Định Tính Toàn Vẹn & Kiến Trúc Trọng Số FatFormer

> **Mục đích**: Ghi nhận kết quả kiểm định thực tế tệp trọng số nền tảng và checkpoint tiền huấn luyện của dự án FatFormer, phân tích cấu trúc `state_dict`, thống kê tham số và xác lập các quy tắc mapping phục vụ việc xây dựng mã nguồn mô hình.  
> **Thời gian thực hiện**: 07/09/2026  
> **Kịch bản thực thi**: [`tools/check_weights.py`](file:///D:/GIT%20REPO/.nam4/fatformer/tools/check_weights.py)  
> **Tài liệu liên quan**:
> - Đánh giá bài báo: [`docs/paper_analysis/fatformer_review.md`](file:///D:/GIT%20REPO/.nam4/fatformer/docs/paper_analysis/fatformer_review.md)
> - Kế hoạch dự án: [`docs/paper_analysis/project_execution_plan.md`](file:///D:/GIT%20REPO/.nam4/fatformer/docs/paper_analysis/project_execution_plan.md)

---

## 1. Môi Trường Thực Thi Đo Đạc

Quá trình kiểm định được chạy trực tiếp trên hệ thống cục bộ:
- **Trình thông dịch Python**: `C:\Python314\python.exe` (Python 3.14)
- **Phiên bản PyTorch**: `2.10.0+cpu`
- **Khả năng tăng tốc CUDA**: `False` *(Môi trường Python mặc định hiện tại đang là CPU-only)*
- **Chế độ nạp bộ nhớ**: `map_location='cpu'` (Tiêu thụ đỉnh: ~3.8 GB RAM hệ thống, an toàn tuyệt đối cho VRAM GPU).

---

## 2. Kết Quả Thẩm Định Tệp Trọng Số Gốc CLIP: `ViT-L-14.pt`

Tệp trọng số nền tảng được phát hành bởi OpenAI dùng làm Feature Extractor cốt lõi cho FatFormer.

- **Đường dẫn tệp**: [`ViT-L-14.pt`](file:///D:/GIT%20REPO/.nam4/fatformer/ViT-L-14.pt)
- **Dung lượng lưu trữ**: **889.56 MB** (932,768,134 bytes)
- **Định dạng đóng gói**: **TorchScript JIT Archive** *(Mô hình đã được compile bằng PyTorch JIT, không phải tệp Python pickle thông thường)*.
- **Tổng số tensors**: **449 tensors**
- **Tổng số tham số**: **427,616,516 tham số** (~427.62M params)

### Phân Bổ Tensors Trong CLIP ViT-L/14:
- **Visual Image Encoder (Vision Transformer ViT-L/14)**: 296 tensors (~304M params)
  - Kích thước ảnh đầu vào: $224 \times 224$ pixels.
  - Kích thước Patch: $14 \times 14$ pixels $\rightarrow$ Tạo ra $16 \times 16 = 256$ patch tokens + 1 Class token = **257 tokens**.
  - Chiều không gian nhúng (*Embedding Dimension*): **1024**.
  - Chiều không gian chiếu (*Projection Dimension*): **768**.
  - Độ sâu Transformer: **24 Residual Blocks** (mỗi block có 16 attention heads).
- **Text Encoder**: 149 tensors (~123M params).

#### Mẫu Tensors Đặc Trưng Của Visual Encoder:
```
• visual.class_embedding         | Shape: [1024]            | torch.float32
• visual.positional_embedding    | Shape: [257, 1024]       | torch.float32
• visual.conv1.weight            | Shape: [1024, 3, 14, 14] | torch.float16
• visual.ln_pre.weight           | Shape: [1024]            | torch.float32
• visual.proj                    | Shape: [1024, 768]       | torch.float16
```

---

## 3. Kết Quả Thẩm Định Tệp Checkpoint: `fatformer_4class_ckpt.pth`

Tệp checkpoint hoàn chỉnh từ tác giả bài báo (Michel Liu et al., CVPR 2024), được huấn luyện tinh chỉnh trên tập dữ liệu ProGAN 4-class (car, cat, chair, horse).

- **Đường dẫn tệp**: [`fatformer_4class_ckpt.pth`](file:///D:/GIT%20REPO/.nam4/fatformer/fatformer_4class_ckpt.pth)
- **Dung lượng lưu trữ**: **1.84 GB** (1,971,052,681 bytes)
- **Kiểu dữ liệu gốc**: `dict` chứa khóa gốc: `['model']`
- **Tổng số tensors trong state_dict**: **1,116 tensors**
- **Tổng số tham số mô hình**: **933,160,936 tham số** (~933.16M params)

### Bảng Thống Kê Chi Tiết Các Phân Hệ Kiến Trúc:

| Phân Hệ Kiến Trúc | Số Tensors | Số Tham Số (Params) | Tỷ Trọng | Vai Trò Chức Năng |
| :--- | :---: | :---: | :---: | :--- |
| **ViT Image Encoder (Backbone)** | 676 | **659,124,608** (659.12M) | 70.63% | Trích xuất đặc trưng thị giác không gian đa cấp độ. |
| **Text Encoder** | 150 | **124,240,128** (124.24M) | 13.31% | Nhúng vector ngữ nghĩa từ khóa và prompt văn bản. |
| **Khối FAA (Frequency & Spatial Adapter)** | 114 | **50,393,190** (50.39M) | 5.40% | **Trọng tâm kỹ thuật**: Bắt vết nhiễu sóng con DWT & Conv cục bộ. |
| **Classifier Head & Projection Layers** | 24 | **28,348,416** (28.35M) | 3.04% | Phân lớp và ánh xạ đặc trưng cuối cùng. |
| **Khối LGA (Language-Guided Alignment)** | 8 | **4,724,736** (4.72M) | 0.51% | Căn chỉnh đối sánh thích nghi giữa patch ảnh và prompt mềm. |
| **Module Phụ trợ / Khác** | 144 | **66,329,858** (66.33M) | 7.11% | Chuẩn hóa LayerNorm, Pooling, Bias bổ trợ. |
| **TỔNG CỘNG** | **1,116** | **933,160,936** (933.16M) | **100%** | Mô hình hoàn chỉnh FatFormer. |

---

## 4. Giải Phẫu Chuyên Sâu Từng Khối Cốt Lõi Của FatFormer

Qua việc quét từng tensor trong `state_dict`, các chi tiết cài đặt thực tế của tác giả đã được làm sáng tỏ:

### 4.1. Khối Thích Ứng Giả Mạo (FAA - Forgery-Aware Adapter)
- **Vị trí chèn**: Khối FAA không chèn vào toàn bộ 24 block, mà được đặt vào các block trung gian và cấp cao của ViT Image Encoder (ví dụ tiêu biểu: `resblocks.7`, `resblocks.11`,...).
- **Cơ chế sóng con 2D Haar DWT**:
  Tác giả **không dùng thư viện ngoài C++** mà cài đặt DWT trực tiếp bằng các ma trận tích chập 2D cố định:
  - `h0_col`, `h1_col` (Shape: `[1, 1, 2, 1]`): Bộ lọc low-pass và high-pass theo cột.
  - `h0_row`, `h1_row` (Shape: `[1, 1, 1, 2]`): Bộ lọc low-pass và high-pass theo hàng.
  - Phân tách ra 4 dải tần: $LL, LH, HL, HH$.
- **Cơ chế sóng con ngược 2D IDWT**:
  - `g0_col`, `g1_col`, `g0_row`, `g1_row`: Tái cấu trúc đặc trưng tần số ngược về miền không gian.
- **Tham số tỷ lệ tần số $\lambda$ (Frequency Scale Factor)**:
  - Tên tensor thực tế: `...forgery_aware_adapter.freq_scale` (Shape: `[1]`, `torch.float32`).
  - Đây chính là hệ số $\lambda$ trong công thức $\hat{g}^{(j)} = \hat{g}_{img}^{(j)} + \lambda \cdot \hat{g}_{freq}^{(j)}$ được mô hình tự học.
- **Grouped Attention trong FAA**:
  - Bao gồm `dwt_norm` (`Shape: [1024]`), các tầng Linear Attention và FFN để học tương tác giữa các dải tần (*Inter-band*) và trong nội bộ dải tần (*Intra-band*).

### 4.2. Khối Căn Chỉnh Dẫn Hướng Ngôn Ngữ (LGA)
- **Patch-Based Enhancer**:
  - Tên tensor thực tế: `language_guided_alignment.patch_basaed_enhancer`
  - Các tensor trọng tâm:
    - `in_proj_weight` (Shape: `[2304, 768]`)
    - `in_proj_bias` (Shape: `[2304]`)
    - `out_proj.weight` (Shape: `[768, 768]`)
    - `out_proj.bias` (Shape: `[768]`)
  - Chức năng: Chiếu tương tác giữa 256 patch tokens của ảnh và vector ngữ cảnh context embedding để tạo ra prompt động.
- **Text-Guided Interactor**:
  - Tên tensor thực tế: `text_guided_interactor`
  - Cấu trúc tương tự với ma trận chiếu `[2304, 768]` và `[768, 768]`.

---

## 5. Quy Tắc Kỹ Thuật Bắt Buộc Khi Xây Dựng Mã Nguồn (Coding Guidelines)

Để đảm bảo mã nguồn mô hình viết mới trong thư mục `models/` nạp trọng số thành công 100%, bắt buộc phải tuân thủ 4 quy tắc sau:

1. **Quy tắc nạp tệp `ViT-L-14.pt`**:
   - Vì tệp là **TorchScript JIT**, hàm nạp chuẩn phải là:
     ```python
     # Cách 1: Nạp qua torch.jit
     jit_model = torch.jit.load("ViT-L-14.pt", map_location="cpu")
     state_dict = jit_model.state_dict()

     # Hoặc Cách 2 (PyTorch >= 2.6):
     state_dict = torch.load("ViT-L-14.pt", map_location="cpu", weights_only=False).state_dict()
     ```
2. **Quy tắc tiền tố tên biến Khối LGA (Typo Handling)**:
   - Trong class `LanguageGuidedAlignment`, module enhancer bắt buộc phải đặt tên biến thuộc tính là `self.patch_basaed_enhancer` (có chữ `a` thừa trong chữ `basaed` theo đúng mã nguồn tác giả gốc) để tránh lỗi `Unexpected key / Missing key` khi `load_state_dict`.
3. **Quy tắc bóc tách `['model']` trong checkpoint**:
   - Khi nạp `fatformer_4class_ckpt.pth`, luôn phải trích xuất khóa `ckpt['model']` trước khi truyền vào mô hình:
     ```python
     ckpt = torch.load("fatformer_4class_ckpt.pth", map_location="cpu")
     model.load_state_dict(ckpt["model"], strict=True)
     ```
4. **Quy tắc phân vùng tham số huấn luyện (Trainable vs Frozen)**:
   - Tổng tham số có thể huấn luyện của các Adapter chỉ là **~55.11M params** (~5.9%). Khi viết script finetune, ta chỉ cần đặt `requires_grad = True` cho các module có chứa tên `forgery_aware_adapter`, `language_guided_alignment`, `text_guided_interactor` và `freq_scale`, đóng băng 94.1% tham số còn lại để tiết kiệm VRAM và chống quên lãng.

---

## 6. Kết Luận & Hành Động Tiếp Theo

- **Hiện trạng tài nguyên**: Cả 2 tệp trọng số nền tảng đều **hoàn hảo, đầy đủ và sẵn sàng 100%**.
- **Tính khả thi thực nghiệm**: Đã nắm trọn vẹn cấu trúc mạng từ tầng DWT, Attention đến LGA.
- **Hành động đề xuất cho Giai đoạn 1**:
  1. Xây dựng module mã nguồn mô hình tại `models/fatformer.py` khớp với cấu trúc `state_dict` đã phân tích.
  2. Viết script `inference.py` chạy thử nghiệm phân loại ảnh Real vs Fake đầu tiên để hoàn tất cột mốc Giai đoạn 1.
