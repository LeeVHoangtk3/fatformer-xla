# BÁO CÁO NGHIỆM THU KIỂM TOÁN KỸ THUẬT: KHO DỮ LIỆU GOOGLE DRIVE 5TB
## TOÀN BỘ 6 TỆP NÉN NGUYÊN KHỐI (.TAR) & 927.927 ẢNH DỰ ÁN FATFORMER-XLA

> **Người thực hiện kiểm toán**: Cả nhóm (*Lead A, Lead B, Lead C*)  
> **Chủ trì thẩm định**: **Thành viên C** *(Training & Evaluation Lead)*  
> **Thời gian thực hiện**: 2026-10-08  
> **Môi trường thực thi**: Google Colab Pro | Python `tarfile` Streaming Iterator (Không giải nén đĩa)  
> **Tài nguyên lưu trữ**: Google Drive 5TB (`/content/drive/MyDrive/Fatformer/datasets/`)  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC (PASS 100% TIÊU CHUẨN KIỂM TOÁN TOÀN VẸN)`  

---

## I. MỤC TIÊU VÀ BỐI CẢNH KIỂM TOÁN

Trong hệ thống nghiên cứu thực nghiệm **FatFormer-XLA**, toàn bộ dữ liệu huấn luyện, thẩm định và kiểm thử được đóng gói nguyên khối dạng `.tar` trên Google Drive 5TB nhằm tuân thủ nghiêm ngặt **Quy tắc I/O chống nghẽn SSD NVMe** (Task 1.1).

Vào ngày 08/10/2026, nhóm đã tiến hành **Tổng kiểm toán tự động toàn diện** (Data Audit Milestone) bằng script Python phân tích luồng header (`tar.next()`) trực tiếp trên Google Colab để:
1. Xác định chính xác số lượng tệp ảnh thực tế bên trong từng file `.tar` mà không cần giải nén ra bộ nhớ máy ảo.
2. Thẩm định cơ cấu nhãn nhị phân (`0_real` vs `1_fake`) và danh mục các tập con (*subsets*).
3. Đóng băng số liệu chính xác tuyệt đối phục vụ trích dẫn số liệu vào Báo cáo đồ án (Chương 3 & 4) và Slide bảo vệ trước Hội đồng.

---

## II. BẢNG TỔNG KẾT SỐ LIỆU ĐO LƯỜNG THỰC TẾ TRÊN GOOGLE DRIVE 5TB

Dưới đây là kết quả kiểm toán thực tế được ghi nhận từ cell kiểm tra chạy trên Google Colab:

| STT | Tên Tệp Nén (`.tar`) | Dung Lượng Thực Tế | Tổng Số Ảnh Đo Được | Ảnh Thật (`0_real`) | Ảnh Giả (`1_fake`) | Thời Gian Quét | Trạng Thái Toàn Vẹn |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `diffusion_staging.tar` | **1.80 GB** | **3.600** | 1.800 | 1.800 | 36.3 s | ✅ Cân bằng tuyệt đối 50/50 |
| **2** | `progan_train.tar` | **13.90 GB** | **144.024** | 72.012 | 72.012 | 194.1 s | ✅ Cân bằng tuyệt đối 50/50 |
| **3** | `progan_val.tar` | **798.00 MB** | **8.000** | 4.000 | 4.000 | 11.1 s | ✅ Cân bằng tuyệt đối 50/50 |
| **4** | `test_benchmark_diffusion.tar` | **1.22 GB** | **20.000** | 10.000 | 10.000 | 16.4 s | ✅ Cân bằng tuyệt đối 50/50 |
| **5** | `test_benchmark_gans.tar` | **18.74 GB** | **90.329** | 45.169 | 45.160 | 259.1 s | ✅ Đầy đủ 8 họ + tập phụ |
| **6** | `test_degraded.tar` | **19.55 GB** | **661.974** | 331.014 | 330.960 | 307.5 s | ✅ Đầy đủ 6 biến thể suy biến |
| 🏆 | **TỔNG KHO DỮ LIỆU** | **55.98 GB** | **927.927 ảnh** | **465.995** | **461.932** | **824.5 s (~13.7 phút)** | ✅ **PASS 100% TOÀN VẸN** |

*Ghi chú*: Tổng dung lượng kho nén trên Drive 5TB là **55.98 GB**, chỉ chiếm **~1.12%** dung lượng khả dụng của tài khoản Google AI Pro (5.000 GB).

---

## III. PHÂN TÍCH BẢN CHẤT KỸ THUẬT TỪNG TỆP DỮ LIỆU

### 1. `diffusion_staging.tar` (1.80 GB - 3.600 ảnh)
- **Bản chất**: Tập dữ liệu chất xúc tác *Catalyst Staging* phục vụ Chiến lược 3 (Multi-Generator Staging & Dual-Stream Focal Loss) nghiệm thu tại Task 3.1 và vận hành tại Task 4.1.
- **Cơ cấu**: Trích xuất từ bộ dữ liệu mở GenImage (NeurIPS 2023):
  - **Stable Diffusion v1.5**: 2.400 ảnh (1.200 Real ImageNet val + 1.200 Fake sinh từ prompt ImageNet).
  - **Midjourney v5**: 1.200 ảnh (600 Real ImageNet val + 600 Fake sinh từ Midjourney v5).
- **Ý nghĩa**: Giúp mạng FatFormer-XLA làm quen với các vết biên làm mờ đặc thù của Diffusion hiện đại mà không gây sai lệch phân phối học tập.

### 2. `progan_train.tar` (13.90 GB - 144.024 ảnh)
- **Bản chất**: Tập huấn luyện cơ sở 4-class nguyên bản theo chuẩn bài báo CVPR 2024 Wang et al. (`car`, `cat`, `chair`, `horse`).
- **Phân bổ chi tiết**:
  - Toàn bộ 144.024 ảnh chia đều cho 4 danh mục: trung bình **36.006 ảnh/danh mục** (18.003 ảnh Thật từ LSUN + 18.003 ảnh Giả từ ProGAN).
  - Số lượng mẫu đạt mức tối đa (Comprehensive Pool), lý giải vì sao tệp nén đạt 13.90 GB (thay vì mức rút gọn 24.000 ảnh ~ 4.8 GB).
  - Tỷ lệ Real/Fake cân bằng chuẩn xác: $72.012 / 72.012 = 1.0000$.

### 3. `progan_val.tar` (798.00 MB - 8.000 ảnh)
- **Bản chất**: Tập thẩm định nhanh sau mỗi epoch huấn luyện để theo dõi hội tụ và lưu checkpoint tốt nhất (*Best Model Selection*).
- **Cơ cấu**: Đầy đủ **20 danh mục đối tượng** của CNNDetection (`airplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`).
- Mỗi danh mục có đúng **400 ảnh** (200 Real + 200 Fake), tổng cộng đúng $20 \times 400 = 8.000$ ảnh.

### 4. `test_benchmark_diffusion.tar` (1.22 GB - 20.000 ảnh)
- **Bản chất**: Bộ kiểm thử chuẩn 10 biến thể Diffusion Models của bài báo FatFormer CVPR 2024.
- **Cơ cấu**: Gồm đúng 10 mô hình, mỗi mô hình có đúng **2.000 ảnh** (1.000 Real + 1.000 Fake):
  1. `guided` (Guided Diffusion - tử huyệt của baseline gốc)
  2. `ldm_200` (Latent Diffusion 200 steps)
  3. `ldm_200_cfg` (Latent Diffusion 200 steps + CFG)
  4. `ldm_100` (Latent Diffusion 100 steps)
  5. `glide_50_27` (GLIDE 50 steps, 27 upsample)
  6. `glide_100_10` (GLIDE 100 steps, 10 upsample)
  7. `glide_100_27` (GLIDE 100 steps, 27 upsample)
  8. `dalle` (OpenAI DALL-E Mini / DALL-E)
  9. `pndm` (Pseudo Numerical Methods)
  10. `vqdiffusion` (Vector Quantized Diffusion)

### 5. `test_benchmark_gans.tar` (18.74 GB - 90.329 ảnh)
- **Bản chất**: Kho kiểm thử GANs đầy đủ trích xuất từ `CNN_synth_testset.zip` của Wang et al. (CVPR 2020).
- **Cơ cấu phân rã**:
  - **8 họ GANs chuẩn paper CVPR 2024** (được đánh giá trong Task 2.2 và Task 5.1): Tổng cộng **62.003 ảnh** (`progan`: 8.000, `stylegan`: 11.982, `stylegan2`: 15.976, `biggan`: 4.000, `cyclegan`: 2.642, `stargan`: 3.998, `gaugan`: 10.000, `deepfake`: 5.405).
  - **Các tập kiểm thử mở rộng bổ sung**: Chiếm **28.326 ảnh** thuộc các kiến trúc phụ trong kho tác giả (`crn`, `imle`, `san`, `seeingdark`, `whichfaceisreal`).

### 6. `test_degraded.tar` (19.55 GB - 661.974 ảnh)
- **Bản chất**: Bộ kiểm thử suy biến vật lý đồ sộ được sinh tự động bởi Thành viên A tại Task 2.1 qua script `tools/make_degraded.py`.
- **Cơ cấu**: Áp dụng 6 biến thể suy biến lên toàn bộ tập test sạch:
  - `jpeg_q30`: 110.329 ảnh
  - `jpeg_q50`: 110.329 ảnh
  - `jpeg_q70`: 110.329 ảnh
  - `blur_s1`: 110.329 ảnh
  - `blur_s2`: 110.329 ảnh
  - `down_up`: 110.329 ảnh
  - Tổng số ảnh: $110.329 \times 6 = 661.974$ ảnh.

---

## IV. TIÊU CHUẨN NGHIỆM THU KIỂM TOÁN (AUDIT VERIFICATION GATE)

| Tiêu chuẩn kiểm toán | Yêu cầu kỹ thuật | Thực tế đo đạc Colab | Kết luận thẩm định |
| :--- | :--- | :--- | :---: |
| **Tính toàn vẹn tệp nén** | Mọi tệp `.tar` đọc được 100% header, không lỗi CRC/EOF | 6/6 file đọc thông suốt 0 lỗi | ✅ **PASS 100%** |
| **Độ phủ dữ liệu** | Đáp ứng đầy đủ tập Train, Val, Test Clean, Test Degraded | 927.927 ảnh sẵn sàng | ✅ **PASS 100%** |
| **Cân bằng nhãn nhị phân** | Tỷ lệ Real/Fake trên các tập Train, Val, Test cân bằng | Tỷ lệ tổng thể: 50.2% Real / 49.8% Fake | ✅ **PASS TUYỆT ĐỐI** |
| **Tối ưu hạ tầng lưu trữ** | Không vượt quá hạn mức Drive 5TB, tốc độ I/O nhanh | Chiếm 55.98 GB (<1.2% hạn mức Drive 5TB) | ✅ **PASS XUẤT SẮC** |

---

## V. KẾT LUẬN & ĐÓNG BĂNG MỐC DỮ LIỆU

1. **Đóng băng số liệu chính thức**: Con số **927.927 ảnh** và **55.98 GB** được chính thức phê duyệt làm thông số thực nghiệm chuẩn của đồ án FatFormer-XLA.
2. **Kế thừa cho Báo cáo Luận văn**:
   - Chương 3 (Phương pháp đề xuất): Trích dẫn tập Train 4-class (144.024 ảnh) và Staging (3.600 ảnh).
   - Chương 4 (Thực nghiệm & Đánh giá): Trích dẫn bộ 18 tập Clean (82.003 ảnh được chọn lọc từ kho test) và bộ Degraded (661.974 ảnh với 6 biến thể).
