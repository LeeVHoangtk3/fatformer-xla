# BÁO CÁO NGHIỆM THU NHIỆM VỤ 3.1 (TASK 3.1)
## CHUẨN HÓA DỮ LIỆU HUẤN LUYỆN STAGING 90/10 (DIFFUSION_STAGING.TAR)

> **Dự án**: FatFormer-XLA *(Generalizable Synthetic Image Detection under Online Degradations)*  
> **Người thực hiện**: **Thành viên A** *(Data & Infrastructure Lead)*  
> **Người nghiệm thu**: Cả nhóm *(Lead A, Lead B, Lead C)*  
> **Ngày hoàn thành**: 2026-10-04  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC (DONE)`  
> **Tài nguyên lưu trữ**: Google Drive 5TB (`/content/drive/MyDrive/Fatformer/datasets/diffusion_staging.tar`)  

---

## I. MỤC TIÊU VÀ TẦM QUAN TRỌNG KỸ THUẬT

1. **Phá vỡ thiên lệch GANs (GANs Overfitting Bias)**:
   - Mô hình FatFormer gốc huấn luyện thuần trên ProGAN 4-class (`car, cat, chair, horse`) có xu hướng phụ thuộc vào các dấu vết nội suy đặc thù của GANs.
   - Khi kiểm thử trên các mô hình khuếch tán (Diffusion Models) hiện đại (Stable Diffusion, Midjourney, DALL-E, Guided Diffusion), độ chính xác nhận diện ảnh giả bị sụt giảm.
2. **Chuẩn hóa tập chất xúc tác 10% (Catalyst Staging Set)**:
   - Trích xuất cân bằng đúng **3.600 ảnh GenImage**:
     - **Stable Diffusion v1.5**: 1.200 ảnh Real + 1.200 ảnh Fake = 2.400 ảnh.
     - **Midjourney v5**: 600 ảnh Real + 600 ảnh Fake = 1.200 ảnh.
3. **Đóng gói hỗn hợp tỷ lệ chuẩn 90/10**:
   - Hợp nhất 3.600 ảnh chất xúc tác với ~32.400 ảnh ProGAN 4-class, tạo thành gói dữ liệu Staging duy nhất tổng cộng **~36.000 ảnh** được nén nguyên khối thành `diffusion_staging.tar` trên Google Drive 5TB.

---

## II. ĐẦU RA NGHIỆM THU THỰC TẾ

| Tệp Dữ Liệu Nén | Dung Lượng | Tổng Số Ảnh | Cơ Cấu Nhãn | Vị Trí Lưu Trữ Trên Drive 5TB | Trạng Thái Toàn Vẹn |
| :--- | :---: | :---: | :---: | :--- | :---: |
| `diffusion_staging.tar` | **1.80 GB** | **36.000 ảnh** | 18.000 Real (0)<br>18.000 Fake (1) | `/content/drive/MyDrive/Fatformer/datasets/diffusion_staging.tar` | ✅ Đã kiểm tra SHA256 & tar integrity |

*Ghi chú*: Tệp đã được nạp sớm và xác nhận trong báo cáo hạ tầng [bao_cao_task_1.1_ha_tang_drive_io.md](../../bao_cao/A/bao_cao_task_1.1_ha_tang_drive_io.md#L46). Script giải nén nội bộ tốc độ cao vào SSD NVMe Colab đã được tích hợp trong notebook [`notebooks/00_setup_drive_and_data_ingestion.ipynb`](../../../notebooks/00_setup_drive_and_data_ingestion.ipynb) (Phần 6.2).

---

## III. BẢO ĐẢM NGUYÊN TẮC I/O CHỐNG NGHẼN SSD NVME

- Tệp `diffusion_staging.tar` tuân thủ nghiêm ngặt quy tắc nén nguyên khối `.tar`:
  1. Copy từ Drive sang SSD Colab trong ~15 giây (`shutil.copy` đạt ~120 MB/s trên GCP mạng nội bộ).
  2. Giải nén sang `/content/dataset_local/diffusion_staging/` mất < 20 giây.
  3. DataLoader đọc ảnh từ SSD NVMe đạt tốc độ **> 1.200 ảnh/giây**, giải phóng 100% GPU utilization trên A100/T4.

---

## IV. ĐỐI SOÁT TIÊU CHUẨN HOÀN THÀNH (DoD CHECKLIST)

- [x] Tệp `diffusion_staging.tar` có sẵn trên Drive 5TB với dung lượng 1.80 GB.
- [x] Tỷ lệ cân bằng nhãn nhị phân: 50% nhãn `0_real` và 50% nhãn `1_fake`.
- [x] Cấu trúc phân chia đúng tỷ lệ chuẩn 90% ProGAN + 10% GenImage catalyst.
- [x] Sẵn sàng bàn giao nguyên liệu cho Thành viên C thực thi Task 3.6 và Task 4.1.
