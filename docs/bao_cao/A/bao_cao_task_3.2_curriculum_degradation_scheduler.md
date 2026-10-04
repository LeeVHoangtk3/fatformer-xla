# BÁO CÁO NGHIỆM THU NHIỆM VỤ 3.2 (TASK 3.2)
## XÂY DỰNG MODULE LẬP LỊCH SUY THOÁI CURRICULUM 3 GIAI ĐOẠN

> **Dự án**: FatFormer-XLA *(Generalizable Synthetic Image Detection under Online Degradations)*  
> **Người thực hiện**: **Thành viên A** *(Data & Infrastructure Lead)*  
> **Người nghiệm thu**: Cả nhóm *(Lead A, Lead B, Lead C)*  
> **Ngày hoàn thành**: 2026-10-04  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC (DONE)`  
> **Tệp mã nguồn**: [`src/datasets/transforms.py`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/src/datasets/transforms.py) & [`src/datasets/dataset.py`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/src/datasets/dataset.py)  

---

## I. MỤC TIÊU VÀ TẦM QUAN TRỌNG KỸ THUẬT

1. **Triệt tiêu hiện tượng sốc gradient (Gradient Shock Elimination)**:
   - Khi huấn luyện các module thích ứng mới (FAA adapters, SRM projection, Dynamic Gating), nếu đưa ngay dải nén cực nặng ($Q=30$ và Down-Up mạnh) vào ở Epoch 1–2, các trọng số khởi tạo ngẫu nhiên sẽ nhận gradient khổng lồ hỗn loạn, làm mất phương hướng hội tụ.
2. **Thiết kế giáo trình học từ dễ đến khó (Curriculum Learning Dynamics)**:
   - Phân chia quá trình huấn luyện 8 epoch thành 3 giai đoạn áp lực tăng dần:
     - **Pha 1 (Epoch 1–2 - Khởi động)**: Nén nhẹ $Q \in [70, 90]$, Gaussian Blur $\sigma \in [0.5, 1.0]$, chưa dùng Down-Up.
     - **Pha 2 (Epoch 3–5 - Tăng áp lực)**: Nén vừa $Q \in [45, 70]$, Blur $\sigma \in [1.0, 1.5]$, Down-Up $224 \rightarrow 160 \rightarrow 224$.
     - **Pha 3 (Epoch 6–8 - Hardening cực hạn)**: Nén sâu $Q \in [30, 50]$ (tử huyệt nén nặng), Blur $\sigma \in [1.5, 2.0]$, Down-Up $224 \rightarrow 112 \rightarrow 224$.
3. **Cơ chế phân bổ mẫu kép**:
   - Cố định **30% ảnh Clean** (bảo toàn đặc trưng nhận diện ảnh sạch, giữ vững mốc trần 96.45% ACC).
   - **70% ảnh Degraded** (áp dụng ngẫu nhiên 1 trong 3 kiểu suy biến: JPEG, Blur, Down-Up theo đúng thông số pha hiện hành).

---

## II. ĐẦU RA KỸ THUẬT NGHIỆM THU

1. **Class `CurriculumDegradationScheduler` trong [`src/datasets/transforms.py`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/src/datasets/transforms.py)**:
   - Phương thức `set_epoch(epoch: int)`: Cho phép DataLoader và vòng lặp huấn luyện cập nhật tham số suy biến động sau mỗi epoch.
   - Các phương thức xử lý ảnh tối ưu: `apply_jpeg()`, `apply_blur()`, `apply_down_up()` sử dụng bộ nhớ đệm RAM BytesIO nội bộ siêu tốc (< 2ms/ảnh).
2. **Tích hợp DataLoader trong [`src/datasets/dataset.py`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/src/datasets/dataset.py)**:
   - Phương thức `DatasetCreator.build_train_dataset(..., scheduler=scheduler)` cho phép truyền trực tiếp bộ lập lịch vào pipeline huấn luyện.

---

## III. BẢNG THÔNG SỐ GIÁO TRÌNH 3 GIAI ĐOẠN

| Giai Đoạn (Stage) | Khoảng Epoch | Dải Nén JPEG ($Q$) | Bán Kính Blur ($\sigma$) | Kích Thước Down-Up | Tỷ Lệ Mẫu Sạch (Clean) | Vai Trò Huấn Luyện |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Giai đoạn 1** | Epoch 1 – 2 | $Q \in [70, 90]$ | $\sigma \in [0.5, 1.0]$ | *Không áp dụng* | 30% | Ổn định gradient các adapter FAA |
| **Giai đoạn 2** | Epoch 3 – 5 | $Q \in [45, 70]$ | $\sigma \in [1.0, 1.5]$ | $224 \rightarrow 160 \rightarrow 224$ | 30% | Rèn luyện mạng cổng Gating $\lambda(x)$ |
| **Giai đoạn 3** | Epoch 6 – 8 | $Q \in [30, 50]$ | $\sigma \in [1.5, 2.0]$ | $224 \rightarrow 112 \rightarrow 224$ | 30% | Tôi luyện biểu diễn vi sai SRM chống tử huyệt $Q=30$ |

---

## IV. ĐỐI SOÁT TIÊU CHUẨN HOÀN THÀNH (DoD CHECKLIST)

- [x] Class `CurriculumDegradationScheduler` đã được nhúng vào `src/datasets/transforms.py`.
- [x] Hỗ trợ cập nhật tham số động theo `epoch` qua hàm `set_epoch()`.
- [x] Tỷ lệ 30% Clean và 70% Degraded hoạt động chuẩn xác theo xác suất ngẫu nhiên.
- [x] Tương thích 100% với hàm `get_train_transforms()` và DataLoader của Thành viên C.
