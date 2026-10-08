# KHO LƯU TRỮ NHẬT KÝ & KẾT QUẢ THỰC NGHIỆM (EXPERIMENTAL LOGS & BENCHMARK REPOSITORY)
> **Dự án**: Nâng cao độ bền vững pháp y cho FatFormer trước biến dạng nén mạng xã hội (FatFormer-XLA)  
> **Cấu trúc phân loại**: Chuẩn hóa theo 4 Cột Mốc Chiến Lược & Mã Task Nghiệm Thu  

---

## 📂 SƠ ĐỒ PHÂN CẤP THƯ MỤC

```
DATASET/logs/
├── m1_baseline_freeze/                     # 🏁 MỐC 1: ĐÓNG BĂNG MỐC SÀN BASELINE (Tuần 1 & Tuần 2)
│   ├── task_1.1_download_datasets.log     # Nhật ký tải và giải nén 36.45 GB dữ liệu ban đầu
│   ├── task_1.4_fast_eval_clean_baseline.csv # Thử nghiệm pipeline Fast-Eval (<8 phút)
│   ├── task_2.2_baseline_clean_results.csv   # Ma trận đo Baseline Clean trên 18 tập test
│   ├── task_2.2_baseline_clean_results.md    # Báo cáo mốc trần phòng lab (ACC: 96.45%)
│   ├── task_2.3_baseline_degraded_results.csv # Ma trận sụt giảm Baseline trên 6 biến thể suy biến
│   ├── task_2.3_baseline_degraded_results.md  # Báo cáo chi tiết mốc sàn sụp đổ (Q=30: 52.89%)
│   └── task_2.3_baseline_degraded_chart.png   # Biểu đồ trực quan hóa tử huyệt nén Baseline
│
├── m3_training_hardening/                  # ❄️ MỐC 3: HUẤN LUYỆN NÂNG CAO & ĐÓNG BĂNG CHECKPOINT (Tuần 4)
│   ├── task_4.2_xla_q30_eval_results.csv  # Kết quả đo phục hồi của FatFormer-XLA tại JPEG Q=30
│   ├── task_4.2_xla_q30_eval_report.md    # Báo cáo phân tích đối chứng cú nhảy vọt Fake ACC (+5.11x)
│   ├── task_4.2_xla_q30_comparison.png    # Biểu đồ so sánh 18 subsets tại Q=30
│   └── task_4.2_xla_q30_grouped_comparison.png # Biểu đồ 3 chỉ số và khắc chế Diffusion hàng đầu
│
├── m4_full_benchmark/                      # 🎓 MỐC 4: TỔNG DUYỆT NGHIỆM THU ĐỒ ÁN (Tuần 5)
│   └── .gitkeep                           # (Sẵn sàng lưu tệp task_5.1_final_full_benchmark_matrix.csv)
│
└── inspections/                            # 🔍 CÔNG CỤ SOI ẢNH CỤC BỘ (INSPECT IMAGE TOOLS)
    └── inspection_history.csv             # Nhật ký ghi nhận từng phiên kiểm thử ảnh đơn lẻ
```

---

## 📋 CHI TIẾT CÁC TỆP THEO TỪNG GIAI ĐOẠN

### 1. Thư mục `m1_baseline_freeze/` (Mốc 1 - Tuần 1 & Tuần 2)
*Mục đích: Thiết lập nền móng khoa học không thể tranh cãi về mốc trần (Clean) và mốc sàn tử huyệt nén (Degraded) của mô hình gốc CVPR 2024.*

| Tên Tệp | Mã Task | Ngày Tạo | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| `task_1.1_download_datasets.log` | Task 1.1 | 2026-09-24 | Ghi nhận tiến trình tải dữ liệu từ Google Drive về máy ảo Colab. |
| `task_1.4_fast_eval_clean_baseline.csv` | Task 1.4 | 2026-09-26 | Kiểm chứng pipeline đánh giá nhanh mẫu ngẫu nhiên đại diện. |
| `task_2.2_baseline_clean_results.csv` | Task 2.2 | 2026-09-28 | **Mốc trần phòng lab**: 96.45% Overall ACC (GANs: 98.39%, Diff: 94.91%). |
| `task_2.2_baseline_clean_results.md` | Task 2.2 | 2026-09-28 | Báo cáo chi tiết mốc trần Clean phục vụ soạn thảo Chương 1 & 2. |
| `task_2.3_baseline_degraded_results.csv` | Task 2.3 | 2026-09-30 | **Mốc sàn sụp đổ**: Đo trên 6 biến thể ($Q=70, 50, 30, \text{Blur}_1, \text{Blur}_2, \text{Down-Up}$). |
| `task_2.3_baseline_degraded_results.md` | Task 2.3 | 2026-09-30 | Vạch trần sự sụp đổ tại $Q=30$: Overall ACC còn 52.89%, Fake ACC còn 5.78%. |
| `task_2.3_baseline_degraded_chart.png` | Task 2.3 | 2026-09-30 | Hình ảnh biểu đồ trực quan đưa vào Slide thuyết trình Mốc 1. |

---

### 2. Thư mục `m3_training_hardening/` (Mốc 3 - Tuần 4)
*Mục đích: Ghi nhận kết quả thực nghiệm sau khi huấn luyện mô hình hoàn thiện FatFormer-XLA qua 8 epoch với SRM 3-Kernels, Dynamic Gating và Curriculum Hardening.*

| Tên Tệp | Mã Task | Ngày Tạo | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| `task_4.2_xla_q30_eval_results.csv` | Task 4.2 | 2026-10-06 | Kết quả thực nghiệm tại $Q=30$ trên 18 tập test (9,000 ảnh). |
| `task_4.2_xla_q30_eval_report.md` | Task 4.2 | 2026-10-06 | Chứng minh giải cứu mô hình: Fake ACC tăng từ 5.78% lên 29.53% (gấp 5.11 lần). |
| `task_4.2_xla_q30_comparison.png` | Task 4.2 | 2026-10-06 | Biểu đồ cột so sánh chi tiết từng subset giữa Baseline và XLA. |
| `task_4.2_xla_q30_grouped_comparison.png`| Task 4.2 | 2026-10-06 | Biểu đồ đối chứng 3 chỉ số và độ bứt phá trên các mô hình Diffusion hàng đầu. |

---

### 3. Thư mục `m4_full_benchmark/` (Mốc 4 - Tuần 5)
*Mục đích: Lưu trữ toàn bộ kết quả đo đạc Full-Benchmark hai chiều (18 tập Clean + 6 biến thể Degraded) của Task 5.1.*

| Tên Tệp | Mã Task | Ngày Tạo | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| `final_full_benchmark_matrix.csv` | Task 5.1 | 2026-10-08 | **Ma trận Master 2 chiều**: 137 dòng, 126 lượt đo đạc phân tầng (Clean + 6 Degraded). |
| `task_5.1_clean_results.csv` | Task 5.1 | 2026-10-08 | Kết quả chi tiết trên 18 tập test sạch phòng lab (Overall ACC: 50.31%). |
| `task_5.1_clean_report.md` | Task 5.1 | 2026-10-08 | Báo cáo Markdown chi tiết kết quả Clean Benchmark. |
| `task_5.1_degraded_results.csv` | Task 5.1 | 2026-10-08 | Kết quả đo đạc trên 6 biến thể suy biến ($Q=30, 50, 70$, Blur, Down-Up). |
| `task_5.1_degraded_report.md` | Task 5.1 | 2026-10-08 | Báo cáo Markdown chi tiết ma trận sụt giảm $\Delta$ so với Clean. |
| `task_5.1_master_benchmark_chart.png` | Task 5.1 | 2026-10-08 | Biểu đồ đối chứng trực quan 7 nhóm cột phục vụ Slide và Báo cáo. |

---

### 4. Thư mục `inspections/` (Công cụ soi ảnh cục bộ)
| Tên Tệp | Mô Tả |
| :--- | :--- |
| `inspection_history.csv` | Tự động ghi lại từng phiên soi ảnh (`inspect_image.py`), gồm tên ảnh, kích thước lưới, checkpoint sử dụng, số lượng patch fake và nhãn dự đoán cuối cùng. |
