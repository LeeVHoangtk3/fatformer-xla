# TASK 5.1: THỰC THI FULL-BENCHMARK TRÊN 18 TẬP TEST HAI CHIỀU

- **Mã nhiệm vụ**: `Task 5.1`
- **Người phụ trách**: **Thành viên C** *(Training & Evaluation Lead)*
- **Môi trường thực hiện**: ☁️ **Colab T4** (~2–3 Compute Units)
- **Thời hạn hoàn thành**: Tuần 5 (Ngày 31)
- **Độ ưu tiên**: 🔴 Cực kỳ khẩn cấp (Cung cấp toàn bộ số liệu cho Báo cáo và Slide)
- **Đầu vào (Input)**:
  - Checkpoint hoàn thiện cuối cùng `fatformer_srm_robust_final.pth`.
  - 18 tập test paper chuẩn (8 GANs + 10 Diffusion) ở cả 2 trạng thái: Sạch (Clean) và Suy biến ($Q=30, 50, 70$, Blur, Down-Up).
- **Đầu ra (Output)**:
  - **Ma trận số liệu Benchmark toàn diện** (ACC, AP, AUC trên từng tập dữ liệu).
  - Tệp kết quả chi tiết `final_full_benchmark_matrix.csv` lưu trên Google Drive 5TB `FatFormer_Hub/logs/`.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Đánh giá kiểm thử toàn diện mô hình đề xuất FatFormer-XLA trên toàn bộ không gian dữ liệu:
   - **Chiều thứ nhất (Cross-Generator Generalization)**: Khả năng tổng quát hóa trên 8 kiến trúc GANs và 10 kiến trúc Diffusion chưa từng gặp trong huấn luyện.
   - **Chiều thứ hai (Degradation Robustness)**: Khả năng chống chịu trước các biến dạng nén lượng tử JPEG ($Q=30, 50, 70$), làm mờ và downsampling.
2. Kỳ vọng số liệu:
   - Trên tập nén sâu $Q=30$: Đạt mức tăng trưởng **$+10\%$ đến $+15\%$** Accuracy so với baseline gốc.
   - Trên tập ảnh sạch (Clean): Bảo toàn hiệu năng vượt trội, không sụt giảm quá $0.5\%$.
   - Trên ảnh Diffusion thế hệ mới: Đạt độ chính xác trung bình trên $90\%$.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Thực thi lệnh chạy Full-Benchmark:
```bash
python tools/full_eval.py \
    --checkpoint /content/drive/MyDrive/FatFormer_Hub/checkpoints/fatformer_srm_robust_final.pth \
    --clip_path ViT-L-14.pt \
    --test_root /content/dataset_local/test_clean \
    --degraded_root /content/dataset_local/test_degraded \
    --output_csv /content/drive/MyDrive/FatFormer_Hub/logs/final_full_benchmark_matrix.csv
```

### Xuất Bảng Ma Trận Tổng Hợp (Markdown Table):
Tạo bảng so sánh trực tiếp song song giữa Mô hình gốc (Baseline) và Mô hình đề xuất (FatFormer-XLA) cho từng miền sinh ảnh.

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Hoàn thành đo đạc đầy đủ trên toàn bộ 18 tập test mà không sót tập nào.
- [ ] Tệp `final_full_benchmark_matrix.csv` được lưu an toàn trên Drive 5TB.
- [ ] Bàn giao bảng số liệu cho toàn nhóm để điền vào Chương 5 của Báo cáo.
