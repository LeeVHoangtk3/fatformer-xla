# TASK 2.3: BENCHMARK BASELINE GỐC TRÊN TẬP DEGRADED (MỐC SÀN SỤT GIẢM)

- **Mã nhiệm vụ**: `Task 2.3`
- **Người phụ trách**: **Thành viên C** *(Training & Evaluation Lead)*
- **Môi trường thực hiện**: ☁️ **Colab T4**
- **Thời hạn hoàn thành**: Tuần 2 (Ngày 13)
- **Độ ưu tiên**: 🔴 Khẩn cấp (Xác lập mốc sàn khoa học và động lực nghiên cứu)
- **Đầu vào (Input)**:
  - Checkpoint gốc `fatformer_4class_ckpt.pth`.
  - Bộ dữ liệu kiểm thử suy biến `test_degraded.tar` từ Thành viên A (chứa $Q=30, 50, 70$, Blur, Down-Up).
- **Đầu ra (Output)**:
  - Bảng số liệu **Baseline Degraded Benchmark** (ACC, AP trên từng mức suy biến).
  - Tệp kết quả `baseline_degraded_results.csv` lưu trên Google Drive 5TB.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Lượng hóa chính xác "tử huyệt" của FatFormer gốc: Đo đạc mức độ sụt giảm hiệu năng khi ảnh bị nén và làm mờ theo các kịch bản mạng xã hội.
2. Kỳ vọng thực nghiệm:
   - Trên tập nén nhẹ $Q=70$: Độ chính xác duy trì tương đối ổn định (~$90\%$).
   - Trên tập nén sâu $Q=30$: Độ chính xác sụt giảm nghiêm trọng (dưới $60\% \sim 65\%$).
   - Trên tập Down-Up và Blur: Sụt giảm từ $15\% \sim 25\%$.
3. Thiết lập **Mốc sàn khoa học (Lower Bound)**: Mục tiêu cải tiến của đồ án FatFormer-XLA là kéo mức sàn $Q=30$ tăng từ $+10\%$ đến $+15\%$.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Thực thi lệnh benchmark trên các biến thể:
```bash
python tools/fast_eval.py \
    --checkpoint /content/drive/MyDrive/FatFormer_Hub/pretrained/fatformer_4class_ckpt.pth \
    --clip_path ViT-L-14.pt \
    --test_root /content/dataset_local/test_degraded \
    --output_csv /content/drive/MyDrive/FatFormer_Hub/logs/baseline_degraded_results.csv
```

### Tổng hợp bảng ma trận sụt giảm:
So sánh trực tiếp:
$$\Delta = \text{ACC}_{\text{Clean}} - \text{ACC}_{\text{Degraded}}$$
Lập biểu đồ cột thể hiện mức sụt giảm nghiêm trọng khi chất lượng nén giảm từ $Q=70 \rightarrow Q=50 \rightarrow Q=30$.

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Bảng số liệu hoàn tất, chỉ rõ mức sụt giảm cụ thể trên từng tập nén và mờ.
- [ ] File `baseline_degraded_results.csv` được lưu trên Drive 5TB.
- [ ] Sẵn sàng làm căn cứ cho cuộc họp chốt Milestone 1 (Task 2.5).
