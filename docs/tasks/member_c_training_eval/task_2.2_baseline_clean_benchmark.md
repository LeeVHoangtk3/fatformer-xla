# TASK 2.2: BENCHMARK BASELINE GỐC TRÊN TOÀN BỘ TẬP CLEAN

- **Mã nhiệm vụ**: `Task 2.2`
- **Người phụ trách**: **Thành viên C** *(Training & Evaluation Lead)*
- **Môi trường thực hiện**: ☁️ **Colab T4 / A100**
- **Thời hạn hoàn thành**: Tuần 2 (Ngày 11)
- **Độ ưu tiên**: 🔴 Khẩn cấp (Tái lập kết quả của bài báo gốc CVPR 2024)
- **Đầu vào (Input)**:
  - Checkpoint gốc `fatformer_4class_ckpt.pth`.
  - 18 tập test Clean paper chuẩn đã giải nén trên SSD Colab:
    - 8 kiến trúc GANs: ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, DeepFake.
    - 10 kiến trúc Diffusion: LDM, GLIDE, DALL-E, Guided Diffusion...
- **Đầu ra (Output)**:
  - Bảng số liệu **Baseline Clean Benchmark** hoàn chỉnh (ACC, AP trên từng tập).
  - Tệp kết quả `baseline_clean_results.csv` lưu trên Google Drive 5TB.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Tái lập tính hợp lệ của bài báo CVPR 2024: Xác nhận checkpoint gốc khi đánh giá trên tập ảnh sạch (Clean) đạt được các chỉ số khớp với công bố của tác giả:
   - Trung bình trên 8 kiến trúc GANs: **$98.4\%$**.
   - Trung bình trên các kiến trúc Diffusion: **$95.0\%$**.
   - Khó nhất (Guided Diffusion): **$76.1\%$**.
2. Thiết lập mốc trần (*Upper Bound Performance*): Đảm bảo rằng trong quá trình cải tiến mô hình ở các tuần sau, hiệu năng trên ảnh sạch không bị sụt giảm quá $0.5\%$.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Thực thi lệnh benchmark trên Colab:
```bash
python tools/full_eval.py \
    --checkpoint /content/drive/MyDrive/FatFormer_Hub/pretrained/fatformer_4class_ckpt.pth \
    --clip_path ViT-L-14.pt \
    --test_root /content/dataset_local/test_clean \
    --output_csv /content/drive/MyDrive/FatFormer_Hub/logs/baseline_clean_results.csv
```

### Tổng hợp và Phân nhóm Kết quả:
- Nhóm GANs (8 datasets): Tính trung bình cộng ACC và AP.
- Nhóm Diffusion (10 datasets): Tính trung bình cộng ACC và AP.
- Xuất bảng markdown lưu vào tài liệu nhóm.

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Bảng số liệu tái lập sai số không quá $\pm 0.5\%$ so với bảng số liệu công bố trong paper gốc CVPR 2024.
- [ ] File `baseline_clean_results.csv` được lưu an toàn trên Drive 5TB.
- [ ] Bàn giao số liệu cho Thành viên B đưa vào Chương 1–2 của Báo cáo.
