# BÁO CÁO NGHIỆM THU NHIỆM VỤ 2.2 (TASK 2.2)
## BENCHMARK BASELINE GỐC TRÊN TOÀN BỘ 18 TẬP CLEAN (CVPR 2024 PAPER REPRODUCTION)

> **Người thực hiện**: Thành viên C *(Training & Evaluation Lead)*  
> **Người nghiệm thu**: Cả nhóm *(Lead A, Lead B, Lead C)*  
> **Thời gian hoàn thành**: 2026-09-27  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH TRIỂN KHAI PIPELINE & SẴN SÀNG THỰC THI BENCHMARK COLAB`  
> **Môi trường & Tài nguyên**: Local Workstation / Google Colab GPU Tesla T4 / A100 (Google Drive 5TB)  

---

## I. MỤC TIÊU VÀ BỐI CẢNH KỸ THUẬT

1. **Tái lập tính hợp lệ khoa học của bài báo CVPR 2024**:
   - Xác nhận checkpoint gốc `fatformer_4class_ckpt.pth` khi đánh giá trên tập ảnh sạch (Clean) đạt được các chỉ số khớp chuẩn mực với công bố của tác giả:
     - Trung bình 8 kiến trúc GANs: $\mathbf{98.4\%}$ (ACC) / $\mathbf{99.5\%}$ (AP).
     - Trung bình 10 kiến trúc Diffusion: $\mathbf{95.0\%}$ (ACC) / $\mathbf{98.0\%}$ (AP).
     - Tập kiểm thử khó nhất (Guided Diffusion): $\mathbf{76.1\%}$ (ACC).
2. **Thiết lập mốc trần hiệu năng (*Upper Bound Performance*)**:
   - Kết quả đánh giá Clean Benchmark đóng vai trò là "mốc trần" tham chiếu cố định. Trong các tuần sau (khi bổ sung SRM 3-Kernels, Dynamic Gating và Curriculum Degradation), hiệu năng trên ảnh sạch không được phép sụt giảm quá $0.5\%$.
3. **Chuẩn bị dữ liệu bàn giao cho Báo cáo**:
   - Toàn bộ kết quả số liệu được tự động phân nhóm và ghi nhận vào tệp `baseline_clean_results.csv` trên Google Drive 5TB để bàn giao cho Thành viên B đưa vào Chương 1–2 của Báo cáo đồ án.

---

## II. KẾ THỪA THÔNG SỐ TỪ CÁC TASK TIỀN ĐỀ (RULE 4)

1. **Kế thừa từ Task 1.1** ([`docs/bao_cao/A/bao_cao_task_1.1_ha_tang_drive_io.md`](../A/bao_cao_task_1.1_ha_tang_drive_io.md) - Lead A):
   - Đường dẫn Drive 5TB: `/content/drive/MyDrive/Fatformer`.
   - Các tệp nén benchmark nguyên khối sẵn sàng trong `datasets/`:
     - `test_benchmark_gans.tar` (**18.74 GB**): 8 họ GANs (`progan`, `stylegan`, `stylegan2`, `biggan`, `cyclegan`, `stargan`, `gaugan`, `deepfake`).
     - `test_benchmark_diffusion.tar` (**1.22 GB**): 10 họ Diffusion (`guided`, `ldm_200`, `ldm_200_cfg`, `ldm_100`, `glide_50_27`, `glide_100_10`, `glide_100_27`, `dalle`, `pndm`, `vqdiffusion`).
   - Đường dẫn giải nén trên SSD NVMe máy ảo Colab: `/content/dataset_local/test_clean` (tốc độ đọc ~1.2 GB/s, loại bỏ nghẽn I/O).
2. **Kế thừa từ Task 1.2** ([`docs/bao_cao/B/bao_cao_task_1.2_fatformer_model_and_weights_loader.md`](../B/bao_cao_task_1.2_fatformer_model_and_weights_loader.md) - Lead B):
   - Checkpoint gốc nạp với cờ `strict=True`, khớp chính xác **1.116/1.116 tensors** (~933.16M tham số) của tác giả CVPR 2024.
3. **Kế thừa từ Task 1.4** ([`docs/bao_cao/C/bao_cao_task_1.4_fast_eval_pipeline.md`](bao_cao_task_1.4_fast_eval_pipeline.md) - Lead C):
   - Đầy đủ 5 chỉ số đo đạc chuẩn hóa: ACC, AP, ROC-AUC, Real ACC, Fake ACC.

---

## III. KẾT QUẢ THỰC TẾ TRIỂN KHAI SCRIPT `tools/full_eval.py`

### 1. Kiến trúc Script Phân tầng Chuyên biệt
Script [`tools/full_eval.py`](../../../tools/full_eval.py) được xây dựng với các tính năng:
- **Đánh giá 100% dữ liệu**: Thiết lập `max_samples_per_subset=None`, duyệt qua toàn bộ số lượng ảnh của từng tập test.
- **Phân nhóm tự động 2 tầng (Hierarchical Grouping)**:
  - Tự động tách biệt và tính toán dòng `MEAN GANS (8)` cho 8 kiến trúc GANs.
  - Tự động tách biệt và tính toán dòng `MEAN DIFF (10)` cho 10 kiến trúc Diffusion.
  - Tự động tính toán dòng tổng hợp `MEAN OVERALL` cho toàn bộ 18 tập test.
- **Cổng Thẩm Định Tự Động (DoD Verification Gate)**:
  - Tự động so sánh kết quả thực tế với mốc paper CVPR 2024 (98.4% GANs, 95.0% Diffusion, 76.1% Guided Diff).
  - Tự động tính toán độ lệch sai số $\Delta = |\text{Result} - \text{Paper}|$ và cảnh báo nếu vượt quá $\pm 0.5\%$.
- **Xuất Dữ liệu Đa Định Dạng**: Hỗ trợ xuất đồng thời cả file CSV (`--output_csv`) và bảng Markdown (`--output_markdown`).

### 2. Kết quả Kiểm thử Smoke Test (Pass 100%)
```bash
python tools/full_eval.py --test_dummy --device cpu --output_csv scratch/dummy_full_eval.csv --output_markdown scratch/dummy_full_eval.md
```
```
======================================================================================
FATFORMER-XLA: PIPELINE BENCHMARK BASELINE TOÀN BỘ 18 TẬP CLEAN (TASK 2.2)
======================================================================================
• Thiết bị tính toán:       CPU (CPU)
• Chế độ đánh giá:          FULL-BENCHMARK (100% số ảnh trong từng tập test)
• Batch size:               32
• Số luồng nạp (Workers):   4
• Chế độ Baseline gốc:      SRM=False, Gating=False, Strict=True

======================================================================================
BẢNG KẾT QUẢ BENCHMARK BASELINE TOÀN DIỆN (CVPR 2024 PAPER REPRODUCTION)
======================================================================================
Index  Subset           ACC (%)    AP (%)     AUC (%)    Real ACC (%)   Fake ACC (%)  
--------------------------------------------------------------------------------------
--- [NHÓM 1: 8 KIẾN TRÚC GANS] ---
(0  )  progan           50.00      50.00      50.00      75.00          25.00         
(1  )  stylegan         50.00      50.00      50.00      75.00          25.00         
--------------------------------------------------------------------------------------
MEAN GANS (8)           50.00      50.00      50.00      75.00          25.00         
--------------------------------------------------------------------------------------
--- [NHÓM 2: 10 KIẾN TRÚC DIFFUSION] ---
(8  )  guided           50.00      50.00      50.00      75.00          25.00         
(12 )  glide_50_27      50.00      50.00      50.00      75.00          25.00         
--------------------------------------------------------------------------------------
MEAN DIFF (10)          50.00      50.00      50.00      75.00          25.00         
======================================================================================
MEAN OVERALL            50.00      50.00      50.00      75.00          25.00         
======================================================================================
[✓] Đã lưu kết quả CSV ra: scratch/dummy_full_eval.csv
[✓] Đã lưu báo cáo Markdown ra: scratch/dummy_full_eval.md
[✓] SMOKE TEST FULL BENCHMARK HOÀN TẤT THÀNH CÔNG!
```

---

## IV. HƯỚNG DẪN THỰC THI FULL BENCHMARK TRÊN GOOGLE COLAB (GPU T4 / A100)

Để thực thi đánh giá toàn diện trên Google Colab Pro và lưu file kết quả nghiệm thu vào Google Drive 5TB:

```bash
# ================================================================================
# BƯỚC 1: GIẢI NÉN TOÀN BỘ 18 TẬP TEST CLEAN SANG Ổ SSD NVME COLAB (~2-3 PHÚT)
# ================================================================================
mkdir -p /content/dataset_local/test_clean

echo "[*] Đang giải nén 10 tập Diffusion (1.22 GB)..."
tar -xf /content/drive/MyDrive/Fatformer/datasets/test_benchmark_diffusion.tar -C /content/dataset_local/test_clean/

echo "[*] Đang giải nén 8 tập GANs (18.74 GB)..."
tar -xf /content/drive/MyDrive/Fatformer/datasets/test_benchmark_gans.tar -C /content/dataset_local/test_clean/

# ================================================================================
# BƯỚC 2: CHẠY BENCHMARK TOÀN DIỆN VÀ TỰ ĐỘNG XUẤT CSV VÀO GOOGLE DRIVE
# ================================================================================
python tools/full_eval.py \
    --checkpoint /content/drive/MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth \
    --clip_path /content/drive/MyDrive/Fatformer/pretrained/ViT-L-14.pt \
    --test_root /content/dataset_local/test_clean \
    --subsets all \
    --batch_size 32 \
    --num_workers 4 \
    --device cuda \
    --output_csv /content/drive/MyDrive/Fatformer/log/baseline_clean_results.csv \
    --output_markdown /content/drive/MyDrive/Fatformer/log/baseline_clean_results.md
```

---

## V. TIÊU CHUẨN NGHIỆM THU (DoD)

| Tiêu chuẩn nghiệm thu (DoD) | Yêu cầu thiết kế | Trạng thái đạt được | Đánh giá |
| :--- | :---: | :---: | :---: |
| **Script `tools/full_eval.py` độc lập** | Chạy qua CLI, nạp 18 tập test | Hoàn thành, đầy đủ tham số CLI | ✅ **ĐẠT (PASS 100%)** |
| **Phân nhóm GANs vs Diffusion** | Tính riêng 2 dòng MEAN nhóm | Cung cấp `MEAN_GANS` và `MEAN_DIFFUSION` | ✅ **ĐẠT (PASS 100%)** |
| **Thẩm định tự động sai số paper** | Sai số $\le \pm 0.5\%$ so với CVPR 2024 | Tích hợp cổng kiểm tra tự động | ✅ **SẴN SÀNG** |
| **Lưu kết quả trên Drive 5TB** | File CSV và Markdown an toàn | Tự động ghi vào `/content/drive/MyDrive/...` | ✅ **SẴN SÀNG** |
