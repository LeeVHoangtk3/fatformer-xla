# Task 7.15 — sửa sau hậu kiểm ngày 09/11/2026

## Đường chạy Colab mới

Dùng [phase7_corrected_rerun_colab.ipynb](../../../../notebooks/task/phase7_corrected/phase7_corrected_rerun_colab.ipynb) và [hướng dẫn](../../../../notebooks/task/phase7_corrected/README.md). Gói `colab/phase7_corrected_colab_bundle.zip` xuất code đang sửa ở local, không phụ thuộc GitHub HEAD 9d798f5. Hai notebook cũ giữ nguyên làm hồ sơ.

Notebook v2 kiểm tra hash, GPU CUDA có đủ VRAM, 400/1.600 ảnh và 82.003 ảnh in-dist trước phần nặng. Probe chọn batch vừa bộ nhớ, rồi đo lại đúng 400 ảnh calibration bằng FP16/cùng batch với benchmark, không train lại model. Không bắt buộc T4. Đóng gói/verification/benchmark/latency chạy trong một thư mục run mới, không ghi đè kết quả cũ. Chưa có best-reference theo điều kiện thì G7.3 vẫn pending.

## Hồ sơ khôi phục local

Bản gốc ở `../task_7.15_original_68cd0a8/`; hai notebook được giữ nguyên. Pool logits gốc 2.000 ảnh chứa cả 400 calibration và 1.600 evaluation. Không dùng pool đó trực tiếp để fit calibration triển khai.

Đã khôi phục đúng 400 ảnh theo frozen split, overlap ID với evaluation bằng 0, rồi fit T/ngưỡng và thay metadata package. Đây là post-hoc recovery; không coi evaluation đã xem là phép kiểm thử mù mới. Độc lập ở đây theo ID subset/class/filename, chưa chứng minh không trùng nội dung ảnh dưới tên khác.

Tái lập tại thư mục gốc repository (PowerShell hoặc Colab, dùng các lệnh một dòng):

```sh
python -m tools.recover_calibration_split --source DATASET/logs/phase7/logits_anchored_v2_clean_calib.npz --calibration_split DATASET/logs/phase7/wise_ft_sweep/logits_baseline_clean_calib_split.npz --evaluation_split DATASET/logs/phase7/wise_ft_sweep/logits_baseline_clean_eval_split.npz --output DATASET/logs/phase7/task_7.15_recovery/logits_anchored_v2_calibration_400.npz --audit_out DATASET/logs/phase7/task_7.15_recovery/partition_audit.json
python -m tools.calibrate_and_select --logits_npz DATASET/logs/phase7/task_7.15_recovery/logits_anchored_v2_calibration_400.npz --calibration_split DATASET/logs/phase7/wise_ft_sweep/logits_baseline_clean_calib_split.npz --evaluation_split DATASET/logs/phase7/wise_ft_sweep/logits_baseline_clean_eval_split.npz --model_name anchored_v2 --output_dir DATASET/logs/phase7/task_7.15_recovery/calibration
python -m tools.package_unified_fatformer --checkpoint DATASET/logs/phase7/task_7.15_original_68cd0a8/unified_fatformer_v2.pth --repackage --calibration DATASET/logs/phase7/task_7.15_recovery/calibration/anchored_v2.json --output DATASET/checkpoints/phase7/unified_fatformer_v2.pth --metadata_out DATASET/checkpoints/phase7/metadata.json
python -m tools.verify_unified_package --checkpoint DATASET/checkpoints/phase7/unified_fatformer_v2.pth --reference DATASET/logs/phase7/task_7.15_original_68cd0a8/unified_fatformer_v2.pth --output DATASET/logs/phase7/task_7.15_recovery/package_verification.json --synthetic_smoke
```

Lệnh đầu ra ghi rõ hash nguồn và partition; không tự sửa hash checkpoint huấn luyện. Repackage kế thừa hash huấn luyện từ package/NPZ cũ và ghi hash container cũ riêng. Tool không tuyên bố đã băm lại file huấn luyện trên Drive.

Nếu trích xuất lại ảnh trên Colab, dùng `tools.extract_calibration_logits` với cả `--split_file` calibration và `--evaluation_split` evaluation. Thiếu hoặc không khớp ảnh sẽ báo lỗi; không có fallback quét toàn thư mục.

Kiểm chứng local đã có: đối chiếu mọi tensor, nạp standalone khi CLIP path không tồn tại, một đầu vào tổng hợp CPU với xác suất tham chiếu. Kết quả ở `package_verification.json`; một đầu vào tổng hợp không thay thế 20 ảnh thật hoặc đánh giá CFE.

Phần chạy tiếp trên Colab:

```sh
python -m tools.verify_unified_package --checkpoint DATASET/checkpoints/phase7/unified_fatformer_v2.pth --reference DATASET/logs/phase7/task_7.15_original_68cd0a8/unified_fatformer_v2.pth --output DATASET/logs/phase7/task_7.15_recovery/real_image_verification.json --device cuda --images /content/verification_images
python -m tools.infer_unified --checkpoint DATASET/checkpoints/phase7/unified_fatformer_v2.pth --device cuda --benchmark_latency --latency_out DATASET/logs/phase7/latency_phase7.json
```

Thư mục verification_images cần ít nhất 20 ảnh thật sự đọc được; giữ cùng preprocessing và precision khi đối chiếu. Lệnh benchmark ghi GPU/VRAM thực tế, 20 warmup, 200 cặp đo/batch, mẫu thô và hash package. Theo yêu cầu cập nhật 09/11/2026, nghiệm thu latency trên GPU CUDA thực tế; không bắt buộc T4 và không suy diễn số đo sang GPU khác.

Còn cần chấm package bằng predict trên đúng 1.600 evaluation clean/Q30 và đối chiếu cấu hình tốt nhất cùng partition. Delta ≤1,5 pp là so với reference theo từng điều kiện, không phải clean trừ Q30. Không dùng evaluation để chọn T/ngưỡng.

`latency_phase7.json` hiện pending cho package mới. Log A100 cũ vẫn được bảo toàn. Chưa có phép đo GPU mới hoặc phép đối đầu theo điều kiện mới trong phiên local này.
