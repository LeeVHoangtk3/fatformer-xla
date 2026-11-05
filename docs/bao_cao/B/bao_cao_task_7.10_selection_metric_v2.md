# Báo cáo Task 7.10 — Selection metric v2

Cập nhật review/sửa đồng bộ: **2026-11-05**. Triển khai ban đầu: Thành viên B.

## Quy tắc được kiểm chứng

`worst_group = min(clean_r, clean_f, q30_r, q30_f)`. Recall tính ở operating point cố định; nhóm thiếu, dữ liệu không hữu hạn hoặc recall ngoài [0,1] gây lỗi. Điểm không được suy ra bằng cách chép recall clean sang Q30.

Hai kỷ lục theo dõi riêng: ứng viên đạt `clean_r >= 0.90` chỉ cập nhật official best khi worst-group tăng; ứng viên không đạt ràng buộc chỉ cập nhật best-effort khi cải thiện kỷ lục riêng của nhóm này. Bằng điểm không ghi đè. Chính sách dùng chung ở epoch 0, dry-run và mỗi epoch; không ép checkpoint cuối thành official best khi chưa có ứng viên đạt chuẩn.

`CrossDomainTrainer.save_checkpoint` còn kiểm tra lại ràng buộc trước ghi để phát hiện cờ `is_best` sai từ caller. Các alias `best.pth` và tên checkpoint cấu hình chỉ cập nhật từ ứng viên official; best-effort có file riêng. CheckpointManager được sửa dùng chung đường ghi atomic cho chế độ latest, lưu từng epoch và filename tùy chỉnh; không còn lỗi `local_path` chưa khởi tạo. Backup sao chép cả latest và loại best tương ứng.

## Đồng bộ ảnh Q30, calibration và trainer

- Q30 nén **sau Resize → CenterCrop**, trước ToTensor/Normalize; dùng `quality=30` cố định. Tối đa 300 ảnh, tối đa 150 mỗi nhãn, seed 42. Khi một lớp ít hơn 150, dùng số sẵn có; vẫn bắt buộc có cả hai nhãn.
- Cache chọn theo chỉ số dataset, hỗ trợ Subset/Concat và sampler có indices. Dataset tensor không có records vẫn lấy mẫu theo nhãn, không lấy 300 phần tử đầu. Mất/hỏng ảnh hoặc cache thiếu nhóm gây lỗi; không bỏ ảnh âm thầm hay gán ảnh clean là Q30.
- Cache từ records đọc ảnh gốc để tránh vòng đổi tensor/PIL làm lệch pixel. Dataset tensor tổng quát phải dùng ImageNet normalization theo hợp đồng eval của dự án. Q30 tensor được cache một lần; startup được tách khỏi phép đo warm validation.
- `OperatingPoint` đọc artifact Task 7.4 có `evidence_level=measured`, `T` và `T_star_default`. Cả clean và Q30 dùng `softmax(logits/T)[:,1] >= threshold`. Không fit trên `val_select`. Source và SHA256 của calibration được lưu trong metadata.
- YAML Phase 7 giữ manifest v2/`val_select` của Task 7.7 và degradation v2 của 7.8; thêm `training.calibration_file: DATASET/logs/phase7/calibration/baseline.json`. Đây là operating point baseline được đóng băng cho run, không phải calibration tự cập nhật theo từng epoch. Có CLI `--calibration_file`, `--selection_threshold`, `--q30_max_samples`.
- `Trainer` thường chọn đúng cặp loader clean/Q30; Q50 hoặc tên mơ hồ không được coi là Q30. Nếu có thêm diffusion_clean, lấy clean cùng miền với Q30. Root `train.py` tạo Q30 post-resize, giới hạn 300 ảnh cân bằng và truyền cấu hình calibration cho Trainer. Loader Q30 được materialize một lần.
- Riêng `Trainer` thường với **loader đơn** giữ chính sách ACC/AP legacy và phát cảnh báo; không báo Q30/worst-group giả, không tạo `val_groups.csv`. Đây là nhánh tương thích cũ, không phải nghiệm thu selection v2. CrossDomainTrainer tự tạo nhánh Q30 từ loader clean nên vẫn dùng đủ bốn nhóm.

`val_groups.csv` lưu đủ `epoch, clean_r, clean_f, q30_r, q30_f, worst, best_effort`, cộng threshold, temperature, calibration source, số mẫu clean/Q30 và nguồn Q30. Không làm tròn score để quyết định. CSV cũ khác schema phải đổi run directory, không append các cột không khớp.

## Test và bằng chứng

Các test mới dùng đường code thật để kiểm chứng:

- Dataset xếp 300 Real trước 20 Fake tạo cache 150 Real/20 Fake; cache thiếu nhóm bị từ chối.
- Tensor Q30 khớp pipeline post-resize, lặp lại tất định, không vượt khỏi Subset/sampler.
- Ngưỡng và temperature từ JSON thay đổi dự đoán cả hai điều kiện và được ghi vào CSV/checkpoint.
- Epoch 0 và cuối pipeline không thể nâng ứng viên `clean_r=.85` thành best; best-effort cao hơn không chặn một official hợp lệ có score thấp hơn; điểm hòa giữ checkpoint cũ.
- Cả bốn tổ hợp chế độ CheckpointManager/filename ghi đúng file và backup; ứng viên vi phạm guard không ghi đè official.
- Trainer nhiều loader chọn Q30 thay vì Q50; loader đơn giữ legacy kèm cảnh báo.
- Test CSV gọi `record_validation`, đọc lại CSV thật; không tự viết một CSV mẫu để kiểm tra chính nó.

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest -q -p no:cacheprovider --basetemp scratch/pytest-task710-final tests
```

Kết quả toàn bộ thư mục `tests` sau sửa: **111 passed in 68.59s**. Có test hồi quy riêng cho temperature lớn với xác suất float32 đã bão hòa; lựa chọn vẫn lấy từ logits gốc, không dựng lại logits từ xác suất đã làm tròn.

## Hiệu năng

```powershell
python -m tools.bench_selection --output DATASET/logs/phase7/selection_benchmark.json
```

Artifact [`selection_benchmark.json`](../../../DATASET/logs/phase7/selection_benchmark.json) ghi phép đo CPU seed 42: 32 ảnh PNG nhiễu 256×256 được dùng lặp cho 400 records cân bằng, model pooled-RGB linear, batch 32, workers 0. Sau hai warmup, chạy năm lượt đảo thứ tự, lấy median:

- Clean-only: **765.91 ms**.
- Clean + cache Q30 (150 Real/150 Fake): **801.32 ms**, tỷ lệ **1.046×**, đạt giới hạn ≤2× trên thử nghiệm này.
- Tạo trainer/cache một lần: **1202.70 ms**, không tính vào warm validation.

Phép đo không chứng minh tốc độ CLIP/T4 hay dữ liệu thật; phải đo lại trên máy train. Công cụ trả exit code 1 nếu vượt ngưỡng và lưu mọi lượt, không chọn lượt nhanh nhất.

## Giới hạn nghiệm thu

Code/test CPU và liên kết 7.7–7.9 được kiểm tra; chưa có smoke checkpoint thật/GPU cho bản sửa này. Không suy từ mock CPU ra chất lượng phát hiện, khả năng resume hoặc việc Task 7.11/G7.2 đã hoàn tất. Benchmark 7.8 hiện được ghi là **+14.08% trên corpus CPU tổng hợp** trong báo cáo 7.8, thay cho số -14.69% cũ.
