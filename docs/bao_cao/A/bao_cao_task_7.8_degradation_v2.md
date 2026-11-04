# Báo cáo Task 7.8 — Pipeline suy biến v2

Cập nhật kiểm tra tích hợp: 2026-11-04. Thành viên A phụ trách triển khai ban đầu.

## Phạm vi và hành vi

`DegradationPipelineV2` tạo cặp `(clean, degraded)` để phục vụ KD/consistency; `return_pair=False` trả một tensor cho trainer đơn luồng. YAML chuẩn đặt clean 60%, JPEG Q30–85 với nhóm nặng Q30–45, WebP, double-JPEG, blur, down-up và composite 25% trong nhóm degraded. Đây là thay đổi augmentation; chưa có phép đo chứng minh cải thiện AUC Q30 sau huấn luyện.

- `DatasetCreator.build_train_dataset()` giữ nguyên transform cũ và đầu ra một tensor. Bật v2 bằng `degradation_config`; bật cặp bằng `return_pair=True`.
- Cross-domain M4/M5 đọc `degradation_config` từ YAML/CLI. Khi không bật v2, train và validation giữ Resize(256) → CenterCrop(224). Khi bật v2, kích thước lấy từ YAML suy biến, trừ khi có override rõ ràng; validation dùng cùng kích thước với train và không áp suy biến.
- `jpeg.q_range` là giới hạn cuối. Các dải `heavy_range`/`light_range` được chặn vào giới hạn đó; dải con không giao được thu về đầu mút gần nhất. Q60 cố định luôn trả Q60. Dải đảo, ngoài [1,100], sai kiểu và heavy_share ngoài [0,1] bị từ chối ngay khi tạo pipeline.
- T5 ép lần lượt cả năm loại suy biến để kiểm tra cùng seed cho cùng tag và tensor, thay vì chỉ kiểm tra nhánh clean.

Ví dụ cấu hình train cross-domain:

```yaml
dataset:
  degradation_config: configs/degradation_v2.yaml
  # img_resolution: 256  # chỉ đặt nếu chủ động ghi đè kích thước trong YAML v2
```

`train.py` vẫn dùng curriculum/transform cũ. Việc bật v2 trong tài liệu này áp dụng cho `src.training.train_crossdomain` hoặc DatasetCreator có truyền config rõ ràng.

## Kiểm tra

Các test bao phủ: nhánh clean chuẩn, phân phối 10.000 mẫu, tất định từng codec, Q60 và các dải tùy chỉnh trên cả dict/YAML cũ, cấu hình không hợp lệ, preprocessing cũ, đầu ra đơn/cặp, YAML → CLI → DataLoader train/validation. Smoke paired đưa ảnh từ loader qua model CPU nhỏ, tính loss KD/consistency, backward và cập nhật trọng số. Đây không phải phép thử GPU/CLIP thực tế.

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest -q -p no:cacheprovider --basetemp scratch/pytest-task78-final tests/test_degradation_v2.py tests/test_smoke_degradation_integration.py
```

Lần chạy cuối sau tối ưu: **31 passed in 133.14s** (9 trường hợp degradation, 22 trường hợp smoke/regression). `--basetemp` đặt thư mục tạm trong workspace để không phụ thuộc quyền truy cập thư mục pytest dùng chung của Windows.

Các số clean=0.605, heavy JPEG=0.367, composite=0.249 thuộc báo cáo lấy mẫu trước đó. Test phân phối chạy lại kiểm tra ngưỡng; không dùng các số cũ làm số đo mới. Histogram hiện lưu tại `DATASET/logs/phase7/degradation_profile.png`; panel là Q JPEG, tỷ lệ loại suy biến, sigma blur và tỷ lệ composite, không phải histogram Q WebP.

## Benchmark DataLoader

Kết quả một tiến trình 50 batch cũ (-14.69%) không đáp ứng phương pháp nghiệm thu và đã được thay thế. Công cụ mới dùng DataLoader thật với 4 worker; hai pipeline cùng nguồn ảnh, resize/crop, batch size và đầu ra paired. Mỗi lượt đo 200 batch sau 20 batch warmup; chạy ba lượt, đảo thứ tự v1/v2, lấy median và loại thời gian khởi động/tắt worker.

```powershell
python -m tools.bench_degradation --config configs/degradation_v2.yaml --batches 200 --num_workers 4 --repeats 3 --output DATASET/logs/phase7/degradation_benchmark.json
```

Nguồn mặc định là 32 ảnh nhiễu RGB tổng hợp có seed, không phải ảnh từ manifest v2. Có thể dùng `--image-dir <thu_muc_anh>` để đo cả đọc/giải mã dữ liệu thật. JSON lưu từng lượt, cấu hình, nguồn ảnh, môi trường và trạng thái ngưỡng; lệnh trả exit code 1 khi overhead vượt 15%. Kết quả trên CPU hiện tại không thay cho benchmark trên máy train.

Phép đo trước tối ưu với phương pháp mới vượt ngưỡng, lưu tại `DATASET/logs/phase7/degradation_benchmark_before_optimization.json`. Profiling cho thấy WebP encode chiếm phần lớn chi phí tăng thêm. Tối ưu đã áp dụng:

- Dùng chung ảnh PIL sau resize/crop cho hai nhánh, bỏ resize lặp; chế độ đơn tensor không tạo tensor clean bị bỏ đi.
- `webp.method=0` chọn encoder nhanh, giữ Q30–70 và tỷ lệ lấy mẫu. Bitrate/pixel sau giải mã có thể khác encoder method=4 trước đây; có thể cấu hình method 0..6. Không thay tỷ lệ clean/composite để đạt tốc độ.

Kết quả sau tối ưu trên CPU Windows hiện tại: v1 median **9.702 ms/batch**, v2 median **11.068 ms/batch**, overhead **+14.08%**, đạt ngưỡng ≤15% trên corpus tổng hợp này. Ngưỡng đạt sát giới hạn; cần đo lại với ảnh thật và máy train trước khi suy rộng hiệu năng. JSON lưu đủ ba lượt, không chọn lượt nhanh nhất.

## Artifact

- Transform: `src/datasets/transforms.py`; cấu hình: `configs/degradation_v2.yaml`.
- Tích hợp: `src/datasets/dataset.py`, `src/training/train_crossdomain.py`.
- Tests: `tests/test_degradation_v2.py`, `tests/test_smoke_degradation_integration.py`.
- Đo/vẽ: `tools/bench_degradation.py`, `tools/plot_degradation_profile.py`.
- Kết quả: `DATASET/logs/phase7/degradation_benchmark.json`, `DATASET/logs/phase7/degradation_profile.png`.
