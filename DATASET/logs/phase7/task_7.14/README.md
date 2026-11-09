# Hồ sơ Task 7.14

**Run package hiện hành:** [corrected_20261109_102954_3d28ad](../task_7.15/current/analysis/README.md), đồng bộ với hồ sơ Task 7.15 cùng hash package `24869052…`. [Benchmark gốc](../task_7.15/current/raw/benchmark/final_report.json), [acceptance](../task_7.15/current/raw/acceptance.json), [metric và CI tái tính](../task_7.15/current/analysis/recomputed_metrics.json).

- Chấm package thực trên 7 điều kiện ×1.600 evaluation và 82.003 lượt ảnh/18 tập in-dist, FP16/CUDA/batch 16, tách 400 calibration.
- Clean AUC **88,66%**, Fake@Real90 **64,12%**; Q30 AUC **79,88%**, Fake@Real90 **42,62%**, ECE **5,08%**.
- In-dist macro balanced ACC **90,101847%**, pooled Real ACC **90,406282%**.
- **G7.2: 6/8, tổng thể FAIL**; chưa đạt Clean Fake@Real90 và worst-subset Fake ACC **30,86%**. **G7.3 pending** reference tốt nhất theo điều kiện; 20 ảnh và overhead L4 đã có.
- Mọi metric/CI95, ID/nhãn, hash artifact và latency đã được hậu kiểm. Weights mới không nằm trong ZIP; vị trí Drive được ghi trong [package reference](../task_7.15/current/artifacts/package_reference.json).

**Run lịch sử:** [locked_benchmark_phase7_20261109_074134](runs/locked_benchmark_phase7_20261109_074134/analysis/README.md), checkpoint v2 tại operating point baseline T=1,6402/ngưỡng=0,9127; G7.2 đạt 5/8 ở protocol cũ. Giữ nguyên dữ liệu đó, không ghép thành số đo package mới.

[index.json](index.json) phân biệt run đo mới nhất với nghiệm thu cuối. Kết quả mới vẫn là xác nhận sau hậu kiểm trên evaluation đã xem trước, không phải phép thử mù mới.
