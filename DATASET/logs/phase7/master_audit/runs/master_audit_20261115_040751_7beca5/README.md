# master_audit_20261115_040751_7beca5

**Execution completed; hậu kiểm verified; G7.2 FAIL 6/8; G7.3 PENDING.**

- [Báo cáo phân tích](analysis/README.md)
- [Run status và cấu hình gốc](raw/run_status.json)
- [Nguồn Git đã chạy](raw/git_source.json)
- [Archive audit](raw/archive_audit_summary.json) — [overlap](raw/content_overlap.json)
- [Benchmark](raw/benchmark_summary.json) — [37 job](raw/job_reports.json) — [175 dòng tổng hợp](raw/measured_condition_and_subset_summary.csv)
- [Gates](raw/gates.json) — [latency L4](raw/latency.json)
- [204 tests liên quan](raw/tests.xml) — [console](raw/tests_console.txt)
- [Metadata package](raw/package_metadata.json)
- [Hậu kiểm local](analysis/verification.json) — [CSV tái tính](analysis/recomputed_summary.csv)
- [Import manifest SHA-256](import_manifest.json)

Package `248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`; source checkpoint `5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c`; source commit `1f39ca8fdc11890fa1daed8cf885506c4b79eba8`. T=`2.7832491397857666`, threshold=`0.7918508648872375`; router tắt. Không train hoặc refit calibration.

Phạm vi: 944.327 bản ghi archive, 815.689 hash file phân biệt, 1.618.061 lượt dự đoán trên nhiều split/điều kiện. Không coi các lượt này là ảnh gốc độc lập hoặc gộp train/validation/test thành một accuracy.

Core18 clean: pooled accuracy **93,46%**, macro balanced accuracy **90,10%**; Q30 pooled accuracy **70,91%**. CFE clean: accuracy **74,38%**, Fake recall tại ngưỡng triển khai **53,12%**; Q30 tương ứng **65,81% / 40,25%**. Hai tiêu chí trượt là clean Fake@Real90 **64,125% <65%** và worst-subset Fake accuracy **30,8642% <45%**.

GPU L4, FP16 backbone / FP32 logits-softmax, batch16, TF32 tắt. Latency mới batch1 **46,15 ms/ảnh**, overhead **0,227768 ms/ảnh**, không gồm I/O hoặc UI. CFE ở cả 7 điều kiện tái lập đúng dự đoán Task 7.15; đây là tăng độ phủ kiểm thử, không phải tăng chất lượng model.

`raw/` giữ nguyên bytes, kể cả đường dẫn Colab trong metadata. SQLite/NPZ chỉ có tại local, được băm trong manifest nhưng không đưa vào Git. Nguồn nhận ban đầu vẫn còn ở `wait/master_audit_20261115_040751_7beca5`; không có checkpoint hoặc archive ảnh Drive trong bộ kết quả.

Giới hạn: hậu kiểm không forward lại toàn bộ ảnh Drive tại local, không bootstrap lại CI95; 204 tests không phải full repository suite; không phải blind holdout mới; chưa chứng minh zero leakage bằng pHash/generator-disjoint; chưa nghiệm thu giao diện Gradio hoặc định vị giả mạo.
