# Master full data audit — bản sửa 15/11/2026

Notebook: [master_full_data_audit_colab.ipynb](../../notebooks/master_full_data_audit_colab.ipynb). Dùng canonical loader, `StrictImageDataset`, `predict_records`, metric và paired latency có sẵn. Mã điều phối: [master_data_audit.py](../../tools/master_data_audit.py).

Bản cũ chỉ ghi bảng số liệu cố định và nạp state dict như model object. Bản sửa bỏ các bảng đó: từng ảnh phải được đọc/giải mã, ghi SHA-256/nhãn/split vào SQLite, sau đó đi qua model. Thiếu dữ liệu hoặc lỗi sẽ dừng và xuất ZIP `failed`; không thay bằng ảnh zero hoặc AUC 50%. Source, weights/calibration và kết quả benchmark lịch sử không bị ghi đè.

## Chuẩn bị

Source được lấy trực tiếp từ repository GitHub, nhánh **hoang**:

```text
https://github.com/LeeVHoangtk3/fatformer-xla.git
```

Mở notebook trên Colab, chọn GPU rồi chạy lần lượt các cell. Cell 2 clone nhánh `hoang`, chốt SHA của nhánh tại lúc bắt đầu và checkout detached. Nếu cần tái lập, đặt `GIT_COMMIT` bằng SHA 40 ký tự đã ghi trong `git_source.json`; notebook chỉ nhận commit thuộc lịch sử nhánh `hoang`. Không lấy source ZIP hoặc yêu cầu upload NPZ. Git cần truy cập được repository; không ghi token vào notebook hoặc URL.

Các file audit phải được commit/push lên `hoang` trước khi chạy. Nếu commit thiếu mã audit/frozen splits, notebook dừng rõ ràng. `git_source.json` ghi URL, commit, branch và SHA từng file source/config/manifest/notebook; source phải sạch và được kiểm lại trước nạp model. Đầu ra runtime nằm ngoài checkout nên không làm source bẩn. Không cập nhật torch/torchvision runtime và không cài TensorRT.

Hai frozen CFE splits đã chuyển sang `configs/master_audit/calibration_split.json` và `evaluation_split.json` để Git theo dõi trực tiếp. ID/nhãn/thứ tự giữ nguyên so với NPZ nguồn; JSON ghi hash NPZ nguồn để truy xuất provenance. Các file manifest train/validation đã có sẵn trong Git. Weights và archive ảnh tiếp tục đọc từ Drive, không đưa vào Git.

Model dùng đúng đường dẫn `Fatformer/checkpoint/phase7/corrected_20261109_102954_3d28ad/unified_fatformer_v2.pth`, SHA `248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`. Nạp cấu trúc chuẩn từ code và calibration nhúng, không `.eval()` dictionary, không chia nhiệt độ thêm lần nữa.

## Danh mục bắt buộc

Theo hồ sơ kho dữ liệu và manifest hiện có (chưa phải kiểm tra lại nội dung Drive trong lượt sửa notebook):

- `datasets/progan_train.tar`: 144.024 bản ghi, diagnostic train.
- `datasets/progan_val.tar`: 8.000 bản ghi, diagnostic validation.
- `datasets/diffusion_staging.tar`: 3.600 bản ghi, tách 3.060 train/540 val theo manifest legacy.
- `phase7/train_crossdomain_v2.tar`: 14.400 bản ghi, tách 12.240 train/1.080 val_select/1.080 val_calib theo SHA trong manifest v2. `train_select` 13.320 là view gộp, không đánh giá thêm để tránh đếm hai lần.
- `datasets/test_communityforensics.tar`: 2.000 bản ghi, frozen 400 calibration/1.600 evaluation. Calibration chỉ đo chẩn đoán tại T/ngưỡng hiện hành, không fit lại.
- `datasets/test_benchmark_gans.tar`: 90.329 bản ghi; 8 tập chính và 5 supplemental `crn`, `imle`, `san`, `seeingdark`, `whichfaceisreal`.
- `datasets/test_benchmark_diffusion.tar`: 20.000 bản ghi, 10 tập chính.
- `datasets/test_degraded.tar`: 661.974 bản ghi, 6 biến thể lưu sẵn, mỗi biến thể 110.329; không áp dụng thêm degradation lần hai.

Sửa cấu hình đường dẫn cell 4 nếu vị trí Drive khác. Dữ liệu gốc không bị thay đổi. Audit duyệt toàn bộ ảnh dạng streaming, sau đó inference giải nén từng archive vào thư mục tạm rồi dọn; cần đủ SSD cho archive giải nén lớn nhất cộng 2 GiB dự phòng. Không giữ cả kho giải nén cùng lúc. Không tự giảm số mẫu/batch khi thiếu tài nguyên; lỗi OOM/SSD phải hiện trong trạng thái run.

## Protocol và báo cáo

Train/validation/calibration được tách hoàn toàn khỏi test, chỉ là kiểm tra chẩn đoán trên model cố định. Không train, không select lại checkpoint và không recalibrate theo kết quả mới. V2 validation đo clean/Q30; train và dữ liệu train/val lịch sử đo clean. CFE evaluation và toàn bộ clean test (18 chính + 5 bổ sung) chạy 7 điều kiện canonical: clean, Q30, Q50, Q75, WebP40, blur1.5, down/up0.5. Các điều kiện này được tạo sau resize/crop, trước normalize.

Archive degraded cũ có Q70, blur1/2 và resize protocol lịch sử; kết quả riêng theo biến thể lưu sẵn. Không trộn chúng với CFE Q75/blur1.5 hoặc gọi hai protocol tương đương. Cùng ảnh nguồn ở nhiều điều kiện/archive không trở thành mẫu độc lập.

GPU dùng FP16 backbone autocast, logits/softmax FP32, TF32 tắt, batch mặc định16. Nếu đổi batch thì ghi khác hợp đồng số đo lịch sử, không cam kết bit-exact. CI95 bootstrap 1.000 lần theo ảnh cho pooled CFE evaluation từng điều kiện và core18 clean/Q30; không mặc định mọi bảng diagnostic đều có CI. Phụ thuộc giữa các real reference/generator chưa được CI theo ảnh mô tả hết.

SQLite lưu mọi bản ghi ảnh; exact SHA kiểm v2 so với CFE/clean test, split v2, split CFE và xung đột nhãn trên cùng bytes. pHash và generator-disjoint audit toàn kho **chưa chạy**, không công bố “zero leakage 100%” từ SHA đơn thuần. Replay/legacy/shared test references có thể trùng nội dung và không được đếm thành ảnh độc lập.

Mặc định notebook chạy các file test loader/partition/calibration/router/metric/loss/degradation liên quan và test audit mới, lưu JUnit/console. Không gọi đó là full repository test. Có thể bật `FULL_REPOSITORY_TESTS` để chạy `pytest tests`; nếu thiếu fixture/dependency hoặc có lỗi, phải xử lý và chạy run mới, không bỏ qua test để công bố đạt.

Mỗi NPZ dự đoán có score/nhãn/target/logits, sample ID, đường dẫn `archive:member`, package SHA và archive SHA; JSON/job report có vai trò/split/protocol. Chỉ core18 được dùng cho macro/G7.2; supplemental, degraded lịch sử, train/validation không góp vào các chỉ số này.

G7.2 tính từ dự đoán mới. G7.0/G7.1 không tự đóng bằng kết quả model cuối. G7.3 vẫn pending best-by-condition reference dù overhead GPU đạt. Latency dùng helper paired hiện có: warmup20, steps200, batch1/8/16/32, median/p95 và overhead; tensor tổng hợp không được tính vào số ảnh thật đã đánh giá.

Run có ID mới, trạng thái và ZIP riêng trong `Fatformer/audit_reports/<run_id>/`. Hoàn tất thực thi không đồng nghĩa mọi gate PASS. Nếu thiếu archive, ảnh lỗi, lệch split, leakage hoặc tests fail thì ZIP có trạng thái `failed`, không phải kết quả full-run.

## Kiểm chứng bản sửa

Cần chạy notebook trên runtime Colab có đủ dữ liệu để có bằng chứng full-run. Kiểm tra local không thay thế việc đó. Đã kiểm trong phiên sửa:

- Schema nbformat và cú pháp toàn bộ code cell hợp lệ; notebook mới không lưu output giả hoặc execution count giả.
- Các test kiểm count/hash/decode, path traversal, duplicate member, leakage, role/split, dự đoán NPZ, loại supplemental khỏi core18, thiếu condition và ngăn finalize sau lỗi. Có thêm test Git cho source/input/notebook đã commit, phát hiện nội dung thay đổi và frozen JSON 400/1.600 không trùng ID. Đây không phải chứng nhận full repository suite hoặc full-data GPU run.
- Bản chuyển sang Git đã qua **39 tests** trong `test_master_data_audit.py`, `test_phase7_git_source.py`, `test_phase7_corrected.py`; 0 lỗi, 0 skip. JUnit local: `artifacts/master_audit/git_workflow_tests.xml`.
- Đọc đủ 14.400 record v2, 3.600 record legacy, 2.000 ID CFE từ các file trong repo. Hai JSON giữ nguyên toàn bộ ID/nhãn của NPZ gốc. Baseline calibration JSON lịch sử chỉ là fixture cho test selection, không dùng làm calibration của model cuối.
- Smoke trên package thật, CPU FP32/batch1, hai ảnh dữ liệu thực (một nguồn REAL, một nguồn FAKE): audit archive → canonical load → `predict_records` → NPZ/metric. Điểm từng ảnh khớp log CPU trước đó. Bằng chứng tại `artifacts/master_audit/local_smoke/`; không gọi hai ảnh là full benchmark hoặc bằng chứng CUDA.
- Một test nguồn notebook lịch sử trước đây đòi output rỗng đã được sửa để chấp nhận output bằng chứng không có error. Notebook có kết quả lịch sử được giữ nguyên, không xóa output để làm test qua.

Chưa có lượt thực thi mới trên toàn kho Drive/Colab, chưa có số đo GPU mới và chưa chạy pHash toàn kho. Vì vậy chỉ xác nhận bản sửa đã qua kiểm tra local nêu trên; trạng thái full-run thực tế vẫn chờ notebook chạy trên dữ liệu đầy đủ.

## Khôi phục lỗi tên ảnh có dấu giờ

Log Colab tại bước audit dừng ở `test/cyclegan/summer/1_fake/2015-12-31 17:23:30_fake.png`. Đây là lỗi bộ lọc đường dẫn cũ chặn mọi dấu `:`, không phải lỗi dữ liệu hoặc model. Bản sửa nhận tên POSIX hợp lệ nhưng vẫn từ chối đường dẫn tuyệt đối, ổ đĩa Windows, `..`, backslash và NUL. Khi staging trên Windows, tên lưu tạm được ánh xạ bằng hash để tránh alternate stream/tên không hợp lệ; tên archive gốc vẫn giữ trong inventory và dự đoán.

Nếu runtime lỗi **vẫn còn hoạt động**, thêm một code cell bên dưới lỗi và chạy:

```python
import subprocess
subprocess.run(['git', 'fetch', 'origin', 'hoang'], cwd=REPO, check=True)
REPAIR_SOURCE_COMMIT = subprocess.check_output(
    ['git', 'rev-parse', 'origin/hoang'], cwd=REPO, text=True).strip()
repair = subprocess.check_output(
    ['git', 'show', REPAIR_SOURCE_COMMIT+':notebooks/resume_master_data_audit_colab.py'],
    cwd=REPO, text=True, encoding='utf-8')
exec(compile(repair, 'resume_master_data_audit_colab.py', 'exec'), globals())
```

Script chỉ dùng cho lỗi audit trước khi nạp model: chốt commit Git mới, xóa cache module Python cũ, tạo run mới, snapshot SQLite đã commit và giữ provenance run trước. Nó kiểm frozen input hashes không đổi; chỉ tái sử dụng archive content audit có report `completed` cùng SHA archive, count/nhãn/split/subset và tổng byte khớp SQLite. Archive dở bị audit lại từ đầu trong inventory mới; run cũ không bị ghi đè. Các phép dự đoán vẫn chạy mới, không tái sử dụng metric/NPZ.

Với log trên, năm archive đầu đã đi qua bước hoàn tất; nếu kiểm chứng cache đạt, không cần giải mã lại ProGAN train/val, staging, data v2 và CFE. `test_gans` sẽ chạy lại từ đầu. Việc tính lại hash archive vẫn cần đọc bytes trên Drive. `audit_resume.json` ghi source/run trước và commit đang chạy.

Không restart runtime nếu muốn dùng inventory còn trên `/content`. Nếu runtime đã mất, đường khôi phục này không có dữ liệu local để tiếp tục; chạy run mới hoặc phục hồi evidence riêng trước. Notebook mới cũng có `RESUME_AUDIT_FROM` để chỉ định một thư mục results cũ còn tồn tại. Không bỏ kiểm tra hash/nhãn hoặc xóa dấu giờ trong dataset để làm pipeline chạy qua.

Bản sửa đã qua **48 test liên quan**, kiểm tên file đúng như log, audit/extract ảnh chứa dấu giờ trên staging Windows an toàn, tái sử dụng archive hoàn tất, khởi động lại archive dở và từ chối cache/input đã đổi. Driver recovery được chạy thử bằng Git repository fixture trong subprocess, xác nhận chốt đúng commit notebook và tạo run mới với provenance cũ; không phải lượt khôi phục thật trên Colab của người dùng.
