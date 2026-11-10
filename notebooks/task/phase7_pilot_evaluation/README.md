# Hiệu chuẩn và benchmark P0 sau pilot — code từ nhánh `hoang`

**Không train lại.** Workflow này chấm checkpoint P0 đã chọn từ validation, đồng thời chấm trọng số incumbent theo cùng protocol. Chưa có số đo mới cho tới khi bạn chạy trên Colab; không thay kết quả 7.14–7.15 đang lưu.

## 1. Hai checkpoint và cách so sánh

- `candidate_p0`: `9d97613c…`, epoch 5 của `pilot_20261109_182359/p0_legacy_full`.
- `incumbent_refit`: source `5cd49252…` của Anchored v2 hiện hành, **fit lại calibration cùng phương pháp** để so với P0. Đây không phải chạy lại chính xác package deployment `24869052…` với calibration cũ.

Mỗi model tự trích logits trên cùng 400 calibration (200 Real/200 Fake), fit T/ngưỡng hội tụ cho FPR mục tiêu 5%, đóng gói và kiểm chứng tensor + 20 ảnh từ calibration. Sau đó chấm 1.600 CFE evaluation và 82.003 record/18 tập in-dist. Hai model dùng cùng GPU, FP16, batch chung được thử vừa bộ nhớ trước calibration, TF32 off, canonical CLIP preprocessing.

Danh sách ID/nhãn ở `configs/phase7_pilot_eval/` được xuất nguyên thứ tự từ split NPZ đã dùng trước đây; không chia lại và không xuất logits cache cũ. Hash ba archive ảnh cũng khóa theo run đã hậu kiểm trước. Đây là post-hoc evaluation vì test đã từng được xem, không phải test mù mới.

## 2. Chuẩn bị Drive

Các file phải có dưới `MyDrive/Fatformer/`:

```text
phase7/improvement_round1/pilot_20261109_182359/p0_legacy_full/best.pth
checkpoint/phase7/anchored_v2_run1/fatformer_anchored_v2_best.pth
datasets/test_communityforensics.tar
datasets/test_benchmark_gans.tar
datasets/test_benchmark_diffusion.tar
```

Hai checkpoint đã chứa đủ backbone; không cần tải lại pretrained CLIP riêng. Hash chính xác được kiểm tra, không tìm tự động các file `best.pth` khác. Source weights, raw ảnh và notebook cũ giữ nguyên.

## 3. Chạy trên Colab

Dùng notebook và runtime GPU mới. Cell 1:

```python
from pathlib import Path
import subprocess

REPO = Path('/content/fatformer-xla')
subprocess.run([
    'git', 'clone', '--branch', 'hoang', '--single-branch',
    'https://github.com/LeeVHoangtk3/fatformer-xla.git', str(REPO)
], check=True)
```

Cell 2 chạy toàn bộ calibration → benchmark → đóng gói/kiểm chứng → so sánh:

```python
EVAL_STAGE = 'full'
EVAL_MAX_BATCH = 16
exec(compile(
    (REPO / 'notebooks/task/phase7_pilot_evaluation/start_colab.py').read_text(),
    'start_colab.py', 'exec'
))
```

Cell tự lưu `EVAL_COMMIT`, `EVAL_RUN_ID`. Cho phép kết nối Drive khi được hỏi. Không cần ZIP code. GPU CUDA bất kỳ đủ dung lượng đều được; probe có thể chọn batch nhỏ hơn 16, nhưng **giữ cùng batch cho cả hai checkpoint** và không hạ batch giữa benchmark. CPU không thay phép đo GPU.

Nếu chỉ muốn chạy phần ngắn trước, đặt `EVAL_STAGE='calibrate'`. Sau khi hoàn tất, giữ cùng phiên/ID/commit và chạy lại cell với `EVAL_STAGE='full'`. Các bước hoàn tất chỉ được dùng lại khi hash và identity khớp; không suy thành công từ việc thấy file. Nếu có output dở hoặc bị thay, giữ run đó làm bằng chứng và tạo `EVAL_RUN_ID` mới. Không `git pull` giữa cùng một run.

Luồng full cần hai lượt chấm, mỗi lượt 11.200 ảnh CFE ở 7 điều kiện + 82.003 record in-dist, tổng 186.406 lượt inference cho benchmark. Thời gian thực phụ thuộc GPU/I/O; chương trình in tiến độ và lưu `commands.log`.

## 4. Kết quả

```text
MyDrive/Fatformer/phase7/improvement_round1/evaluation/<EVAL_RUN_ID>/
  source_identity.json
  eval_identity.json
  batch_contract.json
  candidate_p0/
    calibration/           # NPZ mới, T/ngưỡng/diagnostics và reliability
    package/unified.pth    # package mới, không ghi đè deployment cũ
    benchmark/            # dự đoán từng ảnh, metrics, CI, ROC và gate
    verification.json
    latency.json
    acceptance.json
  incumbent_refit/
  comparison.json
  status.json
  evaluation_results_without_weights.zip
```

Đường dẫn Colab: `/content/phase7_p0_evaluation/<EVAL_RUN_ID>/`. ZIP cuối là ZIP kết quả để tải về, không chứa weights/dataset. Package weights lưu riêng tại đúng thư mục Drive phía trên. Các file `reference` local chỉ là bản sao source đã kiểm hash.

Sau chạy, lưu notebook mới và giải nén ZIP kết quả vào `wait/phase7_pilot_evaluation/<EVAL_RUN_ID>/` để hậu kiểm. Nếu có lỗi, lấy ZIP/log của run đó; không đổi FAIL thành PASS hoặc thay ảnh hỏng bằng ảnh giả.

## 5. Cách đọc kết quả

- `candidate_p0/acceptance.json`: G7.2 của P0 mới; không dùng điểm validation 42% để thay cho gate test.
- `comparison.json`: delta P0 − incumbent refit trên cùng IDs, paired CI cho AUC/Fake@Real90 ở CFE. Bootstrap phân tầng theo nhãn, 1.000 lần/seed 42, cùng chỉ số resample cho cả hai model. CI này điều kiện trên một seed train; chưa xử lý mọi phụ thuộc generator hoặc đa kiểm định.
- Macro in-dist là trung bình balanced ACC của 18 tập. CI chẩn đoán paired theo 18 tập được ghi riêng; không thay luật gate gốc và không chứng minh mọi record ảnh độc lập.
- Latency là overhead của package so với backbone, đo paired; tổng ms/ảnh không phải overhead. Quality claim chỉ áp dụng batch/precision benchmark, không tự chuyển sang mọi batch/backend.
- So với incumbent không tự chứng minh best-by-condition. **G7.3 vẫn pending** phần reference đó; workflow không tự thay deployment dù G7.2 đạt.

Sau hậu kiểm mới quyết định có thay checkpoint cuối và cập nhật 7.14/7.15/7.16 hay không. Nếu P0 không cải thiện, giữ negative result; không quay lại chọn P1 hoặc đổi ngưỡng theo evaluation.
