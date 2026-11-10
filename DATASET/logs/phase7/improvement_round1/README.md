# Phase 7 — vòng cải thiện 1, lấy code từ Git nhánh `hoang`

Cập nhật 10/11/2026. **Đã hậu kiểm pilot train và matched evaluation trên L4. P0 chưa cải thiện đủ để thay model: G7.2 FAIL 4/8, đối chứng cũ hiệu chuẩn lại FAIL 6/8.** Kết quả ở `../task_7.15/current/` giữ nguyên: G7.2 FAIL 6/8, G7.3 pending verified best-by-condition reference.

## Kết quả mới nhất

Evaluation [`p0_eval_20261110_065901`](evaluation/runs/p0_eval_20261110_065901/README.md), source commit `977b282`: notebook chạy hoàn tất, đã nhập 90 file nguyên byte và bản sao notebook. Chỉ số/từng subset, 17 step markers và paired bootstrap tái lập; 2.446 phép kiểm tra dữ liệu đạt. Clean Fake@Real90 P0 61,00% so với 64,125% của đối chứng; Q30 38,875% so với 42,625%; macro in-dist 89,6794% so với 90,1018%. P0 có một số chỉ số phụ tốt hơn, nhưng không đạt mục tiêu thay thế. Dự đoán của đối chứng refit giống hoàn toàn arrays deployment cũ; hash gói refit khác nên không thay package hiện hành. Xem [quyết định](evaluation/runs/p0_eval_20261110_065901/analysis/decision.json).

Pilot train [`pilot_20261109_182359`](runs/pilot_20261109_182359/README.md), source commit `2665014`: P0 worst-group **42,00%**, P1 **35,33%**; cả hai chọn epoch 5. P0 được chọn đúng theo validation và đã được đánh giá tiếp. Kết quả seed này chưa hỗ trợ giả thuyết đổi preprocessing train sang CLIP sẽ tốt hơn; worst-group validation không thay chỉ số test ở đoạn trên.

Đã nhập 27 file từ `wait` vào `runs/<run_id>/raw/`, lưu notebook đã chạy vào `evidence/`, tái tính số cập nhật từ 3.830 record train và đối chiếu 146 source hashes. Có 28.832 phép kiểm tra dữ liệu đạt, không phải số lượng unit test. P0 bỏ qua 1 update, P1 bỏ qua 2; không có update áp dụng norm không hữu hạn. Weights không có trong gói nộp, nên chưa được băm lại hoặc forward tại local.

Xem [hậu kiểm chi tiết](runs/pilot_20261109_182359/README.md), [JSON kiểm tra](runs/pilot_20261109_182359/analysis/verification.json), [tham chiếu model được chọn](runs/pilot_20261109_182359/candidate_reference.json). Các mục dưới đây giữ thiết kế và hướng dẫn chạy; không cần train lại chỉ để hoàn thiện hồ sơ vòng này.

[Hướng dẫn calibration/benchmark P0 và incumbent cùng protocol](../../../../notebooks/task/phase7_pilot_evaluation/README.md) là hồ sơ cách chạy đã dùng. Không cần chạy lại cùng thí nghiệm để chốt vòng này. Muốn nghiên cứu tiếp cần giả thuyết và thiết kế mới trên train/validation; G7.3 vẫn cần baseline từng điều kiện có bằng chứng hợp lệ. Phần hướng dẫn bên dưới là thiết kế và lịch sử chạy, không phải yêu cầu chạy lại.

## 1. Những vấn đề đã xác định

- Log train cũ có `grad_norm=NaN` ở epoch 1 và 4; không có bộ đếm AMP skipped updates nên chưa xác định được nguyên nhân. Không sửa log cũ hoặc suy ngược số cập nhật bị mất.
- Train/validation cũ dùng stretch vuông + ImageNet normalization; benchmark deployment dùng resize cạnh ngắn bicubic + center crop + CLIP normalization. Đây là khác biệt thật, chưa chứng minh là nguyên nhân điểm thấp.
- Manifest có 1.200 ảnh autoregressive: 1.020 train, 90 val_calib, 90 val_select, tất cả mang nhãn `tamingTransformers`. Không thể kết luận train thiếu cả họ autoregressive hoặc đã phủ đủ generator của subset CFE khó.

## 2. Thiết kế pilot đã chốt

- `p0_legacy_full`: train mới bằng tiền xử lý cũ, không lấy A4 cũ thay control.
- `p1_clip_full`: train mới bằng CLIP preprocessing tương ứng deployment.

Cả hai validation bằng **cùng CLIP preprocessing**, cùng IDs/ngưỡng/T baseline đóng băng. Giữ seed 42, 5 epoch/nhánh, dữ liệu, khởi tạo/teacher và loss A4. Batch mặc định 16 × accumulation 2 = 32; có thể chọn 8 × 4 hoặc 4 × 8 trước khi chạy và áp dụng cho cả hai nhánh.

Chọn checkpoint trên `val_select`: worst-group cao nhất với Clean Real ≥90%; hòa điểm chọn `p0`. Worst-group là min của bốn recall clean Real/Fake và Q30 Real/Fake, không phải worst-generator của CFE. Không dùng CFE evaluation để chọn nhánh, epoch hoặc seed. Đây là pilot hai nhánh, không thay ablation bốn nhánh của Task 7.12.

Thiết kế: `preparation/protocol.json`. Input/hash: `preparation/expected_inputs.json`. Audit: `preparation/audit.json`. Cấu hình tham khảo: `preparation/planned_configs/`; runner tạo cấu hình runtime từ code đã checkout.

## 3. Điều kiện kỹ thuật trước khi train dài

Runner kiểm tra ảnh/hash/isolation/split, canonical baseline và các test liên quan. **Cả hai nhánh phải qua smoke 100 mini-batch + 5 warmup trước khi train dài bất kỳ nhánh nào.**

`optimizer_updates.jsonl` lưu từng lần optimizer cập nhật hoặc bị AMP bỏ qua, AMP scale, norm, epoch/minibatch. Bộ đếm dùng optimizer post-step hook với non-fused AdamW; không suy từ giá trị trả về `None` của `step()`.

Chính sách kỹ thuật: có cập nhật dùng được, số đếm khớp, skip fraction phần đo ≤5%, không có cập nhật với norm không hữu hạn. Warmup lưu riêng và có thể cho AMP tự giảm scale. 5% ở đây là chính sách ổn định của vòng mới, không phải FPR hoặc tiêu chí chất lượng G7.2. OOM/gate lỗi thì dừng, lưu log và không train dài.

## 4. Lấy code từ Git trên Colab

Tạo notebook Colab mới, chọn GPU CUDA đủ bộ nhớ. Cell đầu, dành cho runtime mới:

```python
!git clone --branch hoang --single-branch https://github.com/LeeVHoangtk3/fatformer-xla.git /content/fatformer-xla
```

Cell thứ hai:

```python
from pathlib import Path
import subprocess

REPO = Path('/content/fatformer-xla')
PILOT_REPO = str(REPO)
PILOT_COMMIT = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True
).strip()
PILOT_STAGE = 'smoke'
PILOT_BATCH_SIZE = 16
PILOT_GRAD_ACCUM = 2
exec(compile(
    (REPO / 'notebooks/task/phase7_improvement_round1/start_colab.py').read_text(),
    'start_colab.py', 'exec'
))
```

Cell lưu chính xác commit đã checkout, không tự `pull` giữa smoke và train. Nếu có runtime/repo cũ, dùng runtime mới hoặc chủ động checkout đúng commit trước; không clone đè hoặc ép reset thay đổi chưa lưu.

Runner chỉ nhận source/config/input sạch so với Git, trên nhánh `hoang` hoặc một commit được checkout detached còn thuộc lịch sử `origin/hoang`. `source_identity.json` ghi commit và hash file thực tế. Đổi code/commit trong cùng run sẽ bị từ chối; SHA là dấu vết kiểm tra, không phải chữ ký số.

## 5. Input Drive và chạy train

Giữ nguyên cấu trúc Drive của Task 7.11:

- `MyDrive/Fatformer/phase7/train_crossdomain_v2.tar`.
- `MyDrive/Fatformer/pretrained/ViT-L-14.pt`.
- `MyDrive/Fatformer/pretrained/fatformer_4class_ckpt.pth`.

Cell kiểm hash, stage ảnh lên SSD Colab, cài thư viện hỗ trợ còn thiếu rồi chạy preflight/smoke. Không thay bản torch/torchvision của Colab. Kiểm hash archive/weights có thể mất vài phút. GPU không bắt buộc là T4; phép backward smoke xác nhận cấu hình có vừa bộ nhớ hay không.

Sau khi smoke thành công, giữ nguyên `PILOT_RUN_ID` và `PILOT_COMMIT` được in ra. Trong cùng phiên chạy:

```python
PILOT_STAGE = 'train'
exec(compile(
    (REPO / 'notebooks/task/phase7_improvement_round1/start_colab.py').read_text(),
    'start_colab.py', 'exec'
))
```

Runner kiểm tra lại trước khi train. Nếu OOM, tạo ID mới, đặt batch 8/accumulation 4 rồi smoke lại cả hai nhánh. Không đổi batch giữa thí nghiệm. Nếu chuyển runtime để resume, clone và checkout đúng commit đã ghi, dùng cùng đường dẫn/cấu hình và full snapshot; context khác GPU/framework/horizon sẽ yêu cầu run mới. Trace epoch dở được giữ riêng, không cộng trùng khi resume.

## 6. Kết quả và hậu kiểm

```text
MyDrive/Fatformer/phase7/improvement_round1/<PILOT_RUN_ID>/
  source_identity.json
  status.json
  p0_legacy_full/   # config, identity, smoke, CSV, trace, best và latest
  p1_clip_full/
  selection.json   # chỉ sau khi cả hai train/verify hoàn tất
  pilot_results_without_weights.zip
```

ZIP ở đây chỉ là **ZIP kết quả để tải về**, không phải ZIP cấp code. Checkpoint/full-state resume được backup theo epoch và không nằm trong ZIP kết quả. Local Colab: `/content/phase7_pilot_runs/<PILOT_RUN_ID>/`. Lưu notebook mới; giải nén kết quả vào `wait/improvement_round1/<PILOT_RUN_ID>/` để hậu kiểm.

Sau pilot, khóa lựa chọn từ validation trước khi hiệu chuẩn/đánh giá. Khi có model được chọn, tạo gói calibration/evaluation gắn đúng hash mới. Notebook `phase7_corrected_rerun_colab.ipynb` cũ khóa checkpoint `5cd49252…`, không tự nhận model mới. Chỉ cập nhật 7.14/7.15/7.16 sau đo và hậu kiểm; không ghi đè kết quả hiện hành dựa vào điểm validation.

## 7. File và kiểm thử

- [Cell khởi động Git](../../../../notebooks/task/phase7_improvement_round1/start_colab.py).
- [Runner](../../../../tools/phase7_training_pilot.py).
- [Kiểm tra commit/source](../../../../tools/phase7_git_source.py).
- [Chuẩn bị metadata](../../../../tools/prepare_phase7_training_pilot.py): không tạo ZIP code.
- `preparation/validation/git_workflow_tests.xml`: kiểm thử workflow Git và training liên quan, không phải bằng chứng GPU.

32 notebook và 55 file raw lịch sử được bảo vệ bằng hash. Các ZIP code/bundle manifest local đã tạo trước đây chỉ là hồ sơ của phương thức cũ, không cần upload hoặc đưa vào Git. Không hạ gate hoặc tuyên bố chất lượng tăng khi chưa có phép đo mới.
