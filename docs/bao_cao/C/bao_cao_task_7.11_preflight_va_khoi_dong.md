# Rà soát Phase 7 và chuẩn bị khởi động Task 7.11 — bản sửa

Cập nhật: **2026-11-05**. Tài liệu này thay các kết luận “100% sẵn sàng”, “8/8 PASS” và “tự resume khi chạy lại Cell 6” của bản rà soát trước. Chưa chạy huấn luyện CLIP trên GPU trong lần sửa này.

## Trạng thái có bằng chứng

Manifest hiện có 14.400 records: **12.240 train, 1.080 val_calib, 1.080 val_select**. Không phải train trên toàn bộ 14.400 ảnh. Manifest byte-for-byte khớp fingerprint trong báo cáo audit lưu ở repo. Điều này không chứng minh ảnh đã được stage hay GPU đã sẵn sàng.

Audit local mới ghi tại [`preflight_train.json`](../../../DATASET/logs/phase7/preflight_train.json) có `ready_to_train=false`:

- 14.400/14.400 ảnh chưa có ở data root local.
- Config dữ liệu local không khớp SHA256 của config dùng khi audit. Phải stage config gốc trong archive đã audit, không chỉnh báo cáo audit để ép khớp.
- Loss, selection và kiểm tra tách ba split theo path/SHA256 chạy được.
- Không có CUDA; canonical model load trong lượt audit nhanh được ghi **SKIP**, không phải PASS.

Benchmark 7.8 hiện là **+14.08% trên corpus CPU tổng hợp**, thay cho -14.69% cũ. Selection 7.10 có benchmark CPU khoảng **1.046×** clean-only; cả hai không thay số đo model/dữ liệu thật trên GPU. Loss legacy được đối chiếu allclose `atol=1e-6`, không gọi đây là bằng chứng bit-exact trên mọi môi trường.

## Preflight đã sửa

Chế độ train bắt buộc đủ tám gate, exit code khác 0 nếu có gate fail. Công cụ kiểm tra mọi ảnh tồn tại, giải mã được và đúng SHA256; audit phải ghi rõ pass, zero violations, đủ coverage và gắn với đúng manifest/config. Kiểm tra ba split bằng cả path lẫn SHA256, đủ hai nhãn; đọc calibration đo thật, baseline/teacher khớp nhau và fingerprint calibration; canonical load phải thực sự trả 0/0. Gate CUDA yêu cầu **14 GiB VRAM trống**, không đánh dấu máy CPU là PASS.

```bash
python -m tools.preflight_phase7_train --config /content/phase7_runtime/train_runtime.yaml --output /content/phase7_runtime/preflight_train.json
```

Chế độ audit nhanh có thể bỏ chạy test hoặc tải model lớn nhưng chỉ ghi SKIP, luôn `ready_to_train=false`. Bypass gate bị từ chối trong chế độ train:

```bash
python -m tools.preflight_phase7_train --mode audit --skip-model-load --output DATASET/logs/phase7/preflight_train.json
```

Report lưu mode, cấu hình/hash, data root, source hash, phiên bản Python/Torch/CUDA và trạng thái từng gate. Exit 0 của audit chỉ có nghĩa audit không gặp FAIL; không phải quyền bắt đầu train. Notebook chỉ nhận `ready_to_train=true` ở mode train.

## Benchmark tách khỏi train

```bash
python -m src.training.train_crossdomain --config configs/train_phase7_anchored_v2.yaml --benchmark_steps 100 --benchmark_warmup 5 --output DATASET/logs/phase7/train_benchmark_100steps.json --save_dir scratch/phase7_benchmark
```

Benchmark chạy forward/backward/optimizer thực trong process riêng; không chạy epoch-0 validation, không ghi checkpoint, train_history hay val_groups. Với nhiều worker, warmup và phần đo dùng chung pool; train chính vẫn dùng worker không persistent để khôi phục RNG theo epoch. JSON ghi mini-batch steps, ms/step, peak allocated/reserved VRAM, steps/epoch, ước lượng phút/epoch/tổng, device, source/input fingerprints và evidence level. Số phút không bao gồm staging, preflight, validation hoặc backup Drive.

`--max_steps 100` vẫn là train giới hạn, **không phải benchmark**. Không dùng cùng run directory cho thử nghiệm/smoke và lượt train chính. Benchmark không được kết hợp resume hoặc max_steps; nếu warmup/steps vượt số batch của loader thì báo lỗi.

## Resume từ epoch hoàn tất

```bash
python -m src.training.train_crossdomain --config configs/train_phase7_anchored_v2.yaml --resume_from DATASET/checkpoints/phase7/checkpoint_latest.pth
```

Latest mới có schema `phase7-epoch-resume/v1`, chứa model, optimizer, scheduler, scaler, global mini-batch step, RNG Python/NumPy/Torch/CUDA/DataLoader, hai kỷ lục selection, lịch sử và context của run. Khôi phục từ epoch hoàn tất gần nhất rồi chạy epoch kế tiếp. Nếu bị ngắt giữa epoch, phần epoch đó được lặp lại; không tuyên bố tiếp tục đúng mini-batch bị ngắt.

- Ghi latest bằng file tạm rồi replace; lỗi ghi không xóa latest cũ.
- Resume bỏ Step-0 identity/epoch-0 validation vì student đã được train; dùng strict model load và khôi phục LR/scheduler thay vì khởi tạo lịch mới.
- Context kiểm tra code, fingerprints đầu vào thực sự được chọn, Torch/CUDA, horizon epoch, loss, batch, AMP và operating point. Đổi thí nghiệm phải dùng run mới.
- Log được khôi phục từ checkpoint để sửa ghi dở, không thêm epoch 0 hoặc nhân đôi dòng khi resume. History nằm trong run directory, gồm loss/group, LR và thời gian; có `--history_csv` để đổi rõ ràng.
- Official best và best-effort vẫn là model-only cho inference. File model-only cũ và snapshot partial-epoch từ `--max_steps` bị từ chối làm resume.
- Backup kiểm tra checksum, ghi latest và sidecar SHA256; lỗi backup không được báo thành công âm thầm. Không cam kết bit-exact khi đổi phần cứng GPU/framework.

Test CPU đối chiếu chạy liền mạch với ngắt/khôi phục cho kết quả trùng trọng số, optimizer, scheduler và chuỗi RNG; có test resume qua pipeline, sửa log, checkpoint lỗi ghi, scaler serialization và benchmark không tạo artifact train. Đây không phải smoke GPU/checkpoint CLIP thật.

## Notebook Colab

[`task_7.11_train_anchored_v2_colab.ipynb`](../../../notebooks/task/task_7.11/task_7.11_train_anchored_v2_colab.ipynb) dùng lệnh `subprocess.run(check=True)`, không tiếp tục âm thầm sau lỗi shell:

1. Chọn CUDA đủ VRAM và mount Drive. Dùng RUN_NAME mới cho thí nghiệm mới.
2. Checkout commit cố định; run tiếp dùng commit đã lưu. Phải publish commit chứa bản sửa trước khi mở notebook từ Git; code cũ thiếu CLI sẽ bị từ chối.
3. Kiểm tra SHA256 của archive trước giải nén vào SSD bằng extraction filter. Lấy manifest/report/config gốc từ archive; dùng `data_root` để giải quyết path, không viết lại manifest làm hỏng audit fingerprint.
4. Chạy preflight train thật.
5. Run mới phải có benchmark GPU đo thật cùng code/input/hyperparameters. Nếu ngân sách T4 vượt 300 phút train, chỉ cắt 5→4 epoch theo quy tắc; vẫn vượt thì dừng để quyết định ngân sách. Resume giữ nguyên horizon đã lưu.
6. Chạy train hoặc copy full latest đã kiểm checksum từ Drive rồi truyền `--resume_from`. Run có latest nhưng không chỉ định resume bị từ chối ghi đè. Không dùng benchmark làm checkpoint khởi tạo.
7. Đọc CSV và vẽ hai panel (group recalls, loss components); copy chart về Drive.

Artifact Drive mới tách theo run: `Fatformer/checkpoint/phase7/<RUN_NAME>/`. Bản local ở `/content/phase7_runtime/runs/<RUN_NAME>/`. Chạy lại notebook chọn full latest đã có; legacy latest thiếu checksum/schema không được giả làm resume.

## Giới hạn nghiệm thu

Kiểm chứng local sau sửa: toàn bộ suite **131 passed** (73,52 giây). Audit chạy lại có 4 PASS, 2 FAIL (thiếu ảnh, config không khớp audit), 2 SKIP (canonical load, CUDA); `ready_to_train=false`. Hai gate test trong audit chạy riêng có 26 và 31 ca PASS. Notebook có 7 code cell, được kiểm tra cú pháp và cấu trúc lệnh bằng test; chưa chạy Colab thực tế.

Đây là sửa workflow chuẩn bị, không phải báo cáo hoàn tất train 7.11 hay G7.2. Chưa có số đo GPU, epoch hội tụ, checkpoint cuối verify 0/0 hoặc benchmark locked test mới. Ước lượng T4/A100 và thời lượng platform trong báo cáo cũ không phải số đo. Cấu hình mini-batch 16, accumulation 2 hiện cho effective batch 32; warmup/early-stop trong kế hoạch vẫn cần nghiệm thu theo run thực, không được suy là đã đạt vì preflight PASS.
