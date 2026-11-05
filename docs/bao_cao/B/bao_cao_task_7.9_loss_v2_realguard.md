# Báo cáo Task 7.9 — RealGuardLoss và FeatureConsistencyLoss

Cập nhật review và sửa tích hợp: **2026-11-05**. Triển khai ban đầu: Thành viên B; phối hợp huấn luyện: Thành viên C.

**Trạng thái:** đã sửa và kiểm tra CPU; chưa xác nhận smoke với checkpoint CLIP thật trên T4 cho phiên bản hiện tại. Đây là nghiệm thu module loss và tích hợp cơ bản, chưa phải nghiệm thu toàn bộ Task 7.11/G7.2.

## Những vấn đề đã sửa

- KD và consistency từng bị nhân trọng số hai lần: `alpha_kd=.7` thành `.49`, `beta_cons=.2` thành `.04`. Các module cũ đã trả loss có trọng số; tổng mới cộng trực tiếp các giá trị đó.
- CE clean/degraded từng hard-code `.3/.3`, không tôn trọng config. Nay đọc `weight_ce` và `weight_ce_deg` trong chế độ v2.
- Feature cosine với vector zero FP16 từng trả NaN do `eps=1e-8` underflow; Guard với excess 400 từng trả Inf do bình phương FP16. Hai module mới tính toán FP32 cho đầu vào FP16/BF16 và detach nhánh clean/teacher đúng vị trí.
- Bật loss mới nhưng thiếu feature, teacher hoặc cặp ảnh nay báo lỗi rõ ràng, thay vì âm thầm bỏ thành phần loss.
- Guard kiểm tra index lớp và mask. Trainer xác minh mapping hiện có của dự án `REAL=0, FAKE=1`; guard-only không yêu cầu model hỗ trợ feature API.
- Test legacy trước đây so hai lần gọi cùng implementation. Nay đối chiếu với công thức CE/KD selective/MSE độc lập trên ba batch cố định.

## Công thức và API

Với cấu hình Phase 7 hiện tại, KD/Cons/Feat/Guard dưới đây là các giá trị **chưa nhân trọng số**:

```text
total = .3 CE_clean + .3 CE_deg + .7 KD + .2 Cons + .3 Feat + .5 Guard
```

`beta_cons=.2` giữ thêm logit consistency như cấu hình và bước tích hợp §5.3 của task; đặt `beta_cons=0` nếu thí nghiệm chỉ dùng năm thành phần ở công thức đầu task. Không nhân thêm `.7` hoặc `.2` vào các khóa `loss_kd`/`loss_cons` đã có trọng số.

Khi `weight_feat=weight_guard=0`, giữ nguyên công thức legacy:

```text
total = weight_ce * (.7 CE_clean + .3 CE_deg) + loss_kd + loss_cons
```

Không thay đổi `CrossDomainDistillationLoss` và `PairedConsistencyLoss`. Tên tham số cũ `logits`/`logits_degraded` tiếp tục hoạt động; có thêm alias `logits_clean`/`logits_deg` dùng bằng keyword. Không truyền cả hai alias cho cùng một đầu vào.

Các khóa trả về gồm `loss_total`, `ce_clean`, `ce_deg`, `kd`, `cons`, `feat`, `guard`; vẫn giữ các khóa `loss_*` cũ. `loss_ce`, `loss_kd`, `loss_cons`, `weighted_feat`, `weighted_guard` cộng lại thành `loss_total`; `loss_feat` và `loss_guard` là giá trị thô. Có thêm `feat_cos`, `feat_kl`, `guard_active`, `n_real`, `guard_excess_mean` để chẩn đoán.

## Đồng bộ với dự án

- Config [`train_phase7_anchored_v2.yaml`](../../../configs/train_phase7_anchored_v2.yaml) bật degradation v2 (Task 7.8), đọc manifest v2 (Task 7.7), chỉ lấy `val_select` cho validation. `val_calib` không được đưa vào loader chọn checkpoint.
- Warm start đổi về baseline `fatformer_4class_ckpt.pth`, phù hợp Step-0 identity và kế hoạch 7.11. Parser thực sự đọc đường dẫn model, tên checkpoint, LR adapter `1e-5`, CE degraded, AMP và gradient clipping từ YAML; CLI vẫn có quyền ghi đè.
- Vòng train chỉ bật `return_features` khi `weight_feat>0`; Guard dùng logit. Teacher được freeze, `eval()` và chạy trong `no_grad()`.
- Mất/hỏng ảnh thật báo lỗi; ảnh giả lập chỉ được phép trong dry-run. Không thay dữ liệu thiếu bằng ảnh màu khi train thật.
- Có `--max_steps` để giới hạn batch train xuyên các epoch. Gradient accumulation chuẩn hóa đúng cửa sổ cuối chưa đủ batch.
- Log v2 lưu tại `DATASET/logs/phase7/train_history.csv`, gồm các thành phần loss, cosine, KL, tỷ lệ Guard hoạt động và gradient norm trước clip. `n_real` trong metric epoch là số Real trung bình mỗi batch. Luồng legacy vẫn dùng lịch sử M6 cũ.
- Smoke CPU kiểm tra pipeline dry-run với v2; teacher giả lập được sao chép từ student trước audit Step-0. Kết quả giả lập không chứng minh tải checkpoint thật hoặc chất lượng phát hiện.

Wrapper `CLIPModel.forward(return_features=True)` đã có từ triển khai trước. Test mới gọi chính forward đó với encoder nhỏ để so sánh logit thường, dual và feature, đồng thời kiểm tra backward và state-dict keys. Không sửa cấu trúc model hay checkpoint trong lần review này.

## Kiểm tra và số đo

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest -q -p no:cacheprovider --basetemp scratch/pytest-task79-final-alias tests
```

Phạm vi: công thức độc lập, trọng số cấu hình, stop-gradient, scale KL theo batch, vector zero FP16, Guard overflow FP16, cấu hình không hợp lệ, input thiếu, alias API, config → loader → trainer với ảnh tạm thật, cách ly calibration, guard-only, forward CLIP wrapper, dry-run CLI và regression Task 7.2/7.8 cùng các test Phase 7 hiện có.

Lần chạy cuối: **85 passed in 69.49s**, không có test fail. Trong đó `test_loss_v2.py` có 17 trường hợp và `test_smoke_loss_v2_integration.py` có 9 trường hợp; các trường hợp còn lại là regression của dự án.

Smoke CPU có seed 42 lưu ở [`task_7.9_cpu_smoke.json`](../../../DATASET/logs/phase7/task_7.9_cpu_smoke.json): hai batch model nhỏ, loss tổng `0.452720`, cosine `0.9999996`, gradient norm trung bình `0.639332`, tỷ lệ batch có Real cho Guard `0.5`; teacher đóng băng và không nhận gradient. Ví dụ Guard: student Fake logit `1.5`, teacher `.3` cho loss `1.44`, gradient student Fake `+2.4`, gradient teacher không có. KL rất nhỏ có thể âm cỡ `1e-8` do làm tròn FP32.

Báo cáo trước ghi 4 test và forward feature `(1,768)` trên Colab CPU ngày 2026-11-01. Các số đó thuộc phiên bản trước review, không thay cho kiểm tra lại với checkpoint thật sau sửa.

## Bước xác nhận còn lại

Máy kiểm tra hiện tại là Windows/PyTorch CPU, CUDA không khả dụng. Sau khi stage đầy đủ ảnh theo manifest và checkpoint trên máy GPU:

```bash
python -m src.training.train_crossdomain --config configs/train_phase7_anchored_v2.yaml --max_steps 20
```

Xác nhận loss/gradient hữu hạn, Guard có batch hoạt động, feature cosine và đầu ra checkpoint. Logic selection worst-group, preflight/cổng dữ liệu, resume và benchmark GPU thuộc các Task 7.10/7.11; các test CPU ở đây không chứng minh những hạng mục đó đã hoàn tất.
