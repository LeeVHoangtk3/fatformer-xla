# Task 7.15 — sửa hội tụ nhiệt độ, bản local ngày 09/11/2026

**Đã sửa code, fit lại T/ngưỡng từ đúng 400 logits thực nghiệm và đóng gói thành bản mới.** Đây là xác nhận sau hậu kiểm, chưa nghiệm thu toàn bộ G7.3 và chưa thay thế artifact cuối dùng trong báo cáo. Các notebook và run người dùng đã nộp được giữ nguyên.

## Thay đổi và kết quả

`tools/calibrate_and_select.py` thay vòng L-BFGS dừng sớm bằng bộ giải đạo hàm NLL theo nghịch đảo nhiệt độ. NLL lồi theo biến này; kiểm tra dấu đạo hàm chọn được nghiệm trong khoảng hoặc nghiệm biên hợp lệ. Bộ giải kiểm tra điều kiện tối ưu và NLL trước khi ghi kết quả; hết vòng lặp mà chưa hội tụ sẽ báo lỗi. JSON mới ghi `temperature_optimization` để truy vết. Không thay backbone hoặc dùng evaluation để chọn T.

- T cũ **1,46759939** → T mới **2,78635955**; hội tụ sau **35 vòng**.
- Ngưỡng 5% cũ **0,92936361** → mới **0,79532111**. Fit lại cả ngưỡng 1%/5%/10%; không giữ ngưỡng cũ khi đổi T.
- NLL calibration **0,5013 → 0,4230**; ECE **11,70% → 6,06%**.
- Real/Fake ACC calibration vẫn **95%/59%**, FPR vẫn **10/200 = 5%**. Đây là cải thiện hiệu chuẩn, không phải tăng khả năng phân biệt hay chứng minh chất lượng held-out.
- T độc lập bằng tối ưu số một biến cho cùng NLL xác nhận kết quả của bộ giải mới.

Nguồn NPZ là cache FP32/batch 32 mới nộp, hash `3b0f66dfdc7c4655df6aa7fcab7346f0315d2663927559b66005bafa2c657942`. Phân vùng 400/1.600 không trùng ID. Không huấn luyện lại, không sửa ảnh, không đổi nhãn hoặc hash nguồn để ép nghiệm thu.

## Artifact và kiểm chứng

- [Calibration mới](calibration/anchored_v2.json), [reliability](calibration/anchored_v2_reliability.png).
- [Package mới](artifacts/unified_fatformer_v2.pth), [metadata](artifacts/metadata.json).
- SHA256 package: **`16606e16c89901f39ab78c29951c1f83d9412f98f27ab55853c2e2dd13ee5f74`**.
- [Kiểm chứng package](verification/package_verification.json): đủ **1.200 tensor**, toàn bộ tensor bằng package nguồn `8ddcb401…`; metadata có calibration mới. Nạp standalone khi CLIP path không tồn tại, kiểm tra một tensor tổng hợp CPU: sai số xác suất **0**, nhãn khớp reference cùng T/ngưỡng.
- **116 tests pass** trong [JUnit XML](verification/tests.xml), gồm nghiệm có công thức giải, nghiệm biên, dữ liệu không hợp lệ, chưa hội tụ, export calibration, phân vùng, router fallback, packaging và pipeline Colab. Ba warning router bị tắt là hành vi dự kiến.
- Bundle đã giải nén vào thư mục mới, kiểm tra hash đủ 140 tệp và chạy **38 tests pass từ chính code trong ZIP**; xem [JUnit của bản export](verification/export_tests.xml). Không cộng hai lần các test trùng nhau thành 154 test riêng biệt.
- [Kiểm chứng hiệu chuẩn độc lập và trạng thái còn thiếu](verification/repair_validation.json), [phân vùng](verification/partition_audit.json), [manifest](artifact_manifest.json), [snapshot code sửa](code_snapshot/).

Package mới được **repackage từ bản nộp** đã kiểm tra. Hash checkpoint huấn luyện `5cd49252…` được kế thừa đúng nguồn; không tuyên bố đã băm lại file huấn luyện trên Drive trong phiên local. Metadata lưu riêng hash container nguồn và việc repackage.

`verification/historical_logits_rescore.csv` chỉ là tính lại điểm từ cache backbone 7.14 với calibration mới. Không gọi nó là lần chạy `package.predict()` mới, không dùng các điểm này để chọn calibration, không tự đổi phán quyết G7.2/G7.3.

## Chạy tiếp trên Colab bằng đúng code đã sửa

Sau khi gom thư mục, chỉ dùng đường dẫn `DATASET/logs/phase7/task_7.15/current/`. `code_snapshot/` và các đường dẫn trong JSON kiểm chứng ghi nhận thời điểm sửa nhiệt độ trước khi đổi bố cục; tra `../PATH_MIGRATION.json` khi đọc bằng chứng cũ. Code để chạy hiện tại nằm trong ZIP `colab/` đã xuất lại theo đường dẫn mới, không lấy lệnh từ các archive.

Đã chuẩn bị [phase7_corrected_colab_bundle.zip](colab/phase7_corrected_colab_bundle.zip) kèm SHA256. Gói gồm snapshot code sửa và inputs đã xác minh, không kèm weights 2 GB hoặc datasets. Không cần commit/push để Colab nhận bản sửa.

1. Đưa **ZIP mới trong thư mục này** vào `MyDrive/Fatformer/logs/phase7/phase7_corrected_colab_bundle.zip`. Notebook ưu tiên file tại đường dẫn này, vì vậy phải kiểm tra không còn nạp ZIP cũ. Khi chạy, đối chiếu SHA256 in ra với tệp `.zip.sha256` cạnh ZIP mới.
2. Mở notebook hiện có `notebooks/task/phase7_corrected/phase7_corrected_rerun_colab.ipynb` và chạy từ đầu với GPU CUDA đủ VRAM. **Không sửa nội dung notebook**. Notebook này nạp ZIP snapshot thay cho clone GitHub.
3. Workflow kiểm tra dataset/VRAM, chọn batch vừa bộ nhớ; dùng 400 calibration để fit, kiểm chứng 20 ảnh thật, đóng gói, chấm đúng 1.600 evaluation và in-dist, rồi đo latency cùng package. Giữ nguyên FAIL/pending nếu thiếu reference hoặc không đạt tiêu chí.

**Phân biệt protocol:** bản local ở đây dùng logits FP32/softmax CPU, có thể tái sử dụng khi giữ protocol đó. Notebook tổng hợp dùng FP16/CUDA để đồng bộ accuracy với latency và sẽ forward lại đúng 400 calibration theo batch/precision đã chọn. Bước này không phải huấn luyện lại. T/ngưỡng/package hash trên GPU có thể khác bản local; không ép số phải bằng nhau và không ghép số đo của hai hash.

Dữ liệu Drive cần checkpoint gốc `checkpoint/phase7/anchored_v2_run1/fatformer_anchored_v2_best.pth` và ba archive `datasets/test_communityforensics.tar`, `datasets/test_benchmark_gans.tar`, `datasets/test_benchmark_diffusion.tar`. Hướng dẫn chi tiết về workflow nằm ở `notebooks/task/phase7_corrected/README.md`.

Để xuất lại ZIP từ repo, giữ đường dẫn inputs đúng run này:

```sh
python -m tools.build_phase7_colab_bundle --calibration_logits DATASET/logs/phase7/task_7.15/current/inputs/calibration_400.npz --calibration DATASET/logs/phase7/task_7.15/current/calibration/anchored_v2.json --partition_audit DATASET/logs/phase7/task_7.15/current/verification/partition_audit.json --output DATASET/logs/phase7/task_7.15/current/colab/phase7_corrected_colab_bundle.zip
```

## Còn chờ, không tự gán PASS

Chưa chạy lại GPU trong phiên local, chưa có kiểm tra ≥20 ảnh thật hay benchmark package mới/reference trên evaluation. Latency L4 **0,119 ms overhead** thuộc package cũ `8ddcb401…`, không đưa sang package mới. G7.3 của bản mới vẫn **pending**. Chỉ cập nhật Task 7.16/báo cáo cuối sau khi nhận đúng các bằng chứng còn thiếu với cùng hash package.
