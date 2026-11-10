# Calibration transfer — vòng cải thiện 2

Trạng thái 10/11/2026: **đã hậu kiểm run `calib_transfer_20261110_155948`**; fit đúng split/budget nhưng hai point clean-only mới không đạt guard Real Q30≥90%. Xem [kết quả](../../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/README.md). Không cần chạy lại y nguyên. File notebook trong repo vẫn là template; chưa có bản thực thi để đối chiếu output. Các hướng dẫn dưới đây giữ để tái lập, không phải yêu cầu train hoặc thay package `24869052…`.

Mở [notebook đã tạo](phase7_calibration_transfer_colab.ipynb) trên Colab, chọn GPU CUDA đủ bộ nhớ và chạy ba code cell theo thứ tự. Dùng runtime mới để giữ repo/run cũ nguyên vẹn. Notebook lấy code Git nhánh `hoang` và khóa source đã kiểm thử **`253942a5262432fbed7ce7ca8b12b4e455bfc2ce`**; không cần ZIP code hoặc sửa notebook diagnostics đã chạy.

[Mở trực tiếp trên Google Colab](https://colab.research.google.com/github/LeeVHoangtk3/fatformer-xla/blob/hoang/notebooks/task/phase7_calibration_transfer/phase7_calibration_transfer_colab.ipynb).

## Input Drive

```text
MyDrive/Fatformer/
  phase7/train_crossdomain_v2.tar
  phase7/improvement_round1/pilot_20261109_182359/p0_legacy_full/best.pth
  checkpoint/phase7/anchored_v2_run1/fatformer_anchored_v2_best.pth
  phase7/improvement_round2/diagnostics/val_diag_20261110_142828/
    identity.json
    status.json
    candidate_p0/predictions_clean.npz
    candidate_p0/predictions_q30_post_crop.npz
    incumbent/predictions_clean.npz
    incumbent/predictions_q30_post_crop.npz
    ...các JSON/CSV/log còn lại của run đã hậu kiểm...
```

Nếu thư mục reference chỉ còn `validation_diagnostics_without_weights.zip`, runner đọc các file đã khóa từ ZIP này. Không cần chạy lại diagnostics hoặc upload dataset/weights vào Git. Thiếu/sai một file thì dừng và in đường dẫn cụ thể; không thay bằng dummy hoặc đoán một run khác.

Control calibration CFE đã được sao nguyên byte vào [`configs/phase7_calibration_transfer/controls/`](../../../configs/phase7_calibration_transfer/controls/), cùng hash/checkpoint tương ứng. Toàn bộ input được khóa trong [plan](../../../configs/phase7_calibration_transfer/plan.json).

## Ba cell làm gì?

1. Kiểm tra CUDA và VRAM.
2. Clone Git `hoang`, checkout commit đã khóa và xác nhận thuộc lịch sử nhánh. Không ghi đè repo đã tồn tại hoặc `pull` giữa phiên chạy.
3. Mount Drive, hash/extract archive lên SSD Colab, kiểm nguồn reference/weights/ảnh và probe cả hai checkpoint. Trích xuất logits **clean `val_calib` 1.080 ảnh** cho mỗi model, fit T và threshold FPR≤5%, rồi so với CFE calibration sẵn có trên `val_select` clean/Q30.

Dataset v2 có 590 real/490 fake ở mỗi split. Fit chỉ dùng `val_calib`, không dùng labels/scores `val_select` để chọn ngưỡng. Comparator `>=` và float32; 5% của 590 cho phép tối đa **29 false positives**, tức khoảng 4,9153%, không phải làm tròn thành 30. Một ngưỡng fit clean được áp dụng cho cả clean/Q30; không có router hoặc cờ chọn ngưỡng theo điều kiện.

Runner chọn một batch chung trong [1,16], FP16 backbone/float32 probabilities, TF32 tắt. Chỉ tái dùng raw validation logits đã hậu kiểm khi **inference source, tên GPU, Torch/CUDA và batch cùng khớp**. Nếu khác bất kỳ điều kiện nào, tự đo lại validation clean/Q30 cho **cả hai model** trước khi chấm hai operating point. Vì vậy có thể dùng T4/L4/A100 hoặc GPU CUDA phù hợp khác; trường hợp phải đo lại sẽ tốn thêm thời gian forward, vẫn không train. Nếu OOM/lỗi, giữ kết quả và dùng runtime/run ID mới, không giảm batch giữa run.

Đây là các điều kiện môi trường mà run cũ ghi nhận, không chứng minh mọi thư viện tiền xử lý đều giống nhau. Khi reuse, kết luận là so hai operating point trên cùng fixed logits đã có; chưa phải kiểm chứng lại toàn bộ pipeline ở mọi môi trường.

## Kết quả và cách đọc

```text
MyDrive/Fatformer/phase7/improvement_round2/calibration_transfer/<TRANSFER_RUN_ID>/
  identity.json
  plan.json
  batch_contract.json          # reuse_frozen_logits hoặc remeasure_both_models
  status.json
  reference_validation/        # snapshot 16 file đầu vào nhỏ, không có weights
  candidate_p0/
    calibration_logits.npz
    validation_logits_clean.npz
    validation_logits_q30_post_crop.npz
    comparison/
      v2_calibration.json      # research schema; deployable=false
      summary.csv              # hai point, hai điều kiện
      paired_deltas.csv        # V2 minus CFE control + CI
      group_metrics.csv
      errors.csv
      comparison.json
      cfe_control_clean.npz     # cùng các arrays chấm điểm khác
  incumbent/                   # cấu trúc tương tự
  report.json
  calibration_transfer_without_weights.zip
```

Đọc `summary.csv` của từng model: so `cfe_control` và `v2_clean_fit` trên **cùng logits validation**, xem recall Real/Fake, FPR, NLL/ECE và nhóm lỗi. CI paired 1.000 lần/seed42 có điều kiện trên hai fit đã có, không bao gồm độ bất định của fit calibration hoặc phụ thuộc generator. ECE là confidence ECE 10 bins với argmax 0,5, không phải ECE của quyết định ở ngưỡng vận hành.

`raw_margin_auc_pct` được tính từ cùng raw logits, không thay bởi calibration. `score_auc_pct` có thể đổi do ties/bão hòa float32; không gọi đó là cải thiện biểu diễn. Thí nghiệm đồng thời đổi domain, cỡ mẫu và tỷ lệ lớp calibration; chưa tách nguyên nhân của ba yếu tố này. Real guard là chẩn đoán, không phải G7.2/G7.3 PASS. File calibration mới dùng schema nghiên cứu riêng, chưa được xuất vào deployment.

Sau khi chạy, lưu notebook có outputs thành:

```text
notebooks/task/phase7_calibration_transfer/phase7_calibration_transfer_executed.ipynb
```

Giải nén ZIP kết quả (hoặc chép folder Drive) vào:

```text
wait/calibration_transfer_without_weights/
```

Gửi lại kết quả để hậu kiểm trước khi quyết định augmentation hay model tiếp. `val_select` đã dùng chọn P0 và chẩn đoán; đây không phải test mù mới. Không thay package hoặc chạy lại 7.14/7.15 ở bước này.
