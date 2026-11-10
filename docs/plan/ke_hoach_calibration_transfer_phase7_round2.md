# Phase 7 round 2 — calibration transfer có đối chứng

Cập nhật 10/11/2026. **Đã chạy và hậu kiểm calibration transfer; điểm clean-only mới không được chọn vì trượt guard Real Q30≥90%.** [Kết quả](../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/README.md) ghi đầy đủ tradeoff; thiết kế dưới đây được giữ nguyên như đã khai báo trước run. Không train hoặc thay model `24869052…`; G7.2 FAIL/đạt 6/8, G7.3 pending.

## Câu hỏi thực nghiệm

Trên cùng checkpoint và validation logits, calibration fit trên clean `val_calib` v2 có thay đổi tradeoff Real/Fake, FPR và confidence so với calibration CFE hiện có không? Đây là thử recipe calibration khác, không chứng minh backbone đủ năng lực hoặc giải thích nguyên nhân CFE. Domain, cỡ mẫu và tỷ lệ lớp calibration cùng thay đổi, chưa tách được đóng góp riêng.

## Quy trình và ràng buộc

1. Giữ hai source checkpoints P0 `9d97613c…` và incumbent `5cd49252…`. Kiểm hash toàn bộ input; dùng archive v2 cũ và reference diagnostics run `val_diag_20261110_142828` trên Drive.
2. Fit NLL temperature T trong [0,05;10], tối đa 100 iteration bằng solver đã có, trên **1.080 clean `val_calib`**, 590 real/490 fake. Fit threshold float32/comparator `>=`, empirical FPR≤5%, tối đa **29 false positives**. Không dùng `val_select` để fit hoặc chọn ngưỡng.
3. Với mỗi model, so `cfe_control` và `v2_clean_fit` trên cùng `val_select` clean/Q30. Không fit theo condition; một point dùng cho cả hai. Cùng FP16 backbone, float32 softmax, TF32 tắt, batch chung chốt trước bằng probe.
4. Reuse validation raw logits chỉ nếu source inference/GPU/Torch/CUDA/batch đều khớp. Nếu khác, đo lại hai model và cả hai điều kiện trước khi so điểm, không đổi batch giữa run. GPU không bắt buộc T4.
5. Lưu raw calibration/validation logits và scored arrays, kiểm provenance/IDs, summary/group/error CSV, paired delta/CI. CI bootstrap 1.000 lần/seed42 theo ảnh, phân tầng nhãn, có điều kiện trên fit hiện tại; không bao quát fit uncertainty, generator dependence hay nhiều seed.

Đọc recall Real/Fake và FPR cùng nhau; không lấy tăng Fake làm thắng nếu vi phạm guard Real. Guard ≥90% clean/Q30 chỉ là điều kiện chẩn đoán, không thay cổng G7.2/G7.3. Report raw-margin AUC/AP dùng cùng raw logits; mọi thay đổi score AUC do ties số thực không phải cải thiện biểu diễn. Không xuất research calibration sang deployment, không chấm CFE evaluation hoặc thay package trong run này.

## File chạy và bàn giao

- [Notebook Colab](../../notebooks/task/phase7_calibration_transfer/phase7_calibration_transfer_colab.ipynb) và [hướng dẫn](../../notebooks/task/phase7_calibration_transfer/README.md): dùng runtime mới, ba code cell, lấy source Git `hoang`, khóa commit đã kiểm thử `253942a5262432fbed7ce7ca8b12b4e455bfc2ce`.
- [Protocol máy đọc được](../../configs/phase7_calibration_transfer/plan.json), [runner](../../tools/phase7_calibration_transfer.py).
- [Preparation và kiểm thử](../../DATASET/logs/phase7/improvement_round2/calibration_transfer/README.md): 30 test CPU/code đạt, chưa phải GPU run. Không sửa notebook diagnostics đã thực thi.
- Kết quả Drive: `MyDrive/Fatformer/phase7/improvement_round2/calibration_transfer/<TRANSFER_RUN_ID>/`.
- Notebook thực thi: `notebooks/task/phase7_calibration_transfer/phase7_calibration_transfer_executed.ipynb`; folder kết quả: `wait/calibration_transfer_without_weights/`.

Sau hậu kiểm mới đánh giá việc điều chỉnh operating point có đủ ý nghĩa trên development split. Nếu vẫn có giới hạn ranking/robustness, thiết kế augmentation riêng với control phù hợp. `val_select` đã được dùng chọn P0 và xem lỗi; không phải holdout mù, không cho phép tuyên bố cải thiện CFE hoặc deployment chỉ từ experiment này. Cần bằng chứng độc lập cho model cuối; không tiếp tục tuning theo test CFE đã xem.

## Kết quả hậu kiểm và quyết định

Run `calib_transfer_20261110_155948`, source `253942a`, L4/FP16/batch16, dùng raw validation logits đã kiểm đúng bit. P0 T1,9718821/ngưỡng0,3712310; incumbent T1,7667391/ngưỡng0,5055273. Cả hai fit 29/590 false positives trên clean calibration, đúng FPR≤5%. Fake recall validation tăng nhưng FPR Q30 lên 13,05%/12,71%, Real còn 86,95%/87,29%, không đạt guard90. Raw ranking không đổi; không chọn hai point mới hoặc tự đổi G7.2.

Nếu cần operating point chung robust, một thí nghiệm riêng có thể fit clean/Q30 paired `val_calib`, dùng một threshold bảo thủ đáp ứng budget mỗi condition. Nếu mục tiêu là tăng CFE Fake@Real90/G7.2, cần weights/data/robustness có control; calibration không tạo ranking gain. [Kế hoạch sau kết quả](../../DATASET/logs/phase7/improvement_round2/calibration_transfer/runs/calib_transfer_20261110_155948/analysis/post_result_plan.json) mới là hướng đề xuất, chưa triển khai experiment kế tiếp. 30 test audit/workflow đạt, không cộng chồng với 30 test preparation có trùng lặp.
