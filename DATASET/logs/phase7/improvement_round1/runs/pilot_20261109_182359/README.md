# Hậu kiểm pilot pilot_20261109_182359

Ngày tổng hợp: 10/11/2026 (Asia/Saigon). **Hai nhánh đã hoàn tất 5 epoch trên NVIDIA L4; chọn `p0_legacy_full` theo validation. Chưa có kết quả chất lượng cuối cùng cho checkpoint mới.**

## 1. Tính toàn vẹn và phạm vi xác minh

- `raw/`: 27 file nộp từ `wait`, giữ nguyên byte, gồm cả `commands.log` và các trace smoke cũ/mới.
- `evidence/executed_notebook.ipynb`: bản sao notebook đã chạy; 4 cell có execution 1–4, không có error output.
- Commit source: `26650142e76f790c0642c091a05dfc2dc3770cef`. Cả 146 hash file khớp byte Git trên Linux; fingerprint `580ef49cf8af6d5aa750f0489957c8a16c604b1269f20892f7edfa05fcaaad44`. Khi hậu kiểm trên Windows đã tắt phép chuyển EOL của `git archive`, không chỉnh source để ép hash.
- **28,832 phép đối chiếu dữ liệu đạt**, không phải số lượng unit test. Xem `analysis/verification.json` và `import_manifest.json`.
- Preflight GPU báo ảnh staged/audit/split/canonical PASS. Local kiểm tra ràng buộc hash và log; không có ảnh staged hoặc weights mới trong gói để tự chạy lại forward.

## 2. So sánh theo thiết kế đã chốt

Hai nhánh cùng seed 42, 5 epoch, 765 mini-batch/epoch, batch 16 × accumulation 2, cùng manifest/khởi tạo/teacher/loss/LR. Cả hai **validation CLIP**, 1.080 ảnh clean (590 Real, 490 Fake), cache Q30 300 ảnh (150/150), T=1,6402 và ngưỡng=0,9127 đóng băng. Chỉ training preprocessing khác nhau; epoch 0 giống hệt nhau.

| Chỉ số trên val_select | P0: train legacy | P1: train CLIP |
| :--- | ---: | ---: |
| Clean Real ACC | 99,32% | 99,83% |
| Clean Fake ACC | 60,41% | 53,88% |
| Q30 Real ACC | 98,00% | 98,00% |
| Q30 Fake ACC | 42,00% | 35,33% |
| Worst-group | 42,00% | 35,33% |

Cả hai chọn epoch 5 đúng luật worst-group với Clean Real ≥90%. Worst-group ở đây là min bốn recall clean/Q30 × Real/Fake, không phải worst-subset của CFE.

P0 hơn P1 **6.67 điểm phần trăm** worst-group: Q30 Fake đúng **63/150 so với 53/150**, thêm 10 ảnh. Clean Fake đúng **296/490 so với 264/490**, hơn 6.53 điểm phần trăm. P1 giữ Real tốt hơn: **589/590 so với 586/590**; không có thắng tuyệt đối ở mọi chỉ số.

**Kết luận công tâm:** giả thuyết đổi training preprocessing sang CLIP chưa được kết quả seed này hỗ trợ; giữ P0 làm ứng viên cho bước đo tiếp. Không khẳng định CLIP luôn kém hoặc chênh lệch có ý nghĩa thống kê: chỉ một seed, không có dự đoán từng ảnh để ước lượng paired CI. Các ảnh Q30 là biến thể từ val_select, không phải 300 ảnh gốc độc lập bổ sung. Validation được xem qua nhiều epoch nên điểm checkpoint đã chọn có độ lạc quan do selection.

## 3. Gradient và cập nhật optimizer

- P0: **1914/1915** lần optimizer thực sự được gọi; bỏ qua 1 lần, tại epoch 1/minibatch 4. AMP scale 2048→1024.
- P1: **1913/1915** lần; bỏ qua 2 lần, tại epoch 1/minibatch 4 và epoch 4/minibatch 102. AMP scale 2048→1024→512.
- Tất cả ba norm không hữu hạn được AMP bỏ qua; **không có cập nhật áp dụng norm không hữu hạn**, không có epoch toàn bộ update bị bỏ qua. CSV gradient là trung bình các norm hữu hạn, được tính lại từ trace.
- Hai nhánh có cùng 1.915 lượt thử, nhưng số update thực khác nhau một lần; đây là giới hạn cần ghi nhận. Không được mô tả gradient luôn hữu hạn hoặc cả hai có số update thực bằng nhau.
- Có hai vòng smoke cho mỗi nhánh (cell smoke riêng và kiểm tra lại trước train). Từng vòng có 3 update warmup + 50 update phần đo. Ở phần đo cuối: P0 skip 0/50, P1 skip 1/50=2%, đều đạt chính sách kỹ thuật ≤5%; warmup được lưu riêng.

Trace mới giải thích được overflow/skip của **run này**. Không suy ngược rằng mọi NaN trong run huấn luyện cũ chắc chắn cùng nguyên nhân.

## 4. Thời gian và bộ nhớ

Smoke L4: P0 881.98 ms/mini-batch, P1 883.28 ms/mini-batch; peak allocated khoảng 18.18 GiB. Đây là thời gian huấn luyện gồm backward, không phải inference latency.

Tổng thời gian các epoch được CSV ghi: P0 58.44 phút, P1 58.28 phút. Metadata notebook ghi cell smoke 20.94 phút, cell train 150.38 phút; hai phạm vi khác nhau vì cell còn gồm preflight/smoke lặp lại/validation/backup. Không dùng tổng epoch time làm tổng wall time phiên Colab.

## 5. Model được chọn và giới hạn bàn giao

Ứng viên: `p0_legacy_full`, epoch 5. File Drive:

```text
MyDrive/Fatformer/phase7/improvement_round1/pilot_20261109_182359/p0_legacy_full/best.pth
```

SHA-256 được GPU report ghi:

```text
9d97613c5f271801a30ff514326557f262a372abeb4d52ce864c78f20fc1069b
```

`run_verification.json` ràng buộc hash model/log tại GPU; local xác minh được hash CSV và sự nhất quán metadata/selection. ZIP không có `.pth`, vì vậy **chưa tự băm lại hoặc nạp model mới tại local**. Xem `candidate_reference.json` và `analysis/model_references.json`.

Chưa có logits/probabilities test, AUC/AP/ECE/FPR cho model này, chưa có CFE 1.600/18 tập in-dist hoặc benchmark đóng gói. Không so trực tiếp worst-group 42% ở đây với 46,67% của run cũ vì validation preprocessing đã khác. Checkpoint deployment hiện hành và phán quyết G7.2 FAIL 6/8, G7.3 pending vẫn giữ nguyên.

## 6. Việc tiếp theo

1. Khóa P0/hash trên như ứng viên theo validation; không cần train lại chỉ để sửa báo cáo của vòng này.
2. Trên Colab, kiểm tra hash và canonical load chính file P0 `best.pth`; trích logits cho đúng 400 calibration đã đóng băng, fit T/ngưỡng bằng bộ tối ưu có kiểm tra hội tụ. T/ngưỡng lựa chọn ở trên không phải calibration cuối của model mới.
3. Đánh giá P0 và model deployment hiện hành bằng protocol phù hợp để so sánh: cùng IDs, tiền xử lý, batch/precision/backend; lưu dự đoán từng ảnh. Chạy CFE/18 tập và các gate, không chọn lại model bằng test.
4. Tạo gói đo/đóng gói mới gắn hash P0. Notebook `phase7_corrected_rerun_colab.ipynb` cũ khóa source `5cd49252…`, không dùng nguyên nó để chấm weights mới. Chỉ thay model hiện hành sau khi có bằng chứng tương ứng; nếu cần kết luận tổng quát, bổ sung seed/test chưa xem theo kế hoạch riêng.

Gói này đủ để chốt **pilot training đã hoàn tất và đã hậu kiểm log**, chưa đủ để gọi toàn bộ Phase 7 PASS hoặc tuyên bố model mới tốt hơn incumbent trên test.
