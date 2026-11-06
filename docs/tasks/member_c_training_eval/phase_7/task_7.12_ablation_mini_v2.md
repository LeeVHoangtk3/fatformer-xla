# TASK 7.12 — ABLATION GUARD / FEATURE TRÊN DỮ LIỆU v2

- **Phụ trách**: Thành viên C; Thành viên B review loss và cách diễn giải.
- **Trạng thái**: `[~] Có log bốn cấu hình A100 40GB, mỗi cấu hình 5 epoch. A3 tốt nhất theo worst-group validation; chưa chứng minh lợi ích tổng quát của hai loss, hậu kiểm checkpoint/logits còn mở và G7.2 chưa đạt đầy đủ.` Xem [báo cáo kết quả](../../../bao_cao/C/bao_cao_task_7.12_ablation.md).
- **Notebook**: [`task_7.12_ablation_mini_colab.ipynb`](../../../../notebooks/task/task_7.12/task_7.12_ablation_mini_colab.ipynb).
- **Kế hoạch liên quan**: [Phase 7](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md). Quy định cụ thể dưới đây sửa các điểm mâu thuẫn về epoch, tương tác và đóng góp dữ liệu của đặc tả cũ.

## 1. Câu hỏi và giới hạn thiết kế

So sánh việc thêm **RealGuard** và **FeatureConsistency** khi giữ nguyên dữ liệu v2, CE, KD, logit consistency, khởi tạo, luật chọn checkpoint và ngân sách huấn luyện.

| Cấu hình | Feature | Guard | Thành phần chung |
| :--- | ---: | ---: | :--- |
| A1 `a1_data` | 0.0 | 0.0 | CE clean/deg + KD + Consistency, dữ liệu v2 |
| A2 `a2_data_guard` | 0.0 | 0.5 | Như A1 |
| A3 `a3_data_featcons` | 0.3 | 0.0 | Như A1 |
| A4 `a4_full` | 0.3 | 0.5 | Như A1 |

Tên file A1 được giữ để tương thích; **không gọi A1 là “Data only”**. Cả bốn cùng dùng dữ liệu v2 nên không tách được đóng góp riêng của dữ liệu mới. Không thêm run ngoài thiết kế để lựa chọn kết quả thuận lợi.

Mặc định **train cả bốn cấu hình trong một thí nghiệm mới**. Không tự tái sử dụng A4 từ Task 7.11: checkpoint, gradient NaN và lịch sử phiên của run cũ còn cần kiểm chứng. Đối chiếu với 7.11 chỉ là thông tin tham khảo, không dùng số của run đó ghép vào bảng mới. Cùng seed không bảo đảm bit-exact khi đổi GPU/framework.

## 2. Cấu hình chung

Nguồn: `configs/ablation_v2/*.yaml`; [`audit_ablation_configs.py`](../../../../tools/audit_ablation_configs.py) kiểm cả bốn cấu hình đầy đủ, không dùng `inherits` chưa được CLI hỗ trợ.

- `loss.mode: v2` bắt buộc, **kể cả A1**. Không dùng việc tắt Guard/Feature để chọn công thức legacy.
- CE: `0.3 CE_clean + 0.3 CE_deg`; KD `0.7`; Consistency `0.2`; nhiệt độ loss `2.0`; label smoothing `0.05`.
- Baseline làm khởi tạo và teacher; SRM/gating bật, backbone đóng băng theo pipeline hiện có.
- Seed 42, mini-batch 16, accumulation 2, AMP, hai LR `1e-4` và `1e-5`, cùng `num_workers=2`.
- **5 epoch cho cả bốn**, khớp YAML hiện tại. Bảng 6/3 epoch của đặc tả cũ không áp dụng cho thí nghiệm này. Nếu cần đổi ngân sách, sửa đồng loạt bốn config **trước khi chạy**, dùng experiment mới; không ghép A4 5 epoch với các run 3/6 epoch.
- Train/selection đọc manifest v2 gốc: `train` và `val_select`; không đưa `val_calib` vào train/selection.
- Chọn checkpoint theo worst-group với `clean_r >= 0.90`; dùng checkpoint do trainer chọn và metadata của chính tệp đó, không chọn lại epoch từ CSV để ghép với trọng số khác.

Audit YAML chỉ xác nhận cấu hình; không chứng minh dữ liệu staged, GPU đủ bộ nhớ hoặc train đã hoàn thành.

## 3. Chuẩn bị và chạy

1. Commit/push mã đã sửa trước khi dùng Colab. Notebook khóa SHA cho cả experiment; chạy tiếp dùng SHA đã lưu. Không reset source tree có thay đổi.
2. Dùng tên mới, mặc định `ablation_v2_fixed_run1`; không dùng thư mục các run cũ thiếu identity. Runtime và Drive tách theo experiment/cấu hình.
3. Stage `train_crossdomain_v2.tar` đã audit, kiểm sidecar SHA256, giữ manifest/report/config nguyên bản; copy backbone và baseline có kiểm hash.
4. Stage CFE tar gốc và `phase7_measured_artifacts.zip` của Task 7.6. Trích baseline calibration cùng `logits_baseline_clean_calib_split.npz` và `..._eval_split.npz`. Thiếu artifact gốc thì dừng; không tự fit lại hay chia split mới.
5. Runner [`run_ablation_v2.py`](../../../../tools/run_ablation_v2.py) chạy preflight đủ 8 PASS và benchmark 100 mini-batch trên GPU hiện tại cho từng cấu hình. Ngưỡng đầu vào >=20 GiB trống là mức kiểm tra thận trọng dựa trên số đo cũ >18 GiB; không bảo đảm mọi GPU đều chạy được. Không khẳng định T4 16 GB phù hợp.
6. Chạy train/full-state resume. Gradient không hữu hạn sẽ dừng, không lưu optimizer update đó. Nếu điều tra AMP/batch/LR, tạo experiment riêng và giữ cùng thay đổi cho cả bốn cấu hình trước khi so sánh.
7. Xác minh đầy đủ horizon, steps/epoch, loss/gradient hữu hạn, checksum, best metadata và CSV. Eval nạp best qua canonical loader; aggregator kiểm missing/unexpected/layout và SHA256 của đúng tệp.
8. Chấm CFE clean/Q30 trên **cùng evaluation IDs độc lập**, sau đó tổng hợp. Không coi việc tạo được file đầu ra là tự động đạt G7.2.

Runner không suy “đã xong” từ best + CSV. Full latest từ Drive phải có checksum, schema và context khớp; tiếp tục từ epoch hoàn tất. Horizon, code hoặc đầu vào đổi thì từ chối resume. Metadata ghi commit, cấu hình, fingerprints và seed thực tế. Các lỗi không được đổi thành số liệu mặc định.

## 4. Khế ước evaluation và calibration

Baseline calibration hiện có được fit trên **400 ảnh CFE calibration**. Không chấm tổng hợp chính thức trên toàn bộ 2.000 ảnh bao gồm lại 400 ảnh đó. Eval đọc **1.600 evaluation IDs gốc** trong artifact Task 7.6, kiểm tag split, nhãn, ID duy nhất và không giao calibration; SHA256 calibration NPZ phải khớp artifact calibration đã đo.

Cả bốn model dùng operating point cố định từ calibration baseline để so sánh. Nhiệt độ này **không phải nhiệt độ đã fit riêng cho từng model mới**. Vì vậy, ECE được ghi rõ là `ece_q30_frozen_temperature_pct`, không tuyên bố là ECE đã tối ưu sau hiệu chuẩn từng model. Không fit calibration trên evaluation để giảm ECE.

Evaluator xuất `metrics.auc`, `metrics.fake_at_real90`, `metrics.ece`, `metrics.ci95` theo **phần trăm**. Aggregator không nhân thêm 100, không dùng `metrics_pooled/roc_auc` sai schema. Thiếu tệp/key/CI, nguồn synthetic, source dirty hoặc checkpoint không khớp đều bị từ chối; không thay bằng 0.

`Fake@Real90` ở đây giữ đúng định nghĩa quantile 90% trên Real của evaluator. Không thay ngầm bằng TPR nội suy tại FPR 10% vì hai phép tính có thể khác nhau khi có ties. CI của Fake@Real90 tính lại ngưỡng trong từng bootstrap resample.

## 5. Chỉ số và diễn giải

CSV ghi epoch horizon thực, epoch best và tám chỉ số:

1. `worst_group` trên val_select.
2. `clean_r` trên val_select.
3. `q30_r` trên val_select.
4. `q30_f` trên val_select.
5. `fake_at_real90_q30` trên CFE held-out.
6. `auc_clean` trên CFE held-out.
7. `auc_q30` trên CFE held-out.
8. `ece_q30_frozen_temperature_pct` trên CFE held-out.

Các tương phản theo điểm phần trăm:

- `guard = A2 - A1`.
- `feature = A3 - A1`.
- `interaction_2x2 = A4 - A2 - A3 + A1`.
- `full_over_best_single = A4 - max(A2, A3)`; đây là mức hơn cấu hình đơn tốt nhất, không gọi là tương tác 2×2.

Worst-group lấy trên tập dùng chọn checkpoint nên các delta của nó được ghi **mô tả**, `ci95=null`, chưa kết luận có ý nghĩa thống kê hay loss đóng góp lớn nhất. Không biến việc chưa có CI thành “CI không chồng”.

CFE dùng 1.000 bootstrap resample phân tầng theo nhãn, cùng chỉ số ảnh cho cả bốn model; clean/Q30 và các model phải khớp sample IDs/targets. JSON ghi CI của các tương phản CFE. CI này chỉ đo bất định theo ảnh **với seed huấn luyện đã chạy**, không đo biến thiên giữa các seed huấn luyện. Báo cả delta âm và trường hợp chưa đủ bằng chứng; không ép một câu kết luận thắng/thua khi dữ liệu không hỗ trợ.

## 6. Đầu ra và điều kiện nghiệm thu

Mỗi run có `run_identity.json`, cấu hình runtime, full latest, best, history, val_groups, preflight, benchmark, `run_verification.json`, JSON evaluation và logits NPZ. Summary tại `<run_root>/summary/`, sao lưu vào thư mục experiment trên Drive; không tự chép sang đường dẫn kết quả chính thức khác.

- [x] Có bốn YAML explicit v2 và công cụ audit chỉ khác hai trọng số loss.
- [x] Có kiểm thử CPU cho công thức CE, schema/đơn vị, split, resume, checkpoint, delta và xử lý thiếu bằng chứng.
- [x] Có preflight của bốn run ghi đủ 8 PASS và benchmark đo trên A100-SXM4-40GB tại `DATASET/logs/phase7/ablation_v2_fixed_run1/`; chưa gọi đây là lần chạy lại độc lập.
- [x] History ghi đủ 5 epoch/run, 765 mini-batch/epoch; context giống nhau ngoài hai trọng số loss. Chưa có bộ đếm optimizer updates bị GradScaler bỏ qua để xác nhận số cập nhật thực tế bằng nhau.
- [ ] Hậu kiểm trực tiếp checkpoint: JSON đã lưu ghi canonical 0/0, layout n=3, dropped=[] và hash best khớp verification; local chưa có tệp `.pth` để nạp lại hoặc kiểm checksum tệp thực.
- [x] Có CSV/JSON/PNG tổng hợp; tám JSON CFE ghi cùng evaluation partition 1.600 mẫu với fingerprints khớp staged inputs. Bảng số khớp nguồn trong phạm vi đã đối chiếu.
- [ ] Hậu kiểm sample IDs/targets và tái tính CI từ logits NPZ gốc. Bộ hiện vật local chưa kèm NPZ; chưa kết luận đã tái lập các khoảng tin cậy độc lập.
- [x] Báo cáo thực nghiệm đã ghi cả delta âm, giới hạn thống kê và phần chưa xác minh. A3 cao nhất trên worst-group validation ở seed 42; chưa xác định loss đóng góp lớn nhất một cách tổng quát.

Công cụ tổng hợp không có chế độ dummy xuất cùng tên/kho kết quả thật. Test dùng fixture trong thư mục tạm. JSON ablation dùng schema riêng `ablation-v2/v1`; `gate_eligible=false` vì nó chưa kiểm mọi yêu cầu G7.2, kể cả 18 tập in-dist. Không nâng trạng thái G7.2 chỉ từ bảng ablation.

## 7. Liên kết

- [Báo cáo số đo GPU và phạm vi hậu kiểm](../../../bao_cao/C/bao_cao_task_7.12_ablation.md).
- [Task 7.11](task_7.11_train_anchored_v2.md).
- [Khế ước CFE](../../../plan/ke_hoach_test_communityforensics_eval.md).
- [Task 7.14 — locked benchmark](task_7.14_locked_benchmark_phase7.md).
