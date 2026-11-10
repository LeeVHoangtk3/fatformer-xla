# Chẩn đoán validation trước vòng train tiếp theo

Trạng thái 10/11/2026: đã kiểm kê dữ liệu và chuẩn bị code; **chưa chạy forward chẩn đoán trên GPU**. Không cần sửa hoặc chạy lại notebook P0 evaluation cũ. Mở một notebook Colab mới để giữ nguyên bằng chứng đã chạy.

Luồng này lấy code trực tiếp từ Git nhánh `hoang`, không cần ZIP code. Nó chỉ đo hai checkpoint đã đóng băng trên `val_select`, không train, không hiệu chuẩn lại, không đọc ảnh CFE/test benchmark và không thay gói deployment.

## 1. Input Drive phải giữ nguyên

```text
MyDrive/Fatformer/
  phase7/train_crossdomain_v2.tar
  phase7/improvement_round1/pilot_20261109_182359/p0_legacy_full/best.pth
  checkpoint/phase7/anchored_v2_run1/fatformer_anchored_v2_best.pth
```

Hash được khóa trong [`configs/phase7_validation_diagnostics/plan.json`](../../../configs/phase7_validation_diagnostics/plan.json). Không đổi tên model khác để lấp vào các đường dẫn này. Checkpoint có đầy đủ backbone nên không yêu cầu tải lại `ViT-L-14.pt` riêng.

## 2. Chạy ba cell trên runtime Colab mới

Chọn GPU CUDA đủ bộ nhớ; có thể là T4, L4, A100 hoặc GPU CUDA tương thích khác. Runner probe cả hai model, chốt batch chung trước khi đo; không tự giảm batch giữa chừng.

**Cell 1 — kiểm tra GPU:**

```python
import torch
assert torch.cuda.is_available(), 'Chọn Runtime → Change runtime type → GPU trước khi chạy'
print(torch.cuda.get_device_name(0))
```

**Cell 2 — lấy code Git nhánh `hoang` và ghi commit:**

```python
from pathlib import Path
import subprocess

REPO = Path('/content/fatformer-xla')
assert not REPO.exists(), 'Dùng runtime mới để không ghi đè repo hoặc thay đổi chưa lưu'
subprocess.run([
    'git', 'clone', '--branch', 'hoang', '--single-branch',
    'https://github.com/LeeVHoangtk3/fatformer-xla.git', str(REPO)
], check=True)
assert (REPO / 'notebooks/task/phase7_validation_diagnostics/start_colab.py').is_file(), 'Nhánh Git chưa có code chẩn đoán'
DIAG_COMMIT = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True
).strip()
print('Commit đã khóa cho phiên này:', DIAG_COMMIT)
```

**Cell 3 — mount Drive và đo validation:**

```python
DIAG_REPO = str(REPO)
DIAG_MAX_BATCH = 16
exec(compile(
    (REPO / 'notebooks/task/phase7_validation_diagnostics/start_colab.py').read_text(),
    'start_colab.py', 'exec'
))
```

Cell 3 sẽ yêu cầu mount Drive. Nó kiểm hash archive, giải nén lên SSD Colab, kiểm hash/giải mã đủ 1.080 ảnh validation, kiểm hash checkpoint, probe bộ nhớ rồi forward. Có thể mất vài phút chỉ để đọc/hash archive Drive; không có bước backward hoặc optimizer.

Không `git pull` giữa một phiên đo. Nếu đổi GPU/commit hoặc gặp OOM, giữ log cũ, dùng runtime và run ID mới. Batch được chọn chung trong [1,16], FP16, TF32 tắt; không cần ép dùng T4.

## 3. Đo gì, ở đâu?

- 1.080 ảnh `val_select`: 590 real, 490 fake, giữ nguyên split và nhãn của manifest v2.
- Hai checkpoint: P0 và incumbent nguồn cũ, cùng geometry CLIP resize cạnh ngắn 224, center crop 224 và CLIP normalization.
- Clean và JPEG Q30 **sau resize/crop**. Đây là protocol validation lịch sử; không đồng nhất nó với mọi benchmark JPEG nén trước resize.
- Q30 chạy đủ 1.080 ảnh. 300 ảnh Q30 lịch sử (150 real/150 fake, seed 42) được chạy trước và đánh dấu riêng; không trộn batch cuối của phần lịch sử với 780 ảnh bổ sung.
- Ghi logits, xác suất, nhãn, ID bằng SHA-256 ảnh, family, generator, path cho từng ảnh. Ghi recall theo nhóm/generator, phân bố điểm, AUC/AP và danh sách lỗi. AUC từng nhóm fake được tính so với cùng 590 ảnh real validation; không tính AUC trên một lớp đơn lẻ. Những nhóm này dùng chung ảnh real và có nhóm rất ít fake, nên các chỉ số không độc lập và chỉ dùng chẩn đoán.
- Dùng chung **ngưỡng chọn model lịch sử** T=1,6402, threshold=0,9127 để chẩn đoán. Đây không phải calibration deployment riêng của P0 hoặc incumbent. Không fit ngưỡng trên validation để xuất ra deployment.

Real90 được tính trên validation để mô tả khả năng xếp hạng; đó không phải test mới, không phải ngưỡng triển khai. Khác GPU/batch/thư viện có thể đổi một số quyết định FP16; phần đối chiếu với log cũ được ghi thành báo cáo, không sửa CSV lịch sử.

## 4. Sau khi chạy xong

Kết quả Drive:

```text
MyDrive/Fatformer/phase7/improvement_round2/diagnostics/<DIAG_RUN_ID>/
  identity.json
  plan.json
  batch_contract.json
  status.json
  candidate_p0/
    canonical_load.json
    probe.json
    predictions_clean.npz
    predictions_q30_post_crop.npz
  incumbent/
    canonical_load.json
    probe.json
    predictions_clean.npz
    predictions_q30_post_crop.npz
  analysis/
    diagnosis.json
    group_metrics.csv
    errors.csv
  validation_diagnostics_without_weights.zip
```

Notebook sẽ tải ZIP **kết quả** về; đây không phải ZIP code và không chứa checkpoint. Có thể lấy trực tiếp thư mục kết quả Drive thay ZIP nếu thuận tiện.

Lưu notebook đã chạy thành:

```text
notebooks/task/phase7_validation_diagnostics/phase7_validation_diagnostics_executed.ipynb
```

Giải nén ZIP hoặc chép thư mục kết quả vào:

```text
wait/phase7_validation_diagnostics/<DIAG_RUN_ID>/
```

Chưa train thêm hoặc chạy lại 7.14/7.15 ở bước này. Dữ liệu chẩn đoán sẽ dùng để phân biệt: khả năng xếp hạng yếu, ngưỡng bảo thủ, độ bền Q30 kém, hay lỗi tập trung ở family/generator cụ thể. Sau hậu kiểm mới khóa giả thuyết, control và tiêu chí dừng cho thí nghiệm tiếp.

Hồ sơ local đã chuẩn bị: [`DATASET/logs/phase7/improvement_round2/diagnostics/README.md`](../../../DATASET/logs/phase7/improvement_round2/diagnostics/README.md).
