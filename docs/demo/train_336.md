# Huấn luyện và master audit ở 336×336

Notebook: [train_336_master_audit_colab.ipynb](../../notebooks/train_336_master_audit_colab.ipynb).

Mở trên [Colab, nhánh test](https://colab.research.google.com/github/LeeVHoangtk3/fatformer-xla/blob/test/notebooks/train_336_master_audit_colab.ipynb), chọn GPU và chạy tuần tự. Notebook không có kết quả điền sẵn. Source được clone từ Git và chốt commit; dữ liệu và checkpoint đọc từ Drive.

## Phạm vi thay đổi

Ảnh đầu vào của lượt mới là **336×336**. ViT-L/14 tạo lưới **24×24 patch**. Mã hóa vị trí được nội suy khi forward; nhánh SRM xuất cùng lưới patch. Tensor và khóa state dict cũ được giữ để nạp checkpoint nghiêm ngặt. Luồng suy luận 224 vẫn là mặc định cho package cũ.

Cấu hình riêng: [train_anchored_336.yaml](../../configs/train_anchored_336.yaml) và [degradation_336.yaml](../../configs/degradation_336.yaml). Giữ seed 42, 5 epoch, trọng số loss và phân phối suy biến. Khởi tạo student và teacher từ baseline gốc như lượt cũ.

Batch thực **4**, tích lũy **8** lần, batch hiệu dụng danh định **32**. Nếu không đủ VRAM, tạo run mới với batch 2 và tích lũy 16. Tích lũy gradient không làm thống kê BatchNorm của SRM tương đương batch thực 16; vì vậy đây không phải thí nghiệm chỉ đổi một yếu tố tuyệt đối.

Train/validation giữ resize vuông và chuẩn hóa ImageNet như cấu hình train cũ. Hiệu chuẩn cuối và master audit dùng resize cạnh ngắn, center crop 336 và chuẩn hóa CLIP, tương ứng cách triển khai 224 trước đây. Sai khác chuẩn hóa train–triển khai đã có ở bản cũ; thay đổi này chưa kiểm chứng hay giải quyết riêng nguyên nhân đó.

## Đầu vào trên Drive

Gốc mặc định: `MyDrive/Fatformer`.

- `pretrained/ViT-L-14.pt` và `pretrained/fatformer_4class_ckpt.pth`.
- `phase7/train_crossdomain_v2.tar` và checksum `.sha256` đi kèm. Bên trong cần manifest, báo cáo cách ly và cấu hình dữ liệu gốc đã được audit.
- Bảy archive còn lại trong danh mục master audit cũ: ProGAN train/val, diffusion staging, CFE, GAN test, diffusion test và ảnh suy biến lưu sẵn. Notebook kiểm tra đủ cả tám trước train.

Không cần đưa source ZIP lên Drive. Không đổi tên hoặc ghi đè package demo 224.

## Trình tự chạy

1. Mount Drive, kiểm tra GPU, clone nhánh `test`, chốt commit và ghi nhận fingerprint source.
2. Kiểm checksum archive train, giải nén bằng bộ lọc tar an toàn và dựng cấu hình runtime ngoài checkout.
3. Audit 2.000 ảnh CFE; dùng đúng 400 ảnh calibration để hiệu chuẩn baseline ở 336. Đóng băng điểm vận hành trước khi chọn epoch; 1.600 ảnh evaluation không tham gia fit.
4. Chạy preflight trên toàn bộ 14.400 ảnh Data v2. Đo riêng 100 mini-batch forward/backward thực trước train. Không tự đổi từ 5 xuống 4 epoch.
5. Train trên 12.240 ảnh; chọn checkpoint bằng 1.080 ảnh `val_select`. Checkpoint resume đầy đủ được backup lên Drive sau epoch hoàn tất. Epoch đang dở được chạy lại khi resume.
6. Hiệu chuẩn lại checkpoint mới trên 400 ảnh calibration. Đóng gói với resolution 336 và checksum mới; kiểm chứng tensor và dự đoán trên 20 ảnh calibration thật, gồm cả Real và Fake.
7. Chạy master audit: **8 archive, 37 job**, không lấy mẫu bớt ảnh. Ghi hash package 336 thực tế vào từng kết quả, lưu metric, bootstrap, latency và các tiêu chí chất lượng.

Nếu không epoch nào đạt điều kiện Real validation, notebook có thể audit `best_effort` để phân tích, nhưng ghi **FAIL** cho điều kiện chọn checkpoint và không coi đó là thành phẩm đạt yêu cầu.

## Đầu ra và cách kết luận

- Checkpoint train: thư mục Drive `checkpoint/phase7/anchored_336_run1`.
- Mỗi lần đóng gói: một thư mục mới trong `packages`; có `run_contract.json`, calibration và kiểm chứng cùng checksum.
- Master audit: thư mục mới `audit_reports/master_audit_336_…`; ZIP ở đây chỉ chứa bằng chứng đầu ra.

Đọc `run_status.json`, `gates.json`, `benchmark_summary.json` và CSV. Notebook xuất thêm `comparison_224_vs_336.csv` và `comparison_conclusion.json`, đối chiếu số đo mới với hồ sơ master audit 224 đã lưu. **Hoàn tất 37 job chỉ xác nhận đủ phạm vi chạy, không xác nhận mô hình tốt hơn.** Bảng chênh lệch mang tính mô tả; không tự kết luận khác biệt có ý nghĩa thống kê hoặc thay package demo hiện hành.

Ảnh suy biến lưu sẵn giữ nguyên byte trước khi được tiền xử lý lên 336. CFE là hậu kiểm đã được xem trong nghiên cứu; kiểm trùng ở master audit là SHA-256, chưa phải kiểm trùng theo cảm nhận.

## Trạng thái kiểm chứng

Xem [logs/resolution336](../../logs/resolution336). Các kiểm thử CPU và forward trên checkpoint thật chỉ chứng minh tương thích kỹ thuật. **Chưa có kết quả train 336 hoặc master audit 336 cho đến khi notebook thực sự hoàn tất trên Colab GPU.**
