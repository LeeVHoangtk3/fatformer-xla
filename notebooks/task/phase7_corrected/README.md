# Chạy lại Phase 7 bằng đúng code đã sửa

Mở `phase7_corrected_rerun_colab.ipynb`, dùng gói `phase7_corrected_colab_bundle.zip` được xuất từ working tree local. Không chạy tiếp hai notebook cũ: chúng được giữ nguyên làm hồ sơ cho các run commit 9d798f5.

## Chuẩn bị

1. Tạo/cập nhật gói tại thư mục gốc repo:

   ```sh
   python -m tools.build_phase7_colab_bundle
   ```

2. ZIP nằm tại `DATASET/logs/phase7/task_7.15/current/colab/phase7_corrected_colab_bundle.zip`; có tệp SHA-256 đi kèm. Gói chứa code hiện có, BPE tokenizer và các input 400/1.600 đã kiểm tra; không chứa model 2 GB hay dataset.
3. Đưa ZIP **v2 mới** vào `MyDrive/Fatformer/logs/phase7/`, thay ZIP v1 nếu đã có; hoặc chọn ZIP local khi cell upload hỏi. Mở notebook mới trên Colab, chọn một GPU CUDA tương thích còn đủ VRAM, Run all. Không bắt buộc T4.

Không cần commit/push để Colab nhận bản sửa. Notebook không clone/pull GitHub và không giải nén artifact cũ đè lên code. Từng tệp code/input được đối chiếu SHA-256 theo manifest trước khi chạy.

## Input Drive bắt buộc

- `checkpoint/phase7/anchored_v2_run1/fatformer_anchored_v2_best.pth`, hash đầy đủ bắt đầu `5cd49252...`; không tìm tự động các file `best.pth` khác.
- `datasets/test_communityforensics.tar`.
- `datasets/test_benchmark_gans.tar`.
- `datasets/test_benchmark_diffusion.tar`.

Các đường dẫn trên nằm dưới `MyDrive/Fatformer/`, đúng cấu trúc các lần chạy đã có. Không cần CLIP weights rời: source checkpoint chứa đủ tensor để loader khởi tạo.

## Trình tự chạy

- Dừng sớm khi không có CUDA tương thích, code/input sai hash, calibration không đúng 400 ảnh hoặc dữ liệu không đủ. Không kiểm tra bắt buộc tên GPU là T4.
- Thử bộ nhớ bằng input tổng hợp trước phần chạy dài: bắt đầu từ `MAX_BATCH_SIZE=16`, giảm batch khi gặp CUDA out-of-memory. Nếu batch 1 cũng không vừa, dừng và ghi VRAM/lỗi. Probe chỉ kiểm tra dung lượng, không tạo số đo accuracy/latency nghiệm thu.
- Stage dữ liệu vào một thư mục run mới; kiểm tra đủ 400 calibration, 1.600 evaluation và 82.003 ảnh/18 tập in-dist. Thiếu ảnh/ảnh hỏng là lỗi, không thay bằng ảnh xám, tensor 0 hay số liệu dummy.
- Trích xuất lại đúng 400 ảnh bằng FP16 autocast/batch đã chọn, rồi fit T trên calibration. Threshold dùng float32 softmax CUDA/cùng batch như wrapper. Bước này bảo đảm protocol khớp; không huấn luyện backbone và không dùng evaluation để fit.
- T mới có thể khác T của cache FP32 local; kiểm tra nguồn và FPR, không ép con số phải giống hoặc điều chỉnh để gate PASS.
- Đóng gói từ checkpoint gốc, kiểm tra toàn bộ tensor và 20 ảnh (hai real + hai fake từ mỗi subset, đều lấy từ calibration).
- Chấm thực tế bằng `package.predict()` trên CFE clean và sáu biến thể, cùng 18 tập in-dist. Lưu logits, xác suất, nhãn, ID và hash package. In-dist dùng CLIP Resize cạnh ngắn 224 bicubic + CenterCrop như package; protocol cũ stretch 224x224 khác, nên không gán mọi chênh lệch cho calibration.
- Đo 20 warmup + 200 cặp backbone/package ở batch 1, batch đã chọn, 8 và 32; giữ delta có dấu và mẫu thô. Batch latency quá lớn được ghi `out_of_memory`, không gán số giả và không bỏ các batch nhỏ đã đo được.
- Ghép báo cáo chỉ khi benchmark, verification và latency cùng hash package. So sánh với model tốt nhất theo từng điều kiện vẫn pending nếu chưa có reference đủ bằng chứng; không thay bằng phép clean trừ Q30.

Theo yêu cầu cập nhật ngày 09/11/2026, phần cứng chấp nhận là GPU CUDA tương thích có đủ dung lượng cho cấu hình chạy. Latency <5 ms overhead được chấm trên GPU thực tế được ghi trong báo cáo; T4 ban đầu chỉ còn là tham chiếu lịch sử. Không suy diễn số đo L4/A100 thành số đo T4. Không cần đổi cờ `REQUIRE_T4`; cờ này đã được bỏ.

Batch được chọn trước calibration để giữ cùng batch/precision/backend giữa calibration và benchmark. Nếu VRAM thay đổi hoặc hết bộ nhớ trong benchmark thật, run dừng; không tự giảm batch giữa chừng rồi trộn kết quả với calibration cũ.

## Lấy kết quả

Cell cuối lưu weights theo run mới tại `MyDrive/Fatformer/checkpoint/phase7/corrected_<run_id>/`, không ghi đè model cũ. ZIP kết quả nằm ở `MyDrive/Fatformer/logs/phase7/corrected_<run_id>/` và được tải về trình duyệt; ZIP không kèm weights 2 GB, nhưng có code bundle, manifest, JSON, NPZ, ROC và stdout log.

Lưu notebook sau khi chạy và đưa ZIP vào `wait` để hậu kiểm. Có thể chạy cell backup riêng sau lỗi để thu log. Đây là xác nhận sau thay calibration (post-hoc); không coi evaluation đã xem là tập kiểm thử mù mới. Kết quả FAIL/pending hợp lệ phải được giữ nguyên.

Code đã được kiểm thử local và kiểm tra lại sau giải nén ZIP, gồm môi trường không có Git. Chưa có lần chạy GPU mới từ notebook này trong phiên local; các con số model/gate phải lấy từ lần chạy Colab thực tế.
