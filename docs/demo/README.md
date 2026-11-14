# Demo FatFormer XLA — Windows và Colab GPU

Một app Gradio dùng package đã chọn, loader canonical và preprocessing hiện có. Upload ảnh JPEG/PNG/WebP, xem crop 224×224, nhãn REAL/FAKE, điểm Fake sau calibration và thời gian backend. Giao diện giải thích quyết định theo ngưỡng và có tùy chọn đo ảnh hưởng từng vùng. Có chế độ clean/JPEG Q30, batch 1, queue tối đa 4 và một inference tại một thời điểm. Ảnh mẫu không cache kết quả dự đoán.

## Chạy Windows CPU

Từ thư mục gốc dự án, môi trường `.venv-demo` đã được tạo trên máy hiện tại:

```powershell
.venv-demo/Scripts/python.exe demo.py
```

Chờ dòng `Running on local URL`, mở **http://127.0.0.1:7860**. Terminal phải tiếp tục chạy; dùng Ctrl+C để dừng. Máy hiện tại dùng Python 3.14.0, torch 2.10.0+cpu, torchvision 0.25.0+cpu, Gradio 6.32.0. App tự giới hạn tối đa 4 CPU threads.

Nếu dựng lại môi trường trên máy đã có torch/torchvision tương thích:

```powershell
python -m venv --system-site-packages .venv-demo
.venv-demo/Scripts/python.exe -m pip install -r requirements-demo.txt
.venv-demo/Scripts/python.exe demo.py
```

Trên máy mới chưa có PyTorch, tạo venv thông thường rồi cài cặp CPU đã kiểm chứng trước khi cài requirements:

```powershell
python -m venv .venv-demo
.venv-demo/Scripts/python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.venv-demo/Scripts/python.exe -m pip install -r requirements-demo.txt
```

Yêu cầu đủ RAM để nạp model gần 2 GB cùng các tensor trung gian; số đo trên máy hiện tại nằm trong [báo cáo kiểm chứng](validation.md). Chưa xác nhận cấu hình RAM tối thiểu trên máy khác. Giữ file tokenizer BPE trong source. App không cần tải thêm CLIP weights. Chỉ lần cài dependencies cần Internet; mặc định app chạy loopback, không tạo link công khai.

Model mặc định:

```text
DATASET/checkpoints/phase7/demo/unified_fatformer_v2.pth
SHA256: 248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0
```

Không thay bằng file cùng tên ở thư mục cha. App kiểm hash trước khi nạp; không khớp thì dừng. Có thể dùng `--checkpoint <đường_dẫn>` khi file được chuyển vị trí, nhưng nội dung phải giữ đúng hash.

## Chạy Colab GPU

1. Tạo snapshot đúng source đang dùng trên Windows:

   ```powershell
   python -m tools.build_demo_bundle
   ```

2. Mở [notebook](../../notebooks/demo_fatformer_xla_colab.ipynb) bằng Google Colab; chọn GPU. Không cần clone GitHub vì source demo có thể chưa được commit/push.
3. Cell 1: nhập hash từ `artifacts/demo/fatformer_xla_demo.sha256`, upload `artifacts/demo/fatformer_xla_demo.zip` (~60 MB, source và 20 ảnh mẫu). Notebook kiểm manifest/hash rồi cài Gradio, giữ PyTorch CUDA của runtime.
4. Cell 2: mount Drive, copy package từ `/MyDrive/Fatformer/checkpoint/phase7/corrected_20261109_102954_3d28ad/unified_fatformer_v2.pth` vào SSD runtime, kiểm hash, nạp GPU, kiểm chứng 20 ảnh clean/Q30 và tải log JSON. Sửa đường dẫn nếu Drive của bạn khác. Cần khoảng 2 GB cho package, ngoài môi trường và bộ nhớ model.
5. Cell 3: đặt thông tin đăng nhập của phiên rồi mở link Gradio. Không lưu mật khẩu vào source/notebook. Dừng runtime sau khi trình diễn.

**Trạng thái:** notebook đã kiểm cú pháp và bundle trên Windows; GPU/Colab cần một lần chạy thực tế để nghiệm thu. Không dùng kết quả GPU lịch sử làm bằng chứng notebook mới đã chạy. Nếu runtime thiếu GPU, notebook dừng rõ ràng, không âm thầm chuyển CPU. Lỗi import sau pip trên runtime đã dùng nhiều thư viện: restart runtime rồi chạy lại notebook từ đầu.

## Kiểm chứng và diễn giải

```powershell
.venv-demo/Scripts/python.exe demo.py --verify-only
.venv-demo/Scripts/python.exe -m pytest tests/test_demo.py -q
```

Log ở `logs/demo/cpu_verification.json`. 20 ảnh có nhãn nguồn từ CSV verification Task 7.15; hash ảnh được kiểm trước khi chạy. So sánh 40 lượt clean/Q30 với reference cùng backend/precision, `atol=1e-6, rtol=1e-5`, giữ nhãn và threshold. Không dùng bộ minh họa nhỏ này để tuyên bố accuracy mới. CPU dùng FP32; GPU dùng autocast FP16 và softmax FP32 theo wrapper. Sai khác CPU/GPU không được mặc định bằng 0.

T = **2.7832491398**, threshold = **0.7918508649**, layout `[7,15,23]`, router **tắt**. Nhãn lấy từ output model theo threshold, không dùng ngưỡng 0.5. Điểm Fake không phải độ chính xác của một ảnh cụ thể. Nhãn nguồn của ảnh mẫu được tra theo hash độc lập với dự đoán.

Pipeline: PIL→RGB, resize cạnh ngắn 224 bicubic, center crop 224, normalize CLIP. Không tự xoay theo EXIF; ảnh RGBA/grayscale chuyển RGB đúng đường reference. JPEG Q30 được mã hóa **sau crop, trước normalize**, bằng Pillow `quality=30`. Ảnh tối đa 32 MiB/25 triệu pixel; từ chối ảnh động và file hỏng. Giới hạn đã tăng từ dự kiến 20 MiB vì ảnh verification gốc lớn nhất có 23.310.018 byte. Vùng ảnh ngoài crop không được model quan sát.

Backend timing gồm đọc ảnh, preprocess, inference và tra nguồn; không gồm thời gian upload, queue hay render trình duyệt. Forward timing riêng có trong phần JSON chi tiết. Q30 chạy thêm một forward, không thể so trực tiếp latency cặp với latency một ảnh.

## Kịch bản trình diễn 3–5 phút

- Chọn một ảnh Real có nguồn, bấm Phân tích và chỉ rõ vùng crop/điểm/nhãn.
- Chọn một ảnh Fake có nguồn; threshold giữ nguyên. Nếu dự đoán sai, trình bày đúng giới hạn model.
- Bật so sánh Q30 trên cùng ảnh và quan sát hai kết quả thực tế.
- Upload ảnh tự chọn; mở accordion để xem device, precision và kết quả chi tiết.

## Quyết định rõ ràng và giải thích vùng ảnh

Mỗi kết quả có nhãn tiếng Việt kèm REAL/FAKE, phép so sánh điểm với **79,185086%**, khoảng cách đến ngưỡng tính bằng **điểm phần trăm** và quy tắc phân loại cố định. Ví dụ điểm Fake 70% vẫn cho **REAL**, vì 70% < 79,185086%. Khoảng cách đến ngưỡng không phải độ tin cậy hay xác suất model đoán đúng. Không thêm ngưỡng mới để chia “chắc chắn/không chắc chắn”. Nhãn REAL cũng không xác minh nguồn gốc ảnh.

Trong chế độ Q30, app ghi rõ nhãn giữ nguyên hay đổi và điểm Fake tăng/giảm bao nhiêu. Phản ứng với JPEG trên một ảnh không chứng minh ảnh đó bị chỉnh sửa.

Để xem chi tiết vùng ảnh:

1. Chọn ảnh, bật **Giải thích vùng ảnh**, bấm **Phân tích ảnh**.
2. Mở mục **Vùng ảnh ảnh hưởng đến điểm Fake**. Tính năng luôn giải thích **crop gốc**, kể cả khi đang so sánh Q30.
3. Crop được chia 4×4 ô, mỗi ô 56×56 pixel. App chạy thêm 16 forward batch 1; mỗi lần thay một ô bằng giá trị 0 sau normalize (tương đương màu trung bình CLIP trong RGB), giữ nguyên các ô khác. Đo `delta_fake_pp = 100 × (p_fake_gốc − p_fake_sau_che)`.
4. **Đỏ / số dương:** khi che ô đó, điểm Fake giảm. **Xanh / số âm:** khi che ô đó, điểm Fake tăng. Số trên ô là delta có dấu; màu dùng cùng thang cho mọi ảnh, tăng độ đậm tới 10 điểm phần trăm. Ô được đánh số từ trái sang phải, từ trên xuống dưới. JSON giữ điểm của cả 16 lần che và tọa độ crop `xyxy` (biên cuối không thuộc ô).

Đây là phép đo độ nhạy theo phương pháp [Occlusion](https://captum.ai/api/occlusion.html), thực hiện trực tiếp bằng PyTorch đã có; không cài Captum. Tùy chọn mặc định tắt vì tốn thêm 16 lượt chạy. Trên hai ảnh mẫu đã chạy ở Windows CPU, phần giải thích mất thêm khoảng **29,6–31,0 giây**; đây không phải cam kết latency. Thời gian backend khi bật tùy chọn đã gồm phần giải thích.

**Bản đồ không phải mặt nạ giả mạo.** Model hiện tại phân loại crop, không trả segmentation/ground truth vị trí sửa ảnh. Kết quả che phụ thuộc màu thay thế/kích thước ô; phép che có thể tạo ảnh ngoài phân phối. Không kết luận “mắt/tóc/tay là giả” từ một ô đỏ, không suy rộng sang phần ngoài crop và không đổi quyết định gốc theo bản đồ. Nếu không ô nào làm điểm Fake giảm thì app nói đúng điều đó.

Trong repo có `tools/visualize_cam.py` cho baseline với loader và resize khác, cùng bản đồ độ lớn activation trong `tools/inspect_image.py`. Không dùng trực tiếp các hình này như bằng chứng định vị của package đã chọn. Để xác nhận “phần nào là giả” cần nhãn vị trí và đánh giá localization riêng; phần đó chưa được nghiệm thu.

Nếu thiếu model: kiểm đường dẫn. Sai SHA: lấy lại đúng package đã chọn, không sửa hằng số để bỏ qua. Port bận: chạy `demo.py --port 7861`. Upload lỗi: chọn lại ảnh tĩnh hợp lệ. GPU OOM: khởi động lại runtime hoặc dùng Windows CPU.
