# Kiểm chứng demo — 14/11/2026

Đã triển khai `demo.py` dùng chung cho Windows CPU và Colab GPU. Đã chạy inference thật và kiểm HTTP trên Windows; Colab GPU và kiểm tra trực quan trình duyệt chưa nghiệm thu. Không thay đổi weights, calibration, quyết định chọn model hoặc trạng thái các gate nghiên cứu.

## Model và môi trường đã chạy

- Package SHA-256: `248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0`, 1.972.728.367 byte; canonical strict load, không cần CLIP checkpoint rời.
- Source checkpoint SHA-256: `5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c`; layout `[7,15,23]`; router tắt; T `2.7832491397857666`; threshold `0.7918508648872375`.
- Windows, Python 3.14.0, torch 2.10.0+cpu, torchvision 0.25.0+cpu, Pillow 12.0.0, Gradio 6.32.0 trong `.venv-demo`; CPU FP32, tối đa 4 torch threads.
- Dùng fallback Haar DWT native và fallback tokenizer hiện có trong repo khi thiếu `pytorch_wavelets`/`ftfy`. Không sửa kiến trúc hoặc bỏ qua mismatch keys để nạp model.

## Bằng chứng CPU

`demo.py --verify-only` kiểm SHA của đủ 20 ảnh Task 7.15, warmup 3 ảnh rồi so 40 lượt clean/Q30 với reference batch 1, cùng backend/precision. Log: [cpu_verification.json](../../logs/demo/cpu_verification.json), console tại `logs/demo/cpu_verification_console.txt`.

- Sai lệch xác suất lớn nhất: **0.0**; cả 40 nhãn khớp reference; tolerance `atol=1e-6`, `rtol=1e-5`.
- Cold load đo trong app: **5,47 giây**, gồm hash/load/kiểm metadata, không gồm khởi động Python/import và dựng UI. Lần chạy app HTTP mới đo **5,31 giây**.
- Backend một cặp clean+Q30: **p50 4.162 ms**, **p95 4.861 ms**, n=20. Timing không gồm reference forward, upload, queue hoặc trình duyệt. Đây là số đo của lượt validation trên máy hiện tại, có kiểm thử/import khác chạy trong một phần thời gian; chưa phải benchmark máy rảnh hoặc cam kết latency.
- Có 4 dự đoán clean và 8 dự đoán Q30 khác nhãn nguồn trong bộ minh họa này. Giữ nguyên toàn bộ kết quả, không đổi threshold. Các con số này không thay thế benchmark accuracy của nghiên cứu.

Ở bản demo ban đầu, đã chạy **30 tests đạt** trong `tests/test_demo.py`, `tests/test_unified_packaging.py`, `tests/test_canonical_loader.py` (8 test demo mới tại thời điểm đó). Kiểm tra ảnh dọc/ngang/nhỏ, RGB/RGBA/grayscale, EXIF, ảnh động/hỏng/quá giới hạn, pipeline Q30 sau crop, threshold khác 0.5 và checkpoint cùng tên sai nội dung. Hai warning router-disabled là hành vi dự kiến.

Lệnh đã dùng trên sandbox Windows để tránh cache/temp cũ không ghi được:

```powershell
.venv-demo/Scripts/python.exe -m pytest tests/test_demo.py tests/test_unified_packaging.py tests/test_canonical_loader.py -q --basetemp scratch/demo_tests_final_20261114 -p no:cacheprovider
```

## HTTP, nguồn Colab và phần chưa nghiệm thu

Kết quả upload thật, kiểm dữ liệu lỗi và chuỗi request lặp được lưu tại [http_verification.json](../../logs/demo/http_verification.json). Bộ kiểm tra gọi app đang phục vụ loopback sau một lần nạp model mới, không gọi trực tiếp callback hoặc dùng dummy model. Số đo RSS server ghi trong log này là các mẫu tại ranh giới request, không phải RAM tối thiểu hoặc peak liên tục. Profile RSS ban đầu chỉ đo venv launcher nên đã đánh dấu không hợp lệ trong `logs/demo/cpu_memory_profile.json`, không sử dụng để kết luận bộ nhớ model.

HTTP `/` trả 200; đường phục vụ file model trả 403. Upload ảnh đầu, clean/Q30 giữ cùng điểm clean, upload ảnh khác và đầu vào rỗng/hỏng đều qua kiểm tra. Sau lỗi, cả ảnh preview, crop và JSON kết quả được xóa. Thêm 20 request phân tích lặp lại cùng ảnh giữ nguyên điểm, không nạp lại model. RSS trước request đầu **2,30 GiB**, cao nhất trong 22 mẫu **2,414 GiB**; các mẫu cuối khoảng **2,41 GiB**, chưa thấy tăng không kiểm soát trong chuỗi ngắn này. Không suy rộng thành bảo đảm không rò bộ nhớ khi chạy lâu.

Notebook không có outputs/credentials, đã kiểm schema nbformat và cú pháp các code cell. Source ZIP có manifest SHA-256, BPE và 20 ảnh/CSV gốc; không chứa model, NPZ hay toàn bộ dataset. Notebook kiểm hash ZIP và từng file, giữ PyTorch CUDA của runtime, copy package từ Drive vào SSD, kiểm hash package và tự chạy verification GPU trước khi mở UI.

Chưa có phiên Colab GPU thực tế để xác nhận VRAM, latency, upload UI hoặc sai khác CPU/GPU. Cần chạy notebook từ runtime GPU mới và giữ `cuda_verification.json` để nghiệm thu. Không dùng log GPU lịch sử thay kết quả mới.

Không kiểm tra trực quan được trong phiên này: Browser Use báo `No browser is available`; Windows Computer Use báo native pipe không tồn tại. Vì vậy chưa xác nhận layout bằng screenshot, thao tác bàn phím, clear trong trình duyệt hoặc màn hình nhỏ. HTTP/API không thay thế các kiểm tra này. Chưa tắt mạng hệ điều hành để nghiệm thu offline hoàn toàn; app dùng weights/BPE local, analytics tắt và font hệ thống.

Giới hạn upload được chốt 32 MiB thay vì dự kiến 20 MiB, giữ mức 25 triệu pixel, để hai ảnh RAISE trong bộ verification gốc được chạy không cần sửa file. Không công bố link công khai hoặc upload source/model lên dịch vụ ngoài trong lượt triển khai Windows.

## Bổ sung quyết định và giải thích vùng ảnh — 14/11/2026

Theo yêu cầu làm rõ kết luận và chỉ ra vùng ảnh ảnh hưởng, app bổ sung quy tắc điểm/ngưỡng, khoảng cách đến ngưỡng và thay đổi nhãn/điểm sau JPEG Q30. Không đổi weights, threshold, calibration hoặc nhãn phân loại.

Tùy chọn giải thích chia crop gốc thành 16 ô 56×56, che từng ô bằng giá trị 0 sau normalize rồi đo thay đổi `prob_fake`. Bản đồ đỏ/xanh thể hiện delta có dấu và ô được đánh số. Dùng thang màu cố định; không kéo giãn tín hiệu nhỏ để mọi ảnh đều có vùng đỏ đậm. Không đặt tên bản đồ là vùng giả mạo hoặc suy ra lỗi mắt/tóc/tay. Cơ sở phương pháp: [Occlusion, tài liệu Captum](https://captum.ai/api/occlusion.html). Dùng PyTorch/Pillow hiện có, không thêm dependency hay sửa mô hình để lấy gradient.

Kiểm chứng mới:

- **14 tests demo đạt** (gồm các test cũ và 6 trường hợp mới). Kiểm điểm dưới/bằng/trên threshold, diễn giải đổi nhãn Q30, tọa độ ô và dấu delta bằng model có công thức biết trước; model hằng không bị gán vùng nghi vấn. Lệnh: `.venv-demo/Scripts/python.exe -m pytest tests/test_demo.py -q --basetemp scratch/demo_decision_tests_20261114 -p no:cacheprovider`.
- Hai ảnh mẫu nguồn REAL/FAKE chạy trên **model thật**; mỗi ảnh có 16 lượt che. Điểm gốc bằng log CPU trước đó; reference che ô đầu tính độc lập khớp; phân tích lại sau giải thích giữ nguyên điểm và nhãn. Log: [decision_explanation_verification.json](../../logs/demo/decision_explanation_verification.json).
- CPU FP32: phần giải thích mất **29,60 giây** và **30,95 giây** trên hai ảnh, ngoài lượt inference gốc. Không lấy hai số này làm p50/p95 hoặc suy ra thời gian Colab.
- Đã xem trực tiếp hai PNG overlay được tạo bởi model, kiểm số ô, dấu và màu: `logs/demo/influence_0.png`, `logs/demo/influence_2.png`. Đây là kiểm ảnh xuất, chưa phải kiểm layout app bằng trình duyệt.
- HTTP app mới trả 200; upload ảnh có bật giải thích trả đủ overlay/16 vùng, giữ cùng điểm gốc với chế độ tắt; Q30 có mô tả đổi điểm/nhãn. Đầu vào rỗng xóa cả bản đồ, văn bản giải thích cũ và kết quả. Log [decision_http_verification.json](../../logs/demo/decision_http_verification.json). Phần giải thích đo ở lượt HTTP này mất 29,48 giây trên CPU; GPU và tương tác trình duyệt vẫn chưa nghiệm thu.

Đầu ra JSON ghi `localization_validated=false`. Chưa có ground truth/mask và đánh giá localization cho tính năng này; phép che có thể làm ảnh ngoài phân phối và kết quả phụ thuộc baseline/kích thước ô. Muốn kết luận “phần nào là giả” cần một hạng mục định vị giả mạo riêng với dữ liệu và kiểm chứng tương ứng. Tùy chọn tắt mặc định để đường demo thông thường vẫn chạy như trước.
