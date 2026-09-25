# TASK 5.5: XÂY DỰNG ỨNG DỤNG LIVE DEMO WEB KÉO THẢ ẢNH

- **Mã nhiệm vụ**: `Task 5.5`
- **Người phụ trách**: **Thành viên A** *(Data & Infra Lead)*
- **Môi trường thực hiện**: 🖥️ **Local** (chạy trên máy tính cá nhân để trình chiếu hội đồng)
- **Thời hạn hoàn thành**: Tuần 5 (Ngày 33)
- **Độ ưu tiên**: 🟢 Trung bình (Tăng tính ứng dụng thực tế và điểm cộng bảo vệ)
- **Đầu vào (Input)**:
  - Checkpoint hoàn thiện cuối cùng `fatformer_srm_robust_final.pth`.
  - Backbone `ViT-L-14.pt` và cấu trúc mô hình hoàn chỉnh.
  - Các thư viện giao diện web: `streamlit` (hoặc `gradio`), `torch`, `Pillow`, `matplotlib`.
- **Đầu ra (Output)**:
  - Ứng dụng web trực quan [`tools/inference_demo.py`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/tools/inference_demo.py).

---

## 1. MỤC TIÊU KỸ THUẬT
1. Xây dựng một giao diện người dùng (UI) trực quan bằng Streamlit cho phép người dùng kéo thả trực tiếp một bức ảnh bất kỳ từ máy tính hoặc tải về từ mạng xã hội (Facebook, Telegram).
2. Tích hợp thanh trượt suy biến động (Dynamic Degradation Slider): Cho phép hội đồng kéo thanh trượt nén JPEG ($Q \in [10, 100]$) hoặc Blur để quan sát xem mô hình có giữ vững dự đoán chính xác hay không.
3. Hiển thị song song: Kết quả dự đoán (Real/Fake kèm % tự tin) + Ảnh nhiệt Grad-CAM giải thích vùng tạo tác.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Cài đặt `tools/inference_demo.py` bằng Streamlit:
```python
import streamlit as st
import torch
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="FatFormer-XLA Robust Forensics Demo", layout="wide")
st.title("🛡️ FatFormer-XLA: AI-Generated Image Forensics Live Demo")
st.markdown("Hệ thống phát hiện ảnh giả mạo AI bền vững trước biến dạng nén mạng xã hội.")

uploaded_file = st.sidebar.file_uploader("Tải lên ảnh cần giám định...", type=["jpg", "jpeg", "png"])
apply_jpeg = st.sidebar.slider("Mô phỏng nén mạng xã hội (JPEG Quality)", 10, 100, 95)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")
    
    # Áp dụng nén thử nghiệm nếu người dùng kéo thanh trượt
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=apply_jpeg)
    buf.seek(0)
    test_img = Image.open(buf)
    
    with col1:
        st.subheader("Ảnh Đầu Vào (Đã qua xử lý)")
        st.image(test_img, use_column_width=True)
        
    with col2:
        st.subheader("Kết Quả Phân Tích Pháp Y")
        with st.spinner("Đang trích xuất đặc trưng vi sai SRM & Tần số DWT..."):
            # Gọi hàm suy luận từ pipeline mô hình
            # dummy logic: prob_fake = model(test_img)
            prob_fake = 0.94  # Thay bằng inference thật
            if prob_fake >= 0.5:
                st.error(f"⚠️ Phát Hiện: **ẢNH SINH BỞI AI (FAKE)**")
                st.metric(label="Độ tin cậy AI", value=f"{prob_fake * 100:.1f}%")
            else:
                st.success(f"✅ Phát Hiện: **ẢNH CHỤP THẬT (REAL)**")
                st.metric(label="Độ tin cậy Thật", value=f"{(1 - prob_fake) * 100:.1f}%")
```

Chạy demo trên máy Local:
```bash
streamlit run tools/inference_demo.py
```

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Ứng dụng chạy mượt mà trên Localhost, không gặp lỗi nạp checkpoint.
- [ ] Trực tiếp demo được trên sân khấu khi bảo vệ đồ án: nén ảnh xuống $Q=30$ mô hình vẫn phát hiện chuẩn xác Real/Fake.
