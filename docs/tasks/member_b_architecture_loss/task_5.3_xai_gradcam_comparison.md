# TASK 5.3: XUẤT BỘ ẢNH GRAD-CAM ĐỐI ĐẦU MINH CHỨNG TÍNH GIẢI THÍCH (XAI)

- **Mã nhiệm vụ**: `Task 5.3`
- **Người phụ trách**: **Thành viên B** *(Architecture & Loss Lead)*
- **Môi trường thực hiện**: 🖥️ **Local** / ☁️ **Colab T4**
- **Thời hạn hoàn thành**: Tuần 5 (Ngày 32)
- **Độ ưu tiên**: 🟠 Cao (Bằng chứng trực quan đắt giá nhất cho Báo cáo và Slide bảo vệ)
- **Đầu vào (Input)**:
  - Checkpoint mô hình gốc `fatformer_4class_ckpt.pth`.
  - Checkpoint hoàn thiện tối ưu `fatformer_srm_robust_final.pth`.
  - Cặp ảnh kiểm thử thuộc các generator hiện đại (Midjourney, Stable Diffusion, ProGAN) ở 2 trạng thái: Sạch (Clean) và Nén sâu ($Q=30$).
- **Đầu ra (Output)**:
  - Bộ 10 ảnh Grad-CAM độ phân giải cao so sánh song song đối đầu (Side-by-side comparison).
  - Tệp hình ảnh lưu trong `docs/assets/gradcam_comparison/`.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Chứng minh tính giải thích được (Explainable AI - XAI): Mô hình FatFormer-XLA không chỉ cải thiện số liệu Accuracy trên giấy tờ, mà cơ chế ra quyết định thực sự đã học được các đặc trưng vi sai bất biến trước biến dạng nén.
2. Trực quan hóa sự khác biệt cốt lõi:
   - **Mô hình gốc (Baseline)**: Khi ảnh bị nén $Q=30$, vùng kích hoạt nhiệt bị phân tán hoặc bám dính vào các đường lưới biên khối vuông của chuẩn nén JPEG ($8 \times 8$ blocking artifacts) $\rightarrow$ Đoán sai hoặc tự tin ảo.
   - **FatFormer-XLA (Đề xuất)**: Nhờ có nhánh SRM 3-Kernels và Cổng $\lambda(x)$ chủ động đóng dải tần nhiễu, bản đồ kích hoạt nhiệt vẫn hội tụ chuẩn xác vào các vùng tạo tác vi mô sinh ảnh (mắt, khóe môi, kết cấu da, viền tóc) $\rightarrow$ Dự đoán Real/Fake chính xác tuyệt đối.

---

## 2. HƯỚNG DẪN TRIỂN KHAI CHI TIẾT (STEP-BY-STEP)

### Xuất ảnh so sánh song song:
```python
import matplotlib.pyplot as plt
import cv2
import numpy as np

def save_side_by_side(orig_img, cam_baseline, cam_proposed, save_path, title):
    """Vẽ ảnh gốc, Grad-CAM baseline và Grad-CAM mô hình cải tiến song song."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Ảnh gốc bị nén
    axes[0].imshow(orig_img)
    axes[0].set_title("Ảnh Nén JPEG Q=30", fontsize=12)
    axes[0].axis('off')
    
    # 2. Grad-CAM Baseline gốc
    heatmap_base = cv2.applyColorMap(np.uint8(255 * cam_baseline), cv2.COLORMAP_JET)
    overlay_base = cv2.addWeighted(np.array(orig_img), 0.6, heatmap_base, 0.4, 0)
    axes[1].imshow(cv2.cvtColor(overlay_base, cv2.COLOR_BGR2RGB))
    axes[1].set_title("FatFormer Gốc (Bị nhiễu khối JPEG)", fontsize=12, color='red')
    axes[1].axis('off')
    
    # 3. Grad-CAM FatFormer-XLA
    heatmap_prop = cv2.applyColorMap(np.uint8(255 * cam_proposed), cv2.COLORMAP_JET)
    overlay_prop = cv2.addWeighted(np.array(orig_img), 0.6, heatmap_prop, 0.4, 0)
    axes[2].imshow(cv2.cvtColor(overlay_prop, cv2.COLOR_BGR2RGB))
    axes[2].set_title("FatFormer-XLA (Bền vững SRM+Gating)", fontsize=12, color='green')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Xuất đủ 10 cặp ảnh đối đầu rõ nét (300 DPI), đặt tên khoa học: `cam_comp_mj_01.png`, `cam_comp_sd_01.png`...
- [ ] Ảnh trực quan minh chứng rõ ràng hiện tượng đóng cổng tần số và bảo toàn vùng tập trung vào dị thường tạo tác AI.
- [ ] Chèn vào Chương 5 của Báo cáo đồ án và đưa vào slide thuyết trình chính trước hội đồng.
