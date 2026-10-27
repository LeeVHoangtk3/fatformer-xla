# TASK 2.5: CHỦ TRÌ HỌP NHÓM MILESTONE 1 - CHỐT MỐC SÀN THỰC NGHIỆM

- **Mã nhiệm vụ**: `Task 2.5`
- **Người phụ trách**: **Thành viên C** *(Chủ trì)*, **Cả 3 thành viên**
- **Môi trường thực hiện**: 🖥️ **Họp trực tuyến / Offline**
- **Thời hạn hoàn thành**: Tuần 2 (Ngày 14)
- **Độ ưu tiên**: 🔴 Khẩn cấp (Cổng kiểm soát tiến độ Milestone 1)
- **Đầu vào (Input)**:
  - Bảng số liệu Baseline Clean (Task 2.2).
  - Bảng số liệu Baseline Degraded (Task 2.3).
  - Bộ ảnh Grad-CAM đối chứng mốc sàn (Task 2.4).
- **Đầu ra (Output)**:
  - **Biên bản kỹ thuật Milestone 1** (ký duyệt bởi cả 3 thành viên, lưu trong `docs/milestones/milestone_1_freeze.md`).
  - Lượng hóa mục tiêu cải tiến cho giai đoạn huấn luyện finetune.

---

## 1. MỤC TIÊU KỸ THUẬT
1. Thống nhất và đóng băng (*freeze*) số liệu thực nghiệm ban đầu của mô hình gốc: Không thay đổi hay chỉnh sửa số liệu mốc sàn sau thời điểm này.
2. Lượng hóa mục tiêu định lượng (*Quantitative Target*) cho đồ án:
   - Mục tiêu 1: Cải thiện độ chính xác trên tập nén sâu $Q=30$ đạt mức $+10\%$ đến $+15\%$ so với baseline.
   - Mục tiêu 2: Duy trì độ chính xác trên tập ảnh sạch (Clean) không bị suy giảm quá $0.5\%$.
   - Mục tiêu 3: Mở rộng khả năng nhận diện ảnh Diffusion thế hệ mới (Midjourney, SD v1.5) vượt trên $90\%$.
3. Rà soát tiến độ viết Báo cáo Chương 1 và Chương 2 của Thành viên B.

---

## 2. NỘI DUNG BIÊN BẢN HỌP MILESTONE 1
- Xác nhận trạng thái hoàn thành của 6 task đầu tiên (Tuần 1 & Tuần 2).
- Phê duyệt bảng chỉ số mốc sàn:
  - Clean Baseline: GANs 98.4%, Diffusion 95.0%.
  - Degraded Baseline ($Q=30$): Ghi nhận sụt giảm xuống còn bao nhiêu %.
- Phê duyệt kế hoạch chuyển giao sang Tuần 3: Chuẩn bị dữ liệu Staging, hoàn thiện module SRM + Gating và vượt qua cổng Smoke Test Gate.

---

## 3. TIÊU CHUẨN NGHIỆM THU (DEFINITION OF DONE - DoD)
- [ ] Biên bản `milestone_1_freeze.md` được cả 3 thành viên ký duyệt và commit lên Git.
- [ ] Chính thức kích hoạt giai đoạn can thiệp kiến trúc mô hình (Tuần 3).
