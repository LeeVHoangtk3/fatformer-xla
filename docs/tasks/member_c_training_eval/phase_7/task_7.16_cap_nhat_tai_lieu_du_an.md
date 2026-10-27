# TASK 7.16 — CẬP NHẬT TÀI LIỆU DỰ ÁN & CHƯƠNG 5 / HẠN CHẾ

- **Mã nhiệm vụ**: `Task 7.16` *(Phase 7 — D10–D12)*
- **Người thực hiện**: **Cả 3 thành viên** — **C chủ trì & chịu trách nhiệm cuối**
- **Phân công**: A = tài liệu dữ liệu; B = tài liệu kiến trúc/checkpoint; C = kế hoạch, báo cáo, slide
- **Môi trường thực thi**: 🖥️ **Local**
- **Độ ưu tiên**: 🟠 Cao — điều kiện nghiệm thu Phase 7 (kế hoạch §IX.6)
- **Cổng liên quan**: **G7.3** (D10) — tài liệu phải khớp số của G7.3
- **Trạng thái**: `[ ] Sẵn sàng`
- **Kế hoạch gốc**: [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §VIII

---

## 1. Mục tiêu

Bảo đảm **không còn một câu nào trong dự án** nói sai về kết quả, sau khi Phase 7 đã thay đổi:

- Gen 1 bị `invalid_layout`;
- Ngưỡng không còn thích ứng theo batch;
- Metric chuyển từ "ACC" sang `AUC / AP / TPR@FPR`;
- Gen 2 thay thế WiSE-FT cũ;
- Có thể G7.2 **không đạt** — khi đó tài liệu phải nói đúng, không tô hồng.

> **Trạng thái 27/10/2026**: 11/11 file ở §VIII **đã được cập nhật cấu trúc** (banner, mục đính chính, bảng Gate, ĐIỂM DỪNG 6). Task 7.16 lúc này chỉ còn **điền số thật** từ Task 7.1 / 7.5 / 7.6 / 7.14 và **rà soát chéo**.

---

## 2. Đầu vào

| # | Đầu vào | Đường dẫn | Vai trò |
| :---: | :--- | :--- | :--- |
| I1 | Kết quả audit | `logs/phase7/checkpoint_structure_audit.json`, `wise_ft_legacy_reinterpretation.json` | Số cho Gen 1 |
| I2 | Kết quả Gen 2 | `logs/phase7/wise_ft_v2_sweep.csv` | Số cho Gen 2 |
| I3 | Re-baseline | `logs/phase7/cfe_v3_baseline_matrix.{json,md}` | Số cho G7.1 |
| I4 | Kết quả cuối | `logs/phase7/final_phase7_report.{json,md}` | Số cho G7.2/G7.3 |
| I5 | Đóng gói | `logs/phase7/latency_phase7.json` | Số độ trễ |
| I6 | 11 file mục tiêu | xem §3 | Đối tượng cập nhật |

---

## 3. Danh mục 11 file phải cập nhật

| # | File | Nội dung cần điền / sửa | Chủ trì |
| :---: | :--- | :--- | :---: |
| 1 | [`docs/plan/ke_hoach.md`](../../../plan/ke_hoach.md) | Bảng Tuần 7–8: số thật của 16 task; trạng thái cổng G7.0–G7.3; 2 rủi ro mới đã có — cập nhật "thực tế xảy ra" | C |
| 2 | [`docs/plan/ke_hoach_cac_cot_moc_va_diem_dung.md`](../../../plan/ke_hoach_cac_cot_moc_va_diem_dung.md) | ĐIỂM DỪNG 6: ghi PASS/FAIL từng cổng; sửa "chung cuộc" theo kết quả thật | C |
| 3 | [`docs/plan/tong_hop_checkpoints_va_lich_su_thuc_nghiem.md`](../../../plan/tong_hop_checkpoints_va_lich_su_thuc_nghiem.md) | §5.4: số đo `n=3` vs `n=8`; §7 Gen 2: điền `α`, `β`, số thật | B |
| 4 | [`docs/bao_cao/overall/bao_cao_kiem_dinh_communityforensics_eval.md`](../../../bao_cao/overall/bao_cao_kiem_dinh_communityforensics_eval.md) | Phần IX: bổ sung số v3; đánh dấu rõ bảng A/B cũ đã hết hiệu lực | C |
| 5 | [`docs/plan/ke_hoach_test_communityforensics_eval.md`](../../../plan/ke_hoach_test_communityforensics_eval.md) | Phụ lục A: cập nhật E1–E6 → "đã vá"; thêm kết quả harness v3 | C |
| 6 | [`docs/tasks/member_a_data_infra/README_A.md`](../../member_a_data_infra/README_A.md) | Điền trạng thái 7.7/7.8; số mẫu, dung lượng, thời gian stream | A |
| 7 | [`docs/tasks/member_b_architecture_loss/README_B.md`](../../member_b_architecture_loss/README_B.md) | Điền trạng thái 7.2/7.9/7.10/7.13; kết quả test | B |
| 8 | [`docs/tasks/member_c_training_eval/README_C.md`](../README_C.md) | Điền trạng thái 10 task; kết quả 4 cổng | C |
| 9 | [`docs/bao_cao/overall/README.md`](../../../bao_cao/overall/README.md) | Thêm liên kết báo cáo Phase 7 + báo cáo đính chính | C |
| 10 | [`docs/bao_cao/overall/tong_quan_du_an.md`](../../../bao_cao/overall/tong_quan_du_an.md) | Ràng buộc vị trí adapter (đã có) — bổ sung hệ quả thực nghiệm Gen 2 | B |
| 11 | [`docs/bao_cao/C/bao_cao_task_6.6_r_locked_recovery.md`](../../../bao_cao/C/bao_cao_task_6.6_r_locked_recovery.md) | Banner phạm vi (đã có) — cập nhật khi có kết quả Gen 2 | C |

---

## 4. Đầu ra

| # | Artifact | Đường dẫn | Nội dung |
| :---: | :--- | :--- | :--- |
| O1 | 11 file đã cập nhật | §3 | Không còn số cũ vô hiệu lực |
| O2 | Báo cáo rà soát | `docs/bao_cao/overall/bao_cao_ra_soat_so_lieu_phase7.md` | Bảng "số cũ → số mới → nguồn" |
| O3 | Chương 5 + Hạn chế | `docs/bao_cao/overall/chuong_5_ket_luan_va_han_che.md` | Kết luận, đóng góp, hạn chế, hướng phát triển |
| O4 | Slide 5 phút | `docs/bao_cao/overall/slide_phase7_5phut.md` | 6 slide, số liệu khớp §I4 |
| O5 | Biên bản nghiệm thu | `docs/milestones/milestone_phase7_acceptance.md` | 4 cổng + chữ ký 3 người |

---

## 5. Các bước thực hiện

### Bước 1 — Bảng "số cũ → số mới" (D10, ~1 h, C)

Trước khi sửa bất kỳ file nào, lập bảng nguồn:

| Số cũ | Xuất hiện ở | Số mới | Nguồn (file + trường JSON) | Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| `52.05%` | báo cáo CFE | … | `final_phase7_report.json` → `cfe.q30.fake_at_real90` | Số cũ dùng ngưỡng batch ⇒ **vô hiệu** |
| `4.10%` | báo cáo CFE | … | `cfe.clean.real_acc` (nếu là Real ACC) | |
| `96.58%` | Task 5.1/6.3 | … | `indist.macro_acc` | Số cũ là overall, không phải macro |
| `91.5%` | Phase 6-R | … | `cfe.clean.acc` | |

**Luật**: mỗi số mới **phải** trỏ được tới một trường JSON cụ thể. Không có nguồn ⇒ không được viết.

### Bước 2 — Cập nhật theo phân công (D10–D11, ~3 h)

**A (1 h)** — file #6:

- Trạng thái 7.7/7.8 (`[x]` / `[~]`).
- Số mẫu thật, số generator bị loại, dung lượng, thời gian stream.
- Nếu dùng kill criteria: ghi rõ "hạ xuống 8.000 ảnh vì …".

**B (1,5 h)** — file #3, #7, #10:

- §5.4: bảng `n=3` vs `n=8` với số thật + `verdict`.
- §7: Gen 2 với `α`, `β_srm`, `n_keys`, `sha256`.
- README_B: kết quả 4 task (test xanh, `0/0`, router thắng/không).

**C (2 h)** — file #1, #2, #4, #5, #8, #9, #11 + Chương 5 + slide:

- Kế hoạch & điểm dừng: PASS/FAIL từng cổng.
- Báo cáo CFE: Phần IX bổ sung số v3, nhấn mạnh **không trích dẫn số cũ**.
- Chương 5: 3 đóng góp + 3 hạn chế + 3 hướng phát triển.

### Bước 3 — Viết Chương 5 & Hạn chế (D11, ~1,5 h, C)

Cấu trúc bắt buộc:

```markdown
## 5.1. Kết luận
- Trả lời 3 câu hỏi nghiên cứu, mỗi câu 1 đoạn, kèm số + CI.

## 5.2. Đóng góp
1. Khế ước đo lường v3 (AUC/AP/TPR@FPR + CI) — thay cho ACC.
2. Chẩn đoán và sửa lỗi layout adapter (bằng chứng: 10 khối, không biểu diễn được).
3. Quy trình tái lập: canonical loader + calibration tách rời + ablation 4 cấu hình.

## 5.3. Hạn chế (bắt buộc trung thực)
- Nếu G7.2 trượt: nêu rõ **thiếu bao nhiêu pp** ở điều kiện nào.
- Chỉ có 2.000 ảnh CFE ⇒ CI rộng; không kết luận cho generator ngoài 5 subset.
- Backbone đóng băng ⇒ trần năng lực; đã thử đòn bẩy nào, kết quả ra sao.
- Router thắng/không thắng, kèm số.

## 5.4. Hướng phát triển
```

**Cấm**: viết "đạt 96,58%" mà không nói đó là overall ACC của tập nào; viết "vượt baseline" khi chênh < nửa CI.

### Bước 4 — Rà soát chéo & grep (D12, ~0,5 h, cả 3)

```bash
# 1. Số cũ không được xuất hiện như kết quả hiện hành
grep -rn "52\.05\|4\.10\|96\.58" docs/ | grep -v "invalid_layout" | grep -v "chỉ để tham chiếu"

# 2. Không còn strict=False
grep -rn "strict=False" src/ tools/ docs/

# 3. Không còn mô tả WiSE-FT cũ như đang hiệu lực
grep -rn "WiSE-FT" docs/plan/ docs/bao_cao/ | grep -v "invalid_layout" | grep -v "Gen 2"

# 4. Mọi bảng có CI
grep -rLn "CI 95%" docs/bao_cao/**/*.md
```

Mỗi kết quả grep phải được xử lý hoặc giải thích trong biên bản.

### Bước 5 — Slide & biên bản (D12, ~1 h, C)

Slide 6 trang:

1. Vấn đề: Gen 1 `invalid_layout` (1 hình).
2. Cách sửa: canonical loader + khế ước v3.
3. Kết quả G7.0/G7.1: bảng 4 cấu hình.
4. Kết quả G7.2: 6 chỉ số vs ngưỡng.
5. Kết quả G7.3: 1 checkpoint, độ trễ.
6. Hạn chế + hướng phát triển.

Biên bản `milestone_phase7_acceptance.md`: 4 cổng, PASS/FAIL, người xác nhận, ngày, **gắn đường dẫn tới bằng chứng**.

---

## 6. Definition of Done

- [ ] **11/11** file ở §3 đã cập nhật (đánh dấu trong biên bản).
- [ ] `bao_cao_ra_soat_so_lieu_phase7.md` có bảng "số cũ → số mới → nguồn JSON" cho **mọi** số bị thay đổi.
- [ ] `grep "52.05\|4.10\|96.58"` không còn kết quả nào **không** kèm nhãn `invalid_layout` hoặc "chỉ để tham chiếu lịch sử".
- [ ] Chương 5 có đủ 4 mục 5.1–5.4; mục Hạn chế nêu **số pp thiếu** nếu G7.2 trượt.
- [ ] Slide 6 trang, số liệu khớp `final_phase7_report.json` **100%** (không làm tròn khác).
- [ ] Biên bản nghiệm thu có 4 cổng + đường dẫn bằng chứng + 3 người xác nhận.
- [ ] `grep "strict=False"` trong `src/`, `tools/`, `docs/` — không còn trong đường báo cáo.
- [ ] Không còn câu nào mô tả WiSE-FT cũ là đang hiệu lực.
- [ ] **Rà soát chéo**: mỗi thành viên đọc ít nhất 1 file của người khác và ký nhận.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
| :--- | :---: | :---: | :--- |
| Tài liệu trôi khỏi số liệu | **Cao** | Phản biện bắt lỗi | Bảng nguồn bắt buộc; grep tự động ở bước 4 |
| Số trong slide khác số trong báo cáo | Cao | Mất uy tín | Sinh slide **từ** JSON, không gõ tay |
| G7.2 trượt nhưng tài liệu vẫn viết như đạt | Trung bình | Vi phạm học thuật | Câu chốt bắt buộc ở §5.3: nêu số pp thiếu |
| 3 người sửa cùng file | Trung bình | Xung đột | Phân công theo §3, mỗi file **một** chủ trì |
| Hết thời gian D12 | Trung bình | Nộp thiếu | Ưu tiên #1, #2, #4, #8, Chương 5; slide có thể làm sau |
| Số cũ nằm trong ảnh/hình (không grep được) | Trung bình | Lọt số sai | Rà hình thủ công: ROC, bảng chụp màn hình |

---

## 8. Liên kết

- Kế hoạch Phase 7: [`../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) — §VIII, §IX
- Kết quả cuối: [`task_7.14_locked_benchmark_phase7.md`](task_7.14_locked_benchmark_phase7.md)
- Đóng gói: [`task_7.15_unified_packaging.md`](task_7.15_unified_packaging.md)
- Báo cáo đính chính: [`../../../bao_cao/overall/bao_cao_kiem_dinh_communityforensics_eval.md`](../../../bao_cao/overall/bao_cao_kiem_dinh_communityforensics_eval.md) — Phần IX
- Chỉ mục giai đoạn: [`README.md`](README.md)
