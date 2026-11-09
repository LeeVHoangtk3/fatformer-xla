# Task 7.16 — hồ sơ tài liệu Phase 7

**Biên soạn và kiểm tra tự động hoàn tất; chờ cross-review/chữ ký thật A/B/C.** Không xác nhận toàn Phase 7 PASS.

| Cổng | Kết quả hiện hành | Bằng chứng / phần còn thiếu |
| :--- | :--- | :--- |
| G7.0 | **PENDING** theo DoD toàn repo | Canonical core: 3 checkpoint nạp 0/0, Gen 1 bị từ chối; còn loader legacy và CI chẩn đoán n3/n8 chưa đủ. |
| G7.1 | **FAIL** | Gen 2: Clean Fake@Real90 48,88% <55%; Q30 AUC 43,67% <55%. |
| G7.2 | **FAIL — 6/8** | Clean Fake@Real90 64,12% <65%; worst-subset Fake ACC 30,86% <45%. |
| G7.3 | **PENDING** | 20 ảnh và overhead L4 0,188055 ms/ảnh đạt; thiếu reference tốt nhất theo điều kiện. |

Hồ sơ để đọc theo thứ tự:

1. [Đối chiếu số cũ → mới → JSON pointer](../../../../docs/bao_cao/overall/bao_cao_ra_soat_so_lieu_phase7.md).
2. [Chương 5: kết luận và hạn chế](../../../../docs/bao_cao/overall/chuong_5_ket_luan_va_han_che.md).
3. [Slide 6 trang / 5 phút](../../../../docs/bao_cao/overall/slide_phase7_5phut.md).
4. [Biên bản, 16 task và 11 tài liệu](../../../../docs/milestones/milestone_phase7_acceptance.md).

`evidence_catalog.json` / `source_values.csv`: số và hash nguồn. `final_phase7_report.json`: view tài liệu v3, **không phải raw GPU report mới**. `gen2_verified_metadata.json`: hash actual Gen 2 và xung đột sidecar cũ. `training_summary.json`: có NaN gradient telemetry. `strict_loading_review.json`: các ngoại lệ/legacy còn mở. `legacy_number_review.json`: mọi số cũ còn giữ kèm phạm vi lịch sử. `original_documents/`: snapshot trước sửa. `human_signoff.json`: chưa có chữ ký.

Kiểm tra tái lập từ root repo:

```powershell
python -m tools.sync_phase7_documentation --check
```

Kiểm thử: 103 case pipeline + 53 case loss/selection/degradation = **156 case trên 9 file**, không phải toàn bộ repository. XML và kiểm tra tài liệu ở `validation/`. Không sửa notebook, weights hoặc kết quả đo; model mới chưa được gửi về local nên không chạy lại forward model tại đây.
