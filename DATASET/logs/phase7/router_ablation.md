# Task 7.13 — code đã kiểm thử, thực nghiệm router hoãn

Trạng thái ablation: `awaiting_measured_evaluation`; evidence: `unverified`; router chưa có bằng chứng thắng.

Lựa chọn triển khai: calibrated fixed fallback. Chưa xác minh deployment hoặc đóng gói trên checkpoint v2; fallback cần calibration đo thật đúng checkpoint. FPR <=5% là ràng buộc thực nghiệm trên calibration, không bảo đảm evaluation/production.

Kết quả cũ tại commit 13749f6 bị hủy vì sinh tín hiệu giả lập và nội suy ablation/CI. Chia sẻ logits không chứng minh router không thể đổi ranking; chưa kết luận thắng hoặc thua khi chưa đo.

Wrapper yêu cầu report đủ năm subset, metric/CI/count và fingerprint nguồn, tính lại verdict; report thiếu hoặc mâu thuẫn không bật router. Đây là kiểm tra hash/tính nhất quán, không phải chữ ký số hay xác minh độc lập số đo.

Implementation đã kiểm thử không thay thế ablation đo thật hoặc nghiệm thu Task 7.15/G7.3. Marker JSON tương ứng tiếp tục ghi unverified, không chứa số đo thay thế.

[Báo cáo và giới hạn](../../../docs/bao_cao/B/bao_cao_task_7.13_degradation_router.md).
