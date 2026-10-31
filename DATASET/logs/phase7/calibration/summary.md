# BẢNG DUMMY — KHÔNG DÙNG NGHIỆM THU TASK 7.4
> **CẢNH BÁO**: Nguồn fit là logits synthetic; `evidence_level=synthetic`, `gate_eligible=false`.

## BẢNG TỔNG HỢP HIỆU CHUẨN NHIỆT ĐỘ & NGƯỠNG
> **Thời gian xuất**: 2026-10-28 16:11:50 | **Quy chuẩn**: Phụ lục A.2

| Model | Nguồn Fit | n_calib | Nhiệt Độ T | T* (FPR=5%) | T* (FPR=1%) | ECE Trước | ECE Sau | Real ACC | Fake ACC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | val_calib | 540 | 1.2511 | **0.8468** | 0.9480 | 7.30% | **4.38%** | 94.81% | 60.00% |

> **Ghi chú**: T* được quét độc lập theo phân vị Real score. ECE sau hiệu chuẩn giảm rõ rệt so với ban đầu.
