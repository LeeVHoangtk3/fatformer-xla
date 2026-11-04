# BÁO CÁO NGHIỆM THU KỸ THUẬT: TASK 7.7
## XÂY DỰNG KHO DỮ LIỆU ĐA BỘ SINH v2 (CFE-SMALL PHÂN TẦNG + ẢNH THẬT ĐA NGUỒN) & KIỂM TOÁN CÔ LẬP CHỐT CHẶN G7.2

> **Mã nhiệm vụ**: `Task 7.7` *(Phase 7 — Tái lập đo lường & Hợp nhất, D1–D3)*  
> **Người thực hiện**: **Thành viên A (Data & Infrastructure Lead)**  
> **Người phối hợp & Nghiệm thu**: **Thành viên C (Training Lead)**  
> **Thời gian nghiệm thu**: 04/11/2026 (Hoàn thành sớm hơn mốc G7.2 ngày 05/11/2026)  
> **Môi trường thực thi**: Google Colab CPU | Google Drive 5TB (Stream Parquet theo batch + SSD Caching) + Local Audit  
> **Trạng thái**: `[x] ĐÃ HOÀN THÀNH XUẤT SẮC (PASS 100% TIÊU CHUẨN DoD & CHỐT CHẶN G7.2)`  
> **Hiện vật bàn giao chính thức**:
> - Gói dữ liệu nén nhị phân: `Fatformer/phase7/train_crossdomain_v2.tar` trên Google Drive
> - Chữ ký băm SHA-256 của gói: `16e5952892035ea0058cc6e996be191c43f6b3af608a6c92fb0a75ebc0a78dff`
> - Tệp kê khai toàn diện: [`DATASET/manifests/data_manifest_v2.json`](../../../DATASET/manifests/data_manifest_v2.json) (`c1e883fed2e0fdd55ad9732ddc3bd44e1d21438b009100134f91788cf6af218a`)
> - Tệp kê khai chia split huấn luyện: [`DATASET/manifests/data_manifest_v2_train.json`](../../../DATASET/manifests/data_manifest_v2_train.json) ($12.240$ mẫu)
> - Tệp kê khai tuyển chọn (Train + Val_Select): [`DATASET/manifests/data_manifest_v2_train_select.json`](../../../DATASET/manifests/data_manifest_v2_train_select.json) ($13.320$ mẫu)
> - Tệp kê khai hiệu chuẩn độc lập: [`DATASET/manifests/data_manifest_v2_val_calib.json`](../../../DATASET/manifests/data_manifest_v2_val_calib.json) ($1.080$ mẫu)
> - Tệp kê khai tuyển chọn mô hình: [`DATASET/manifests/data_manifest_v2_val_select.json`](../../../DATASET/manifests/data_manifest_v2_val_select.json) ($1.080$ mẫu)
> - Báo cáo kiểm toán cô lập đa chiều: [`DATASET/logs/phase7/data_isolation_report.json`](../../../DATASET/logs/phase7/data_isolation_report.json)
> - Danh sách bộ sinh bị cấm (Deny-list): [`DATASET/logs/phase7/generator_denylist.json`](../../../DATASET/logs/phase7/generator_denylist.json)
> - Notebook Colab hoàn chỉnh: [`notebooks/tasks/task_7.7/task_7.7_build_data_v2_colab.ipynb`](../../../notebooks/tasks/task_7.7/task_7.7_build_data_v2_colab.ipynb)

---

## I. MỤC TIÊU & CƠ SỞ KHOA HỌC

Kho dữ liệu Phase 6 cũ (`DATASET/manifests/data_manifest.json`, $3.600$ mẫu) chỉ có một nguồn duy nhất là `diffusion_staging` (gồm Stable Diffusion v1.5 và Midjourney v5). Khi đánh giá tổng thể trên $18$ tập test của Task 5.1 và $2.000$ ảnh của CommunityForensics (CFE), mô hình bị hổng kiến thức nghiêm trọng ở các họ kiến trúc mới giai đoạn 2024–2025 (`modern_gans`, `autoregressive`, `commercial`, `pixel_diffusion`).

Mục tiêu của **Task 7.7** là tái thiết toàn diện kho dữ liệu huấn luyện Gen 2 quy mô **$14.400$ mẫu** nhằm thỏa mãn đồng thời 4 điều kiện khắt khe:
1. **Đa dạng hoá cấu trúc sinh (5 họ Fake hiện đại)**: Phủ đều 5 họ kiến trúc sinh ảnh tiên tiến giai đoạn 2024–2025 (`autoregressive`, `commercial`, `latent_diffusion`, `modern_gans`, `pixel_diffusion`), khống chế độ lệch mất cân bằng giữa các họ $\le 1,50\times$.
2. **Đa dạng hoá nguồn ảnh Thật (Real Diversity)**: Phân bổ $\ge 3$ nguồn ảnh thật độc lập (COCO, FFHQ, LandscapesHQ, VISION, ImageNet) để triệt tiêu hiện tượng học vẹt dấu vết nén camera (Domain Shortcut Learning).
3. **Cô lập 100% học thuật (Zero Leakage)**: Tuyệt đối không trùng lặp byte (SHA-256) và tương đồng tri giác (pHash khoảng cách Hamming $\le 8$) với $2.000$ ảnh test CFE và $110.329$ ảnh thuộc $18$ tập test chuẩn của Task 5.1.
4. **Cô lập định danh bộ sinh (Generator-Disjoint)**: Không một bộ sinh nào trong tập huấn luyện trùng tên/phiên bản với các bộ sinh đã bị deny-list khóa ở tập test.

---

## II. KẾ THỪA THÔNG SỐ TỪ CÁC TASK TIỀN ĐỀ (RULE 4)

| Thông số kế thừa | Giá trị thực tế đã nghiệm thu | Nguồn dẫn xuất / Báo cáo tiền đề |
| :--- | :--- | :--- |
| **Danh mục 18 tập test bị khóa** | $8$ GANs ($62.003$ ảnh) + $10$ Diffusion ($20.000$ ảnh) + $5$ supplemental ($28.326$ ảnh) = $110.329$ ảnh | [Báo cáo Task 5.1](../../bao_cao/C/bao_cao_task_5.1_full_benchmark_18_datasets.md) |
| **Tập test chuẩn CFE** | $2.000$ ảnh CFE Test ($16$ bộ sinh) | [Báo cáo Task 1.3](../../bao_cao/A/bao_cao_task_1.3_chuẩn_bị_dữ_liệu_cfe.md) |
| **Nguồn dữ liệu Replay In-dist** | `diffusion_staging.tar` ($3.600$ mẫu cũ, giữ lại các mẫu sạch không trùng test) | [Báo cáo Task 3.1](../../bao_cao/A/bao_cao_task_3.1_prepare_genimage_staging.md) |
| **Hạ tầng lưu trữ Drive & Cache** | `Fatformer/datasets/` & `Fatformer/phase7/` trên Google Drive 5TB | [Báo cáo Task 1.1](../../bao_cao/A/bao_cao_task_1.1_ha_tang_drive_io.md) |
| **Cổng năng lực đích** | **Chốt chặn G7.2**: Yêu cầu `violations = 0` trước khi mở Task 7.11 | [Kế hoạch Phase 7](../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §II.3 & §IV.2 |

---

## III. KẾT QUẢ TRIỂN KHAI KỸ THUẬT & PHÂN BỐ KHO DỮ LIỆU v2

### 1. Quy Mô Tổng Thể & Cơ Cấu Phân Chia Split

Kho dữ liệu đã được tổng hợp, khử trùng lặp và phân tầng thành công đạt đúng **$14.400$ mẫu** (tổng dung lượng byte ảnh gốc: **$8.107.529.958$ bytes** $\approx 8,11\text{ GB}$):

- **Tập Train ($85\%$)**: **$12.240$ mẫu** ($6.683$ Real, $5.557$ Fake).
- **Tập Val Calibration ($7,5\%$)**: **$1.080$ mẫu** ($590$ Real, $490$ Fake) — *Dành riêng cho Task 7.4 fit nhiệt độ và Task 7.13 fit Degradation Router; cấm dùng để chọn checkpoint*.
- **Tập Val Selection ($7,5\%$)**: **$1.080$ mẫu** ($590$ Real, $490$ Fake) — *Dành riêng cho Task 7.10 chọn checkpoint theo Worst-Group*.
- **Tập Train Select (Ghép Train + Val_Select)**: **$13.320$ mẫu** — *Dành cho loader hiện tại của Task 7.11 để không bị trộn lẫn tập Calibration*.

### 2. Ma Trận Phân Bố Đa Chiều ($14.400$ Mẫu: `Family × Split × Label`)

$$\begin{array}{|l|r|r|r||r|r|r||r|r|r||r|}
\hline
\textbf{Kiến trúc (Family)} & \multicolumn{3}{c||}{\textbf{Train (85\%)}} & \multicolumn{3}{c||}{\textbf{Val\_Calib (7,5\%)}} & \multicolumn{3}{c||}{\textbf{Val\_Select (7,5\%)}} & \textbf{Tổng} \\
\cline{2-10}
& \text{Real} & \text{Fake} & \text{Tổng} & \text{Real} & \text{Fake} & \text{Tổng} & \text{Real} & \text{Fake} & \text{Tổng} & \\
\hline
\text{autoregressive} & 0 & 1.020 & 1.020 & 0 & 90 & 90 & 0 & 90 & 90 & \mathbf{1.200} \\
\text{commercial} & 0 & 1.020 & 1.020 & 0 & 90 & 90 & 0 & 90 & 90 & \mathbf{1.200} \\
\text{latent\_diffusion} & 0 & 1.488 & 1.488 & 0 & 131 & 131 & 0 & 131 & 131 & \mathbf{1.750} \\
\text{modern\_gans} & 0 & 1.020 & 1.020 & 0 & 90 & 90 & 0 & 90 & 90 & \mathbf{1.200} \\
\text{pixel\_diffusion} & 0 & 1.009 & 1.009 & 0 & 89 & 89 & 0 & 89 & 89 & \mathbf{1.187} \\
\text{real} & 6.683 & 0 & 6.683 & 590 & 0 & 590 & 590 & 0 & 590 & \mathbf{7.863} \\
\hline
\textbf{Tổng cộng} & \mathbf{6.683} & \mathbf{5.557} & \mathbf{12.240} & \mathbf{590} & \mathbf{490} & \mathbf{1.080} & \mathbf{590} & \mathbf{490} & \mathbf{1.080} & \mathbf{14.400} \\
\hline
\textbf{Tỷ lệ Real (\%)} & \multicolumn{3}{c||}{\mathbf{54,60\%}} & \multicolumn{3}{c||}{\mathbf{54,63\%}} & \multicolumn{3}{c||}{\mathbf{54,63\%}} & \mathbf{54,60\%} \\
\hline
\end{array}$$

### 3. Đánh Giá Độ Cân Bằng & Tính Hợp Đồng Dữ Liệu

1. **Tỷ lệ Real/Fake**: 
   - Đạt **$54,60\%$ Real** trên toàn bộ kho và dao động trong khoảng $[54,60\%, 54,63\%]$ ở cả 3 split $\rightarrow$ Nằm hoàn toàn trong khung mục tiêu khắt khe $[45\%, 55\%]$.
2. **Mức độ mất cân bằng giữa 5 họ sinh (Family Imbalance)**:
   - Họ lớn nhất: `latent_diffusion` ($1.750$ ảnh).
   - Họ nhỏ nhất: `pixel_diffusion` ($1.187$ ảnh).
   - Tỷ lệ chênh lệch: $\frac{1.750}{1.187} = \mathbf{1,4743\times} \le \mathbf{1,500\times}$ $\rightarrow$ **ĐẠT CHUẨN** ở toàn bộ 3 split (Train: $1,4748\times$, Val_calib: $1,4719\times$, Val_select: $1,4719\times$).
3. **Thành phần Replay In-dist**:
   - Tổng cộng **$2.991$ ảnh** mang nhãn `is_replay: true` ($1.792$ Real, $1.199$ Fake Stable Diffusion v1.5).
   - Chiếm **$20,77\%$** tổng dung lượng kho $\rightarrow$ Nằm chính xác trong dải yêu cầu $[20\%, 25\%]$.
   - Đã loại bỏ sạch $9$ ảnh Replay bị trùng pHash với 18 tập test Task 5.1 trước khi đưa vào kho.
4. **Chi tiết $17$ bộ sinh Fake & $5$ nguồn ảnh Thật**:
   - **Autoregressive ($1.200$)**: `tamingTransformers` ($1.200$).
   - **Commercial ($1.200$)**: `GPT-4o` ($1.200$).
   - **Pixel Diffusion ($1.187$)**: `DeepFloyd` ($1.187$).
   - **Modern GANs ($1.200$)**: $6$ bộ sinh đại diện cân bằng: `CIPS` ($200$), `Gansformer` ($200$), `GigaGAN` ($200$), `ProjectedGAN` ($200$), `StyleSANXL` ($200$), `StyleSwin` ($200$).
   - **Latent Diffusion ($1.750$)**: `stable_diffusion_v1_5` ($1.199$), `DiT` ($200$), `LFM` ($200$), cùng $5$ biến thể SDXL ($151$ ảnh): `animagine-xl-31` ($31$), `dreamshaperxlturbov21fix5` ($30$), `realcartoon-anime-v11` ($30$), `acorn-is-boning-xl` ($30$), `samaritan-3d-cartoon-v40` ($30$).
   - **Real Sources ($7.863$)**: `COCO` ($1.518$), `FFHQ` ($1.518$), `LandscapesHQ` ($1.518$), `VISION` ($1.517$), và `diffusion_staging` Replay ($1.792$).

---

## IV. BÁO CÁO KIỂM TOÁN CÔ LẬP ĐỘC LẬP CHỐT CHẶN G7.2

Báo cáo kiểm toán độc lập [data_isolation_report.json](../../../DATASET/logs/phase7/data_isolation_report.json) được thực thi bởi công cụ `tools/audit_manifest_v2.py` với thời gian kiểm định $512,7$ giây, đối chiếu $14.400$ bản ghi với toàn bộ kho băm tham chiếu.

### 1. Kết Quả Đối Chiếu 3 Cặp Độc Lập

```json
{
  "pass": true,
  "violations": 0,
  "integrity": {
    "failures": 0,
    "problems": []
  },
  "pairs": {
    "v2_vs_cfe_test": {
      "threshold": 8,
      "n_checked": 14400,
      "sha256_matches": 0,
      "phash_matches": 0,
      "violations": 0
    },
    "v2_vs_task51_18sets": {
      "threshold": 8,
      "n_checked": 14400,
      "sha256_matches": 0,
      "phash_matches": 0,
      "violations": 0
    },
    "internal_splits": {
      "threshold": 3,
      "violations": 0,
      "pairs": {
        "train__vs__val_calib": {"sha256_matches": 0, "phash_matches": 0},
        "train__vs__val_select": {"sha256_matches": 0, "phash_matches": 0},
        "val_calib__vs__val_select": {"sha256_matches": 0, "phash_matches": 0}
      }
    }
  }
}
```

### 2. Tuyên Bố Pháp Lý Học Thuật (Academic Compliance Declaration)

> **XÁC NHẬN CHÍNH THỨC**:
> - **Tổng số vi phạm (Violations)**: **`0`** (Không có bất kỳ sự trùng lặp nào về byte SHA-256 hay tương đồng thị giác pHash).
> - **Lỗi toàn vẹn (Integrity Failures)**: **`0`** (Mọi đường dẫn tương đối đều hợp lệ, không thiếu byte, không hỏng checksum).
> - **Phủ kín tập kiểm thử (Coverage Complete)**: **`true`** (Bảo đảm đối chứng đủ $18$ tập test Task 5.1 và $5$ tập phụ trợ với tổng cộng $110.329$ ảnh).
> - **Kết luận Chốt Chặn G7.2**: **`PASS` — ĐỦ ĐIỀU KIỆN TUYỆT ĐỐI ĐỂ MỞ TASK 7.11**.

---

## V. TỔNG KẾT TÀI NGUYÊN & HIỆU NĂNG THỰC THI TRÊN COLAB

- **Thời gian quét siêu dữ liệu (Metadata Indexing)**: $0,8$ giây (Tận dụng DuckDB / Parquet Footers).
- **Thời gian tải & xử lý $14$ Shard Parquet**: $937,6$ giây (~$15,6$ phút).
- **Băng thông mạng tiêu thụ thực tế**: $21,81\text{ GB}$ (Tiết kiệm $63,65\%$ so với ngân sách trần $60\text{ GB}$).
- **Thời gian chạy kiểm toán độc lập**: $512,7$ giây (~$8,5$ phút).
- **Thời gian đóng gói Tar & Tính SHA-256**: ~$60$ giây.

---

## VI. BIÊN BẢN BÀN GIAO CHO THÀNH VIÊN C (TASK 7.11)

Kho dữ liệu v2 đã sẵn sàng trên Google Drive và repo cục bộ. **Thành viên C (Training Lead)** có thể bắt đầu ngay lập tức **Task 7.11 (Anchored Recovery v2 Training)** theo các chỉ dẫn kỹ thuật sau:

### 1. Lệnh Giải Nén Dữ Liệu Trên Môi Trường Colab GPU
```bash
# Giải nén trực tiếp vào thư mục gốc của repo trên SSD Colab (/content/fatformer-xla)
tar -xf /content/drive/MyDrive/Fatformer/phase7/train_crossdomain_v2.tar -C /content/fatformer-xla
```

### 2. Hợp Đồng Đọc Dữ Liệu Cho Training Pipeline (Loader Contract)
- **Tập Huấn Luyện & Chọn Lọc Checkpoint**:
  Truyền đối số `--train_manifest DATASET/manifests/data_manifest_v2_train_select.json` (chứa $12.240$ ảnh train và $1.080$ ảnh val_select). Tuyệt đối **không** đưa `val_calib` vào loader này.
- **Tập Hiệu Chuẩn Ngưỡng & Router (Task 7.4 & Task 7.13)**:
  Chỉ đọc riêng tệp `DATASET/manifests/data_manifest_v2_val_calib.json` ($1.080$ ảnh).
- **Hàm phạt bảo vệ ảnh thật (`RealGuard Loss` - Task 7.9)**:
  Tỷ lệ ảnh thật trong tập train đạt $54,60\%$ ($6.683$ ảnh), cung cấp đủ độ dồi dào về mẫu âm tính sạch cho hàm `guard_loss`.

---

*Báo cáo được lập và ký nhận điện tử bởi Thành viên A (Data & Infrastructure Lead).*  
*Trạng thái Phase 7 Dashboard: Task 7.7 [x] Hoàn thành $\rightarrow$ Mở khóa Task 7.11.*

