# BÁO CÁO NGHIỆM THU KỸ THUẬT: KIỂM CHUẨN NGOẠI VI COMMUNITYFORENSICS-EVAL (CVPR 2025)
## ĐÁNH GIÁ ĐA CHIỀU: NĂNG LỰC TỔNG QUÁT HÓA ZERO-SHOT & ĐỘ BỀN VỮNG MẠNG XÃ HỘI

> **Mã nhiệm vụ**: `Task CFE-Benchmark` *(External Zero-shot OOD Generalization)*  
> **Người thực hiện**: **Thành viên C (Training & Evaluation Lead)**  
> **Người phối hợp**: **Thành viên B (Architecture Lead)**, **Thành viên A (Data & Infrastructure Lead)**  
> **Tệp mã nguồn kiểm chuẩn**: [`tools/eval_communityforensics.py`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/tools/eval_communityforensics.py)  
> **Notebook thực thi**: [`notebooks/eval_communityforensics.ipynb`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/notebooks/eval_communityforensics.ipynb)  
> **Môi trường thực thi**: Google Colab GPU (NVIDIA Tesla T4, 14.56 GB VRAM, PyTorch `2.11.0+cu130`)  
> **Tập dữ liệu kiểm định**: `test_communityforensics.tar` (4.86 GB, 2.000 ảnh chuẩn hóa 224×224 RGB)  
> **Thời điểm nghiệm thu**: 26/10/2026 (lúc 18:59:15)  
> **Trạng thái nghiệm thu**: `[x] ĐÃ HOÀN THÀNH — SỐ LIỆU ĐO ĐẠC THỰC TẾ, CÓ KÈM PHÂN TÍCH VẤN ĐỀ KỸ THUẬT`

> 🚨 **ĐÍNH CHÍNH 27/10/2026 — BẮT BUỘC ĐỌC PHẦN IX TRƯỚC KHI TRÍCH DẪN**
> Sau khi kiểm chứng trực tiếp cấu trúc 4 checkpoint bằng `torch.load`, báo cáo này được xác nhận là **có giá trị cho Baseline và Anchored Recovery**, nhưng **mọi số liệu của cột WiSE-FT bị vô hiệu** (`status: invalid_layout`):
> - Checkpoint `fatformer_wise_ft_best.pth` chứa **10 khối adapter** `{2,5,7,8,11,14,15,17,20,23}`; kiến trúc chỉ biểu diễn được `n ∈ {3,4,6,8,12,24}` ⇒ **không nạp đầy đủ được**. Tại `7,15,23` nó còn mất cả nhánh SRM + Gating (66 vs 94 tensor/khối).
> - Vì vậy **các số `52,05%` (clean overall), `4,10%` (clean Fake), `0,20%` (Q30 Fake), `100,00%` (Real)** và cả **`96,58%` in-dist** đều **không phải kết quả hiện hành**. Không dùng để kết luận, không dùng để so sánh "hơn/kém", không đưa vào slide.
> - Nguyên nhân gốc không phải "146 unexpected keys" mà là **ánh xạ khối adapter**. Chi tiết: **Phần IX** dưới đây và [`ke_hoach_phase_7_dieu_chinh_toan_dien.md`](../../plan/ke_hoach_phase_7_dieu_chinh_toan_dien.md) §I.

---

## I. ĐỐI CHIẾU & KẾ THỪA BÁO CÁO NHIỆM VỤ TIỀN ĐỀ (TUÂN THỦ QUY TẮC 4)

Trước khi đánh giá, cần đối chiếu với hai bộ số liệu nền đã được nghiệm thu chính thức:

> 🚫 **Cột WiSE-FT trong bảng dưới đây là `invalid_layout`** — xem banner đầu báo cáo và Phần IX. Giá trị được giữ lại chỉ để truy vết lịch sử, **không phải kết quả hiện hành**.

| Chỉ Số Kỹ Thuật | Baseline CVPR 2024 | WiSE-FT (Task 6.1-R) 🚫 `invalid_layout` | Anchored Recovery (Task 6.3-R) |
| :--- | :---: | :---: | :---: |
| **Clean Macro ACC (18 tập in-dist.)** | 96.45% | ~~96.58%~~ | 77.31% |
| **Clean Real ACC (in-dist.)** | 99.37% | ~~98.10%~~ | 57.01% |
| **Clean Fake ACC (in-dist.)** | 93.54% | ~~95.07%~~ | **97.61% 🔥** |
| **Q=30 Fake ACC (in-dist.)** | 5.78% | ~~6.73%~~ | **40.42% 🏆** |
| **Q=30 Fake ACC (OOD-CFE)** | 0.90% | ~~0.20%~~ | **40.80%** |
| **Clean ROC-AUC (in-dist.)** | 99.10% | ~~99.15%~~ | 96.71% |

*Nguồn in-distribution: [`bao_cao_task_6.6_r_locked_recovery.md`](file:///d:/GIT%20REPO/.nam4/fatformer-xla/docs/bao_cao/C/bao_cao_task_6.6_r_locked_recovery.md)*  
*Nguồn OOD-CFE: Kết quả thực nghiệm tại notebook ngày 26/10/2026.*

---

## II. TỔNG QUAN TẬP KIỂM CHUẨN NGOẠI VI (OOD Dataset)

Toàn bộ 2.000 ảnh được trích xuất từ **CommunityForensics-Eval (CVPR 2025)** — bộ sinh AI thế hệ 2024–2025 hoàn toàn ngoài phạm vi huấn luyện:

| Tập Con | Họ Bộ Sinh (Đại Diện) | Số Mẫu (Real / Fake) | Mức Độ OOD |
| :--- | :--- | :---: | :---: |
| `1_commercial` | Midjourney v6, DALL-E 3 | 600 (300 / 300) | **Rất cao** — API thương mại đóng hộp |
| `2_latent_diffusion` | SDXL, Flux.1 | 600 (300 / 300) | **Cao** — Thế hệ mới của LDM |
| `3_pixel_diffusion` | DeepFloyd-IF, ADM | 400 (200 / 200) | **Trung bình** — Gần với GLIDE/LDM cũ |
| `4_modern_gans` | StyleGAN3, GigaGAN | 200 (100 / 100) | **Thấp** — Gần với tập train ProGAN/SG2 |
| `5_other_autoregressive` | Muse, LlamaGen | 200 (100 / 100) | **Rất cao** — Kiến trúc hoàn toàn mới |
| **Tổng cộng** | **5 họ kiến trúc** | **2.000 (1.000 / 1.000)** | |

---

## III. KẾT QUẢ THỰC NGHIỆM ĐẦY ĐỦ

### Bảng A: Điều Kiện Ảnh Sạch Phòng Lab (Clean OOD)

| Tập Kiểm Chuẩn Con | Baseline ($T^*=0.50$) | WiSE-FT 🚫 `invalid_layout` ($T=1.08, T^*=0.50$) | Anchored ($T=1.28, T^*=0.40$) |
| :--- | :---: | :---: | :---: |
| ACC `1_commercial` | 56.17% | ~~50.00%~~ | **73.17%** |
| ACC `2_latent_diffusion` | 64.50% | ~~50.00%~~ | **72.00%** |
| ACC `3_pixel_diffusion` | **65.00%** | ~~50.00%~~ | 62.75% |
| ACC `4_modern_gans` | **95.00%** | ~~70.50%~~ | 71.50% |
| ACC `5_other_autoregressive` | 59.50% | ~~50.00%~~ | **60.50%** |
| **Subset-Macro Mean ACC** | **68.03%** | ~~54.10%~~ | 67.98% |
| **Overall Pooled Micro ACC** | 64.65% | ~~52.05%~~ | **69.30%** |
| **Real ACC** | 94.00% | ~~100.00%~~ | 48.10% |
| **Fake ACC** | 35.30% | ~~4.10%~~ | **90.50%** |
| **ROC-AUC** | 76.21% | ~~57.07%~~ | **79.28%** |
| **Average Precision (AP)** | **78.22%** | ~~60.41%~~ | 73.53% |

> 🚫 Toàn bộ giá trị ở cột WiSE-FT đã bị gạch bỏ (vô hiệu từ 27/10/2026). Xem Phần IX.2–IX.3 để biết nguyên nhân (10 khối adapter, fallback `n=3`).

### Bảng B: Điều Kiện Nén Sâu Mạng Xã Hội (JPEG Q=30 OOD)

| Chỉ Số Đo Đạc | Baseline | WiSE-FT 🚫 `invalid_layout` | Anchored |
| :--- | :---: | :---: | :---: |
| Fake ACC `1_commercial` | 0.00% | 0.00% | **66.67%** |
| Fake ACC `2_latent_diffusion` | 0.00% | 0.00% | **37.67%** |
| Fake ACC `3_pixel_diffusion` | 0.50% | 0.00% | **20.00%** |
| Fake ACC `4_modern_gans` | 8.00% | 2.00% | **19.00%** |
| Fake ACC `5_other_autoregressive` | 0.00% | 0.00% | **36.00%** |
| **Subset-Macro Fake ACC** | 1.70% | 0.40% | **35.87%** |
| **Overall Pooled Fake ACC** | 0.90% | 0.20% | **40.80%** |
| Overall Macro ACC | 50.35% | 50.10% | 49.60% |
| Real ACC (Q=30) | 99.80% | **100.00%** | 58.40% |
| ROC-AUC (Q=30) | 50.00% | 50.93% | **50.07%** |
| AP (Q=30) | 55.10% | 54.11% | 50.57% |

> 🚫 Cột WiSE-FT của Bảng B cũng bị vô hiệu (`invalid_layout`). Các giá trị được giữ nguyên (không gạch) chỉ để đối chiếu với báo cáo 26/10; **không dùng làm kết quả**.
> ℹ️ **Lưu ý phương pháp**: `ROC-AUC (Q=30)` của cả 3 mô hình đều xấp xỉ 50% ⇒ **thứ hạng xác suất gần như ngẫu nhiên** dưới nén sâu. Điều này có nghĩa "Fake ACC 40,80%" của Anchored chủ yếu đến từ việc **hạ ngưỡng** (đánh đổi Real ACC xuống 58,40%), không phải từ năng lực phân tách thật. Đây là lý do Phase 7 đặt `AUC Q30 ≥ 60%` làm tiêu chí G7.2 thay vì chỉ nhìn Fake ACC.

---

## IV. PHÂN TÍCH KỸ THUẬT CHUYÊN SÂU & CÁC VẤN ĐỀ CẦN LƯU Ý

> [!WARNING]
> Phần này ghi nhận các **điểm bất thường kỹ thuật** trong kết quả thực nghiệm. Đây là phân tích trung thực phục vụ báo cáo đồ án.

### 1. Hiện Tượng Suy Biến Phân Loại (Classifier Collapse) của WiSE-FT trên OOD

Kết quả WiSE-FT trên 4/5 tập con Clean cho thấy pattern **mang dấu hiệu Classifier Collapse**:

```
Tất cả tập con ngoại trừ 4_modern_gans:
  ACC = 50.00% | Real ACC = 100.00% | Fake ACC = 0.00%
```

Đây không phải là "bảo toàn ảnh thật tốt" — đây là mô hình dự đoán **100% toàn bộ mẫu là Real**, tức là hệ số phân loại Fake hoàn toàn bằng 0. Điều này có nghĩa:
- Trên 4/5 tập con OOD (Commercial, LatentDiff, PixDiff, Autoregressive): WiSE-FT **mù hoàn toàn** với ảnh giả mới.
- Riêng `4_modern_gans` (StyleGAN3, GigaGAN): gần với tập train → mô hình bắt được 41.00% Fake.

**Nguyên nhân kỹ thuật nghi vấn**: Trong log thực thi, WiSE-FT báo **146 unexpected keys** khi nạp checkpoint:
```
Unexpected keys (146): ['srm.proj.4.weight', 'srm.proj.1.bias', 'srm.k1',
                         'srm.proj.0.weight', 'gating.mlp.0.weight']...
```

> 🚨 **CHẨN ĐOÁN NÀY ĐÃ BỊ SỬA (27/10/2026)** — xem Phần IX.3. 146 key thực chất chỉ là `srm.*`/`gating.*` top-level đã chết (`CLIPModel.srm = None`), **không phải** nguyên nhân. Nguyên nhân thật: checkpoint có **10 khối adapter**, eval cũ fallback `len(indices) > 8 → n=3` nên **chỉ nạp 3/10 khối** và bỏ 5 khối M4 thuần. Nói cách khác, mô hình chạy với phần lớn thân mạng **không được nạp từ checkpoint**, không phải "backbone CLIP thuần túy".

~~Điều này cho thấy các trọng số **SRM và Gating** trong checkpoint WiSE-FT có kiến trúc chiều sâu khác với định nghĩa model hiện tại trong `build_model()` → các module SRM/Gating **không được nạp thực sự**, mô hình hoạt động như backbone CLIP + FAA adapter thuần túy. Khi thiếu module lọc tần số, mô hình không có cơ chế phân biệt OOD → mặc định dự đoán "Real" cho mọi ảnh OOD theo phân phối xác suất bảo thủ.~~

**Đối chiếu**: ~~Trên 18 tập in-distribution (Task 6.6-R), WiSE-FT đạt 96.58% ACC và 95.07% Fake ACC — chứng tỏ in-distribution WiSE-FT hoạt động bình thường.~~ 🚫 Các số này thuộc checkpoint `invalid_layout`, không dùng để lập luận. Việc "in-distribution ổn" của Gen 1 chưa từng được đo trên một lần nạp đầy đủ.

### 2. Phân Tích Baseline CVPR 2024: Phụ Thuộc Phân Phối GAN Huấn Luyện

Baseline cho thấy sự lệch phân phối rõ rệt theo họ kiến trúc:

| Họ Bộ Sinh | Clean Fake ACC (OOD) | Lý Giải |
| :--- | :---: | :---: |
| `4_modern_gans` (StyleGAN3, GigaGAN) | **98.00%** | Gần với ProGAN/SG1/SG2 trong tập train |
| `3_pixel_diffusion` (DeepFloyd, ADM) | **36.50%** | Có GLIDE/ADM trong tập train (guided, ldm_*) |
| `2_latent_diffusion` (SDXL, Flux.1) | 34.33% | Xa hơn với DALL-E 1, LDM cũ |
| `5_other_autoregressive` (Muse, LlamaGen) | 25.00% | Kiến trúc tự hồi quy hoàn toàn mới |
| `1_commercial` (MJ v6, DALL-E 3) | 18.00% | Mô hình đóng hộp, dấu vết artifact khác biệt nhất |

**Kết luận**: Baseline bị phụ thuộc trực tiếp vào phân phối tập huấn luyện. Độ sụt giảm tuyến tính theo mức độ OOD là bằng chứng thực nghiệm điển hình của hiện tượng **In-distribution Bias**.

### 3. Anchored Recovery: Điểm Mạnh Thực Sự & Đánh Đổi Cần Minh Bạch

**Điểm mạnh được xác nhận thực nghiệm:**
- Fake ACC trên Clean OOD đạt **90.50%** — vượt Baseline (35.30%); ~~vượt WiSE-FT (4.10%)~~ — không so sánh với checkpoint `invalid_layout`.
- Q=30 Fake ACC đạt **40.80%** — nhất quán với in-distribution (Task 6.6-R: 40.42%), xác nhận tính ổn định của checkpoint. ⚠️ Nhưng `AUC Q30 = 50,07%` ⇒ **thứ hạng gần như ngẫu nhiên**; 40,80% chủ yếu là hệ quả hạ ngưỡng (`T*=0.40`), đánh đổi Real ACC còn 58,40%. Phase 7 phải nâng **AUC**, không chỉ dịch ngưỡng.
- Kết quả nổi bật nhất: **Commercial AI (MJ v6, DALL-E 3) Q=30 Fake ACC = 66.67%** — họ bộ sinh khó nhận diện nhất.

**Đánh đổi cần minh bạch trong báo cáo:**
- **Clean Real ACC = 48.10%**: Hơn một nửa ảnh thật (51.9%) bị phân loại nhầm thành Fake. Đây là tỷ lệ **False Alarm** quá cao cho ứng dụng thực tế.
- **Q=30 Real ACC = 58.40%**: Cải thiện hơn dưới nén sâu nhưng vẫn còn 41.6% False Alarm.
- Nguyên nhân: Ngưỡng $T^*=0.40$ (thấp hơn chuẩn 0.50) và nhiệt độ $T=1.28$ khiến mô hình nhạy cảm quá mức với ảnh thật OOD.
- **ROC-AUC Q=30 = 50.07%** (gần bằng đồng xu ngẫu nhiên): Phản ánh giới hạn vật lý thực sự — tín hiệu pháp y bị xóa sổ gần hoàn toàn bởi JPEG Q=30 trên ảnh thế hệ mới. Con số 40.80% Fake ACC chủ yếu đến từ sự thiên lệch ngưỡng chứ không phải từ AUC cao.

---

## V. MA TRẬN ĐỐI CHIẾU IN-DISTRIBUTION vs. OOD (ĐO LƯỜNG DEGRADATION GAP)

| Chỉ Số | WiSE-FT In-dist (18 tập) 🚫 | WiSE-FT OOD (CFE) 🚫 | Gap 🚫 | Anchored In-dist | Anchored OOD (CFE) | Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean Macro ACC | ~~96.58%~~ | ~~54.10%~~ | ~~-42.5 pp~~ | 77.31% | **67.98%** | **-9.3 pp** |
| Clean Fake ACC | ~~95.07%~~ | ~~4.10%~~ | ~~-90.97 pp~~ | **97.61%** | **90.50%** | **-7.1 pp** |
| Q=30 Fake ACC | ~~6.73%~~ | ~~0.20%~~ | ~~-6.5 pp~~ | **40.42%** | **40.80%** | **+0.4 pp** |
| Clean Real ACC | ~~98.10%~~ | ~~100.00%~~ | ~~+1.9 pp~~ | 57.01% | 48.10% | -8.9 pp |

> [!IMPORTANT]
> 🚨 **Bảng này không còn dùng được để so sánh WiSE-FT vs Anchored** (27/10/2026). Toàn bộ cột WiSE-FT dựa trên checkpoint `invalid_layout`; "gap 90,97 pp" chỉ phản ánh việc **nạp thiếu 7/10 khối**, không phản ánh năng lực tổng quát hóa. Kết luận đúng đắn duy nhất hiện có: **Anchored Recovery** giữ được Fake ACC cao trên OOD (90,50%) nhưng **đánh đổi Real ACC xuống 48,10%** và **AUC Q30 chỉ 50,07%**. Việc so sánh "ai tổng quát hóa tốt hơn" phải chờ Gen 2 (Task 7.5) và kết quả G7.1/G7.2.

---

## VI. MA TRẬN KHUYẾN NGHỊ TRIỂN KHAI ĐA PHƯƠNG ÁN (TUÂN THỦ QUY TẮC 2)

| Tiêu Chí | Baseline | WiSE-FT 🚫 `invalid_layout` | Anchored | Hệ Thống Kép |
| :--- | :---: | :---: | :---: | :---: |
| **Phát hiện AI thế hệ mới (Clean OOD Fake ACC)** | 35.30% ⚠️ | ~~4.10%~~ | **90.50%** ✅ | ✅ Kích hoạt Anchored |
| **Bảo vệ ảnh thật (Clean Real ACC)** | 94.00% ✅ | ~~100.00%~~ | 48.10% ❌ | ⚠️ Cần hiệu chuẩn lại (Task 7.4) |
| **Kháng nén sâu Q=30 (Fake ACC)** | 0.90% ❌ | ~~0.20%~~ | **40.80%** ✅ | ✅ Kích hoạt Anchored |
| **Ổn định OOD vs. in-dist** | Trung bình | ~~Rất kém~~ | **Xuất sắc** ✅ | ⚠️ Chờ đo lại ở G7.1 |
| **False Alarm Rate (ảnh thật bị báo nhầm)** | 6% ✅ | ~~0%~~ | **51.9%** ❌ | ✅ Điều phối ngưỡng |
| **Ngữ cảnh phù hợp nhất** | Đối chuẩn học thuật | Pháp đình / Tư pháp | Kiểm duyệt mạng xã hội | **Hệ thống sản xuất thực tế** |

---

## VII. KẾT LUẬN KHOA HỌC & ĐÓNG GÓP CHO ĐỒ ÁN / PAPER

### Kết luận 1: Minh chứng thực nghiệm về "Khoảng Trống OOD Deepfake" (OOD Forensic Gap)

Benchmark CFE-Eval (CVPR 2025) xác nhận một khoảng trống lớn tồn tại giữa hiệu năng in-distribution và OOD của tất cả các mô hình phát hiện deepfake hiện tại. Cụ thể, Baseline CVPR 2024 sụt giảm từ **93.54% → 35.30% Fake ACC** khi chuyển từ tập kiểm chuẩn quen thuộc sang các bộ sinh 2024–2025 — đây là bằng chứng thực nghiệm mạnh nhất để thuyết phục hội đồng phản biện về tính cấp thiết của đề tài.

### Kết luận 2: Anchored Recovery Tổng Quát Hóa Vượt Trội — Nhưng Cần Hiệu Chỉnh Ngưỡng

Anchored Recovery chứng minh khả năng tổng quát hóa OOD ổn định nhất (OOD Degradation Gap chỉ -7.1 pp trên Fake ACC). Tuy nhiên, tỷ lệ False Alarm **51.9%** trên ảnh thật OOD Clean là vấn đề kỹ thuật cần được giải quyết ở bước triển khai sản xuất. Giải pháp đề xuất: hiệu chỉnh lại ngưỡng $T^*$ từ 0.40 lên 0.45–0.48 trên tập val OOD, hoặc sử dụng cơ chế **ensemble voting** với WiSE-FT làm bộ lọc xác nhận ảnh thật.

### Kết luận 3: Tính Đúng Đắn của Kiến Trúc Hệ Thống Kép — ⚠️ ĐÃ BỊ ĐÍNH CHÍNH

> 🚨 **Kết luận này không còn hiệu lực** (xem Phần IX). Tiền đề của nó là checkpoint WiSE-FT Gen 1 — vốn **không nạp được đầy đủ** (10 khối adapter). Do đó "không có bộ trọng số đơn lẻ nào đáp ứng đồng thời" là **giả thuyết chưa được kiểm chứng**, không phải kết quả. Phase 7 kiểm chứng lại bằng Gen 2 + `DegradationRouter` (cổng G7.3).

~~Kết quả trên cả hai điều kiện (Clean + Q=30) và trên cả hai phân phối (in-dist + OOD) đồng loạt xác nhận: không có bộ trọng số đơn lẻ nào đáp ứng đồng thời 100% Real ACC, 90%+ OOD Fake ACC, và 40%+ Q=30 Fake ACC. Chiến lược **Hệ Thống Kép** (WiSE-FT ↔ Anchored Recovery) do nhóm thiết kế là giải pháp khoa học tất yếu và tối ưu về mặt lý thuyết lẫn thực nghiệm.~~

### Kết luận 4: Phát Hiện Kỹ Thuật — Cần Kiểm Tra Compatibility SRM của WiSE-FT — ✅ ĐÃ XÁC NHẬN VÀ CHẨN ĐOÁN LẠI

> ✅ Cảnh báo này **đã được xác nhận là đúng hướng**, nhưng chẩn đoán cụ thể đã được sửa: vấn đề không chỉ là SRM/Gating mà là **tập khối adapter không biểu diễn được**. Xem Phần IX, mục F1/F2.

Cảnh báo 146 unexpected keys khi nạp WiSE-FT checkpoint cần được điều tra thêm trong các thực nghiệm tương lai. Nếu xác nhận SRM/Gating không nạp đúng, cần rebuild lại `eval_communityforensics.py` để khớp chính xác kiến trúc checkpoint, và tái chạy thực nghiệm để có số liệu WiSE-FT OOD chính xác. → **Đã thực hiện ngày 27/10/2026; xem Phần IX.**

---

## VIII. PHỤ LỤC: THÔNG SỐ THỰC THI THỰC TẾ

| Thông Số | Giá Trị |
| :--- | :--- |
| GPU | NVIDIA Tesla T4 (14.56 GB VRAM) |
| Thời gian chạy toàn bộ | ~18.7 phút (1.120 giây — Cell 4) |
| Batch size | 32 |
| Num workers | 2 |
| Transform | Resize(224, BICUBIC) → CenterCrop(224) → ToTensor → Normalize (CLIP mean/std) |
| JPEG Q=30 | In-memory via `io.BytesIO` (không ghi disk) |
| Baseline checkpoint | `fatformer_4class_ckpt.pth` — nạp strict=False, 0 unexpected keys |
| WiSE-FT checkpoint | `fatformer_wise_ft_best.pth` — nạp **`strict=False`** ⚠️; thực chất **nạp thiếu khối** (xem Phần IX) |
| Anchored checkpoint | `fatformer_anchored_recovery_best.pth` — nạp strict=False, 0 unexpected keys |

---

## IX. ĐÍNH CHÍNH KỸ THUẬT NGÀY 27/10/2026

### IX.1. Phạm vi hiệu lực của báo cáo

| Thành phần báo cáo | Trạng thái sau đính chính |
| :--- | :--- |
| Số liệu **Baseline CVPR 2024** (Bảng A/B) | ✅ **Giữ nguyên hiệu lực** (nạp `0/0`) |
| Số liệu **Anchored Recovery** (Bảng A/B) | ✅ **Giữ nguyên hiệu lực** (nạp `0/0`), nhưng phần diễn giải "Real 48,10%" phải đọc kèm ngưỡng `T*=0.40` và sẽ được tái hiệu chuẩn ở Task 7.4 |
| Số liệu **WiSE-FT** (mọi cột) | 🚫 **`invalid_layout` — vô hiệu** |
| Kết luận 3 (hệ thống kép tất yếu) | 🚫 **Hạ xuống giả thuyết**, chờ G7.3 |
| Kết luận 4 (cảnh báo SRM) | ✅ Đúng hướng, chẩn đoán đã sửa (F1/F2) |

**Không được trích dẫn như kết quả hiện hành**: `52,05%`, `4,10%`, `0,20%`, `100,00% Real`, `96,58% in-dist`, `6,73% Q30 Fake` của WiSE-FT. Nếu cần nhắc, phải ghi kèm: *"số liệu lịch sử của checkpoint `invalid_layout`, không tái lập được"*.

### IX.2. Phát hiện F1 — WiSE-FT Gen 1 không biểu diễn được bằng kiến trúc hiện tại

Đo trực tiếp bằng `torch.load(map_location='cpu')`:

| Checkpoint | Số tensor | Khối adapter | `n` biểu diễn được | SRM/Gating trong adapter | Alias `image_encoder` ↔ `clip_model.visual` | Kết luận |
| :--- | ---: | :--- | :--- | :---: | :---: | :--- |
| `fatformer_4class_ckpt.pth` | 1.116 | `{7,15,23}` | `n=3` | ❌ 66 tensor/khối | 395/395 khớp | ✅ Hợp lệ |
| `fatformer_crossdomain_weights_only.pth` (M4) | 1.684 | `{2,5,8,11,14,17,20,23}` | `n=8` | ✅ 94 tensor/khối | 672/672 khớp | ✅ Hợp lệ |
| `fatformer_wise_ft_best.pth` (Gen 1) | 1.816 | `{2,5,7,8,11,14,15,17,20,23}` | **∅** | ⚠️ 66/94 tại `7,15,23` | 738/738 khớp | 🚫 `invalid_layout` |
| `fatformer_anchored_recovery_best.pth` | 1.200 | `{7,15,23}` | `n=3` | ✅ 94 tensor/khối | 437/437 khớp | ✅ Hợp lệ |

Ràng buộc kiến trúc (`src/models/clip/model.py`):

```python
n_spilt = layers // num_vit_adapter        # layers = 24
add_adapter[(i + 1) * n_spilt - 1] = True  # i = 0 .. n-1
# n=3  -> {7, 15, 23}
# n=8  -> {2, 5, 8, 11, 14, 17, 20, 23}
# Không n ∈ {3,4,6,8,12,24} nào sinh ra được 10 khối {2,5,7,8,11,14,15,17,20,23}
```

**Hệ quả**: Gen 1 là sản phẩm của việc trộn hai checkpoint **khác `num_vit_adapter`**. Không thể "sửa bằng loader"; chỉ có thể **cách ly** và **tái tạo Gen 2** từ hai checkpoint cùng `n=3`.

### IX.3. Phát hiện F2 — Nguyên nhân sụp đổ là ánh xạ khối, không phải khoá thừa

- 146 "unexpected keys" thực chất là `srm.*` (10) và `gating.*` (4) top-level — các module đã bị vô hiệu trong `CLIPModel` (`self.srm = None`). Chúng **không phải** nguyên nhân làm Fake ACC sụp.
- Nguyên nhân thật: eval cũ gặp 10 khối, rơi vào fallback `len(indices) > 8 → 3`, tức dựng mô hình với `n=3` và **chỉ nạp 3/10 khối** (`{7,15,23}`), đồng thời **bỏ 5 khối M4 thuần** `{2,5,8,11,14,17,20}`. Kết quả 4,10% là hệ quả tất yếu của một mô hình gần như ngẫu nhiên ở phần lớn thân mạng.
- Kiểm tra alias: **100%** key `image_encoder.*` bằng đúng tensor `clip_model.visual.*` ⇒ không có ghi đè chéo.

### IX.4. Sáu khuyết tật công cụ (E1–E6) và bản vá tương ứng

| Mã | File | Khuyết tật (bản cập nhật hiện tại) | Rủi ro | Bản vá (Phase 7) |
| :--- | :--- | :--- | :--- | :--- |
| **E1** | `tools/eval_communityforensics.py` | `load_state_dict(filtered_sd, strict=False)`, chỉ lọc theo shape, không assert `missing` | Nạp thiếu vẫn in "✓" | **Task 7.2–7.3**: `strict=True` + assert `0/0`, thiếu thì `raise` |
| **E2** | `tools/eval_communityforensics.py` | `if len(indices) > 8: return 3` | **Chọn đúng tập khối tệ nhất** cho Gen 1 | Trả `adapter_block_indices` tường minh hoặc raise `IncompatibleAdapterLayout` |
| **E3** | `tools/eval_communityforensics.py` | `adaptive_thresh = 0.50·(1−mean_w) + 0.42·mean_w`, `mean_w` tính trên **chính tập đang chấm** | Rò rỉ thông tin tập test; không triển khai được | Ngưỡng **cố định** fit trên tập hiệu chuẩn tách rời (**Task 7.4**) |
| **E4** | `tools/eval_communityforensics.py` | Hệ số router (`12.0`, `1.18`, clamp `[0.10,0.90]`, `0.50/0.42`) đặt tay | "Unified" chỉ là trộn heuristic | Fit logistic trên `val_calib`, lưu `router_params.json` (**Task 7.13**) |
| **E5** | `tools/eval_communityforensics.py` | Nhiệt độ `1.0811/1.2763` hard-code; chỉ có `TPR@FPR=5%`; không CI; AP có edge-case chia 0 | Không tái lập, thiếu chuẩn học thuật | `TPR@FPR 1/5/10`, `ECE`, bootstrap CI 1.000 lần, `git_commit` + `sha256` (**Task 7.3**) |
| **E6** | `tools/wise_ft_merge.py` | `sweep_threshold_constrained()` **không được gọi**; không có verify sau trộn; `is_fixed_buffer` khớp chuỗi con `"k1"/"k2"/"k3"`; grid chỉ quét `α`, **không quét `β_srm`**; vẫn copy `srm.*`/`gating.*` top-level | Trộn ra file không ai kiểm chứng; ngưỡng "khuyến nghị" vô căn cứ | Thêm `--verify` (dựng model + assert `0/0`), lọc khoá chết, lưới `α×β`, gọi calibration (**Task 7.5**) |

### IX.5. Việc phải làm để báo cáo này có hiệu lực trở lại

1. **Task 7.1**: chấm lại Gen 1 hai cấu hình (`n=3` và `n=8`) trên 2.000 ảnh CFE clean, xuất `wise_ft_legacy_reinterpretation.json`. Mục đích: chứng minh bằng số rằng sụp đổ do loader, không do trộn khác basin.
2. **Task 7.5**: tạo **Gen 2** từ `Baseline (n=3)` + `Anchored (n=3)`, verify `0/0`.
3. **Task 7.3–7.4**: chuyển toàn bộ metric sang `report_schema v3`.
4. **Task 7.6**: re-baseline 4 cấu hình, xuất bảng đối chiếu với báo cáo 26/10 và ghi rõ chênh lệch do đâu (loader / ngưỡng / metric).
5. Chỉ sau đó mới cập nhật lại các bảng A/B của báo cáo này.

> **Ghi chú về JPEG Q=30**: tập CFE hiện tại **đã là 224×224** (mục VIII), nên thao tác `Resize(224) → CenterCrop(224)` là no-op và việc đặt JPEG sau resize **không thay đổi kết quả**. Thay đổi `--degrade_order` trong bản cập nhật chỉ có ý nghĩa phòng thủ cho ảnh độ phân giải gốc, **không được trình bày như nguyên nhân cải thiện**.

| Báo cáo JSON đầu ra | `/content/drive/MyDrive/Fatformer/logs/cfe_benchmark/communityforensics_benchmark_report.json` |
| Báo cáo Markdown đầu ra | `/content/drive/MyDrive/Fatformer/logs/cfe_benchmark/communityforensics_benchmark_report.md` |
