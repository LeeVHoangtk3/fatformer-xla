1. Mã Nguồn & Trọng Số Mô Hình FatFormer
Mã nguồn chính thức của FatFormer (GitHub):

https://github.com/Michel-liu/FatFormer

[cite: 1]

Công dụng: Chứa toàn bộ kiến trúc mô hình, các khối bộ thích ứng (FAA, LGA), kịch bản nạp dữ liệu (data loader) và lệnh chạy kiểm thử/đánh giá.  

Trọng số tiền huấn luyện FatFormer (Google Drive):

https://drive.google.com/file/d/1Q_Kgq4ygDf8XEHgAf-SgDN6Ru_IOTLkj/view?usp=sharing

[cite: 1, 2]

Công dụng: Tải tệp checkpoint model.pth đã được tinh chỉnh hoàn chỉnh trên tập dữ liệu ProGAN 4-class từ tác giả. Dùng để làm baseline hoặc tiếp tục fine-tune.  

Trọng số gốc CLIP ViT-L/14 (Azure Storage):

https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt

[cite: 1]

Công dụng: Tệp trọng số gốc CLIP từ OpenAI. Bắt buộc phải tải và lưu vào thư mục pretrained/ trước khi chạy huấn luyện/đánh giá FatFormer.  

2. Tập Dữ Liệu Gốc Dùng Trong Bài Báo FatFormer
CNNDetection (Tập huấn luyện ProGAN gốc):

https://github.com/peterwang512/CNNDetection

[cite: 1, 2]

Gương Baidu Netdisk (Mã: 9i5w): https://pan.baidu.com/s/1obzmrCsWvGyUlmH8MkSTLA

[cite: 1]

Công dụng: Cung cấp tập dữ liệu ảnh sinh bởi ProGAN (cấu hình 2-class và 4-class: car, cat, chair, horse) dùng để huấn luyện gốc FatFormer theo chuẩn.  

Tập kiểm thử GANs (StyleGAN, BigGAN, CycleGAN,...):

https://github.com/peterwang512/CNNDetection#testset

[cite: 1]

Gương Baidu Netdisk (Mã: 75cz): https://pan.baidu.com/s/1aAiW8oMQcIZIaLYuQIOAjg

[cite: 1]

Gương OneDrive: https://1drv.ms/u/s!Aqkrc9gPuk8jqaM2khCvAejz_K4Jow?e=e0yeDQ

[cite: 1]

Công dụng: Dùng để đánh giá khả năng tổng quát hóa của mô hình trên 8 họ thuật toán GAN khác nhau.  

DIRE & DiffusionForensics (Tập kiểm thử Diffusion Models):

https://github.com/ZhendongWang6/DIRE

[cite: 1, 2, 4]

Gương Baidu Netdisk (Mã: dire): https://pan.baidu.com/s/1zoubPr5n_mGI27En9uyL8Q

[cite: 1, 5]

Công dụng: Chứa các mẫu ảnh thử nghiệm từ các mô hình Diffusion tiên tiến (LDM, ADM, Glide, PNDM) và các cặp ảnh tái cấu trúc.  

UniversalFakeDetect (Dữ liệu thử nghiệm Diffusion mở rộng):

https://github.com/Yuheng-Li/UniversalFakeDetect

[cite: 1, 2]

Công dụng: Cung cấp bổ sung dữ liệu kiểm thử từ DALL-E 2, VQ-Diffusion nhằm thử nghiệm tính tổng quát hóa.  

3. Tập Dữ Liệu Quy Mô Lớn Đề Xuất Cho Việc Fine-Tune
GenImage Benchmark (NeurIPS 2023):

https://github.com/GenImage-Dataset/GenImage

[cite: 7, 8]

Liên kết tải trực tiếp (Google Drive): https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS

[cite: 8]

Gương Baidu Yunpan (Mã: ztf1): Lưu tại kho Github của dự án.  

Công dụng: Hơn 1 triệu cặp ảnh thực/giả thu thập từ 8 mô hình sinh mạnh nhất (Midjourney, Stable Diffusion v1.4/v1.5, BigGAN, GLIDE...). Thích hợp nhất để fine-tune FatFormer nhằm nhận biết đa dạng phong cách ảnh thực tế.  

DRCT-2M Dataset (ICML 2024):

https://github.com/beibuwandeluori/DRCT

[cite: 11]

Công dụng: 2 triệu ảnh sinh từ 16 biến thể Diffusion khác nhau (SDXL, SD-Turbo, ControlNet...). Rất tốt để tinh chỉnh sâu bộ thích ứng FAA trên các vết nhiễu vi mô của mô hình khuếch tán.  

CommunityForensics-Eval (Hugging Face / CVPR 2025):

https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval

[cite: 15]

Công dụng: Tập dữ liệu dạng Parquet chứa hơn 50.000 ảnh từ 21+ mô hình thương mại và Hugging Face kèm theo văn bản nhúng (prompt). Rất phù hợp để đánh giá khả năng nhận diện ảnh thương mại thực tế và tinh chỉnh nhánh LGA.  

4. Tổng Hợp Công Cụ & Khung Đánh Giá Mở Rộng
Awesome AIGC Image Detection (GitHub):

https://github.com/graydove/Awesome-AIGC-Image-Detection

[cite: 7]

Công dụng: Kho tổng hợp liên kết tải về của hầu hết các tập dữ liệu phát hiện ảnh AI phổ biến nhất hiện nay cùng mã nguồn của các phương pháp đối chứng (UniFD, LGrad, NPR, FreqNet).