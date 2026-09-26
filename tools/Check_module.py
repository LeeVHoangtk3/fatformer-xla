import os
import sys
import torch

# Đảm bảo Python nhận diện được thư mục gốc của dự án
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import build_model

def verify_loader(ckpt_path="fatformer_4class_ckpt.pth", clip_path="ViT-L-14.pt"):
    class Args:
        backbone = "CLIP:ViT-L/14"
        clip_path = os.path.join(project_root, "ViT-L-14.pt")
        num_classes = 2
        num_vit_adapter = 3
        num_context_embedding = 8
        init_context_embedding = ""
        hidden_dim = 768
        clip_vision_width = 1024
        frequency_encoder_layer = 2
        decoder_layer = 4
        num_heads = 12
        use_srm = False
        use_gating = False

    print("[*] Khởi tạo kiến trúc FatFormer...")
    model = build_model(Args())
    
    ckpt_full_path = os.path.join(project_root, ckpt_path)
    print(f"[*] Nạp checkpoint từ {ckpt_full_path}...")
    ckpt = torch.load(ckpt_full_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    
    # Nạp nghiêm ngặt
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    
    print(f"[+] Missing keys: {len(missing)}")
    print(f"[+] Unexpected keys: {len(unexpected)}")
    
    if len(missing) == 0 and len(unexpected) == 0:
        print("[✓] TUYỆT VỜI: Khớp 100% (1.116/1.116 tensors) pass strict=True!")
    else:
        print("[!] Cảnh báo: Vẫn còn tensor chưa khớp!")

if __name__ == "__main__":
    verify_loader()