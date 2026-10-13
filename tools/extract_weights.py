"""
Trích xuất trọng số suy luận (Weights-only & PEFT-only) từ full checkpoint FatFormer-XLA
"""
import os
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    src_path = "DATASET/checkpoints/fatformer_plan_a_best.pth"
    weights_path = "DATASET/checkpoints/fatformer_plan_a_best_weights.pth"
    peft_path = "DATASET/checkpoints/fatformer_plan_a_peft_only.pth"

    if not os.path.isfile(src_path):
        print(f"[!] Không tìm thấy tệp: {src_path}")
        return

    print(f"[*] Đang nạp full checkpoint từ {src_path}...")
    ckpt = torch.load(src_path, map_location="cpu")
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # 1. Trích xuất weights-only
    print("[*] Đang lưu model weights-only...")
    torch.save(sd, weights_path)
    sz_weights = os.path.getsize(weights_path) / (1024**2)
    print(f"[✓] Đã tạo thành công: {weights_path} ({sz_weights:.2f} MB, {len(sd)} tensors)")

    # 2. Trích xuất PEFT-only
    print("[*] Đang lọc tham số PEFT...")
    peft_prefixes = ["language_guided_alignment", "text_guided_interactor", "srm", "gating", "ctx"]
    peft_sd = {k: v for k, v in sd.items() if any(p in k for p in peft_prefixes)}
    peft_export = {
        "epoch": ckpt.get("epoch", 5),
        "best_val_score": ckpt.get("best_val_score", 83.09),
        "best_val_ap": ckpt.get("best_val_ap", 80.39),
        "val_metrics": ckpt.get("val_metrics", {}),
        "peft_state_dict": peft_sd
    }
    torch.save(peft_export, peft_path)
    sz_peft = os.path.getsize(peft_path) / (1024**2)
    print(f"[✓] Đã tạo thành công: {peft_path} ({sz_peft:.2f} MB, {len(peft_sd)} tensors)")

if __name__ == "__main__":
    main()
