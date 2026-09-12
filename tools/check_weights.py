"""
Kiểm tra tính toàn vẹn và phân tích cấu trúc state_dict của:
1. ViT-L-14.pt (OpenAI CLIP Backbone)
2. fatformer_4class_ckpt.pth (FatFormer Pre-trained Checkpoint)
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from collections import defaultdict

def format_size(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def count_params(tensor_dict):
    total = 0
    for name, tensor in tensor_dict.items():
        if hasattr(tensor, 'numel'):
            total += tensor.numel()
    return total

def inspect_vit_clip(path):
    print("=" * 70)
    print(f"1. PHÂN TÍCH FILE CLIP BACKBONE: {os.path.basename(path)}")
    print("=" * 70)
    if not os.path.exists(path):
        print(f"[LỖI] Không tìm thấy file tại: {path}")
        return None
    
    file_size = os.path.getsize(path)
    print(f"- Kích thước tệp: {format_size(file_size)}")

    import torch
    try:
        # Thử load như checkpoint state_dict thông thường
        ckpt = torch.load(path, map_location='cpu')
        is_jit = False
    except Exception as e:
        print(f"- Không nạp được bằng torch.load thông thường ({e}), thử torch.jit.load...")
        try:
            ckpt = torch.jit.load(path, map_location='cpu')
            is_jit = True
        except Exception as e2:
            print(f"[LỖI] Không thể đọc tệp bằng PyTorch: {e2}")
            return None

    if is_jit:
        print("- Định dạng: TorchScript JIT Model")
        state_dict = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            print("- Định dạng: Dictionary chứa 'state_dict'")
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            print("- Định dạng: Dictionary chứa 'model'")
            state_dict = ckpt['model']
        else:
            print("- Định dạng: Raw state_dict (Dictionary)")
            state_dict = ckpt
    else:
        print(f"- Định dạng khác: {type(ckpt)}")
        state_dict = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else {}

    total_params = count_params(state_dict)
    print(f"- Tổng số tensors: {len(state_dict)}")
    print(f"- Tổng số tham số (Parameters): {total_params:,} ({total_params / 1e6:.2f}M params)")

    # Phân loại keys cơ bản
    visual_keys = [k for k in state_dict.keys() if 'visual' in k]
    text_keys = [k for k in state_dict.keys() if 'visual' not in k and any(w in k for w in ['transformer', 'token_embedding', 'text', 'ln_final'])]
    print(f"- Số tensors Image Encoder (Visual ViT): {len(visual_keys)}")
    print(f"- Số tensors Text Encoder / Khác: {len(text_keys)}")

    print("\n* 5 Tensors mẫu đầu tiên của Visual Encoder:")
    for k in visual_keys[:5]:
        t = state_dict[k]
        shape_str = list(t.shape) if hasattr(t, 'shape') else 'N/A'
        dtype_str = str(t.dtype) if hasattr(t, 'dtype') else 'N/A'
        print(f"  • {k:45s} | Shape: {str(shape_str):20s} | {dtype_str}")

    return state_dict

def inspect_fatformer(path):
    print("\n" + "=" * 70)
    print(f"2. PHÂN TÍCH FILE FATFORMER CHECKPOINT: {os.path.basename(path)}")
    print("=" * 70)
    if not os.path.exists(path):
        print(f"[LỖI] Không tìm thấy file tại: {path}")
        return None

    file_size = os.path.getsize(path)
    print(f"- Kích thước tệp: {format_size(file_size)}")

    import torch
    try:
        ckpt = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"[LỖI] Không thể đọc tệp checkpoint: {e}")
        return None

    print(f"- Kiểu dữ liệu gốc: {type(ckpt)}")
    if isinstance(ckpt, dict):
        top_keys = list(ckpt.keys())
        print(f"- Top-level keys: {top_keys}")
        
        # Kiểm tra metadata phụ (epoch, optimizer,...)
        for meta_k in ['epoch', 'iter', 'best_acc', 'best_ap', 'args', 'opt', 'scaler']:
            if meta_k in ckpt:
                val = ckpt[meta_k]
                if isinstance(val, (int, float, str, bool)):
                    print(f"  • {meta_k}: {val}")
                else:
                    print(f"  • {meta_k}: [Object {type(val).__name__}]")

        if 'model' in ckpt and isinstance(ckpt['model'], dict):
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else {}

    total_params = count_params(state_dict)
    print(f"- Tổng số tensors trong state_dict: {len(state_dict)}")
    print(f"- Tổng số tham số (Parameters): {total_params:,} ({total_params / 1e6:.2f}M params)")

    # Phân nhóm chuyên sâu theo các khối kiến trúc FatFormer
    groups = defaultdict(dict)
    for k, v in state_dict.items():
        k_lower = k.lower()
        if any(w in k_lower for w in ['faa', 'freq', 'wavelet', 'dwt', 'inter_band', 'intra_band', 'scale_factor', 'lambda']):
            groups['Khối FAA (Frequency & Spatial Adapter)'][k] = v
        elif any(w in k_lower for w in ['lga', 'prompt', 'context', 'pbe', 'tgi', 'enhancer', 'interactor']):
            groups['Khối LGA (Language-Guided Alignment)'][k] = v
        elif 'visual' in k_lower or 'image_encoder' in k_lower:
            groups['ViT Image Encoder (Backbone)'][k] = v
        elif any(w in k_lower for w in ['text_encoder', 'token_embedding', 'text_projection']):
            groups['Text Encoder'][k] = v
        elif any(w in k_lower for w in ['fc', 'head', 'classifier']):
            groups['Classifier Head'][k] = v
        else:
            groups['Khác / Module chung'][k] = v

    print("\n* BẢNG THỐNG KÊ CHI TIẾT CÁC PHÂN HỆ:")
    print(f"  {'Tên Phân Hệ':<40s} | {'Số Tensors':<10s} | {'Số Tham Số':<15s}")
    print("  " + "-" * 70)
    for g_name, g_dict in groups.items():
        p_count = count_params(g_dict)
        print(f"  {g_name:<40s} | {len(g_dict):<10d} | {p_count / 1e6:8.2f}M ({p_count:,})")

    # Hiển thị các tensor then chốt của FAA và LGA
    print("\n* CÁC TENSOR TRỌNG TÂM CỦA KHỐI FAA:")
    faa_items = list(groups['Khối FAA (Frequency & Spatial Adapter)'].items())
    if faa_items:
        for k, t in faa_items[:10]:
            shape_str = list(t.shape) if hasattr(t, 'shape') else 'N/A'
            dtype_str = str(t.dtype) if hasattr(t, 'dtype') else 'N/A'
            print(f"  • {k:50s} | Shape: {str(shape_str):20s} | {dtype_str}")
        if len(faa_items) > 10:
            print(f"  ... (còn {len(faa_items) - 10} tensors khác trong FAA)")
    else:
        print("  [Lưu ý] Không phát hiện tiền tố 'faa' trực tiếp, kiểm tra các keys Adapter khác...")
        adapter_keys = [k for k in state_dict.keys() if 'adapter' in k.lower() or 'conv' in k.lower()]
        for k in adapter_keys[:8]:
            print(f"  • {k}")

    print("\n* CÁC TENSOR TRỌNG TÂM CỦA KHỐI LGA (Prompts / Alignment):")
    lga_items = list(groups['Khối LGA (Language-Guided Alignment)'].items())
    if lga_items:
        for k, t in lga_items[:10]:
            shape_str = list(t.shape) if hasattr(t, 'shape') else 'N/A'
            dtype_str = str(t.dtype) if hasattr(t, 'dtype') else 'N/A'
            print(f"  • {k:50s} | Shape: {str(shape_str):20s} | {dtype_str}")
        if len(lga_items) > 10:
            print(f"  ... (còn {len(lga_items) - 10} tensors khác trong LGA)")
    else:
        print("  [Lưu ý] Đang nằm trong không gian ngữ cảnh tự động hoặc prompt module.")

    return state_dict

def main():
    import torch
    print("=" * 70)
    print("THẨM ĐỊNH MÔI TRƯỜNG & TRỌNG SỐ DỰ ÁN FATFORMER")
    print("=" * 70)
    print(f"- Python executable: {sys.executable}")
    print(f"- PyTorch version:   {torch.__version__}")
    print(f"- CUDA khả dụng:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- GPU thiết bị:      {torch.cuda.get_device_name(0)}")
        print(f"- VRAM khả dụng:     {format_size(torch.cuda.get_device_properties(0).total_memory)}")
    else:
        print("- Chế độ thực thi:   CPU")

    vit_path = os.path.join(os.getcwd(), 'ViT-L-14.pt')
    fatformer_path = os.path.join(os.getcwd(), 'fatformer_4class_ckpt.pth')

    inspect_vit_clip(vit_path)
    inspect_fatformer(fatformer_path)

    print("\n" + "=" * 70)
    print("HOÀN THÀNH KIỂM ĐỊNH TÍNH TOÀN VẸN CỦA TRỌNG SỐ!")
    print("=" * 70)

if __name__ == '__main__':
    main()
