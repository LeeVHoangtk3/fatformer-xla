from .clip_models import CLIPModel

VALID_NAMES = [
    'CLIP:ViT-B/32', 
    'CLIP:ViT-B/16', 
    'CLIP:ViT-L/14', 
]

def build_model(args):
    """
    Khởi tạo mô hình FatFormer với CLIP backbone.
    Hỗ trợ cấu hình bổ sung: --use_srm, --use_gating, v.v.
    """
    backbone = getattr(args, "backbone", "CLIP:ViT-L/14")
    if backbone.startswith("CLIP:"):
        model_name = backbone[5:]
        assert backbone in VALID_NAMES, f"Backbone {backbone} không hợp lệ. Chọn từ {VALID_NAMES}"
        return CLIPModel(model_name, args)
    else:
        raise NotImplementedError(f"Backbone {backbone} chưa được hỗ trợ.")
