"""Explicit training profiles; legacy behavior stays the default for old runs."""
import torchvision.transforms as T

PROFILES = ("legacy_imagenet", "clip")
IMAGENET_MEAN = (.485, .456, .406)
IMAGENET_STD = (.229, .224, .225)
CLIP_MEAN = (.48145466, .4578275, .40821073)
CLIP_STD = (.26862954, .26130258, .27577711)


def preprocessing_parts(profile="legacy_imagenet", img_resolution=224, crop_resolution=224):
    if profile not in PROFILES:
        raise ValueError(f"Unknown preprocessing profile: {profile}")
    if profile == "clip":
        geometry = T.Compose([T.Resize(img_resolution, interpolation=T.InterpolationMode.BICUBIC),
                              T.CenterCrop(crop_resolution)])
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        geometry = T.Compose([T.Resize((img_resolution, img_resolution)), T.CenterCrop(crop_resolution)])
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    tensor = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return geometry, tensor, mean, std
