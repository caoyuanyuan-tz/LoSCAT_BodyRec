# models/vit_12cls.py
import torch
import torch.nn as nn

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 384

CLASSES = [
    'SCFA','NECK','CHEST','ABDM','UBCK','LBCK',
    'ARM','FRM','HND','THI','LEG','FT'
]

def build_vit(model_name: str = "vit_base_patch16_384", num_classes: int = 12):
    """
    Returns a timm ViT with a 12-class head.
    Must match the architecture used during training.
    """
    try:
        import timm
    except ImportError as e:
        raise RuntimeError("Please pip install timm") from e

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    return model

def load_state_dict_safely(model: nn.Module, state_dict_path: str, map_location="cpu"):
    state = torch.load(state_dict_path, map_location=map_location)
    model.load_state_dict(state, strict=True)
    return model
