from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet_ibn import resnet18_ibn_a, resnet34_ibn_a, resnet50_ibn_a, resnet101_ibn_a
from .swin_transformer_v2 import swin_transformer_v2_base, swin_transformer_v2_small, swin_transformer_v2_tiny
from .mobilenetv3 import mobilenetv3_small


model_dict = {
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "ResNet101": resnet101,
    "ResNet18_ibn_a": resnet18_ibn_a,
    "ResNet34_ibn_a": resnet34_ibn_a,
    "ResNet50_ibn_a": resnet50_ibn_a,
    "ResNet101_ibn_a": resnet101_ibn_a,
    "MobileNetV3_Small": mobilenetv3_small,
    "Swin_Transformer_V2_Base": swin_transformer_v2_base,
    "Swin_Transformer_V2_Small": swin_transformer_v2_small,
    "Swin_Transformer_V2_Tiny": swin_transformer_v2_tiny
}
