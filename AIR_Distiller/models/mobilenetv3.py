import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.class_block import ClassBlock
from torchvision import models
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

__all__ = ["mobilenetv3_small", "mobilenetv3_large"]


model_urls = {
    "mobilenetv3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
    "mobilenetv3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth"
}


class MobileNetV3(nn.Module):
    def __init__(self, last_stride = 2, num_classes= 1000, mode="small", pretrained=True):
        super(MobileNetV3, self).__init__()

        # 根据 mode 选择模型
        if mode == "large":
            base_model = mobilenet_v3_large(pretrained=pretrained)
            self.out_channels = 960  # MobileNetV3-Large 的输出通道数
        elif mode == "small":
            base_model = mobilenet_v3_small(pretrained=pretrained)
            self.out_channels = 576  # MobileNetV3-Small 的输出通道数
        else:
            raise ValueError("Invalid mode: choose either 'large' or 'small'")

    
        self.model = nn.Sequential(*list(base_model.features.children())) 
        
        
        # 获取该模块的所有子模块
        for idx, (name, module) in enumerate(self.model[9].block.named_children()):
            # 定位到第二个 Conv2dNormActivation 模块
            if idx == 1:
                # 修改步长为 (1, 1)
                module[0].stride = (last_stride, last_stride)
      
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = ClassBlock(self.out_channels, num_classes, num_bottleneck=512)

    def forward(self, x):
        
        feat1 = self.model[0:2](x)  # 第1次下采样后的特征
        feat2 = self.model[2:4](feat1)  # 第1次下采样后的特征
        feat3 = self.model[4:9](feat2) # 第2次下采样后的特征
        feat4 = self.model[9:](feat3) # 第2次下采样后的特征

        x = self.avgpool(feat4)
        x = x.view(x.size(0), -1)
        avg = x
        out, retrieval_feat = self.fc(x)

        feats = {}
        feats["pooled_feat"] = avg
        feats["feats"] = [feat1, feat2, feat3, feat4]
        feats["retrieval_feat"] = retrieval_feat

        return out, feats
    


def mobilenetv3_small(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
   
    if pretrained:
        model = MobileNetV3(last_stride=kwargs["last_stride"] , num_classes=kwargs["num_classes"], mode="small", pretrained=True)

    else:
        model = MobileNetV3(last_stride=kwargs["last_stride"] , num_classes=kwargs["num_classes"], mode="small", pretrained=False)
       
    return model