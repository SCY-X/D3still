import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from timm.models import create_model
from .utils.class_block import ClassBlock



class Swin_Transformer_V2(nn.Module):
    def __init__(self, pretrained = False, pretrained_path = "", model_select="swinv2_small_window8_256", num_classes=1000):
        super(Swin_Transformer_V2, self).__init__()

        if pretrained_path != "":
            self.model = create_model(model_select, pretrained=pretrained, 
                                        pretrained_cfg_overlay=dict(file=pretrained_path))

        else:
            self.model = create_model(model_select, pretrained=pretrained)
        
        self.model.head = nn.Identity()

        if model_select == "swinv2_small_window8_256":
            num_channel = 768
        elif model_select == "swinv2_tiny_window8_256":
            num_channel = 768
        elif model_select == "swinv2_base_window8_256":
            num_channel = 1024

        self.fc = ClassBlock(num_channel, num_classes, num_bottleneck=512)

    def forward(self, x):

        x, intermediates_feats = self.model.forward_intermediates(x)  # Shape: (1, H1 * W1, C1)
        avg = x.mean(dim=(1, 2))
        out, retrieval_feat = self.fc(avg)

        feats = {}
        feats["pooled_feat"] = avg
        feats["feats"] = intermediates_feats
        feats["retrieval_feat"] = retrieval_feat

        return out, feats
      
    

def swin_transformer_v2_base(pretrained=False, pretrained_path="", **kwargs):

    model = Swin_Transformer_V2(pretrained, pretrained_path,  model_select="swinv2_base_window8_256", num_classes=kwargs["num_classes"])
   
    return model


def swin_transformer_v2_small(pretrained=False, pretrained_path="", **kwargs):
    
    model = Swin_Transformer_V2(pretrained, pretrained_path, model_select="swinv2_small_window8_256", num_classes=kwargs["num_classes"])
   
    return model


def swin_transformer_v2_tiny(pretrained=False, pretrained_path="", **kwargs):
    model = Swin_Transformer_V2(pretrained, pretrained_path, model_select="swinv2_tiny_window8_256", num_classes=kwargs["num_classes"])
   
    return model
