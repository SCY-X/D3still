import math
import warnings

import torch
import torch.nn as nn
from .utils.class_block import ClassBlock

__all__ = ['ResNet_IBN', 'resnet18_ibn_a', 'resnet34_ibn_a', 'resnet50_ibn_a', 'resnet101_ibn_a', 'resnet152_ibn_a',
           'resnet18_ibn_b', 'resnet34_ibn_b', 'resnet50_ibn_b', 'resnet101_ibn_b', 'resnet152_ibn_b']


model_urls = {
    'resnet18_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
    'resnet34_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
    'resnet50_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'resnet101_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'resnet18_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_b-bc2f3c11.pth',
    'resnet34_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_b-04134c37.pth',
    'resnet50_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_b-9ca61e85.pth',
    'resnet101_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_b-c55f6dba.pth',
}

class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class BasicBlock_IBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(BasicBlock_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.IN = nn.InstanceNorm2d(planes, affine=True) if ibn == 'b' else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class ResNet_IBN(nn.Module):
    def __init__(self, block, layers, ibn_cfg=('a', 'a', 'a', None), last_stride=2, num_classes=1000, frozen_stages=-1):

        self.inplanes = 64
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ibn=ibn_cfg[3])
        self.avgpool = nn.AvgPool2d(7)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = ClassBlock(512 * block.expansion, num_classes, num_bottleneck=512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn,
                            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks-1) else ibn))

        return nn.Sequential(*layers)
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            print('layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        x = self.avgpool(feat4)
        x = x.view(x.size(0), -1)
        avg = x
        out, retrieval_feat = self.fc(x)

        feats = {}
        feats["pooled_feat"] = avg
        feats["feats"] = [feat1, feat2, feat3, feat4]
        feats["retrieval_feat"] = retrieval_feat

        return out, feats




def resnet18_ibn_a(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-18-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[2, 2, 2, 2],
                       ibn_cfg=('a', 'a', 'a', None), last_stride=kwargs["last_stride"],
                       num_classes=kwargs["num_classes"])
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet18_ibn_a'])
        
        for key, parm in pretrained_dict.items():
            if key.startswith("fc"):
                continue
            else:
                model.state_dict()[key].copy_(parm)
        
    return model



def resnet34_ibn_a(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-34-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('a', 'a', 'a', None), last_stride=kwargs["last_stride"],
                       num_classes=kwargs["num_classes"])
    
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet34_ibn_a'])
        
        for key, parm in pretrained_dict.items():
            if key.startswith("fc"):
                continue
            else:
                model.state_dict()[key].copy_(parm)
    return model


def resnet50_ibn_a(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-50-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('a', 'a', 'a', None), last_stride=kwargs["last_stride"],
                       num_classes=kwargs["num_classes"])
    
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_a'])
        
        for key, parm in pretrained_dict.items():
            if key.startswith("fc"):
                continue
            else:
                model.state_dict()[key].copy_(parm)
    return model


def resnet101_ibn_a(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-101-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 23, 3],
                       ibn_cfg=('a', 'a', 'a', None), last_stride=kwargs["last_stride"],
                       num_classes=kwargs["num_classes"])
    
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet101_ibn_a'])
        
        for key, parm in pretrained_dict.items():
            if key.startswith("fc"):
                continue
            else:
                model.state_dict()[key].copy_(parm)

    return model


def resnet152_ibn_a(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-152-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 8, 36, 3],
                       ibn_cfg=('a', 'a', 'a', None), last_stride=kwargs["last_stride"],
                       num_classes=kwargs["num_classes"])
    
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
            for key, parm in pretrained_dict.items():
                if key.startswith("fc"):
                    continue
                else:
                    model.state_dict()[key].copy_(parm)
        else:
            warnings.warn("Pretrained model not available for ResNet-152-IBN-a!")
    return model



def resnet18_ibn_b(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-18-IBN-b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[2, 2, 2, 2],
                       ibn_cfg=('b', 'b', None, None), last_stride=kwargs["last_stride"],
                       num_classes=kwargs["num_classes"])
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet18_ibn_b'])
        
        for key, parm in pretrained_dict.items():
            if key.startswith("fc"):
                continue
            else:
                model.state_dict()[key].copy_(parm)
    return model



def resnet34_ibn_b(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-34-IBN-b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('b', 'b', None, None), last_stride=kwargs["last_stride"],
                       num_classes=kwargs["num_classes"])
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet34_ibn_b'])
        
        for key, parm in pretrained_dict.items():
            if key.startswith("fc"):
                continue
            else:
                model.state_dict()[key].copy_(parm)
    return model


def resnet50_ibn_b(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-50-IBN-b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('b', 'b', None, None), last_stride=kwargs["last_stride"],
                       num_classes=kwargs["num_classes"])
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_b'])
        
        for key, parm in pretrained_dict.items():
            if key.startswith("fc"):
                continue
            else:
                model.state_dict()[key].copy_(parm)
    return model


def resnet101_ibn_b(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-101-IBN-b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 23, 3],
                       ibn_cfg=('b', 'b', None, None), last_stride=kwargs["last_stride"],
                       num_classes=kwargs["num_classes"])
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet101_ibn_b'])
        
        for key, parm in pretrained_dict.items():
            if key.startswith("fc"):
                continue
            else:
                model.state_dict()[key].copy_(parm)
    return model


def resnet152_ibn_b(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-152-IBN-b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 8, 36, 3],
                       ibn_cfg=('b', 'b', None, None), last_stride=kwargs["last_stride"],
                       num_classes=kwargs["num_classes"])
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
            for key, parm in pretrained_dict.items():
                if key.startswith("fc"):
                    continue
                else:
                    model.state_dict()[key].copy_(parm)
        else:
            warnings.warn("Pretrained model not available for ResNet-152-IBN-b!")
    return model





