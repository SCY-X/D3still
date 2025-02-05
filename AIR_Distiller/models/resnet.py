import math
import torch
from torch import nn
from .utils.class_block import ClassBlock

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], last_stride=2, num_classes=1000, frozen_stages=-1):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.frozen_stages = frozen_stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = ClassBlock(512 * block.expansion, num_classes, num_bottleneck=512)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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


    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




def resnet18(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], last_stride=kwargs["last_stride"] , num_classes=kwargs["num_classes"])
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls["resnet18"])

        for key, parm in pretrained_dict.items():
            if key.startswith("fc"):
                continue
            else:
                model.state_dict()[key].copy_(parm)
    return model


def resnet34(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], last_stride=kwargs["last_stride"] , num_classes=kwargs["num_classes"])
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls["resnet34"])
            
        for key, parm in pretrained_dict.items():
            if key.startswith("fc"):
                continue
            else:
                model.state_dict()[key].copy_(parm)
    return model


def resnet50(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], last_stride=kwargs["last_stride"] , num_classes=kwargs["num_classes"])
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls["resnet50"])
            
        for key, parm in pretrained_dict.items():
            if key.startswith("fc"):
                continue
            else:
                model.state_dict()[key].copy_(parm)

    return model


def resnet101(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], last_stride=kwargs["last_stride"] , num_classes=kwargs["num_classes"])
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls["resnet101"])
            
        for key, parm in pretrained_dict.items():
            if key.startswith("fc"):
                continue
            else:
                model.state_dict()[key].copy_(parm)

    return model


def resnet152(pretrained=False, pretrained_path="", **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3], last_stride=kwargs["last_stride"] , num_classes=kwargs["num_classes"])
    if pretrained:
        if pretrained_path != "":
            pretrained_dict = {k.replace('module.', ""): v for k, v in torch.load(pretrained_path, weights_only=True).items()}
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls["resnet152"])
            
        for key, parm in pretrained_dict.items():
            if key.startswith("fc"):
                continue
            else:
                model.state_dict()[key].copy_(parm)

    return model