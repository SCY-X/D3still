import torch.nn as nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, num_bottleneck=256):
        super(ClassBlock, self).__init__()
        add_block1 = [nn.BatchNorm1d(input_dim), nn.ReLU(inplace=True),
                      nn.Linear(input_dim, num_bottleneck, bias=False),
                      nn.BatchNorm1d(num_bottleneck)]
        add_block = nn.Sequential(*add_block1)
        add_block.apply(weights_init_kaiming)

        classifier = nn.Linear(num_bottleneck, class_num, bias=False)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        y = self.classifier(x)
        return y, x