import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        for i in self.bn3.parameters():
            i.requires_grad = False
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
    def __init__(self, block, layers):
        self.freeze_bn = True
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        tmp_x = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_x.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        tmp_x.append(x)
        x = self.layer2(x)
        tmp_x.append(x)
        x = self.layer3(x)
        tmp_x.append(x)
        x = self.layer4(x)
        tmp_x.append(x)

        return tmp_x


class ResNet_PPM(nn.Module):
    def __init__(self, block, layers):
        super(ResNet_PPM,self).__init__()
        self.resnet = ResNet(block, layers)

        self.in_planes = 256

        self.ppm_pre = nn.Sequential(
            nn.Conv2d(2048, self.in_planes, 1, 1, bias=False),
            nn.GroupNorm(32, self.in_planes),
        )
        ppms = []
        for ii in [1, 3, 5]:
            ppms.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ii), 
                nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False), 
                nn.GroupNorm(32, self.in_planes),
                ))
        self.ppms = nn.ModuleList(ppms)

        self.ppm_cat = nn.Sequential(
            nn.Conv2d(self.in_planes, self.in_planes, 3, 1, 1, bias=False),
            nn.GroupNorm(32, self.in_planes),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.resnet(x)

        x_pre = self.ppm_pre(x[-1])
        x_ppm = x_pre
        for k in range(len(self.ppms)):
            x_ppm = torch.add(x_ppm, F.interpolate(self.ppms[k](x_pre), x_pre.size()[2:], mode='bilinear', align_corners=True))
        x_ppm = self.ppm_cat(self.relu(x_ppm))
        x.append(x_ppm)

        return x

def resnet50_ppm():
    model = ResNet_PPM(Bottleneck, [3, 4, 6, 3])
    return model
