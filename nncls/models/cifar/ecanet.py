import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from thop import profile
from thop import clever_format

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class BasicResidualECABlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride, k_size=3):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.eca = eca_layer(out_channels, k_size)

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)

        residual = self.eca(residual)

        x = residual + shortcut

        return nn.ReLU(inplace=True)(x)

class BottleneckResidualECABlock(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride, k_size=3):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.eca = eca_layer(out_channels * self.expansion, k_size)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):

        shortcut = self.shortcut(x)

        residual = self.residual(x)
        residual = self.eca(residual)

        x = residual + shortcut

        return nn.ReLU(inplace=True)(x)

class ECAResNet(nn.Module):

    def __init__(self, block, block_num, class_num=100, k_size=[3, 3, 3, 3]):
        super().__init__()

        self.in_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(block, block_num[0], 64, 1, int(k_size[0]))
        self.stage2 = self._make_stage(block, block_num[1], 128, 2, int(k_size[1]))
        self.stage3 = self._make_stage(block, block_num[2], 256, 2, int(k_size[2]))
        self.stage4 = self._make_stage(block, block_num[3], 512, 2, int(k_size[3]))

        self.linear = nn.Linear(self.in_channels, class_num)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x


    def _make_stage(self, block, num, out_channels, stride, k_size):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, k_size))
        self.in_channels = out_channels * block.expansion

        while num - 1:
            layers.append(block(self.in_channels, out_channels, 1, k_size))
            num -= 1

        return nn.Sequential(*layers)

def ecaresnet18():
    return ECAResNet(BasicResidualECABlock, [2, 2, 2, 2])

def ecaresnet34():
    return ECAResNet(BasicResidualECABlock, [3, 4, 6, 3])

def ecaresnet50():
    return ECAResNet(BottleneckResidualECABlock, [3, 4, 6, 3])

def ecaresnet101():
    return ECAResNet(BottleneckResidualECABlock, [3, 4, 23, 3])

def ecaresnet152():
    return ECAResNet(BottleneckResidualECABlock, [3, 8, 36, 3])

if __name__ == '__main__':
    net = ecaresnet50()
    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(net, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)