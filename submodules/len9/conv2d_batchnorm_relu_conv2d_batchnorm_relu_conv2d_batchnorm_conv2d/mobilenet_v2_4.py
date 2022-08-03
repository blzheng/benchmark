import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d30 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu620 = ReLU6(inplace=True)
        self.conv2d31 = Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.batchnorm2d31 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu621 = ReLU6(inplace=True)
        self.conv2d32 = Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d33 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x86):
        x87=self.conv2d30(x86)
        x88=self.batchnorm2d30(x87)
        x89=self.relu620(x88)
        x90=self.conv2d31(x89)
        x91=self.batchnorm2d31(x90)
        x92=self.relu621(x91)
        x93=self.conv2d32(x92)
        x94=self.batchnorm2d32(x93)
        x95=self.conv2d33(x94)
        return x95

m = M().eval()
x86 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x86)
end = time.time()
print(end-start)
