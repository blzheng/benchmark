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
        self.relu20 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.batchnorm2d31 = BatchNorm2d(384, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(384, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(72, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.conv2d33 = Conv2d(72, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(432, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x88):
        x89=self.relu20(x88)
        x90=self.conv2d31(x89)
        x91=self.batchnorm2d31(x90)
        x92=self.relu21(x91)
        x93=self.conv2d32(x92)
        x94=self.batchnorm2d32(x93)
        x95=self.conv2d33(x94)
        x96=self.batchnorm2d33(x95)
        return x96

m = M().eval()
x88 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x88)
end = time.time()
print(end-start)
