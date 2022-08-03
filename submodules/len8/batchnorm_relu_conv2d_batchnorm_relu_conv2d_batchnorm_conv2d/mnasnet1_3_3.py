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
        self.batchnorm2d21 = BatchNorm2d(336, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu14 = ReLU(inplace=True)
        self.conv2d22 = Conv2d(336, 336, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=336, bias=False)
        self.batchnorm2d22 = BatchNorm2d(336, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(336, 104, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(104, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.conv2d24 = Conv2d(104, 624, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x61):
        x62=self.batchnorm2d21(x61)
        x63=self.relu14(x62)
        x64=self.conv2d22(x63)
        x65=self.batchnorm2d22(x64)
        x66=self.relu15(x65)
        x67=self.conv2d23(x66)
        x68=self.batchnorm2d23(x67)
        x69=self.conv2d24(x68)
        return x69

m = M().eval()
x61 = torch.randn(torch.Size([1, 336, 28, 28]))
start = time.time()
output = m(x61)
end = time.time()
print(end-start)
