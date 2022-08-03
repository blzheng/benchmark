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
        self.conv2d9 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu66 = ReLU6(inplace=True)
        self.conv2d10 = Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
        self.batchnorm2d10 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu67 = ReLU6(inplace=True)
        self.conv2d11 = Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d12 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x25):
        x26=self.conv2d9(x25)
        x27=self.batchnorm2d9(x26)
        x28=self.relu66(x27)
        x29=self.conv2d10(x28)
        x30=self.batchnorm2d10(x29)
        x31=self.relu67(x30)
        x32=self.conv2d11(x31)
        x33=self.batchnorm2d11(x32)
        x34=self.conv2d12(x33)
        return x34

m = M().eval()
x25 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x25)
end = time.time()
print(end-start)
