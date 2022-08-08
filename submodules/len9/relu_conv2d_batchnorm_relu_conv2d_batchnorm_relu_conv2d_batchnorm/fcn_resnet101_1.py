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
        self.relu4 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d10 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x27):
        x28=self.relu4(x27)
        x29=self.conv2d8(x28)
        x30=self.batchnorm2d8(x29)
        x31=self.relu7(x30)
        x32=self.conv2d9(x31)
        x33=self.batchnorm2d9(x32)
        x34=self.relu7(x33)
        x35=self.conv2d10(x34)
        x36=self.batchnorm2d10(x35)
        return x36

m = M().eval()
x27 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x27)
end = time.time()
print(end-start)
