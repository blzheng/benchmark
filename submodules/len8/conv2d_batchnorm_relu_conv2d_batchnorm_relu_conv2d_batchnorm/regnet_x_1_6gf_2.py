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
        self.conv2d9 = Conv2d(72, 168, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(168, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(168, 168, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=7, bias=False)
        self.batchnorm2d10 = BatchNorm2d(168, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(168, 168, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(168, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x25):
        x28=self.conv2d9(x25)
        x29=self.batchnorm2d9(x28)
        x30=self.relu7(x29)
        x31=self.conv2d10(x30)
        x32=self.batchnorm2d10(x31)
        x33=self.relu8(x32)
        x34=self.conv2d11(x33)
        x35=self.batchnorm2d11(x34)
        return x35

m = M().eval()
x25 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x25)
end = time.time()
print(end-start)
