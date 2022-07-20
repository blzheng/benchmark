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
        self.conv2d31 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
        self.batchnorm2d32 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x100):
        x101=self.conv2d31(x100)
        x102=self.batchnorm2d31(x101)
        x103=self.relu21(x102)
        x104=self.conv2d32(x103)
        x105=self.batchnorm2d32(x104)
        return x105

m = M().eval()
x100 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x100)
end = time.time()
print(end-start)
