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
        self.conv2d30 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x96, x90):
        x97=self.conv2d30(x96)
        x98=self.batchnorm2d30(x97)
        x99=operator.add(x98, x90)
        x100=self.relu25(x99)
        x101=self.conv2d31(x100)
        x102=self.batchnorm2d31(x101)
        return x102

m = M().eval()
x96 = torch.randn(torch.Size([1, 512, 14, 14]))
x90 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x96, x90)
end = time.time()
print(end-start)
