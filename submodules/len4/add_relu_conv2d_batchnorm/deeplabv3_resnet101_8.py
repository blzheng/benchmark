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
        self.relu25 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x100, x92):
        x101=operator.add(x100, x92)
        x102=self.relu25(x101)
        x103=self.conv2d31(x102)
        x104=self.batchnorm2d31(x103)
        return x104

m = M().eval()
x100 = torch.randn(torch.Size([1, 1024, 28, 28]))
x92 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x100, x92)
end = time.time()
print(end-start)