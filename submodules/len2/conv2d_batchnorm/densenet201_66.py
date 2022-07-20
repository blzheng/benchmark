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
        self.conv2d133 = Conv2d(1760, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d134 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x472):
        x473=self.conv2d133(x472)
        x474=self.batchnorm2d134(x473)
        return x474

m = M().eval()
x472 = torch.randn(torch.Size([1, 1760, 14, 14]))
start = time.time()
output = m(x472)
end = time.time()
print(end-start)
