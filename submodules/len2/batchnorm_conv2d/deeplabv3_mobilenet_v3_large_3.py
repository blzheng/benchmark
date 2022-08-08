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
        self.batchnorm2d32 = BatchNorm2d(112, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d41 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x120):
        x121=self.batchnorm2d32(x120)
        x122=self.conv2d41(x121)
        return x122

m = M().eval()
x120 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x120)
end = time.time()
print(end-start)
