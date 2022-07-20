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
        self.batchnorm2d102 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu97 = ReLU(inplace=True)
        self.conv2d103 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x337, x330):
        x338=self.batchnorm2d102(x337)
        x339=operator.add(x338, x330)
        x340=self.relu97(x339)
        x341=self.conv2d103(x340)
        return x341

m = M().eval()
x337 = torch.randn(torch.Size([1, 1024, 14, 14]))
x330 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x337, x330)
end = time.time()
print(end-start)
