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
        self.conv2d157 = Conv2d(1200, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d93 = BatchNorm2d(344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x463, x468):
        x469=operator.mul(x463, x468)
        x470=self.conv2d157(x469)
        x471=self.batchnorm2d93(x470)
        return x471

m = M().eval()
x463 = torch.randn(torch.Size([1, 1200, 7, 7]))
x468 = torch.randn(torch.Size([1, 1200, 1, 1]))
start = time.time()
output = m(x463, x468)
end = time.time()
print(end-start)
