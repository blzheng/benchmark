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
        self.conv2d92 = Conv2d(52, 1248, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()
        self.conv2d93 = Conv2d(1248, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x282, x279):
        x283=self.conv2d92(x282)
        x284=self.sigmoid18(x283)
        x285=operator.mul(x284, x279)
        x286=self.conv2d93(x285)
        x287=self.batchnorm2d55(x286)
        return x287

m = M().eval()
x282 = torch.randn(torch.Size([1, 52, 1, 1]))
x279 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x282, x279)
end = time.time()
print(end-start)
