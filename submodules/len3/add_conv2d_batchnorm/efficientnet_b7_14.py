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
        self.conv2d87 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x272, x257):
        x273=operator.add(x272, x257)
        x274=self.conv2d87(x273)
        x275=self.batchnorm2d51(x274)
        return x275

m = M().eval()
x272 = torch.randn(torch.Size([1, 80, 28, 28]))
x257 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x272, x257)
end = time.time()
print(end-start)
