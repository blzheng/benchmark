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
        self.conv2d89 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x272, x257):
        x273=operator.add(x272, x257)
        x274=self.conv2d89(x273)
        x275=self.batchnorm2d53(x274)
        return x275

m = M().eval()
x272 = torch.randn(torch.Size([1, 192, 7, 7]))
x257 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x272, x257)
end = time.time()
print(end-start)
