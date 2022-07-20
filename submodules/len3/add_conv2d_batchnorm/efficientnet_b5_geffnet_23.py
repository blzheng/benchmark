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
        self.conv2d148 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d88 = BatchNorm2d(1824, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x440, x426):
        x441=operator.add(x440, x426)
        x442=self.conv2d148(x441)
        x443=self.batchnorm2d88(x442)
        return x443

m = M().eval()
x440 = torch.randn(torch.Size([1, 304, 7, 7]))
x426 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x440, x426)
end = time.time()
print(end-start)
