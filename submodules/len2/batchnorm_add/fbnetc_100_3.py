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
        self.batchnorm2d36 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x118, x110):
        x119=self.batchnorm2d36(x118)
        x120=operator.add(x119, x110)
        return x120

m = M().eval()
x118 = torch.randn(torch.Size([1, 64, 14, 14]))
x110 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x118, x110)
end = time.time()
print(end-start)