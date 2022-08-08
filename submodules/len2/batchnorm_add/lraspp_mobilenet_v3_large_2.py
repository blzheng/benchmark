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
        self.batchnorm2d8 = BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x24, x17):
        x25=self.batchnorm2d8(x24)
        x26=operator.add(x25, x17)
        return x26

m = M().eval()
x24 = torch.randn(torch.Size([1, 24, 56, 56]))
x17 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x24, x17)
end = time.time()
print(end-start)
