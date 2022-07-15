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
        self.batchnorm2d22 = BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x119):
        x120=self.batchnorm2d22(x119)
        return x120

m = M().eval()
x119 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x119)
end = time.time()
print(end-start)
