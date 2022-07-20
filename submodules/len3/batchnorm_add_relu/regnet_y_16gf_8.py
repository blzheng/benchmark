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
        self.batchnorm2d21 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)

    def forward(self, x104, x119):
        x105=self.batchnorm2d21(x104)
        x120=operator.add(x105, x119)
        x121=self.relu28(x120)
        return x121

m = M().eval()
x104 = torch.randn(torch.Size([1, 1232, 14, 14]))
x119 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x104, x119)
end = time.time()
print(end-start)
