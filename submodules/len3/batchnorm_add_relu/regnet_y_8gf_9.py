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
        self.batchnorm2d24 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)

    def forward(self, x118, x105):
        x119=self.batchnorm2d24(x118)
        x120=operator.add(x105, x119)
        x121=self.relu28(x120)
        return x121

m = M().eval()
x118 = torch.randn(torch.Size([1, 896, 14, 14]))
x105 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x118, x105)
end = time.time()
print(end-start)
