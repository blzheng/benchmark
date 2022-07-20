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
        self.batchnorm2d42 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU(inplace=True)

    def forward(self, x214, x201):
        x215=self.batchnorm2d42(x214)
        x216=operator.add(x201, x215)
        x217=self.relu52(x216)
        return x217

m = M().eval()
x214 = torch.randn(torch.Size([1, 896, 14, 14]))
x201 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x214, x201)
end = time.time()
print(end-start)
