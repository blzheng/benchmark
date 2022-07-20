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
        self.batchnorm2d60 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu76 = ReLU(inplace=True)

    def forward(self, x310, x297):
        x311=self.batchnorm2d60(x310)
        x312=operator.add(x297, x311)
        x313=self.relu76(x312)
        return x313

m = M().eval()
x310 = torch.randn(torch.Size([1, 2904, 14, 14]))
x297 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x310, x297)
end = time.time()
print(end-start)
