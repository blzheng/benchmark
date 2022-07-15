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
        self.batchnorm2d38 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x142):
        x143=self.batchnorm2d38(x142)
        return x143

m = M().eval()
x142 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x142)
end = time.time()
print(end-start)
