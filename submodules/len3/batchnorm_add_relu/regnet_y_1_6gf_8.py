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
        self.batchnorm2d23 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)

    def forward(self, x116, x103):
        x117=self.batchnorm2d23(x116)
        x118=operator.add(x103, x117)
        x119=self.relu28(x118)
        return x119

m = M().eval()
x116 = torch.randn(torch.Size([1, 120, 28, 28]))
x103 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x116, x103)
end = time.time()
print(end-start)
