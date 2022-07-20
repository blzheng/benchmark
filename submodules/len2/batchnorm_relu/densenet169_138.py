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
        self.batchnorm2d138 = BatchNorm2d(1184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu138 = ReLU(inplace=True)

    def forward(self, x489):
        x490=self.batchnorm2d138(x489)
        x491=self.relu138(x490)
        return x491

m = M().eval()
x489 = torch.randn(torch.Size([1, 1184, 7, 7]))
start = time.time()
output = m(x489)
end = time.time()
print(end-start)
