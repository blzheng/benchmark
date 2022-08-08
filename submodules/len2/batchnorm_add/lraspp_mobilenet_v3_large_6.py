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
        self.batchnorm2d35 = BatchNorm2d(112, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x132, x119):
        x133=self.batchnorm2d35(x132)
        x134=operator.add(x133, x119)
        return x134

m = M().eval()
x132 = torch.randn(torch.Size([1, 112, 14, 14]))
x119 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x132, x119)
end = time.time()
print(end-start)
