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
        self.batchnorm2d99 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu94 = ReLU(inplace=True)

    def forward(self, x327, x320):
        x328=self.batchnorm2d99(x327)
        x329=operator.add(x328, x320)
        x330=self.relu94(x329)
        return x330

m = M().eval()
x327 = torch.randn(torch.Size([1, 1024, 14, 14]))
x320 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x327, x320)
end = time.time()
print(end-start)
