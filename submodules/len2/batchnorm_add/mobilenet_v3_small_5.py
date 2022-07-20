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
        self.batchnorm2d32 = BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x146, x133):
        x147=self.batchnorm2d32(x146)
        x148=operator.add(x147, x133)
        return x148

m = M().eval()
x146 = torch.randn(torch.Size([1, 96, 7, 7]))
x133 = torch.randn(torch.Size([1, 96, 7, 7]))
start = time.time()
output = m(x146, x133)
end = time.time()
print(end-start)
