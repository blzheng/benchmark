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
        self.batchnorm2d22 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x114, x101):
        x115=self.batchnorm2d22(x114)
        x116=operator.add(x115, x101)
        return x116

m = M().eval()
x114 = torch.randn(torch.Size([1, 48, 28, 28]))
x101 = torch.randn(torch.Size([1, 48, 28, 28]))
start = time.time()
output = m(x114, x101)
end = time.time()
print(end-start)