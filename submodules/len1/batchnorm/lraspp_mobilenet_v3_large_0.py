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
        self.batchnorm2d0 = BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x1):
        x2=self.batchnorm2d0(x1)
        return x2

m = M().eval()
x1 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x1)
end = time.time()
print(end-start)
