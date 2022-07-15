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
        self.batchnorm2d10 = BatchNorm2d(48, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x29):
        x30=self.batchnorm2d10(x29)
        return x30

m = M().eval()
x29 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x29)
end = time.time()
print(end-start)
