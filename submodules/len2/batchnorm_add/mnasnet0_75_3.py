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
        self.batchnorm2d32 = BatchNorm2d(72, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x93, x102):
        x94=self.batchnorm2d32(x93)
        x103=operator.add(x102, x94)
        return x103

m = M().eval()
x93 = torch.randn(torch.Size([1, 72, 14, 14]))
x102 = torch.randn(torch.Size([1, 72, 14, 14]))
start = time.time()
output = m(x93, x102)
end = time.time()
print(end-start)
