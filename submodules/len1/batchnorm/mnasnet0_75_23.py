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
        self.batchnorm2d23 = BatchNorm2d(64, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x67):
        x68=self.batchnorm2d23(x67)
        return x68

m = M().eval()
x67 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x67)
end = time.time()
print(end-start)
