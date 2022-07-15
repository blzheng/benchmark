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
        self.batchnorm2d12 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x67):
        x68=self.batchnorm2d12(x67)
        return x68

m = M().eval()
x67 = torch.randn(torch.Size([1, 40, 56, 56]))
start = time.time()
output = m(x67)
end = time.time()
print(end-start)
